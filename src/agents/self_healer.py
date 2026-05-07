# src/agents/self_healer.py
# Node 4: Self-Healer — Universal Edition
#
# Changes from original:
#   - Removed import of VALID_ORDER_STATUSES — no longer used
#   - LLM prompt: replaced "replace_invalid_status" with domain-neutral
#     "replace_invalid_categorical" + always passes valid_values/default_value
#   - Deterministic fallback: uses semantic_types from state to choose the
#     correct action (abs() only for amount-semantics; categorical fix
#     reads value_set from the failing expectation kwargs; out_of_bounds
#     passes actual min/max from the expectation to mcp_tools)
#   - healing_history: cumulative per-iteration record added to state;
#     repeated zero-effect (col, action_type) pairs are skipped
#   - repair_confidence: 0.0–1.0 score per column action, persisted in state
#   - LLM plan validation: every LLM-proposed action is screened against
#     ALLOWED_ACTION_TYPES before execution; unknown actions are replaced
#     by the deterministic fallback for that failure type

import json
import logging

from src.config import OLLAMA_MODEL
from src.agents.state import AgentState
from src.agents.profiler_agent import call_ollama, extract_json
from src.mcp_tools import apply_healing_action, send_alert

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Canonical GE type synonym map — kept here so the healer normalises
# LLM-emitted failure_type strings the same way rule_generator does.
GE_NORMALIZATION_MAP = {
    "within_range":    "expect_column_values_to_be_between",
    "unique":          "expect_column_values_to_be_unique",
    "not_null":        "expect_column_values_to_not_be_null",
    "no_nulls":        "expect_column_values_to_not_be_null",
    "before_or_equal_to": "expect_column_values_to_be_between",
    "in_set":          "expect_column_values_to_be_in_set",
}

# Whitelist of action_types the healer may actually execute.
# Any LLM output outside this set is replaced by the deterministic fallback.
ALLOWED_ACTION_TYPES = {"quarantine", "fix_value", "deduplicate"}

# Base repair confidence scores by action type + operation.
# These reflect how safe/reversible each action is.
REPAIR_CONFIDENCE = {
    ("quarantine",  "is_null"):        0.99,
    ("quarantine",  "is_duplicate"):   0.97,
    ("quarantine",  "is_future_date"): 0.92,
    ("quarantine",  "out_of_bounds"):  0.75,
    ("fix_value",   "abs"):            0.82,
    ("fix_value",   "replace_invalid_categorical"): 0.70,
    ("fix_value",   "replace_invalid_status"):      0.70,  # legacy alias
    ("fix_value",   "replace_empty_string"):        0.65,
    ("deduplicate", None):             0.95,
}

# Keywords that indicate a numeric column should NEVER have abs() applied
# (i.e., negative values are semantically valid, e.g. balance, delta, temp)
SIGNED_AMOUNT_KEYWORDS = (
    "balance", "delta", "diff", "change", "variance",
    "temperature", "temp", "offset", "gain", "loss",
    "latitude", "longitude", "lat", "lon", "lng",
    "profit_loss", "return",
)

# Keywords that indicate a column's negative values are errors
UNSIGNED_AMOUNT_KEYWORDS = (
    "price", "amount", "cost", "fee", "charge", "revenue",
    "salary", "wage", "total", "sum", "payment", "earnings",
    "spend", "budget", "quantity", "qty", "count", "distance",
    "weight", "height", "age",
)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _should_apply_abs(col: str, sem_type: str) -> bool:
    """
    Decide whether abs() is a safe healing operation for a given column.

    abs() is ONLY safe for columns where negative values are definitively
    wrong (prices, counts, quantities). It must NOT be applied to columns
    where negative values are semantically valid (balance, temperature, lat/lon).
    """
    col_lower = col.lower()

    # Explicitly signed — never apply abs
    if any(kw in col_lower for kw in SIGNED_AMOUNT_KEYWORDS):
        return False

    # Explicitly unsigned — safe to apply abs
    if any(kw in col_lower for kw in UNSIGNED_AMOUNT_KEYWORDS):
        return True

    # Semantic type fallback
    if sem_type in ("currency", "integer", "float"):
        # For unknown numeric columns: conservative — don't abs
        return False

    return False


def _compute_repair_confidence(action_type: str, operation: str | None) -> float:
    """Look up base confidence, fallback to 0.50 for unknown combinations."""
    return REPAIR_CONFIDENCE.get((action_type, operation), 0.50)


def _deterministic_action_for_failure(failure: dict, semantic_types: dict) -> dict | None:
    """
    Return a deterministic healing action dict for a single validation failure.

    Uses the failure's expectation_type, its kwargs (for bounds/value_set),
    and the column's semantic type to choose the correct action.

    Returns None if no sensible action can be determined.
    """
    col = failure.get("kwargs", {}).get("column")
    exp_type = failure.get("expectation_type")
    if not col:
        return None

    sem_type = semantic_types.get(col, "unknown")
    kwargs = failure.get("kwargs", {})

    if exp_type == "expect_column_values_to_not_be_null":
        return {
            "action_type": "quarantine",
            "condition": "is_null",
            "reason": f"Missing required value in '{col}' (null check failed)",
        }

    elif exp_type == "expect_column_values_to_be_between":
        # Route date vs numeric
        if sem_type == "datetime" or "date" in col.lower() or "time" in col.lower():
            return {
                "action_type": "quarantine",
                "condition": "is_future_date",
                "reason": f"Invalid date in '{col}' (out of valid range)",
            }
        else:
            # Check whether abs() is a safer fix than quarantine
            if _should_apply_abs(col, sem_type):
                return {
                    "action_type": "fix_value",
                    "operation": "abs",
                    "ensure_numeric": True,
                    "reason": f"Negative values in '{col}' are definitively wrong — applying abs()",
                }
            else:
                # Pass actual bounds from the expectation kwargs
                return {
                    "action_type": "quarantine",
                    "condition": "out_of_bounds",
                    "min_bound": kwargs.get("min_value"),
                    "max_bound": kwargs.get("max_value"),
                    "reason": (
                        f"'{col}' values outside expected range "
                        f"[{kwargs.get('min_value')}, {kwargs.get('max_value')}]"
                    ),
                }

    elif exp_type == "expect_column_values_to_be_unique":
        return {"action_type": "deduplicate"}

    elif exp_type == "expect_column_values_to_be_in_set":
        # Read valid set from the expectation itself — NOT from hardcoded config
        value_set = kwargs.get("value_set", [])
        if value_set:
            return {
                "action_type": "fix_value",
                "operation": "replace_invalid_categorical",
                "valid_values": value_set,
                "default_value": value_set[0],
                "reason": (
                    f"Invalid categorical value in '{col}' — "
                    f"replacing with default '{value_set[0]}'"
                ),
            }
        else:
            # No valid set available → quarantine
            return {
                "action_type": "quarantine",
                "condition": "is_null",
                "reason": (
                    f"Invalid categorical value in '{col}' — "
                    "no value_set available, quarantining"
                ),
            }

    elif exp_type == "expect_column_values_to_match_regex":
        # Quarantine rows that fail regex validation
        regex = kwargs.get("regex", "")
        return {
            "action_type": "quarantine",
            "condition": "fails_regex",
            "regex": regex,
            "reason": f"Value in '{col}' failed regex validation — quarantining non-matching rows",
        }

    else:
        # Unknown expectation type → safe quarantine
        return {
            "action_type": "quarantine",
            "condition": "is_null",
            "reason": f"Unhandled failure type '{exp_type}' on '{col}' — quarantining",
        }


def _validate_llm_action(action: dict, failure: dict, semantic_types: dict) -> dict:
    """
    Screen an LLM-proposed action against the allowed menu.

    If the action_type is unknown or the operation is unknown,
    replace it with the deterministic action for this failure.
    This prevents hallucinated action types from reaching apply_healing_action.
    """
    action_type = action.get("action_type")

    if action_type not in ALLOWED_ACTION_TYPES:
        logger.warning(
            f"[SELF-HEALER] LLM proposed unknown action_type '{action_type}' — "
            "replacing with deterministic fallback"
        )
        return _deterministic_action_for_failure(failure, semantic_types) or action

    # For fix_value: validate the operation
    if action_type == "fix_value":
        op = action.get("operation")
        valid_ops = {"abs", "replace_invalid_categorical",
                     "replace_invalid_status", "replace_empty_string"}
        if op not in valid_ops:
            logger.warning(
                f"[SELF-HEALER] LLM proposed unknown operation '{op}' — "
                "replacing with deterministic fallback"
            )
            return _deterministic_action_for_failure(failure, semantic_types) or action

        # replace_invalid_categorical is only valid for be_in_set failures.
        # For match_regex failures (e.g. email), it destroys valid data.
        exp_type = failure.get("expectation_type", "")
        if op in ("replace_invalid_categorical", "replace_invalid_status"):
            if exp_type != "expect_column_values_to_be_in_set":
                logger.warning(
                    f"[SELF-HEALER] replace_invalid_categorical invalid for "
                    f"'{exp_type}' — replacing with deterministic fallback"
                )
                return _deterministic_action_for_failure(failure, semantic_types) or action

    # For quarantine actions on match_regex / be_between failures,
    # the LLM consistently picks wrong conditions (not_matches_regex
    # without a regex, is_future_date on non-date columns).
    # Always use the deterministic fallback which has correct logic.
    exp_type = failure.get("expectation_type", "")
    if action_type == "quarantine" and exp_type in (
        "expect_column_values_to_match_regex",
        "expect_column_values_to_be_between",
    ):
        det = _deterministic_action_for_failure(failure, semantic_types)
        if det:
            logger.info(
                f"[SELF-HEALER] Overriding LLM quarantine for '{exp_type}' "
                f"with deterministic fallback (condition: {det.get('condition')})"
            )
            return det

    return action


# =============================================================================
# SELF-HEALER NODE
# =============================================================================

def self_heal(state: AgentState) -> AgentState:
    """
    LangGraph Node 4: Self-heal data quality failures.

    Design:
      1. Normalize failure expectation types (synonym map)
      2. Call LLM for a high-level healing strategy
      3. Validate every LLM action against the allowed menu
      4. For zero-output LLM or post-screen failures: activate deterministic fallback
      5. Skip (col, action_type) pairs already tried with 0 rows affected
      6. Execute each action via apply_healing_action MCP tool
      7. Record healing_history and repair_confidence in state
    """
    logger.info("=" * 60)
    logger.info("🔧 [SELF-HEALER] Analyzing failures and generating fix plan...")
    logger.info("=" * 60)

    validation_result = state.get("validation_result", {})
    failures = validation_result.get("failures", [])
    iteration = state.get("iteration", 0) + 1
    csv_path = state.get("csv_path")
    semantic_types = state.get("semantic_types", {})
    healing_history = list(state.get("healing_history", []))  # mutable copy
    repair_confidence = dict(state.get("repair_confidence", {}))

    logger.info(f"[SELF-HEALER] Iteration {iteration}: {len(failures)} failure(s)")

    # ── Step 1: Normalise GE synonym names in failure list ────────────────────
    for f in failures:
        original = f.get("expectation_type", "")
        normalised = GE_NORMALIZATION_MAP.get(original, original)
        if normalised != original:
            logger.debug(f"[SELF-HEALER] Normalised expectation type: {original} → {normalised}")
        f["expectation_type"] = normalised

    # ── Step 2: Build set of already-tried zero-effect (col, action) pairs ───
    already_tried_zero = {
        (h["column"], h["action_type"])
        for h in healing_history
        if h.get("rows_affected", 0) == 0
    }
    if already_tried_zero:
        logger.info(
            f"[SELF-HEALER] Skipping previously zero-effect actions: {already_tried_zero}"
        )

    # ── Step 3: LLM healing strategy ─────────────────────────────────────────
    # Build a compact failure summary for the LLM (no raw indices)
    llm_failures = [
        {
            "column": f.get("kwargs", {}).get("column"),
            "expectation_type": f.get("expectation_type"),
            "failing_count": f.get("failing_count", 0),
            "kwargs": {
                k: v for k, v in f.get("kwargs", {}).items()
                if k != "column"
            },
        }
        for f in failures
    ]

    system_prompt = """\
You are an automated data quality engineer.
Your ONLY job is to receive validation failures and return a JSON healing plan.

You MUST choose actions ONLY from this allowed menu:

ALLOWED ACTIONS:
  {"action_type": "quarantine", "condition": "is_null"}
  {"action_type": "quarantine", "condition": "is_future_date"}
  {"action_type": "quarantine", "condition": "out_of_bounds",
   "min_bound": <number_or_null>, "max_bound": <number_or_null>}
  {"action_type": "fix_value", "operation": "abs"}
  {"action_type": "fix_value", "operation": "replace_invalid_categorical",
   "valid_values": [...], "default_value": "..."}
  {"action_type": "deduplicate"}

RULES:
- For expect_column_values_to_be_between on date columns: use is_future_date
- For expect_column_values_to_be_between on numeric columns:
    * If column name suggests amounts/prices/counts (never negative): use abs
    * Otherwise: use out_of_bounds with min_bound/max_bound from the kwargs
- For expect_column_values_to_be_in_set: use replace_invalid_categorical,
  setting valid_values to the value_set from the failure kwargs
- For expect_column_values_to_be_unique: use deduplicate
- For expect_column_values_to_not_be_null: use quarantine/is_null
- For out_of_bounds: always set min_bound and max_bound from the
  failure's kwargs min_value and max_value

Return ONLY valid JSON in this exact structure:
{
  "strategy_summary": "...",
  "healing_plan": [
    {
      "column": "...",
      "failure_type": "...",
      "action": { <one action from the menu above> },
      "reasoning": "..."
    }
  ]
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": json.dumps(llm_failures, indent=2)},
    ]

    logger.info(f"[SELF-HEALER] Calling Ollama ({OLLAMA_MODEL})...")
    raw_response = call_ollama(messages)
    healing_plan_response = extract_json(raw_response)

    healing_plan = healing_plan_response.get("healing_plan", [])
    strategy_summary = healing_plan_response.get(
        "strategy_summary", "LLM returned empty strategy"
    )

    # ── Step 4: Validate each LLM action against the allowed menu ─────────────
    # Build a lookup from column → its failure for context in validation
    failure_by_col = {
        f.get("kwargs", {}).get("column"): f
        for f in failures
    }

    validated_plan = []
    for item in healing_plan:
        col = item.get("column")
        action = item.get("action", {})
        if not col or not action:
            continue
        ref_failure = failure_by_col.get(col, {})
        validated_action = _validate_llm_action(action, ref_failure, semantic_types)
        validated_plan.append({**item, "action": validated_action})

    healing_plan = validated_plan

    # ── Step 5: Deterministic fallback if LLM produced nothing usable ─────────
    if not healing_plan and failures:
        logger.warning(
            "[SELF-HEALER] LLM returned empty or entirely-invalid plan — "
            "activating deterministic fallback"
        )
        strategy_summary = "Deterministic fallback (LLM produced no valid plan)"

        for f in failures:
            col = f.get("kwargs", {}).get("column")
            exp_type = f.get("expectation_type")
            if not col:
                continue

            action = _deterministic_action_for_failure(f, semantic_types)
            if action:
                healing_plan.append({
                    "column": col,
                    "failure_type": exp_type,
                    "action": action,
                    "reasoning": f"Deterministic fallback for {exp_type} on '{col}'",
                })

    logger.info(f"[SELF-HEALER] Strategy: {strategy_summary}")
    logger.info(f"[SELF-HEALER] Healing plan ({len(healing_plan)} actions):")
    for item in healing_plan:
        act = item.get("action", {})
        logger.info(
            f"  🔧 [{item.get('column')}] "
            f"{act.get('action_type')} / {act.get('operation') or act.get('condition', '')} "
            f"— {item.get('reasoning', '')[:80]}"
        )

    # ── Step 6: Execute actions ───────────────────────────────────────────────
    executed_actions = []

    for item in healing_plan:
        action = item.get("action", {})
        column = item.get("column")

        if not action or not column:
            logger.warning(f"[SELF-HEALER] Skipping malformed plan item: {item}")
            continue

        action_type = action.get("action_type")
        operation = action.get("operation")

        if not action_type:
            logger.warning(f"[SELF-HEALER] Missing action_type — skipping: {action}")
            continue

        # ── Skip previously tried zero-effect actions ─────────────────────
        if (column, action_type) in already_tried_zero:
            logger.info(
                f"[SELF-HEALER] Skipping ({column}, {action_type}) — "
                "was tried before with 0 rows affected"
            )
            continue

        # ── Ensure column is set in the action dict ───────────────────────
        if not action.get("column"):
            action["column"] = column

        # ── Derive quarantine condition if missing ────────────────────────
        if action_type == "quarantine" and not action.get("condition"):
            failure_type = item.get("failure_type", "")
            sem_type = semantic_types.get(column, "unknown")
            if sem_type == "datetime" or "date" in column.lower():
                action["condition"] = "is_future_date"
            elif "null" in failure_type:
                action["condition"] = "is_null"
            elif "unique" in failure_type:
                action["condition"] = "is_duplicate"
            elif "between" in failure_type:
                action["condition"] = "out_of_bounds"
            else:
                action["condition"] = "is_null"

        # ── Ensure quarantine reason is set ──────────────────────────────
        if action_type == "quarantine" and not action.get("reason"):
            action["reason"] = item.get("reasoning", "Data quality failure")

        # ── Pass bounds from the expectation kwargs to out_of_bounds ─────
        if action.get("condition") == "out_of_bounds":
            ref_failure = failure_by_col.get(column, {})
            ref_kwargs = ref_failure.get("kwargs", {})
            if "min_bound" not in action and "min_value" in ref_kwargs:
                action["min_bound"] = ref_kwargs["min_value"]
            if "max_bound" not in action and "max_value" in ref_kwargs:
                action["max_bound"] = ref_kwargs["max_value"]

        # ── For categorical fix: ensure valid_values is populated ─────────
        # Bug 2 fix: ALWAYS prefer the ground-truth value_set from the
        # expectation kwargs over LLM-hallucinated values (e.g. ['...']).
        if operation in ("replace_invalid_categorical", "replace_invalid_status"):
            ref_failure = failure_by_col.get(column, {})
            value_set = ref_failure.get("kwargs", {}).get("value_set", [])
            if value_set:
                action["valid_values"] = value_set
                # Only use LLM's default_value if it's actually in the value_set
                llm_default = action.get("default_value")
                if llm_default in value_set:
                    action["default_value"] = llm_default
                else:
                    action["default_value"] = value_set[0]
            elif not action.get("valid_values"):
                # No ground-truth value_set AND LLM provided nothing → quarantine
                logger.warning(
                    f"[SELF-HEALER] No valid_values for replace_invalid_categorical "
                    f"on '{column}' — switching to quarantine"
                )
                action = {
                    "action_type": "quarantine",
                    "condition": "is_null",
                    "column": column,
                    "reason": "No valid categorical set available — quarantining",
                }
                action_type = "quarantine"

        # ── Guard abs() with semantic type check ──────────────────────────
        if operation == "abs":
            if not _should_apply_abs(column, semantic_types.get(column, "unknown")):
                # Column may have legitimately signed values — use out_of_bounds instead
                logger.info(
                    f"[SELF-HEALER] abs() rejected for '{column}' (semantic type: "
                    f"{semantic_types.get(column, 'unknown')}) — switching to out_of_bounds"
                )
                ref_failure = failure_by_col.get(column, {})
                ref_kwargs = ref_failure.get("kwargs", {})
                action = {
                    "action_type": "quarantine",
                    "condition": "out_of_bounds",
                    "column": column,
                    "min_bound": ref_kwargs.get("min_value"),
                    "max_bound": ref_kwargs.get("max_value"),
                    "reason": (
                        f"Signed numeric '{column}' — quarantining out-of-range "
                        f"values instead of applying abs()"
                    ),
                }
                action_type = "quarantine"
                operation = None
            else:
                action["ensure_numeric"] = True

        logger.info(
            f"\n[SELF-HEALER] Applying {action_type}"
            f"{'/' + str(action.get('operation') or action.get('condition', '')) }"
            f" on '{column}'"
        )

        # ── Execute ───────────────────────────────────────────────────────
        # Pass semantic type so apply_healing_action can guard conditions
        action["semantic_type"] = semantic_types.get(column, "unknown")
        try:
            result = apply_healing_action(action, csv_path)

            rows_affected = result.get("rows_affected", 0)
            conf_key = action.get("operation") or action.get("condition")
            confidence = _compute_repair_confidence(action_type, conf_key)

            executed_actions.append({
                "action": action,
                "result": result,
                "reasoning": item.get("reasoning", ""),
                "confidence": confidence,
            })

            # Update cumulative repair_confidence (keep highest seen per column)
            existing_conf = repair_confidence.get(column, 0.0)
            repair_confidence[column] = max(existing_conf, confidence)

            # Record in healing_history
            healing_history.append({
                "iteration": iteration,
                "column": column,
                "action_type": action_type,
                "operation_or_condition": action.get("operation") or action.get("condition"),
                "rows_affected": rows_affected,
                "success": result.get("success", False),
                "confidence": confidence,
            })

            logger.info(
                f"[SELF-HEALER] ✅ Applied: {rows_affected} rows affected, "
                f"{result.get('remaining_rows', 'N/A')} remaining "
                f"(confidence: {confidence:.0%})"
            )

        except Exception as e:
            logger.error(f"[SELF-HEALER] ❌ Failed to apply action: {e}", exc_info=True)
            healing_history.append({
                "iteration": iteration,
                "column": column,
                "action_type": action_type,
                "rows_affected": 0,
                "success": False,
                "error": str(e),
            })
            continue

    # ── Step 7: Update state ──────────────────────────────────────────────────
    messages_log = state.get("messages", [])
    messages_log.append({
        "agent": "self_healer",
        "step": f"healing_iteration_{iteration}",
        "content": (
            f"Iteration {iteration}: applied {len(executed_actions)} healing actions. "
            f"Strategy: {strategy_summary}"
        ),
    })

    logger.info("\n[SELF-HEALER] ✅ Healing complete. Returning to Validator...")

    return {
        **state,
        "healing_actions":  executed_actions,
        "healing_history":  healing_history,
        "repair_confidence": repair_confidence,
        "iteration":        iteration,
        "messages":         messages_log,
    }