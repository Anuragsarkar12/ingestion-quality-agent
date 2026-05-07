# src/agents/rule_generator.py
# Node 2: Rule Generator — Universal Edition
#
# Changes from original:
#   - sanitize_rules(): removed ALL hardcoded column-name overrides
#     (no more "order_date", "status", "order_amount" special-cases)
#   - Uses semantic_types from state for type-appropriate guardrails
#   - Uses profile-inferred bounds instead of hardcoded 0–50000
#   - Rejects LLM-invented expectation types not in the known GE vocabulary
#   - Generates rules for ALL columns (not just WARNING/CRITICAL)
#   - _truncate_profile_for_llm(): prevents token explosion on wide datasets
#   - _generate_deterministic_base_rules(): covers columns excluded from LLM pass
#   - Domain-neutral LLM prompts

import json
import logging

from src.config import OLLAMA_MODEL, GE_SUITE_NAME
from src.agents.state import AgentState
from src.agents.profiler_agent import call_ollama, extract_json
from src.mcp_tools import save_expectation_suite

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Exhaustive list of GE expectation types this system supports.
# Any LLM-invented type outside this list is rejected.
VALID_GE_EXPECTATION_TYPES = {
    "expect_column_values_to_not_be_null",
    "expect_column_values_to_be_unique",
    "expect_column_values_to_be_between",
    "expect_column_values_to_be_in_set",
    "expect_column_values_to_match_regex",
    "expect_column_values_to_be_of_type",
    "expect_column_values_to_be_dateutil_parseable",
}

# Normalize common LLM synonym names → canonical GE type
GE_TYPE_NORMALIZATION_MAP = {
    "within_range": "expect_column_values_to_be_between",
    "unique": "expect_column_values_to_be_unique",
    "not_null": "expect_column_values_to_not_be_null",
    "no_nulls": "expect_column_values_to_not_be_null",
    "before_or_equal_to": "expect_column_values_to_be_between",
    "in_set": "expect_column_values_to_be_in_set",
    "match_regex": "expect_column_values_to_match_regex",
    "is_unique": "expect_column_values_to_be_unique",
    "expect_column_values_to_be_valid_email": "expect_column_values_to_match_regex",
}


# =============================================================================
# HELPERS
# =============================================================================

def _summarize_column_stats(stats: dict) -> dict:
    """
    Return a compact stats dict safe for LLM prompts.
    Omits large fields (top_values full dict) to control token count.
    """
    compact = {k: v for k, v in stats.items()
               if k in ("dtype", "null_pct", "unique_pct", "min", "max",
                        "negative_count", "future_date_pct",
                        "valid_email_count", "invalid_email_count",
                        "has_empty_strings")}
    # Include only the keys of top_values (not counts) to save tokens
    if "top_values" in stats:
        compact["top_value_keys"] = list(stats["top_values"].keys())[:5]
    return compact


def _truncate_profile_for_llm(column_context: dict, max_cols: int = 20) -> dict:
    """
    For wide datasets, select the most important columns for LLM rule generation.

    Priority:
      1. CRITICAL/WARNING columns
      2. Columns with complex semantic types (email, currency, datetime, categorical)
      3. Identifier columns (need uniqueness rules)
      4. Remaining columns up to max_cols

    Omitted columns receive deterministic base rules from
    _generate_deterministic_base_rules() without LLM involvement.
    """
    if len(column_context) <= max_cols:
        return column_context

    PRIORITY_TYPES = {"email", "currency", "datetime", "categorical", "identifier"}
    PRIORITY_STATUSES = {"CRITICAL", "WARNING"}

    priority_cols = {
        col: ctx for col, ctx in column_context.items()
        if (ctx.get("health_status") in PRIORITY_STATUSES or
            ctx.get("semantic_type") in PRIORITY_TYPES)
    }
    normal_cols = {
        col: ctx for col, ctx in column_context.items()
        if col not in priority_cols
    }

    result = dict(list(priority_cols.items())[:max_cols])
    remaining = max_cols - len(result)
    if remaining > 0:
        result.update(dict(list(normal_cols.items())[:remaining]))

    omitted = set(column_context.keys()) - set(result.keys())
    if omitted:
        logger.info(
            f"[RULE GEN] Wide dataset: truncated to {max_cols} cols for LLM. "
            f"{len(omitted)} low-priority columns will receive deterministic rules: {omitted}"
        )
    return result


def _generate_deterministic_base_rules(columns: list, semantic_types: dict,
                                        profile: dict) -> list:
    """
    Generate minimal deterministic rules for columns excluded from the LLM pass.

    Every column gets:
      - not_null  (if null% < 5%)
      - unique    (if identifier type)
      - between   (if numeric, using observed bounds × 1.5 headroom)
    """
    rules = []
    raw_stats = profile.get("raw_stats", {})

    for col in columns:
        stats = raw_stats.get(col, {})
        null_pct = stats.get("null_pct", 0)
        sem_type = semantic_types.get(col, "unknown")

        if null_pct < 5:
            rules.append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": col},
                "reasoning": (
                    f"Deterministic: {null_pct:.1f}% null rate indicates required field"
                ),
            })

        if sem_type == "identifier":
            rules.append({
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": col},
                "reasoning": f"Deterministic: '{col}' is an identifier — must be unique",
            })

        if sem_type in ("currency", "float", "integer"):
            observed_min = stats.get("min")
            observed_max = stats.get("max")
            if observed_min is not None and observed_max is not None:
                rules.append({
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": col,
                        "min_value": min(0.0, float(observed_min)),
                        "max_value": round(float(observed_max) * 1.5, 4),
                    },
                    "reasoning": (
                        f"Deterministic: range from observed "
                        f"[{observed_min}, {observed_max}] with 50% headroom"
                    ),
                })

        if sem_type == "datetime":
            rules.append({
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": col,
                    "min_value": "1900-01-01",
                    "max_value": "today",
                },
                "reasoning": "Deterministic: date column must not be in the future or pre-1900",
            })

    return rules


# =============================================================================
# RULE SANITIZATION (DOMAIN-NEUTRAL)
# =============================================================================

def sanitize_rules(expectations: list, profile: dict, semantic_types: dict) -> list:
    """
    Sanitize and guard LLM-generated expectations.

    This function is DOMAIN-NEUTRAL — it makes no assumptions about column names.
    All guardrails are driven by:
      - semantic_types (inferred in profile_data())
      - observed statistics from profile (for bounds)
      - the VALID_GE_EXPECTATION_TYPES whitelist

    Rejects:
      - Rules for columns not in the dataset
      - Unknown expectation types (hallucinated by LLM)
      - Rules with missing required kwargs

    Repairs:
      - Empty value_set (filled from profile top_values)
      - Missing bounds (filled from observed min/max)
      - Empty reasoning (filled with semantic context)
    """
    cleaned = []
    existing_columns = set(profile.get("raw_stats", {}).keys())

    for exp in expectations:
        exp_type = exp.get("expectation_type", "")
        kwargs = exp.get("kwargs", {})
        col = kwargs.get("column")

        # ── Guard 1: must have column and type ──────────────────────────────
        if not exp_type or not col:
            logger.debug(f"[RULE GEN] Skipping rule — missing type or column: {exp}")
            continue

        # ── Guard 2: column must exist in dataset ────────────────────────────
        if col not in existing_columns:
            logger.warning(f"[RULE GEN] Skipping rule — column '{col}' not in dataset")
            continue

        # ── Guard 3: normalize synonym names ────────────────────────────────
        exp_type = GE_TYPE_NORMALIZATION_MAP.get(exp_type, exp_type)
        exp["expectation_type"] = exp_type

        # ── Guard 4: whitelist check ─────────────────────────────────────────
        if exp_type not in VALID_GE_EXPECTATION_TYPES:
            logger.warning(
                f"[RULE GEN] Rejecting unknown expectation type '{exp_type}' "
                f"on '{col}' — not in supported vocabulary"
            )
            continue

        # ── Guard 5: Semantic-aware kwargs repair ────────────────────────────
        sem_type = semantic_types.get(col, "unknown")
        col_stats = profile.get("raw_stats", {}).get(col, {})

        if exp_type == "expect_column_values_to_be_between":
            if sem_type == "datetime" or "date" in col.lower() or "time" in col.lower():
                # Date columns: use string date bounds
                kwargs.setdefault("min_value", "1900-01-01")
                kwargs.setdefault("max_value", "today")
            elif sem_type in ("currency", "integer", "float"):
                # Numeric columns: use profile-inferred bounds (NOT hardcoded)
                observed_min = col_stats.get("min")
                observed_max = col_stats.get("max")
                if "min_value" not in kwargs and observed_min is not None:
                    kwargs["min_value"] = min(0.0, float(observed_min))
                if "max_value" not in kwargs and observed_max is not None:
                    kwargs["max_value"] = round(float(observed_max) * 1.5, 4)

        if exp_type == "expect_column_values_to_be_in_set":
            # Numeric columns should never get be_in_set — convert to be_between
            if sem_type in ("currency", "integer", "float"):
                observed_min = col_stats.get("min")
                observed_max = col_stats.get("max")
                exp["expectation_type"] = "expect_column_values_to_be_between"
                exp_type = "expect_column_values_to_be_between"
                kwargs.pop("value_set", None)
                if observed_min is not None:
                    kwargs.setdefault("min_value", min(0.0, float(observed_min)))
                if observed_max is not None:
                    kwargs.setdefault("max_value", round(float(observed_max) * 1.5, 4))
                logger.info(
                    f"[RULE GEN] Converted be_in_set → be_between "
                    f"for numeric '{col}' (semantic type: {sem_type})"
                )
            elif not kwargs.get("value_set"):
                # Fill from profile top_values when LLM left it empty
                top_values = col_stats.get("top_values", {})
                if top_values:
                    kwargs["value_set"] = list(top_values.keys())
                    logger.info(
                        f"[RULE GEN] Auto-populated value_set for '{col}' "
                        f"from profile: {kwargs['value_set']}"
                    )
                else:
                    logger.warning(
                        f"[RULE GEN] Skipping in_set rule for '{col}' — "
                        "empty value_set and no top_values in profile"
                    )
                    continue

        if exp_type == "expect_column_values_to_match_regex":
            # Bug 3 fix: LLM sometimes picks regex for categorical columns.
            # Convert to be_in_set using profile top_values.
            if sem_type == "categorical":
                top_values = col_stats.get("top_values", {})
                if top_values:
                    exp["expectation_type"] = "expect_column_values_to_be_in_set"
                    exp_type = "expect_column_values_to_be_in_set"
                    kwargs["value_set"] = list(top_values.keys())
                    kwargs.pop("regex", None)
                    logger.info(
                        f"[RULE GEN] Converted match_regex → be_in_set "
                        f"for categorical '{col}': {kwargs['value_set']}"
                    )
                else:
                    logger.warning(
                        f"[RULE GEN] Skipping regex rule for categorical "
                        f"'{col}' — no top_values in profile"
                    )
                    continue
            # Email regex: use standard pattern; don't rely on LLM regex
            elif sem_type == "email":
                kwargs["regex"] = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

        # ── Guard 6: fill empty reasoning ───────────────────────────────────
        if not exp.get("reasoning"):
            exp["reasoning"] = (
                f"Auto-generated rule for '{col}' "
                f"(semantic type: {sem_type}, check: {exp_type})"
            )

        exp["kwargs"] = kwargs
        cleaned.append(exp)

    return cleaned


# =============================================================================
# RULE GENERATOR NODE
# =============================================================================

def generate_rules(state: AgentState) -> AgentState:
    """
    LangGraph Node 2: Generate validation rules using LLM + deterministic guards.

    Generates rules for ALL columns:
      - LLM generates rules for up to 20 highest-priority columns
      - Deterministic rules cover remaining columns
      - sanitize_rules() validates and repairs all LLM output
    """
    logger.info("=" * 60)
    logger.info("📋 [RULE GENERATOR] Generating validation rules...")
    logger.info("=" * 60)

    profile = state.get("profile", {})
    profile_analysis = profile.get("analysis", {})
    data_info = profile.get("data_info", {})
    semantic_types = state.get("semantic_types", {})
    suite_name = state.get("suite_name", GE_SUITE_NAME)

    all_columns = data_info.get("columns", [])
    raw_stats = profile.get("raw_stats", {})

    # ── Build column context (ALL columns, not just WARNING/CRITICAL) ─────────
    column_context = {}
    for col in all_columns:
        col_analysis = profile_analysis.get("column_analyses", {}).get(col, {})
        col_stats = raw_stats.get(col, {})
        column_context[col] = {
            "health_status": col_analysis.get("health_status", "UNKNOWN"),
            "issues": col_analysis.get("issues_found", []),
            "recommended_validations": col_analysis.get("recommended_validations", []),
            "semantic_type": semantic_types.get(col, "unknown"),
            "stats_summary": _summarize_column_stats(col_stats),
        }

    # ── Truncate for LLM (up to 20 priority columns) ─────────────────────────
    llm_column_context = _truncate_profile_for_llm(column_context, max_cols=20)
    omitted_columns = [c for c in all_columns if c not in llm_column_context]

    # ── Call LLM for primary rules ────────────────────────────────────────────
    system_prompt = """\
You are a data quality engineer generating Great Expectations validation rules.
You receive a dataset profile with column statistics and semantic types.

CRITICAL RULES:
1. Respond ONLY with valid JSON. No text, no markdown.
2. Only use these expectation types:
   - expect_column_values_to_not_be_null
   - expect_column_values_to_be_unique
   - expect_column_values_to_be_between
   - expect_column_values_to_be_in_set
   - expect_column_values_to_match_regex
   - expect_column_values_to_be_of_type
3. Do NOT use regex for date validation — use expect_column_values_to_be_between
4. For date columns: min_value="1900-01-01", max_value="today"
5. For categorical columns (semantic_type=categorical):
   use expect_column_values_to_be_in_set with the top_value_keys as value_set
6. For identifier columns: add expect_column_values_to_be_unique
7. For email columns: add expect_column_values_to_not_be_null and
   expect_column_values_to_match_regex with regex="^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"
8. For currency/integer/float columns with negative values flagged as issues:
   add expect_column_values_to_be_between with min_value=0

Output schema:
{
  "suite_name": "dynamic_quality_suite",
  "expectations": [
    {
      "expectation_type": "...",
      "kwargs": {"column": "...", ...},
      "reasoning": "..."
    }
  ]
}"""

    user_prompt = (
        f"Dataset: {data_info.get('row_count', 0)} rows, "
        f"{data_info.get('column_count', 0)} columns\n"
        f"Columns: {list(llm_column_context.keys())}\n\n"
        f"Column analysis:\n{json.dumps(llm_column_context, indent=2)}\n\n"
        "Generate validation rules for ALL columns listed above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = call_ollama(messages)
    try:
        parsed = extract_json(raw_response)
    except Exception as e:
        logger.error(f"[RULE GEN] JSON parse error: {e}")
        parsed = {}

    if not parsed:
        parsed = {"suite_name": suite_name, "expectations": []}

    llm_expectations = parsed.get("expectations", [])

    # ── Sanitize LLM rules ────────────────────────────────────────────────────
    cleaned_llm_rules = sanitize_rules(llm_expectations, profile, semantic_types)

    # ── Generate deterministic rules for omitted columns ─────────────────────
    deterministic_rules = _generate_deterministic_base_rules(
        omitted_columns, semantic_types, profile
    )

    # ── Merge: LLM rules take precedence; deterministic fills gaps ─────────────
    all_expectations = cleaned_llm_rules + deterministic_rules

    # De-duplicate: same (expectation_type, column) combo → keep first
    seen = set()
    deduped = []
    for exp in all_expectations:
        key = (exp["expectation_type"], exp.get("kwargs", {}).get("column"))
        if key not in seen:
            seen.add(key)
            deduped.append(exp)

    expectation_suite = {
        "suite_name": suite_name,
        "expectations": deduped,
    }

    # ── Log generated rules ───────────────────────────────────────────────────
    logger.info(
        f"[RULE GEN] Generated {len(deduped)} rules "
        f"({len(cleaned_llm_rules)} from LLM, "
        f"{len(deterministic_rules)} deterministic):"
    )
    for exp in deduped:
        col = exp.get("kwargs", {}).get("column", "N/A")
        logger.info(f"  📏 [{col}] {exp['expectation_type']}")
        logger.info(f"      Reason: {exp.get('reasoning', 'N/A')}")

    # ── Save suite to disk ────────────────────────────────────────────────────
    save_expectation_suite(expectation_suite, suite_name)

    messages_log = state.get("messages", [])
    messages_log.append({
        "agent": "rule_generator",
        "step": "rules_generated",
        "content": (
            f"Generated {len(deduped)} validation rules "
            f"({len(cleaned_llm_rules)} LLM + {len(deterministic_rules)} deterministic). "
            f"Suite: '{suite_name}'."
        ),
    })

    return {
        **state,
        "expectation_suite": expectation_suite,
        "suite_name": suite_name,
        "messages": messages_log,
    }