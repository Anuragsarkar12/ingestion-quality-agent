# src/graph/workflow.py
# LangGraph Workflow Assembly — Universal Edition
#
# Changes from original:
#   - mark_success(): now writes the healed CSV to PROCESSED_DATA_PATH so
#     the CLI path (main.py) also gets a clean output file.  Previously only
#     app.py did a shutil.copy AFTER graph.invoke(), meaning CLI runs never
#     produced a clean file.
#   - alert_and_end(): enriched alert with per-column semantic type context
#     and repair_confidence summary so operators know which columns were
#     attempted and what confidence the healer assigned.
#   - build_initial_state(): new helper that constructs a fully-populated
#     AgentState dict with all fields the upgraded architecture requires.
#     Both main.py and app.py use this instead of duplicating the dict.

import logging
import os
import shutil

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.profiler_agent import profile_data
from src.agents.rule_generator import generate_rules
from src.agents.self_healer import self_heal
from src.validation.validator import validate_data
from src.config import MAX_HEALING_ITERATIONS, PROCESSED_DATA_PATH
from src.mcp_tools import send_alert
import src.config as _cfg

logger = logging.getLogger(__name__)


# =============================================================================
# INITIAL STATE FACTORY
# =============================================================================

def build_initial_state(csv_path: str) -> AgentState:
    """
    Construct a fully-populated initial AgentState.

    Centralises initial-state construction so main.py and app.py stay
    in sync with the AgentState TypedDict — a single place to update
    when new fields are added.
    """
    return AgentState(
        # ── Input ────────────────────────────────────────────────────────────
        csv_path           = csv_path,

        # ── Profiler output (populated by Node 1) ────────────────────────────
        profile            = {},
        profile_summary    = "",
        semantic_types     = {},          # new: inferred by profiler

        # ── Rule generator output (populated by Node 2) ──────────────────────
        suite_name         = "",          # new: derived from filename by profiler
        expectation_suite  = {},

        # ── Validator output (populated by Node 3) ───────────────────────────
        validation_result  = {},

        # ── Healer output (populated by Node 4) ──────────────────────────────
        healing_actions    = [],
        healing_history    = [],          # new: cumulative across all iterations
        repair_confidence  = {},          # new: 0.0–1.0 per column

        # ── Workflow control ─────────────────────────────────────────────────
        iteration          = 0,
        final_status       = "PENDING",
        error_message      = None,

        # ── Audit log ────────────────────────────────────────────────────────
        messages           = [],
    )


# =============================================================================
# CONDITIONAL EDGE FUNCTION
# =============================================================================

def should_heal_or_end(state: AgentState) -> str:
    """
    Decide the next node after validate_data.

    Returns:
        "end"           — validation passed, pipeline complete
        "self_heal"     — validation failed, iterations remaining
        "alert_and_end" — validation failed, max iterations exhausted
    """
    validation_passed = state.get("validation_result", {}).get("success", False)
    iteration         = state.get("iteration", 0)

    logger.info("[ROUTER] Routing decision:")
    logger.info(f"  Validation passed : {validation_passed}")
    logger.info(f"  Iteration         : {iteration} / {MAX_HEALING_ITERATIONS}")

    if validation_passed:
        logger.info("[ROUTER] → PASS ✅ → mark_success")
        return "end"
    elif iteration < MAX_HEALING_ITERATIONS:
        logger.info(f"[ROUTER] → FAIL — attempting heal #{iteration + 1}")
        return "self_heal"
    else:
        logger.warning(f"[ROUTER] → MAX RETRIES reached → alert_and_end")
        return "alert_and_end"


# =============================================================================
# TERMINAL NODES
# =============================================================================

def mark_success(state: AgentState) -> AgentState:
    """
    Terminal node: all validations passed.

    Writes the healed CSV to PROCESSED_DATA_PATH (respects any runtime
    override set by app.py via src.config.PROCESSED_DATA_PATH).
    This ensures the CLI path (main.py) also produces a clean output file,
    not just the Streamlit path.
    """
    logger.info("=" * 60)
    logger.info("🎉 [SUCCESS NODE] All validations passed!")
    logger.info("=" * 60)

    csv_path   = state.get("csv_path")
    clean_path = getattr(_cfg, "PROCESSED_DATA_PATH", PROCESSED_DATA_PATH)

    # Write clean output — the working CSV is the healed file at this point
    if csv_path and os.path.exists(csv_path):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(clean_path)), exist_ok=True)
            if os.path.abspath(csv_path) != os.path.abspath(clean_path):
                shutil.copy2(csv_path, clean_path)
                logger.info(f"[SUCCESS NODE] Clean data written → {clean_path}")
            else:
                logger.info(f"[SUCCESS NODE] Source and dest are the same file — skipping copy")
        except Exception as e:
            logger.warning(f"[SUCCESS NODE] Could not write clean output: {e}")
    else:
        logger.warning("[SUCCESS NODE] csv_path missing or file not found — skipping clean write")

    messages = state.get("messages", [])
    messages.append({
        "agent":   "system",
        "step":    "pipeline_complete",
        "content": (
            f"Pipeline completed successfully after "
            f"{state.get('iteration', 0)} healing iteration(s). "
            f"Clean output: {clean_path}"
        ),
    })

    return {**state, "final_status": "PASS", "messages": messages}


def alert_and_end(state: AgentState) -> AgentState:
    """
    Terminal node: max healing iterations exhausted.

    Emits a structured CRITICAL alert that includes:
    - per-column failure details
    - semantic types so operators know the data domain
    - repair_confidence so they know what was attempted
    """
    logger.info("=" * 60)
    logger.info("🚨 [ALERT NODE] Max healing iterations reached!")
    logger.info("=" * 60)

    validation_result  = state.get("validation_result", {})
    failures           = validation_result.get("failures", [])
    semantic_types     = state.get("semantic_types", {})
    repair_confidence  = state.get("repair_confidence", {})
    iterations         = state.get("iteration", 0)

    # Build a structured failure summary for the alert
    failure_lines = []
    for f in failures:
        col   = f.get("kwargs", {}).get("column", "N/A")
        sem   = semantic_types.get(col, "unknown")
        conf  = repair_confidence.get(col)
        conf_str = f"repair confidence {conf:.0%}" if conf is not None else "not attempted"
        failure_lines.append(
            f"  - {f['expectation_type']} on '{col}' "
            f"(semantic type: {sem}, {conf_str}): "
            f"{f.get('failing_count', 0)} rows"
        )

    alert_message = (
        f"DATA QUALITY ALERT: pipeline could not self-heal after "
        f"{iterations} iteration(s).\n\n"
        f"Unresolved failures ({len(failures)}):\n"
        + "\n".join(failure_lines)
        + "\n\nMANUAL INTERVENTION REQUIRED."
    )

    send_alert(alert_message, severity="CRITICAL")

    # Bug 4 fix: write best-effort clean output even on ALERT path.
    # The working CSV has been partially healed — still useful for operators.
    csv_path   = state.get("csv_path")
    clean_path = getattr(_cfg, "PROCESSED_DATA_PATH", PROCESSED_DATA_PATH)
    if csv_path and os.path.exists(csv_path):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(clean_path)), exist_ok=True)
            if os.path.abspath(csv_path) != os.path.abspath(clean_path):
                shutil.copy2(csv_path, clean_path)
                logger.info(f"[ALERT NODE] Best-effort clean data written → {clean_path}")
            else:
                logger.info(f"[ALERT NODE] Source and dest are the same file — skipping copy")
        except Exception as e:
            logger.warning(f"[ALERT NODE] Could not write clean output: {e}")
    else:
        logger.warning("[ALERT NODE] csv_path missing or file not found — skipping clean write")

    messages = state.get("messages", [])
    messages.append({
        "agent":   "system",
        "step":    "alert_sent",
        "content": (
            f"Critical alert sent. "
            f"{len(failures)} unresolvable failure(s) remain after {iterations} iteration(s)."
        ),
    })

    return {**state, "final_status": "ALERT", "messages": messages}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_workflow() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Graph structure:
        profile_data → generate_rules → validate_data
        validate_data --[pass]--------→ mark_success  → END
        validate_data --[fail, retry]-→ self_heal → validate_data  (loop)
        validate_data --[fail, limit]-→ alert_and_end → END
    """
    logger.info("[WORKFLOW] Building LangGraph workflow...")

    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("profile_data",   profile_data)
    workflow.add_node("generate_rules", generate_rules)
    workflow.add_node("validate_data",  validate_data)
    workflow.add_node("self_heal",      self_heal)
    workflow.add_node("alert_and_end",  alert_and_end)
    workflow.add_node("mark_success",   mark_success)

    # ── Unconditional edges ───────────────────────────────────────────────────
    workflow.set_entry_point("profile_data")
    workflow.add_edge("profile_data",   "generate_rules")
    workflow.add_edge("generate_rules", "validate_data")
    workflow.add_edge("self_heal",      "validate_data")
    workflow.add_edge("alert_and_end",  END)
    workflow.add_edge("mark_success",   END)

    # ── Conditional edge after validation ─────────────────────────────────────
    workflow.add_conditional_edges(
        "validate_data",
        should_heal_or_end,
        {
            "end":           "mark_success",
            "self_heal":     "self_heal",
            "alert_and_end": "alert_and_end",
        },
    )

    compiled = workflow.compile()

    logger.info("[WORKFLOW] Graph compiled ✅")
    logger.info("[WORKFLOW]   profile_data → generate_rules → validate_data")
    logger.info("[WORKFLOW]   validate_data --[pass]--> mark_success → END")
    logger.info("[WORKFLOW]   validate_data --[fail, retry]--> self_heal → validate_data")
    logger.info("[WORKFLOW]   validate_data --[fail, limit]--> alert_and_end → END")

    return compiled