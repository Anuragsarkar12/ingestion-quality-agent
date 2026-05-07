# src/validation/validator.py
# Node 3: Validator — Universal Edition
#
# Change from original:
#   - Uses state.get("suite_name", GE_SUITE_NAME) instead of the hardcoded
#     GE_SUITE_NAME constant.  This is the only change needed here: the node
#     is already deterministic and domain-neutral; it just needed to honour
#     the dynamic suite name produced by profiler_agent.profile_data().
#
# Why it matters:
#   The profiler now derives the suite name from the uploaded filename
#   (e.g. "sensor_data_quality_suite" for sensor_data.csv).  Passing
#   GE_SUITE_NAME hard-coded would cause run_ge_validation to look for
#   "dynamic_quality_suite.json" regardless of what the rule generator saved,
#   returning a "suite not found" error for every non-default filename.

import logging

from src.config import GE_SUITE_NAME
from src.agents.state import AgentState
from src.mcp_tools import run_ge_validation

logger = logging.getLogger(__name__)


def validate_data(state: AgentState) -> AgentState:
    """
    LangGraph Node 3: Validate data against the expectation suite.

    Runs the Great Expectations suite whose name was assigned by the profiler
    (state["suite_name"]) and records the full result in state for the router
    and healer to consume.

    This node is PURELY DETERMINISTIC — no LLM, no heuristics.
    It runs the rules that rule_generator created and reports pass/fail counts.
    Routing (what to do with those results) is handled by should_heal_or_end().
    """
    iteration_display = state.get("iteration", 0) + 1
    logger.info("=" * 60)
    logger.info(f"✅ [VALIDATOR] Running validation (pass {iteration_display})...")
    logger.info("=" * 60)

    csv_path  = state.get("csv_path")

    # ── Read suite name dynamically from state ────────────────────────────────
    # Falls back to the config default only if profiler_agent failed to set it
    # (e.g. an interrupted run or a unit-test stub that skips profiling).
    suite_name = state.get("suite_name") or GE_SUITE_NAME
    logger.info(f"[VALIDATOR] Suite: '{suite_name}'  |  File: {csv_path}")

    # ── Run validation via MCP tool ───────────────────────────────────────────
    validation_result = run_ge_validation(csv_path, suite_name)

    # ── Log summary ───────────────────────────────────────────────────────────
    stats   = validation_result.get("statistics", {})
    overall = "✅ PASSED" if validation_result.get("success") else "❌ FAILED"

    logger.info(f"[VALIDATOR] Result   : {overall}")
    logger.info(f"[VALIDATOR] Checked  : {stats.get('evaluated_expectations', 0)} rules")
    logger.info(f"[VALIDATOR] Passed   : {stats.get('successful_expectations', 0)}")
    logger.info(f"[VALIDATOR] Failed   : {stats.get('unsuccessful_expectations', 0)}")
    logger.info(f"[VALIDATOR] Score    : {stats.get('success_percent', 0):.1f}%")

    if not validation_result.get("success"):
        logger.info("[VALIDATOR] Failures:")
        for failure in validation_result.get("failures", []):
            col   = failure.get("kwargs", {}).get("column", "N/A")
            count = failure.get("failing_count", 0)
            logger.info(f"  ❌ {failure['expectation_type']} on '{col}': {count} failing rows")

    # ── Update state ──────────────────────────────────────────────────────────
    messages = state.get("messages", [])
    passed   = stats.get("successful_expectations", 0)
    total    = stats.get("evaluated_expectations",  0)
    messages.append({
        "agent":   "validator",
        "step":    f"validation_pass_{iteration_display}",
        "content": (
            f"Validation {'PASSED' if validation_result['success'] else 'FAILED'} "
            f"({passed}/{total} rules passed, "
            f"{stats.get('success_percent', 0):.1f}%). "
            f"Suite: '{suite_name}'."
        ),
    })

    return {
        **state,
        "validation_result": validation_result,
        "messages":          messages,
    }