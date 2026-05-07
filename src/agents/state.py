# src/agents/state.py
# Defines the shared state that flows through the LangGraph workflow.

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph workflow.

    CHANGE LOG (universal refactor):
    - Added semantic_types: column-name → semantic type string (email, currency, etc.)
    - Added healing_history: per-iteration record to prevent redundant re-healing
    - Added suite_name: dynamic, derived from input filename (not hardcoded to "orders")
    - Added repair_confidence: 0.0–1.0 per column being healed
    """

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------
    csv_path: str
    # File path to the raw CSV being processed

    # -------------------------------------------------------------------------
    # PROFILER OUTPUT (filled by Node 1)
    # -------------------------------------------------------------------------
    profile: Dict[str, Any]
    # {
    #   "raw_stats": {...},      ← per-column statistics from compute_column_profile()
    #   "analysis": {...},       ← LLM interpretation
    #   "data_info": {...},      ← row_count, column_count, columns, dtypes, sample_rows
    #   "semantic_types": {...}  ← copy of semantic_types for reference within profile
    # }

    profile_summary: str
    # Human-readable one-paragraph summary from LLM

    semantic_types: Dict[str, str]
    # Deterministically inferred semantic type per column.
    # Keys:   column name (str)
    # Values: one of:
    #   email | phone | url | datetime | date | boolean |
    #   categorical | identifier | currency | integer | float | free_text | unknown
    #
    # Example: {"user_email": "email", "status": "categorical", "price": "currency"}
    #
    # This field is populated by profile_data() via infer_semantic_types()
    # and consumed by generate_rules() and self_heal() to make domain-neutral decisions.

    # -------------------------------------------------------------------------
    # RULE GENERATOR OUTPUT (filled by Node 2)
    # -------------------------------------------------------------------------
    suite_name: str
    # Dynamic suite name derived from the input CSV filename.
    # Example: "orders_raw_quality_suite" for orders_raw.csv
    # Avoids hardcoded "orders_quality_suite" for universal use.

    expectation_suite: Dict[str, Any]
    # {
    #   "suite_name": "...",
    #   "expectations": [
    #     {
    #       "expectation_type": "expect_column_values_to_not_be_null",
    #       "kwargs": {"column": "email"},
    #       "reasoning": "..."
    #     },
    #     ...
    #   ]
    # }

    # -------------------------------------------------------------------------
    # VALIDATOR OUTPUT (filled by Node 3)
    # -------------------------------------------------------------------------
    validation_result: Dict[str, Any]
    # {
    #   "success": False,
    #   "statistics": {
    #     "evaluated_expectations": 8,
    #     "successful_expectations": 6,
    #     "unsuccessful_expectations": 2,
    #     "success_percent": 75.0
    #   },
    #   "failures": [
    #     {
    #       "expectation_type": "...",
    #       "kwargs": {"column": "order_amount", "min_value": 0, "max_value": 50000},
    #       "failing_count": 25,
    #       "failing_indices": [4, 17, 23, ...]
    #     }
    #   ]
    # }

    # -------------------------------------------------------------------------
    # SELF-HEALER OUTPUT (filled by Node 4)
    # -------------------------------------------------------------------------
    healing_actions: List[Dict[str, Any]]
    # Actions applied in the MOST RECENT healing iteration.
    # Each entry: {"action": {...}, "result": {...}, "reasoning": "..."}

    healing_history: List[Dict[str, Any]]
    # Cumulative record across ALL iterations.
    # Used to detect and skip repeated zero-effect actions.
    # Each entry: {
    #   "iteration": 1,
    #   "column": "price",
    #   "action_type": "fix_value",
    #   "rows_affected": 12,
    #   "success": True
    # }

    repair_confidence: Dict[str, float]
    # Per-column confidence score for the healing action applied.
    # 0.0 = completely uncertain, 1.0 = deterministically correct.
    # Example: {"price": 0.85, "email": 0.99, "status": 0.70}

    # -------------------------------------------------------------------------
    # WORKFLOW CONTROL
    # -------------------------------------------------------------------------
    iteration: int
    # Number of heal → validate cycles completed

    final_status: str
    # Overall outcome: "PENDING" → "PASS" | "FAIL" | "ALERT"

    error_message: Optional[str]
    # Error description if something went wrong

    # -------------------------------------------------------------------------
    # AUDIT LOG
    # -------------------------------------------------------------------------
    messages: List[Dict[str, str]]
    # Log of all reasoning steps by all agents.
    # Each entry: {"agent": "profiler", "step": "profile_complete", "content": "..."}