# src/agents/governance_state.py
# Shared state for the Lineage & Governance Agent (B2).
# Architecturally separate from AgentState (B1).

from typing import TypedDict, List, Dict, Any, Optional


class GovernanceState(TypedDict):
    """
    Shared state for the Lineage & Governance workflow.

    Flows through: parse_lineage → detect_pii → analyze_governance
    """

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------
    sql_input: str
    # Raw SQL transformation(s) provided by the user.

    db_path: str
    # Path to the SQLite database for table introspection.

    # -------------------------------------------------------------------------
    # LINEAGE OUTPUT (filled by Node 1)
    # -------------------------------------------------------------------------
    lineage: Dict[str, Any]
    # {
    #   "source_tables": ["orders_clean"],
    #   "target_tables": ["premium_customers"],
    #   "columns": {
    #     "target_col": {
    #       "source_table": "orders_clean",
    #       "source_col": "customer_id",
    #       "transform": "direct"
    #     }
    #   },
    #   "edges": [
    #     {"src_table": "...", "src_col": "...", "tgt_table": "...", "tgt_col": "...", "transform": "..."}
    #   ]
    # }

    # -------------------------------------------------------------------------
    # PII DETECTION OUTPUT (filled by Node 2)
    # -------------------------------------------------------------------------
    pii_detections: List[Dict[str, Any]]
    # [
    #   {
    #     "column": "email",
    #     "table": "orders_clean",
    #     "pii_type": "EMAIL",
    #     "confidence": 0.95,
    #     "detection_method": "regex+keyword",
    #     "sample_values": ["john@example.com", ...]
    #   }
    # ]

    # -------------------------------------------------------------------------
    # GOVERNANCE OUTPUT (filled by Node 3)
    # -------------------------------------------------------------------------
    governance_report: Dict[str, Any]
    # {
    #   "risk_level": "HIGH",
    #   "risk_score": 0.85,
    #   "summary": "...",
    #   "recommendations": [
    #     {"column": "email", "action": "mask", "priority": "HIGH", "detail": "..."}
    #   ],
    #   "pii_count": 3,
    #   "tables_analyzed": 2
    # }

    # -------------------------------------------------------------------------
    # MASKING OUTPUT (filled on demand)
    # -------------------------------------------------------------------------
    masking_result: Optional[Dict[str, Any]]
    # {
    #   "masked_tables": {"orders_clean": <DataFrame>},
    #   "columns_masked": ["email", "phone"],
    #   "rows_affected": 955
    # }

    # -------------------------------------------------------------------------
    # AUDIT LOG
    # -------------------------------------------------------------------------
    messages: List[Dict[str, str]]
    # Reasoning trace for the governance workflow.
