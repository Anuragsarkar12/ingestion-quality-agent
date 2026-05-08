# src/graph/governance_workflow.py
# LangGraph workflow for the Lineage & Governance Agent (B2).
# Completely separate from the Ingestion Quality workflow (B1).

import logging

from langgraph.graph import StateGraph, END

from src.agents.governance_state import GovernanceState
from src.agents.lineage_agent import parse_lineage
from src.agents.pii_detector import detect_pii
from src.agents.governance_analyzer import analyze_governance

logger = logging.getLogger(__name__)


# =============================================================================
# INITIAL STATE FACTORY
# =============================================================================

def build_governance_initial_state(
    sql_input: str,
    db_path: str = "database/final.db",
) -> GovernanceState:
    """Construct a fully-populated initial GovernanceState."""
    return GovernanceState(
        sql_input=sql_input,
        db_path=db_path,
        lineage={},
        pii_detections=[],
        governance_report={},
        masking_result=None,
        messages=[],
    )


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_governance_workflow() -> StateGraph:
    """
    Build the Lineage & Governance LangGraph workflow.

    Flow: parse_lineage → detect_pii → analyze_governance → END
    """
    graph = StateGraph(GovernanceState)

    # Add nodes
    graph.add_node("parse_lineage", parse_lineage)
    graph.add_node("detect_pii", detect_pii)
    graph.add_node("analyze_governance", analyze_governance)

    # Linear flow
    graph.set_entry_point("parse_lineage")
    graph.add_edge("parse_lineage", "detect_pii")
    graph.add_edge("detect_pii", "analyze_governance")
    graph.add_edge("analyze_governance", END)

    compiled = graph.compile()
    logger.info("[GOVERNANCE] Workflow compiled: parse_lineage → detect_pii → analyze_governance → END")

    return compiled
