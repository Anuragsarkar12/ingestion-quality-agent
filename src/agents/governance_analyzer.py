# src/agents/governance_analyzer.py
# Node 3 of the Governance workflow: Risk Classification & Recommendations.

import logging
from typing import Dict, Any, List

from src.agents.governance_state import GovernanceState

logger = logging.getLogger(__name__)


# =============================================================================
# RISK WEIGHTS
# =============================================================================

PII_RISK_WEIGHTS = {
    "EMAIL":       0.6,
    "PHONE":       0.6,
    "NAME":        0.5,
    "ADDRESS":     0.7,
    "AADHAAR":     0.95,
    "PAN":         0.9,
    "PASSPORT":    0.9,
    "DOB":         0.5,
    "SSN":         0.95,
    "IP":          0.4,
    "CREDIT_CARD": 0.95,
}


def _classify_risk(score: float) -> str:
    """Map numeric risk score to category."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    return "LOW"


def _generate_recommendations(
    pii_detections: List[Dict[str, Any]],
    risk_level: str,
) -> List[Dict[str, Any]]:
    """Generate actionable governance recommendations."""
    recommendations = []

    for det in pii_detections:
        pii_type = det["pii_type"]
        col = det["column"]
        table = det["table"]
        confidence = det.get("confidence", 0.5)

        # Masking recommendation
        recommendations.append({
            "column": col,
            "table": table,
            "pii_type": pii_type,
            "action": "mask",
            "priority": "HIGH" if PII_RISK_WEIGHTS.get(pii_type, 0.5) >= 0.7 else "MEDIUM",
            "detail": f"Apply {pii_type}-specific masking to '{col}' (confidence: {confidence:.0%})",
        })

        # Access control for high-risk PII
        if PII_RISK_WEIGHTS.get(pii_type, 0) >= 0.8:
            recommendations.append({
                "column": col,
                "table": table,
                "pii_type": pii_type,
                "action": "restrict_access",
                "priority": "CRITICAL",
                "detail": f"Restrict access to '{col}' — contains {pii_type} data",
            })

        # Tokenization for identifiers
        if pii_type in ("AADHAAR", "PAN", "SSN", "PASSPORT", "CREDIT_CARD"):
            recommendations.append({
                "column": col,
                "table": table,
                "pii_type": pii_type,
                "action": "tokenize",
                "priority": "HIGH",
                "detail": f"Consider tokenization for '{col}' ({pii_type}) for downstream analytics",
            })

    # Audit logging for any PII presence
    if pii_detections and risk_level in ("HIGH", "CRITICAL"):
        recommendations.append({
            "column": "*",
            "table": "*",
            "pii_type": "GENERAL",
            "action": "audit_logging",
            "priority": "HIGH",
            "detail": "Enable audit logging for all access to tables containing PII",
        })

    return recommendations


# =============================================================================
# NODE: analyze_governance
# =============================================================================

def analyze_governance(state: GovernanceState) -> GovernanceState:
    """
    Governance Node 3: Classify risk and generate recommendations.
    """
    logger.info("=" * 60)
    logger.info("📊 [GOVERNANCE] Analyzing risk and generating recommendations...")
    logger.info("=" * 60)

    pii_detections = state.get("pii_detections", [])
    lineage = state.get("lineage", {})

    if not pii_detections:
        report = {
            "risk_level": "LOW",
            "risk_score": 0.0,
            "summary": "No PII detected. Data governance risk is minimal.",
            "recommendations": [],
            "pii_count": 0,
            "tables_analyzed": len(
                set(lineage.get("source_tables", []) + lineage.get("target_tables", []))
            ),
        }
        logger.info("[GOVERNANCE] No PII → LOW risk")
    else:
        # Calculate risk score
        pii_scores = []
        for det in pii_detections:
            weight = PII_RISK_WEIGHTS.get(det["pii_type"], 0.5)
            confidence = det.get("confidence", 0.5)
            pii_scores.append(weight * confidence)

        risk_score = min(1.0, max(pii_scores) * 0.7 + (len(pii_detections) / 10) * 0.3)

        # Boost risk if data flows through multiple tables
        edge_count = len(lineage.get("edges", []))
        if edge_count > 5:
            risk_score = min(1.0, risk_score + 0.1)

        risk_level = _classify_risk(risk_score)
        recommendations = _generate_recommendations(pii_detections, risk_level)

        # Build summary
        pii_types = set(d["pii_type"] for d in pii_detections)
        tables = set(d["table"] for d in pii_detections)

        report = {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "summary": (
                f"Detected {len(pii_detections)} PII column(s) "
                f"({', '.join(sorted(pii_types))}) across "
                f"{len(tables)} table(s). "
                f"Overall risk: {risk_level} ({risk_score:.0%})."
            ),
            "recommendations": recommendations,
            "pii_count": len(pii_detections),
            "tables_analyzed": len(
                set(lineage.get("source_tables", []) + lineage.get("target_tables", []))
            ),
        }

        logger.info(
            f"[GOVERNANCE] Risk: {risk_level} ({risk_score:.0%}), "
            f"{len(recommendations)} recommendation(s)"
        )

    messages = state.get("messages", [])
    messages.append({
        "agent": "governance",
        "step": "analysis_complete",
        "content": report["summary"],
    })

    return {**state, "governance_report": report, "messages": messages}
