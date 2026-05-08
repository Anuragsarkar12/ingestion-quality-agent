# src/agents/pii_detector.py
# Node 2 of the Governance workflow: Multi-layer PII Detection.
#
# Detection layers:
#   1. Keyword matching on column names
#   2. Regex pattern validation on actual data samples
#   3. Semantic reasoning (column name + data shape)

import logging
import re
import sqlite3
from typing import Dict, Any, List, Optional

import pandas as pd

from src.agents.governance_state import GovernanceState
import src.config as _cfg

logger = logging.getLogger(__name__)


# =============================================================================
# PII KEYWORD PATTERNS (Layer 1)
# =============================================================================

PII_KEYWORDS: Dict[str, List[str]] = {
    "EMAIL":    ["email", "e_mail", "e-mail", "mail_address", "email_address"],
    "PHONE":    ["phone", "mobile", "cell", "telephone", "contact_number", "phone_number"],
    "NAME":     ["name", "first_name", "last_name", "full_name", "customer_name",
                 "patient_name", "user_name", "person_name"],
    "ADDRESS":  ["address", "street", "city", "zip", "zipcode", "postal",
                 "addr", "residence", "location", "home_address"],
    "AADHAAR":  ["aadhaar", "aadhar", "aadhaar_number", "aadhar_number", "uid"],
    "PAN":      ["pan", "pan_number", "pan_card"],
    "PASSPORT": ["passport", "passport_number", "passport_no"],
    "DOB":      ["dob", "date_of_birth", "birth_date", "birthdate"],
    "SSN":      ["ssn", "social_security", "social_security_number"],
    "IP":       ["ip_address", "ip_addr", "client_ip", "source_ip"],
    "CREDIT_CARD": ["credit_card", "card_number", "cc_number", "card_num"],
}

# Columns that look like PII keywords but aren't
PII_EXCLUSIONS = {
    "product_name", "category_name", "table_name", "column_name",
    "file_name", "suite_name", "index_name", "game_name",
    "shipping_address",  # might actually be PII — keep if in doubt
}


# =============================================================================
# PII REGEX PATTERNS (Layer 2)
# =============================================================================

PII_REGEX: Dict[str, str] = {
    "EMAIL":       r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "PHONE":       r"^[\+]?[\d\s\-\(\)]{7,15}$",
    "AADHAAR":     r"^\d{4}\s?\d{4}\s?\d{4}$",
    "PAN":         r"^[A-Z]{5}\d{4}[A-Z]$",
    "PASSPORT":    r"^[A-Z]\d{7}$",
    "SSN":         r"^\d{3}-?\d{2}-?\d{4}$",
    "CREDIT_CARD": r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$",
    "IP":          r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
}


# =============================================================================
# DETECTION LOGIC
# =============================================================================

def _keyword_match(col_name: str) -> Optional[str]:
    """Layer 1: Match column name against PII keywords."""
    col_lower = col_name.lower().strip()

    # Exact exclusions
    if col_lower in PII_EXCLUSIONS:
        return None

    for pii_type, keywords in PII_KEYWORDS.items():
        for kw in keywords:
            if kw == col_lower or col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_"):
                return pii_type
            # Substring match for compound names
            if kw in col_lower and len(kw) >= 4:
                return pii_type

    return None


def _regex_match(values: List[str], pii_type: str) -> float:
    """Layer 2: Test sample values against regex for the given PII type."""
    pattern = PII_REGEX.get(pii_type)
    if not pattern:
        return 0.0

    if not values:
        return 0.0

    matches = sum(1 for v in values if v and re.match(pattern, str(v).strip()))
    return matches / len(values) if values else 0.0


def _semantic_check(col_name: str, values: List[str]) -> Optional[str]:
    """
    Layer 3: Semantic reasoning for ambiguous columns.

    Checks data shape when keyword matching is inconclusive.
    """
    col_lower = col_name.lower()

    # Check if values look like emails even if column isn't named 'email'
    if values:
        email_count = sum(1 for v in values if v and "@" in str(v) and "." in str(v))
        if email_count / max(len(values), 1) > 0.5:
            return "EMAIL"

        # Check if values look like phone numbers
        phone_count = sum(
            1 for v in values
            if v and re.match(r"^[\+]?[\d\s\-\(\)]{7,15}$", str(v).strip())
        )
        if phone_count / max(len(values), 1) > 0.5:
            return "PHONE"

    # Name-like columns with string data
    if any(kw in col_lower for kw in ["name", "naam"]):
        if values and all(isinstance(v, str) and v.replace(" ", "").isalpha() for v in values[:5] if v):
            return "NAME"

    return None


def _get_sample_values(
    table_name: str,
    column_name: str,
    db_path: str,
    limit: int = 20,
) -> List[str]:
    """Fetch sample non-null values from SQLite for a column."""
    try:
        conn = sqlite3.connect(db_path)
        query = (
            f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL AND "{column_name}" != "" '
            f'LIMIT {limit}'
        )
        rows = conn.execute(query).fetchall()
        conn.close()
        return [str(r[0]) for r in rows]
    except Exception:
        return []


def detect_pii_for_table(
    table_name: str,
    columns: List[str],
    db_path: str,
) -> List[Dict[str, Any]]:
    """
    Run multi-layer PII detection on all columns of a table.

    Returns list of PII detection dicts.
    """
    detections = []

    for col in columns:
        # Layer 1: Keyword
        pii_type = _keyword_match(col)
        method = "keyword"
        confidence = 0.7 if pii_type else 0.0

        # Layer 2: Regex on sample data
        samples = _get_sample_values(table_name, col, db_path)

        if pii_type and samples:
            regex_score = _regex_match(samples, pii_type)
            if regex_score > 0.3:
                confidence = min(0.95, confidence + regex_score * 0.3)
                method = "keyword+regex"
            elif regex_score == 0.0 and pii_type in ("AADHAAR", "PAN", "PASSPORT", "SSN", "CREDIT_CARD"):
                # Keyword matched but data doesn't match pattern — lower confidence
                confidence = 0.4
                method = "keyword_only"

        # Layer 3: Semantic fallback
        if not pii_type and samples:
            pii_type = _semantic_check(col, samples)
            if pii_type:
                confidence = 0.6
                method = "semantic"

                # Boost with regex
                regex_score = _regex_match(samples, pii_type)
                if regex_score > 0.3:
                    confidence = min(0.90, confidence + regex_score * 0.3)
                    method = "semantic+regex"

        if pii_type:
            detections.append({
                "column": col,
                "table": table_name,
                "pii_type": pii_type,
                "confidence": round(confidence, 2),
                "detection_method": method,
                "sample_values": samples[:5],
            })

    return detections


# =============================================================================
# NODE: detect_pii
# =============================================================================

def detect_pii(state: GovernanceState) -> GovernanceState:
    """
    Governance Node 2: Detect PII across all tables referenced in lineage.
    """
    logger.info("=" * 60)
    logger.info("🔐 [PII DETECTOR] Scanning for sensitive data...")
    logger.info("=" * 60)

    lineage = state.get("lineage", {})
    db_path = state.get("db_path", getattr(_cfg, "DATABASE_PATH", "database/final.db"))

    if lineage.get("error"):
        logger.warning("[PII] Skipping — lineage has errors")
        return {
            **state,
            "pii_detections": [],
            "messages": state.get("messages", []) + [{
                "agent": "pii_detector",
                "step": "skipped",
                "content": "Skipped PII detection due to lineage errors.",
            }],
        }

    # Collect all tables to scan
    all_tables = set(lineage.get("source_tables", []) + lineage.get("target_tables", []))

    all_detections = []

    for table_name in all_tables:
        # Get columns from DB
        try:
            conn = sqlite3.connect(db_path)
            col_info = pd.read_sql_query(
                f'PRAGMA table_info("{table_name}")', conn
            )
            conn.close()
            columns = col_info["name"].tolist()
        except Exception as e:
            logger.warning(f"[PII] Cannot introspect '{table_name}': {e}")
            continue

        detections = detect_pii_for_table(table_name, columns, db_path)
        all_detections.extend(detections)

        logger.info(
            f"[PII] {table_name}: {len(detections)} PII column(s) detected"
        )

    messages = state.get("messages", [])
    messages.append({
        "agent": "pii_detector",
        "step": "detection_complete",
        "content": (
            f"Scanned {len(all_tables)} table(s). "
            f"Found {len(all_detections)} PII column(s): "
            + ", ".join(f"{d['column']}({d['pii_type']})" for d in all_detections)
        ),
    })

    return {**state, "pii_detections": all_detections, "messages": messages}
