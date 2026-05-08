# src/agents/masking_engine.py
# Active governance enforcement: apply PII masking to data.
#
# Masking patterns:
#   EMAIL:       john@gmail.com   → j***@gmail.com
#   PHONE:       9876543210       → ******3210
#   NAME:        John Doe         → J*** D**
#   ADDRESS:     123 Main St      → REDACTED
#   AADHAAR:     1234 5678 9012   → XXXX XXXX 9012
#   PAN:         ABCDE1234F       → XXXXX1234X
#   PASSPORT:    A1234567         → A******7
#   SSN:         123-45-6789      → ***-**-6789
#   CREDIT_CARD: 4111111111111111 → ************1111
#   DOB:         1990-05-15       → 1990-**-**
#   IP:          192.168.1.1      → 192.168.*.*
#   DEFAULT:     <value>          → first + *** + last

import logging
import re
import sqlite3
from typing import Dict, Any, List

import pandas as pd

from src.mcp_tools import save_df_to_db
import src.config as _cfg

logger = logging.getLogger(__name__)


# =============================================================================
# MASKING FUNCTIONS
# =============================================================================

def _mask_email(val: str) -> str:
    """j***@gmail.com"""
    if not val or "@" not in str(val):
        return str(val)
    val = str(val)
    local, domain = val.split("@", 1)
    if len(local) <= 1:
        return f"*@{domain}"
    return f"{local[0]}***@{domain}"


def _mask_phone(val: str) -> str:
    """******3210"""
    val = str(val).strip()
    digits = re.sub(r"\D", "", val)
    if len(digits) < 4:
        return "****"
    return "*" * (len(digits) - 4) + digits[-4:]


def _mask_name(val: str) -> str:
    """J*** D**"""
    if not val:
        return str(val)
    parts = str(val).strip().split()
    masked = []
    for part in parts:
        if len(part) <= 1:
            masked.append("*")
        else:
            masked.append(part[0] + "*" * (len(part) - 1))
    return " ".join(masked)


def _mask_address(val: str) -> str:
    """REDACTED"""
    return "REDACTED"


def _mask_aadhaar(val: str) -> str:
    """XXXX XXXX 9012"""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) < 4:
        return "XXXX XXXX XXXX"
    return f"XXXX XXXX {digits[-4:]}"


def _mask_pan(val: str) -> str:
    """XXXXX1234X"""
    val = str(val).strip().upper()
    if len(val) == 10:
        return f"XXXXX{val[5:9]}X"
    return "XXXXXXXXXX"


def _mask_passport(val: str) -> str:
    """A******7"""
    val = str(val).strip()
    if len(val) <= 2:
        return "*" * len(val)
    return val[0] + "*" * (len(val) - 2) + val[-1]


def _mask_ssn(val: str) -> str:
    """***-**-6789"""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) < 4:
        return "***-**-****"
    return f"***-**-{digits[-4:]}"


def _mask_credit_card(val: str) -> str:
    """************1111"""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) < 4:
        return "****************"
    return "*" * (len(digits) - 4) + digits[-4:]


def _mask_dob(val: str) -> str:
    """1990-**-**"""
    val = str(val).strip()
    # Try to keep year
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", val)
    if match:
        return f"{match.group(1)}-**-**"
    return "****-**-**"


def _mask_ip(val: str) -> str:
    """192.168.*.*"""
    parts = str(val).strip().split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.*.*"
    return "*.*.*.*"


def _mask_generic(val: str) -> str:
    """First char + *** + last char"""
    val = str(val).strip()
    if len(val) <= 2:
        return "*" * len(val)
    return val[0] + "*" * (len(val) - 2) + val[-1]


# Masking dispatch table
MASKERS = {
    "EMAIL":       _mask_email,
    "PHONE":       _mask_phone,
    "NAME":        _mask_name,
    "ADDRESS":     _mask_address,
    "AADHAAR":     _mask_aadhaar,
    "PAN":         _mask_pan,
    "PASSPORT":    _mask_passport,
    "SSN":         _mask_ssn,
    "CREDIT_CARD": _mask_credit_card,
    "DOB":         _mask_dob,
    "IP":          _mask_ip,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def apply_masking(
    pii_detections: List[Dict[str, Any]],
    db_path: str,
) -> Dict[str, Any]:
    """
    Apply PII masking to all detected columns across all tables.

    Returns:
        {
            "masked_tables": {table_name: masked_df, ...},
            "columns_masked": [col1, col2, ...],
            "rows_affected": N,
            "details": [{col, pii_type, samples_before, samples_after}, ...]
        }
    """
    logger.info("🔐 [MASKING] Applying recommended masking...")

    # Group detections by table
    by_table: Dict[str, List[Dict]] = {}
    for det in pii_detections:
        tbl = det["table"]
        by_table.setdefault(tbl, []).append(det)

    masked_tables = {}
    all_columns_masked = []
    total_rows = 0
    details = []

    for table_name, dets in by_table.items():
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            conn.close()
        except Exception as e:
            logger.warning(f"[MASKING] Cannot read '{table_name}': {e}")
            continue

        for det in dets:
            col = det["column"]
            pii_type = det["pii_type"]

            if col not in df.columns:
                continue

            masker = MASKERS.get(pii_type, _mask_generic)

            # Capture samples before
            samples_before = df[col].dropna().head(3).tolist()

            # Apply masking
            df[col] = df[col].apply(
                lambda v: masker(v) if pd.notna(v) and str(v).strip() else v
            )

            samples_after = df[col].dropna().head(3).tolist()
            all_columns_masked.append(col)

            details.append({
                "column": col,
                "table": table_name,
                "pii_type": pii_type,
                "samples_before": [str(s) for s in samples_before],
                "samples_after": [str(s) for s in samples_after],
            })

            logger.info(f"[MASKING] Masked '{col}' ({pii_type}) in '{table_name}'")

        masked_tables[table_name] = df
        total_rows += len(df)

    result = {
        "masked_tables": masked_tables,
        "columns_masked": all_columns_masked,
        "rows_affected": total_rows,
        "details": details,
    }

    logger.info(
        f"[MASKING] Complete: {len(all_columns_masked)} column(s) masked "
        f"across {len(masked_tables)} table(s), {total_rows} rows"
    )

    return result


def persist_masked_tables(
    masked_tables: Dict[str, pd.DataFrame],
    db_path: str = None,
) -> Dict[str, bool]:
    """
    Persist masked DataFrames as <original_table>_masked into SQLite.
    """
    if db_path is None:
        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

    results = {}
    for table_name, df in masked_tables.items():
        masked_name = f"{table_name}_masked"
        success = save_df_to_db(df, masked_name, if_exists="replace")
        results[masked_name] = success
        if success:
            logger.info(f"[MASKING] Persisted → {masked_name} ({len(df)} rows)")
        else:
            logger.warning(f"[MASKING] Failed to persist {masked_name}")

    return results
