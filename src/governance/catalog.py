# src/governance/catalog.py
# Metadata catalogue persistence for the Governance Agent.
#
# Creates and populates governance metadata tables in SQLite:
#   - catalog_tables:      table-level metadata
#   - catalog_columns:     column-level metadata + PII tags
#   - lineage_edges:       column lineage graph
#   - pii_tags:            detected PII with confidence
#   - governance_reports:  risk assessments

import logging
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List

import src.config as _cfg

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA CREATION
# =============================================================================

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_tables (
    table_name   TEXT PRIMARY KEY,
    source_sql   TEXT,
    created_at   TEXT,
    row_count    INTEGER,
    column_count INTEGER,
    has_pii      INTEGER DEFAULT 0,
    risk_level   TEXT DEFAULT 'LOW'
);

CREATE TABLE IF NOT EXISTS catalog_columns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name   TEXT,
    column_name  TEXT,
    data_type    TEXT,
    is_pii       INTEGER DEFAULT 0,
    pii_type     TEXT,
    pii_confidence REAL,
    sensitivity  TEXT DEFAULT 'LOW',
    masked       INTEGER DEFAULT 0,
    UNIQUE(table_name, column_name)
);

CREATE TABLE IF NOT EXISTS lineage_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    src_table    TEXT,
    src_col      TEXT,
    tgt_table    TEXT,
    tgt_col      TEXT,
    transform    TEXT,
    created_at   TEXT,
    UNIQUE(src_table, src_col, tgt_table, tgt_col)
);

CREATE TABLE IF NOT EXISTS pii_tags (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name   TEXT,
    column_name  TEXT,
    pii_type     TEXT,
    confidence   REAL,
    detection_method TEXT,
    sample_values TEXT,
    created_at   TEXT,
    UNIQUE(table_name, column_name)
);

CREATE TABLE IF NOT EXISTS governance_reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_level   TEXT,
    risk_score   REAL,
    summary      TEXT,
    recommendations TEXT,
    pii_count    INTEGER,
    tables_analyzed INTEGER,
    created_at   TEXT
);
"""


def _get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get a SQLite connection and ensure catalog schema exists."""
    if db_path is None:
        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(CATALOG_SCHEMA)
    return conn


# =============================================================================
# PERSISTENCE FUNCTIONS
# =============================================================================

def persist_lineage(lineage: Dict[str, Any], db_path: str = None) -> bool:
    """Persist lineage edges to the catalog."""
    try:
        conn = _get_conn(db_path)
        edges = lineage.get("edges", [])
        now = datetime.now().isoformat()

        for edge in edges:
            conn.execute(
                """INSERT OR REPLACE INTO lineage_edges
                   (src_table, src_col, tgt_table, tgt_col, transform, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    edge["src_table"], edge["src_col"],
                    edge["tgt_table"], edge["tgt_col"],
                    edge.get("transform", "direct"), now,
                ),
            )

        # Also record target tables in catalog_tables
        for tbl in lineage.get("target_tables", []):
            conn.execute(
                """INSERT OR REPLACE INTO catalog_tables
                   (table_name, source_sql, created_at)
                   VALUES (?, ?, ?)""",
                (tbl, lineage.get("raw_sql", ""), now),
            )

        conn.commit()
        conn.close()
        logger.info(f"[CATALOG] Persisted {len(edges)} lineage edge(s)")
        return True
    except Exception as e:
        logger.warning(f"[CATALOG] Failed to persist lineage: {e}")
        return False


def persist_pii_tags(
    pii_detections: List[Dict[str, Any]],
    db_path: str = None,
) -> bool:
    """Persist PII detections to the catalog."""
    try:
        conn = _get_conn(db_path)
        now = datetime.now().isoformat()

        for det in pii_detections:
            conn.execute(
                """INSERT OR REPLACE INTO pii_tags
                   (table_name, column_name, pii_type, confidence,
                    detection_method, sample_values, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    det["table"], det["column"], det["pii_type"],
                    det.get("confidence", 0.0),
                    det.get("detection_method", ""),
                    json.dumps(det.get("sample_values", [])),
                    now,
                ),
            )

            # Update catalog_columns
            conn.execute(
                """INSERT OR REPLACE INTO catalog_columns
                   (table_name, column_name, is_pii, pii_type,
                    pii_confidence, sensitivity)
                   VALUES (?, ?, 1, ?, ?, ?)""",
                (
                    det["table"], det["column"], det["pii_type"],
                    det.get("confidence", 0.0),
                    "HIGH" if det.get("confidence", 0) >= 0.7 else "MEDIUM",
                ),
            )

            # Mark table as having PII
            conn.execute(
                """UPDATE catalog_tables SET has_pii = 1
                   WHERE table_name = ?""",
                (det["table"],),
            )

        conn.commit()
        conn.close()
        logger.info(f"[CATALOG] Persisted {len(pii_detections)} PII tag(s)")
        return True
    except Exception as e:
        logger.warning(f"[CATALOG] Failed to persist PII tags: {e}")
        return False


def persist_governance_report(
    report: Dict[str, Any],
    db_path: str = None,
) -> bool:
    """Persist a governance report to the catalog."""
    try:
        conn = _get_conn(db_path)
        now = datetime.now().isoformat()

        conn.execute(
            """INSERT INTO governance_reports
               (risk_level, risk_score, summary, recommendations,
                pii_count, tables_analyzed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                report.get("risk_level", "LOW"),
                report.get("risk_score", 0.0),
                report.get("summary", ""),
                json.dumps(report.get("recommendations", [])),
                report.get("pii_count", 0),
                report.get("tables_analyzed", 0),
                now,
            ),
        )

        # Update risk level on catalog_tables
        for rec in report.get("recommendations", []):
            tbl = rec.get("table")
            if tbl and tbl != "*":
                conn.execute(
                    """UPDATE catalog_tables SET risk_level = ?
                       WHERE table_name = ?""",
                    (report["risk_level"], tbl),
                )

        conn.commit()
        conn.close()
        logger.info(f"[CATALOG] Persisted governance report (risk: {report.get('risk_level')})")
        return True
    except Exception as e:
        logger.warning(f"[CATALOG] Failed to persist report: {e}")
        return False


def persist_all(
    lineage: Dict[str, Any],
    pii_detections: List[Dict[str, Any]],
    governance_report: Dict[str, Any],
    db_path: str = None,
) -> Dict[str, bool]:
    """Persist all governance metadata in one call."""
    return {
        "lineage": persist_lineage(lineage, db_path),
        "pii_tags": persist_pii_tags(pii_detections, db_path),
        "report": persist_governance_report(governance_report, db_path),
    }
