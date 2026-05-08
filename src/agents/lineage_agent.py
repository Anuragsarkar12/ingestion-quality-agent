# src/agents/lineage_agent.py
# Node 1 of the Governance workflow: SQL Lineage Extraction.
# Uses sqlglot for deterministic SQL parsing — no LLM needed.

import logging
from typing import Dict, Any, List

import sqlglot
from sqlglot import exp
from sqlglot.lineage import lineage as sg_lineage

from src.agents.governance_state import GovernanceState

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _extract_tables(parsed) -> Dict[str, List[str]]:
    """Extract source and target tables from parsed SQL."""
    sources = set()
    targets = set()

    for stmt in parsed if isinstance(parsed, list) else [parsed]:
        # Target tables (CREATE TABLE ... AS, INSERT INTO)
        if isinstance(stmt, exp.Create):
            tbl = stmt.find(exp.Table)
            if tbl:
                targets.add(tbl.name)
            # Sources come from the inner SELECT
            for inner_tbl in stmt.find_all(exp.Table):
                name = inner_tbl.name
                if name and name not in targets:
                    sources.add(name)
        elif isinstance(stmt, exp.Insert):
            tbl = stmt.find(exp.Table)
            if tbl:
                targets.add(tbl.name)
            select = stmt.find(exp.Select)
            if select:
                for inner_tbl in select.find_all(exp.Table):
                    sources.add(inner_tbl.name)
        elif isinstance(stmt, exp.Select):
            # Pure SELECT — sources only
            for tbl in stmt.find_all(exp.Table):
                sources.add(tbl.name)

    # Remove targets from sources (self-reference)
    sources -= targets

    return {
        "source_tables": sorted(sources),
        "target_tables": sorted(targets),
    }


def _extract_column_lineage(parsed) -> List[Dict[str, str]]:
    """Extract column-level lineage edges."""
    edges = []

    for stmt in parsed if isinstance(parsed, list) else [parsed]:
        # Find the target table name
        target_table = None
        if isinstance(stmt, exp.Create):
            tbl = stmt.find(exp.Table)
            if tbl:
                target_table = tbl.name
        elif isinstance(stmt, exp.Insert):
            tbl = stmt.find(exp.Table)
            if tbl:
                target_table = tbl.name

        # Find the SELECT within
        select = None
        if isinstance(stmt, exp.Create):
            select = stmt.find(exp.Select)
        elif isinstance(stmt, exp.Insert):
            select = stmt.find(exp.Select)
        elif isinstance(stmt, exp.Select):
            select = stmt
            target_table = "__result__"

        if not select:
            continue

        # Build source table lookup from FROM / JOIN
        from_tables = []
        for tbl in select.find_all(exp.Table):
            from_tables.append(tbl.name)

        default_source = from_tables[0] if from_tables else "unknown"

        # Extract column projections
        for projection in select.find_all(exp.Column):
            src_col = projection.name
            src_table = default_source

            # Check if column has explicit table reference
            if projection.table:
                src_table = projection.table

            # Find alias if any
            parent = projection.parent
            tgt_col = src_col
            if isinstance(parent, exp.Alias):
                tgt_col = parent.alias

            edges.append({
                "src_table": src_table,
                "src_col": src_col,
                "tgt_table": target_table or "__result__",
                "tgt_col": tgt_col,
                "transform": "direct",
            })

        # Detect aggregate / function transforms
        for func in select.find_all(exp.Func):
            func_name = func.sql_name() if hasattr(func, 'sql_name') else type(func).__name__
            for col in func.find_all(exp.Column):
                src_table = col.table if col.table else default_source

                parent = func.parent
                tgt_col = func_name.lower()
                if isinstance(parent, exp.Alias):
                    tgt_col = parent.alias

                edges.append({
                    "src_table": src_table,
                    "src_col": col.name,
                    "tgt_table": target_table or "__result__",
                    "tgt_col": tgt_col,
                    "transform": func_name.lower(),
                })

    # Deduplicate
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["src_table"], e["src_col"], e["tgt_table"], e["tgt_col"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges


def _extract_where_conditions(parsed) -> List[str]:
    """Extract WHERE clause conditions as strings."""
    conditions = []
    for stmt in parsed if isinstance(parsed, list) else [parsed]:
        for where in stmt.find_all(exp.Where):
            conditions.append(where.this.sql())
    return conditions


def _extract_joins(parsed) -> List[Dict[str, str]]:
    """Extract JOIN conditions."""
    joins = []
    for stmt in parsed if isinstance(parsed, list) else [parsed]:
        for join in stmt.find_all(exp.Join):
            join_info = {
                "type": join.args.get("side", "INNER"),
                "table": "",
                "on": "",
            }
            tbl = join.find(exp.Table)
            if tbl:
                join_info["table"] = tbl.name
            on_clause = join.find(exp.On)
            if on_clause:
                join_info["on"] = on_clause.this.sql()
            joins.append(join_info)
    return joins


# =============================================================================
# NODE: parse_lineage
# =============================================================================

def parse_lineage(state: GovernanceState) -> GovernanceState:
    """
    Governance Node 1: Parse SQL and extract full lineage.

    Uses sqlglot for deterministic parsing — no LLM.
    """
    logger.info("=" * 60)
    logger.info("🧬 [LINEAGE AGENT] Parsing SQL lineage...")
    logger.info("=" * 60)

    sql_input = state.get("sql_input", "")

    if not sql_input.strip():
        logger.warning("[LINEAGE] Empty SQL input")
        return {
            **state,
            "lineage": {"error": "Empty SQL input"},
            "messages": state.get("messages", []) + [{
                "agent": "lineage",
                "step": "parse_error",
                "content": "No SQL provided.",
            }],
        }

    try:
        # Parse SQL (may be multiple statements)
        parsed = sqlglot.parse(sql_input)

        # Extract lineage components
        tables = _extract_tables(parsed)
        edges = _extract_column_lineage(parsed)
        conditions = _extract_where_conditions(parsed)
        joins = _extract_joins(parsed)

        # Build column mapping
        columns = {}
        for edge in edges:
            columns[edge["tgt_col"]] = {
                "source_table": edge["src_table"],
                "source_col": edge["src_col"],
                "transform": edge["transform"],
            }

        lineage = {
            "source_tables": tables["source_tables"],
            "target_tables": tables["target_tables"],
            "columns": columns,
            "edges": edges,
            "conditions": conditions,
            "joins": joins,
            "statement_count": len(parsed),
            "raw_sql": sql_input,
        }

        logger.info(
            f"[LINEAGE] Extracted: {len(tables['source_tables'])} source(s), "
            f"{len(tables['target_tables'])} target(s), "
            f"{len(edges)} column edge(s)"
        )

        messages = state.get("messages", [])
        messages.append({
            "agent": "lineage",
            "step": "parse_complete",
            "content": (
                f"Parsed {len(parsed)} statement(s). "
                f"Sources: {tables['source_tables']}. "
                f"Targets: {tables['target_tables']}. "
                f"{len(edges)} column lineage edges extracted."
            ),
        })

        return {**state, "lineage": lineage, "messages": messages}

    except Exception as e:
        logger.error(f"[LINEAGE] Parse error: {e}")
        return {
            **state,
            "lineage": {"error": str(e)},
            "messages": state.get("messages", []) + [{
                "agent": "lineage",
                "step": "parse_error",
                "content": f"SQL parse failed: {e}",
            }],
        }
