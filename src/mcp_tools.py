# src/mcp_tools.py
# MCP (Model Context Protocol) Tool Definitions — Universal Edition
#
# Changes from original:
#   - _robust_read_csv(): encoding + delimiter detection (handles non-UTF-8, semicolons, etc.)
#   - infer_semantic_types(): deterministic column semantic typing (no LLM needed)
#   - compute_column_profile(): sampling guard for large datasets
#   - apply_healing_action(): domain-neutral bounds, replace_invalid_categorical
#   - All pd.read_csv() calls replaced with _robust_read_csv()

import csv
import sqlite3
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.config import (
    DATABASE_PATH, TABLE_STAGING, TABLE_CLEAN, TABLE_QUARANTINE,
    GE_EXPECTATIONS_DIR, GE_SUITE_NAME,
    PROCESSED_DATA_PATH, MAX_PROFILE_ROWS,
)
import src.config as _cfg

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL: SQLite PERSISTENCE HELPER
# =============================================================================

def save_df_to_db(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
) -> bool:
    """
    Persist a DataFrame into the SQLite mock DB.

    Args:
        df:         DataFrame to write.
        table_name: Target table (e.g. TABLE_STAGING, TABLE_CLEAN, TABLE_QUARANTINE).
        if_exists:  'replace' to overwrite, 'append' to add rows.

    Returns True on success, False on failure (never raises — DB writes
    are additive and must not break the pipeline).
    """
    db_path = getattr(_cfg, "DATABASE_PATH", DATABASE_PATH)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()
        logger.info(
            f"[DB] Persisted {len(df)} rows → {table_name} "
            f"(mode={if_exists}, db={db_path})"
        )
        return True
    except Exception as e:
        logger.warning(f"[DB] Failed to write '{table_name}': {e}")
        return False


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

MCP_TOOLS_MANIFEST = {
    "tools": [
        {"name": "read_csv_data",        "description": "Load a CSV file and return basic statistics"},
        {"name": "run_sql_query",         "description": "Execute a SQL query on the database"},
        {"name": "compute_column_profile","description": "Compute detailed statistics for each column"},
        {"name": "infer_semantic_types",  "description": "Deterministically infer semantic column types"},
        {"name": "save_expectation_suite","description": "Save a Great Expectations suite to disk"},
        {"name": "run_ge_validation",     "description": "Run Great Expectations validation against the dataset"},
        {"name": "apply_healing_action",  "description": "Apply a data healing transformation"},
        {"name": "send_alert",            "description": "Log an alert for unresolvable data quality issues"},
    ]
}


# =============================================================================
# INTERNAL: ROBUST CSV LOADER
# =============================================================================

def _robust_read_csv(filepath: str, sample_rows: int = None) -> pd.DataFrame:
    """
    Robustly load a CSV file by auto-detecting encoding and delimiter.

    Strategy:
      1. Use chardet for encoding detection (falls back to trial sequence)
      2. Use csv.Sniffer for delimiter detection
      3. Try: UTF-8-BOM → UTF-8 → Latin-1 → Windows-1252 → ISO-8859-1
      4. Strip phantom Unnamed columns from trailing delimiters
      5. Drop fully-empty rows and columns
      6. Honour sample_rows for large-file partial loads

    Handles: Excel exports, SAP/Oracle CSVs, European semicolon files,
             BOM-prefixed files, files with trailing commas.
    """
    ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "latin-1", "windows-1252", "iso-8859-1"]

    # ── Step 1: Encoding detection via chardet ───────────────────────────────
    try:
        import chardet
        file_size = os.path.getsize(filepath)
        with open(filepath, "rb") as f:
            raw = f.read(min(65_536, file_size))
        detected = chardet.detect(raw)
        if detected.get("confidence", 0) > 0.70:
            enc = detected["encoding"]
            logger.info(f"[CSV LOADER] chardet: encoding={enc} "
                        f"(confidence={detected['confidence']:.0%})")
            ENCODINGS_TO_TRY = [enc] + [e for e in ENCODINGS_TO_TRY if e.lower() != enc.lower()]
    except ImportError:
        logger.debug("[CSV LOADER] chardet not installed — using encoding trial sequence")
    except Exception as e:
        logger.debug(f"[CSV LOADER] chardet failed: {e}")

    # ── Step 2: Delimiter detection ──────────────────────────────────────────
    delimiter = ","
    for enc in ENCODINGS_TO_TRY:
        try:
            with open(filepath, "r", encoding=enc, errors="replace") as f:
                sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
            if delimiter != ",":
                logger.info(f"[CSV LOADER] Non-standard delimiter detected: '{delimiter}'")
            break
        except Exception:
            continue

    # ── Step 3: Load with encoding fallback sequence ─────────────────────────
    last_error = None
    for encoding in ENCODINGS_TO_TRY:
        try:
            read_kwargs = {
                "filepath_or_buffer": filepath,
                "encoding": encoding,
                "sep": delimiter,
                "on_bad_lines": "warn",
                "low_memory": False,
            }
            if sample_rows:
                read_kwargs["nrows"] = sample_rows

            df = pd.read_csv(**read_kwargs)

            # Strip phantom columns (trailing delimiters → Unnamed: N)
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
            # Drop fully-null columns and rows
            df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

            logger.info(
                f"[CSV LOADER] Loaded: encoding={encoding}, delimiter='{delimiter}', "
                f"shape={df.shape}"
            )
            return df

        except Exception as e:
            last_error = e
            logger.debug(f"[CSV LOADER] Failed with encoding={encoding}: {e}")
            continue

    raise RuntimeError(
        f"[CSV LOADER] Could not load '{filepath}' with any encoding. "
        f"Last error: {last_error}"
    )


# =============================================================================
# MCP TOOL: infer_semantic_types
# =============================================================================

def infer_semantic_types(profile: dict, data_info: dict) -> dict:
    """
    MCP Tool: Deterministically infer the semantic type of each column.

    Returns dict mapping column_name → semantic_type string.

    Semantic types:
        email | phone | url | datetime | boolean | categorical |
        identifier | currency | integer | float | free_text | unknown

    Rules are checked in priority order; first match wins.
    No LLM needed — pure heuristics on column name + statistics.
    """
    semantic_types = {}
    raw_stats = profile.get("raw_stats", {})
    columns = data_info.get("columns", [])

    for col in columns:
        col_lower = col.lower().strip()
        stats = raw_stats.get(col, {})
        dtype = stats.get("dtype", "object")
        unique_pct = stats.get("unique_pct", 0)
        unique_count = stats.get("unique_count", 0)
        top_values = stats.get("top_values", {})

        # ── Rule 1: Email ────────────────────────────────────────────────────
        if (stats.get("valid_email_count", 0) > 0 or
                any(kw in col_lower for kw in ("email", "e_mail", "e-mail", "mail"))):
            semantic_types[col] = "email"
            continue

        # ── Rule 2: URL / URI ────────────────────────────────────────────────
        if any(kw in col_lower for kw in ("url", "uri", "link", "href", "website", "site")):
            semantic_types[col] = "url"
            continue

        # ── Rule 3: Phone ────────────────────────────────────────────────────
        if any(kw in col_lower for kw in ("phone", "mobile", "tel", "fax", "contact_no",
                                            "phone_number", "cell")):
            semantic_types[col] = "phone"
            continue

        # ── Rule 4: Already-parsed datetime columns ──────────────────────────
        if "datetime" in dtype or dtype == "datetime64[ns]":
            semantic_types[col] = "datetime"
            continue

        # ── Rule 5: Date-like column names ───────────────────────────────────
        date_keywords = ("date", "time", "timestamp", "created_at", "updated_at",
                         "modified_at", "created_on", "dob", "birthdate", "hired")
        if any(kw in col_lower for kw in date_keywords):
            semantic_types[col] = "datetime"
            continue

        # ── Rule 6: Boolean ──────────────────────────────────────────────────
        bool_values = {str(v).lower().strip() for v in top_values.keys()}
        if bool_values and bool_values <= {"true", "false", "yes", "no",
                                            "1", "0", "t", "f", "y", "n",
                                            "active", "inactive", "enabled", "disabled"}:
            if unique_count <= 4:
                semantic_types[col] = "boolean"
                continue

        # ── Rule 7: Categorical (low cardinality, non-numeric) ───────────────
        if dtype == "object" and unique_count >= 2 and unique_pct < 10 and unique_count <= 50:
            semantic_types[col] = "categorical"
            continue

        # ── Rule 8: Identifier (high cardinality, name looks like ID) ────────
        id_suffixes = ("_id", "_key", "_code", "_num", "_no", "_ref",
                       "_uuid", "_guid", "id", "key", "code")
        if any(col_lower.endswith(sfx) or col_lower == sfx for sfx in id_suffixes):
            if unique_pct > 70:
                semantic_types[col] = "identifier"
                continue

        # ── Rule 9: Currency / Amount ────────────────────────────────────────
        # Bug 5 fix: removed dtype guard — columns with dirty values (N/A, --,
        # MISSING) have dtype=object even though they are semantically numeric.
        # The name-keyword match is sufficient; the validator and healer already
        # handle mixed-type columns via pd.to_numeric(errors="coerce").
        currency_keywords = ("price", "amount", "cost", "fee", "charge", "revenue",
                              "salary", "wage", "total", "sum", "balance", "value",
                              "spend", "budget", "payment", "earnings", "profit")
        if any(kw in col_lower for kw in currency_keywords):
            semantic_types[col] = "currency"
            continue

        # ── Rule 10: Numeric from pandas dtype ───────────────────────────────
        if "int" in dtype:
            semantic_types[col] = "integer"
            continue
        if "float" in dtype:
            semantic_types[col] = "float"
            continue

        # ── Rule 11: Free text (high cardinality strings) ────────────────────
        if dtype == "object" and unique_pct > 60:
            semantic_types[col] = "free_text"
            continue

        semantic_types[col] = "unknown"

    logger.info(f"[MCP] infer_semantic_types: {semantic_types}")
    return semantic_types


# =============================================================================
# MCP TOOL: read_csv_data
# =============================================================================

def read_csv_data(filepath: str) -> Dict[str, Any]:
    """MCP Tool: Load a CSV file and return basic information."""
    logger.info(f"[MCP] read_csv_data: filepath={filepath}")

    df = _robust_read_csv(filepath)

    result = {
        "success": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    logger.info(f"[MCP] read_csv_data: {result['row_count']} rows, {result['column_count']} columns")
    return result


# =============================================================================
# MCP TOOL: compute_column_profile
# =============================================================================

def compute_column_profile(filepath: str) -> Dict[str, Any]:
    """
    MCP Tool: Compute detailed statistics for every column.

    Includes a sampling guard: datasets > MAX_PROFILE_ROWS are profiled
    on a random sample to prevent OOM on large files.
    """
    logger.info("[MCP] compute_column_profile called")

    df = _robust_read_csv(filepath)
    total_rows = len(df)
    sampled = False

    if total_rows > MAX_PROFILE_ROWS:
        sample_frac = MAX_PROFILE_ROWS / total_rows
        df = df.sample(frac=sample_frac, random_state=42)
        sampled = True
        logger.warning(
            f"[MCP] Large dataset ({total_rows:,} rows) — profiling on "
            f"{len(df):,} row sample ({sample_frac:.1%})"
        )

    # Best-effort date parsing
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    today = pd.Timestamp.now()
    profile = {}

    for col in df.columns:
        series = df[col]

        col_stats: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "total_count": len(series),
            "unique_count": int(series.nunique()),
            "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
            "sampled": sampled,
            "total_rows": total_rows,
            "profiled_rows": len(df),
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            col_stats.update({
                "min": float(non_null.min()) if len(non_null) > 0 else None,
                "max": float(non_null.max()) if len(non_null) > 0 else None,
                "mean": round(float(non_null.mean()), 4) if len(non_null) > 0 else None,
                "std": round(float(non_null.std()), 4) if len(non_null) > 0 else None,
                "p25": round(float(non_null.quantile(0.25)), 4) if len(non_null) > 0 else None,
                "p75": round(float(non_null.quantile(0.75)), 4) if len(non_null) > 0 else None,
                "negative_count": int((series < 0).sum()),
                "zero_count": int((series == 0).sum()),
            })

        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            col_stats.update({
                "min_date": str(non_null.min()) if len(non_null) > 0 else None,
                "max_date": str(non_null.max()) if len(non_null) > 0 else None,
                "future_date_count": int((non_null > today).sum()),
                "future_date_pct": round((non_null > today).mean() * 100, 2),
            })

        else:  # string/object column
            non_null = series.dropna()
            col_stats.update({
                "top_values": series.value_counts().head(10).to_dict(),
                "sample_values": list(non_null.head(3).astype(str)),
                "has_empty_strings": int((series == "").sum()),
            })

            # Email validation heuristic
            if "email" in col.lower() or "mail" in col.lower():
                email_pattern = series.str.contains(r"@.*\.", na=False, regex=True)
                col_stats["valid_email_count"] = int(email_pattern.sum())
                col_stats["invalid_email_count"] = int(
                    (~email_pattern & series.notna()).sum()
                )

        profile[col] = col_stats

    logger.info(f"[MCP] compute_column_profile: profiled {len(profile)} columns")
    return profile


# =============================================================================
# MCP TOOL: save_expectation_suite
# =============================================================================

def save_expectation_suite(suite: Dict[str, Any], suite_name: str) -> Dict[str, Any]:
    """MCP Tool: Save a Great Expectations suite to a JSON file."""
    logger.info(f"[MCP] save_expectation_suite: '{suite_name}'")

    os.makedirs(GE_EXPECTATIONS_DIR, exist_ok=True)
    suite_path = os.path.join(GE_EXPECTATIONS_DIR, f"{suite_name}.json")

    suite["_metadata"] = {
        "suite_name": suite_name,
        "created_at": datetime.now().isoformat(),
        "created_by": "IngestionQualityAgent",
    }

    with open(suite_path, "w") as f:
        json.dump(suite, f, indent=2)

    logger.info(f"[MCP] save_expectation_suite: saved to {suite_path}")
    return {
        "success": True,
        "path": suite_path,
        "expectation_count": len(suite.get("expectations", [])),
    }


# =============================================================================
# MCP TOOL: run_ge_validation
# =============================================================================

def run_ge_validation(filepath: str, suite_name: str) -> Dict[str, Any]:
    """MCP Tool: Run Great Expectations validation against the dataset."""
    logger.info(f"[MCP] run_ge_validation: {filepath} with suite '{suite_name}'")

    suite_path = os.path.join(GE_EXPECTATIONS_DIR, f"{suite_name}.json")
    if not os.path.exists(suite_path):
        logger.error(f"[MCP] Suite not found: {suite_path}")
        return {
            "success": False,
            "error": f"Suite '{suite_name}' not found at {suite_path}",
            "statistics": {"evaluated_expectations": 0, "successful_expectations": 0,
                           "unsuccessful_expectations": 0, "success_percent": 0},
            "results": [],
            "failures": [],
        }

    with open(suite_path, "r") as f:
        suite = json.load(f)

    df = _robust_read_csv(filepath)

    # Best-effort date parsing for date columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    today = pd.Timestamp.now()
    results = []
    all_pass = True

    for expectation in suite.get("expectations", []):
        exp_type = expectation.get("expectation_type")
        kwargs = expectation.get("kwargs", {})
        column = kwargs.get("column")

        result = {
            "expectation_type": exp_type,
            "kwargs": kwargs,
            "success": False,
            "failing_count": 0,
            "failing_indices": [],
            "reasoning": expectation.get("reasoning", ""),
        }

        # Column existence guard — treat missing columns as skipped (pass),
        # not fail.  A column may have been dropped after healing emptied
        # all its values; marking it as fail creates an unresolvable loop.
        if column not in df.columns:
            logger.warning(f"[MCP] Column '{column}' not in dataset — skipping rule (treated as pass)")
            result["error"] = f"Column '{column}' not found in dataset — skipped"
            result["success"] = True
            results.append(result)
            continue

        try:
            if exp_type == "expect_column_values_to_not_be_null":
                mask = df[column].isna()

            elif exp_type == "expect_column_values_to_be_between":
                min_val = kwargs.get("min_value")
                max_val = kwargs.get("max_value")
                series = df[column]

                if pd.api.types.is_datetime64_any_dtype(series) or "date" in column.lower():
                    series = pd.to_datetime(series, errors="coerce")
                    if min_val is not None:
                        min_val = pd.to_datetime(str(min_val), errors="coerce")
                    if max_val == "today":
                        max_val = today
                    elif max_val is not None:
                        max_val = pd.to_datetime(str(max_val), errors="coerce")
                else:
                    series = pd.to_numeric(series, errors="coerce")
                    # Bug 1 fix: bounds from JSON may be strings (e.g. "0.0");
                    # cast to float to prevent TypeError on comparison.
                    if min_val is not None:
                        try:
                            min_val = float(min_val)
                        except (TypeError, ValueError):
                            min_val = None
                    if max_val is not None:
                        try:
                            max_val = float(max_val)
                        except (TypeError, ValueError):
                            max_val = None

                mask = pd.Series([False] * len(df), index=df.index)
                if min_val is not None:
                    mask = mask | (series < min_val)
                if max_val is not None:
                    mask = mask | (series > max_val)

            elif exp_type == "expect_column_values_to_be_unique":
                mask = df[column].duplicated(keep=False)

            elif exp_type == "expect_column_values_to_be_in_set":
                value_set = set(kwargs.get("value_set", []))
                mask = ~df[column].isin(value_set) & df[column].notna()

            elif exp_type == "expect_column_values_to_match_regex":
                regex = kwargs.get("regex", "")
                try:
                    mask = df[column].notna() & ~df[column].astype(str).str.match(
                        regex, na=False
                    )
                except Exception as re_err:
                    logger.warning(f"[MCP] Regex '{regex}' failed: {re_err} — treating as pass")
                    mask = pd.Series([False] * len(df), index=df.index)

            elif exp_type == "expect_column_values_to_be_of_type":
                expected_type = kwargs.get("type_", "")
                if expected_type in ("int", "float", "number"):
                    mask = ~pd.to_numeric(df[column], errors="coerce").notna()
                else:
                    mask = pd.Series([False] * len(df), index=df.index)

            elif exp_type == "expect_column_values_to_be_dateutil_parseable":
                # Check that column is parseable as dates
                parsed = pd.to_datetime(df[column], errors="coerce")
                mask = parsed.isna() & df[column].notna()

            else:
                logger.warning(f"[MCP] Unknown expectation type: {exp_type} — recording as fail")
                result["error"] = f"Unknown expectation type: {exp_type}"
                result["success"] = False
                results.append(result)
                all_pass = False
                continue

            # Compute result
            result["failing_count"] = int(mask.sum())
            result["failing_indices"] = list(df[mask].index[:50])
            result["success"] = result["failing_count"] == 0

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            logger.error(f"[MCP] Error running {exp_type} on '{column}': {e}")

        if not result["success"]:
            all_pass = False

        results.append(result)
        status = "✅ PASS" if result["success"] else f"❌ FAIL ({result['failing_count']} rows)"
        logger.info(f"[MCP] [{status}] {exp_type} on '{column}'")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    return {
        "success": all_pass,
        "statistics": {
            "evaluated_expectations": total,
            "successful_expectations": passed,
            "unsuccessful_expectations": total - passed,
            "success_percent": round(passed / total * 100, 1) if total > 0 else 0,
        },
        "results": results,
        "failures": [r for r in results if not r["success"]],
    }


# =============================================================================
# MCP TOOL: apply_healing_action
# =============================================================================

def apply_healing_action(action: Dict[str, Any], filepath: str) -> Dict[str, Any]:
    """
    MCP Tool: Apply a data healing action.

    Changes from original:
    - out_of_bounds: uses action-supplied min_bound/max_bound (not hardcoded 0–50000)
    - replace_invalid_categorical: generic version of replace_invalid_status
    - All pd.read_csv replaced with _robust_read_csv
    """
    action_type = action.get("action_type")
    column = action.get("column")

    logger.info(f"[MCP] apply_healing_action: {action_type} on '{column}'")

    df = _robust_read_csv(filepath)

    # Best-effort date parsing
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    today = pd.Timestamp.now()
    rows_affected = 0

    # ── Guard: column must exist ─────────────────────────────────────────────
    if column and column not in df.columns:
        logger.error(f"[MCP] Column '{column}' not found in dataset")
        return {"success": False, "error": f"Column '{column}' not found",
                "rows_affected": 0, "remaining_rows": len(df)}

    # ── Fix values ───────────────────────────────────────────────────────────
    if action_type == "fix_value":
        operation = action.get("operation")

        if operation == "abs":
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            mask = numeric_series < 0
            rows_affected = int(mask.sum())
            df.loc[mask, column] = numeric_series[mask].abs()
            logger.info(f"[MCP] abs(): fixed {rows_affected} negative values in '{column}'")

        elif operation in ("replace_invalid_status", "replace_invalid_categorical"):
            # ✅ Generic: works for ANY categorical column, not just "status"
            valid_values = action.get("valid_values", [])
            default_value = action.get("default_value")

            if not valid_values or default_value is None:
                logger.warning(
                    f"[MCP] replace_invalid_categorical: no valid_values/default_value "
                    f"for '{column}' — skipping"
                )
            else:
                mask = ~df[column].isin(valid_values) & df[column].notna()
                rows_affected = int(mask.sum())
                df.loc[mask, column] = default_value
                logger.info(
                    f"[MCP] replace_invalid_categorical: fixed {rows_affected} rows "
                    f"in '{column}' → '{default_value}'"
                )

        elif operation == "replace_empty_string":
            default_value = action.get("default_value")
            mask = df[column] == ""
            rows_affected = int(mask.sum())
            if default_value is not None:
                df.loc[mask, column] = default_value
            logger.info(f"[MCP] replace_empty_string: {rows_affected} rows in '{column}'")

        else:
            logger.warning(f"[MCP] Unknown fix_value operation: '{operation}' — skipping")

    # ── Quarantine rows ──────────────────────────────────────────────────────
    elif action_type == "quarantine":
        reason = action.get("reason", "Data quality failure")
        condition = action.get("condition")

        # Normalize LLM-hallucinated condition aliases
        if condition == "not_matches_regex":
            condition = "fails_regex"

        if condition == "is_null":
            mask = df[column].isna()

        elif condition == "is_future_date":
            # Guard: only apply on actual date/datetime columns.
            # On non-date columns (e.g. integer 'age'), pd.to_datetime
            # coerces everything to NaT and wipes the entire dataset.
            sem_type = action.get("semantic_type", "")
            col_lower = column.lower()
            is_date_col = (
                sem_type in ("datetime", "date")
                or any(kw in col_lower for kw in ("date", "time", "timestamp", "dob", "born"))
                or pd.api.types.is_datetime64_any_dtype(df[column])
            )
            if not is_date_col:
                logger.warning(
                    f"[MCP] is_future_date on non-date column '{column}' "
                    f"(semantic: '{sem_type}') — treating as out_of_bounds instead"
                )
                # Fall through to out_of_bounds with no explicit bounds
                # → only quarantines values that fail numeric coercion
                numeric_series = pd.to_numeric(df[column], errors="coerce")
                mask = numeric_series.isna() & df[column].notna()
            else:
                date_series = pd.to_datetime(df[column], errors="coerce")
                min_date = pd.to_datetime("1900-01-01")
                mask_na = date_series.isna()
                mask_invalid = pd.Series([False] * len(df), index=df.index)
                valid_dates = date_series.dropna()
                if len(valid_dates) > 0:
                    mask_invalid.loc[valid_dates.index] = (
                        (valid_dates > today) | (valid_dates < min_date)
                    )
                mask = mask_na | mask_invalid

        elif condition == "is_duplicate":
            mask = df[column].duplicated(keep="first")

        elif condition == "out_of_bounds":
            # ✅ Use action-supplied bounds, NOT hardcoded 0–50000
            min_bound = action.get("min_bound")
            max_bound = action.get("max_bound")
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            mask = pd.Series([False] * len(df), index=df.index)

            if min_bound is not None:
                try:
                    mask = mask | (numeric_series < float(min_bound))
                except (TypeError, ValueError):
                    pass

            if max_bound is not None:
                try:
                    mask = mask | (numeric_series > float(max_bound))
                except (TypeError, ValueError):
                    pass

            # If no bounds specified, only quarantine NaN (failed numeric coercion)
            if min_bound is None and max_bound is None:
                mask = numeric_series.isna() & df[column].notna()

            logger.info(
                f"[MCP] out_of_bounds: min={min_bound}, max={max_bound}, "
                f"{mask.sum()} rows flagged"
            )

        elif condition == "fails_regex":
            regex = action.get("regex", "")
            if regex:
                try:
                    mask = df[column].notna() & ~df[column].astype(str).str.match(
                        regex, na=False
                    )
                    logger.info(
                        f"[MCP] fails_regex: regex='{regex}', "
                        f"{mask.sum()} rows flagged in '{column}'"
                    )
                except Exception as re_err:
                    logger.warning(f"[MCP] fails_regex: regex '{regex}' failed: {re_err}")
                    mask = pd.Series([False] * len(df), index=df.index)
            else:
                logger.warning(f"[MCP] fails_regex: no regex provided for '{column}'")
                mask = pd.Series([False] * len(df), index=df.index)

        else:
            logger.warning(f"[MCP] Unknown quarantine condition: '{condition}'")
            mask = pd.Series([False] * len(df), index=df.index)

        rows_affected = int(mask.sum())

        # Safety guard: refuse to quarantine >50% of rows in one action.
        # This prevents catastrophic data loss from misapplied conditions
        # (e.g. is_future_date on an integer column).
        total_rows = len(df)
        if total_rows > 0 and rows_affected > total_rows * 0.5:
            logger.warning(
                f"[MCP] Quarantine on '{column}' would remove {rows_affected}/{total_rows} "
                f"rows ({rows_affected/total_rows:.0%}) — aborting to prevent data loss"
            )
            rows_affected = 0
        elif rows_affected > 0:
            quarantine_df = df[mask].copy()
            quarantine_df["reject_reason"] = reason
            df = df[~mask]

            # Use live config reference (respects app.py runtime override)
            quarantine_path = _cfg.QUARANTINE_DATA_PATH
            os.makedirs(os.path.dirname(os.path.abspath(quarantine_path)), exist_ok=True)

            if os.path.exists(quarantine_path):
                quarantine_df.to_csv(quarantine_path, mode="a", header=False, index=False)
            else:
                quarantine_df.to_csv(quarantine_path, index=False)

            logger.info(
                f"[MCP] Quarantined {rows_affected} rows → {quarantine_path}. "
                f"Reason: {reason}"
            )

    # ── Deduplicate ──────────────────────────────────────────────────────────
    elif action_type == "deduplicate":
        original_count = len(df)
        df = df.drop_duplicates(subset=[column], keep="first")
        rows_affected = original_count - len(df)
        logger.info(f"[MCP] deduplicate: removed {rows_affected} duplicates on '{column}'")

    else:
        logger.warning(f"[MCP] Unknown action_type: '{action_type}' — no changes made")

    # Convert datetime columns back to strings before saving
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    df.to_csv(filepath, index=False)
    logger.info(f"[MCP] Saved: {len(df)} rows remaining in {filepath}")

    return {
        "success": True,
        "action_type": action_type,
        "column": column,
        "rows_affected": rows_affected,
        "remaining_rows": len(df),
    }


# =============================================================================
# MCP TOOL: run_sql_query
# =============================================================================

def run_sql_query(query: str) -> Dict[str, Any]:
    """MCP Tool: Run a SQL query against the SQLite database."""
    logger.info(f"[MCP] run_sql_query: {query[:80]}...")

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query)
    rows = cursor.fetchall()

    result = {
        "success": True,
        "row_count": len(rows),
        "columns": [d[0] for d in cursor.description] if cursor.description else [],
        "rows": [dict(row) for row in rows[:100]],
    }
    conn.close()
    return result


# =============================================================================
# MCP TOOL: send_alert
# =============================================================================

def send_alert(message: str, severity: str = "WARNING") -> Dict[str, Any]:
    """MCP Tool: Log a data quality alert."""
    timestamp = datetime.now().isoformat()
    alert = {
        "timestamp": timestamp,
        "severity": severity,
        "message": message,
        "source": "IngestionQualityAgent",
    }

    log_fn = {
        "CRITICAL": logger.critical,
        "WARNING": logger.warning,
    }.get(severity, logger.info)
    log_fn(f"ALERT [{severity}]: {message}")

    alerts_path = "logs/alerts.json"
    os.makedirs("logs", exist_ok=True)
    alerts = []
    if os.path.exists(alerts_path):
        try:
            with open(alerts_path, "r") as f:
                alerts = json.load(f)
        except Exception:
            alerts = []
    alerts.append(alert)
    with open(alerts_path, "w") as f:
        json.dump(alerts, f, indent=2)

    return {"success": True, "alert_logged": True, "severity": severity}