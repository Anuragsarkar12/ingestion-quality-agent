# src/agents/profiler_agent.py
# Node 1: Profiler Agent — Universal Edition
#
# Changes from original:
#   - call_ollama(): exponential-backoff retry (up to 3 attempts)
#   - profile_data(): calls infer_semantic_types() and stores in state
#   - Removed all "orders" domain references from log messages

import json
import re
import time
import requests
import logging

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, RAW_DATA_PATH
from src.agents.state import AgentState
from src.mcp_tools import compute_column_profile, read_csv_data, infer_semantic_types

logger = logging.getLogger(__name__)


# =============================================================================
# OLLAMA WRAPPER WITH RETRY
# =============================================================================

def call_ollama(messages: list, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """
    Call the local Ollama server with exponential-backoff retry.

    Retries on:
      - ConnectionError  (Ollama not running / restarting)
      - Timeout          (model still loading — can take 30–90s on first call)
      - HTTP 5xx         (server-side errors)

    Does NOT retry on HTTP 4xx (client errors — won't improve with retry).

    Returns:
      The model's text response, or "{}" on complete failure
      (callers must handle the empty-JSON fallback).
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_BASE_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": 4096,
                        "temperature": 0,
                    },
                },
                timeout=300,
            )

            if response.status_code == 200:
                content = response.json()["message"]["content"]
                logger.debug(f"[OLLAMA] Response received ({len(content)} chars)")
                return content

            elif response.status_code < 500:
                # 4xx — don't retry
                response.raise_for_status()

            else:
                logger.warning(
                    f"[OLLAMA] Server error {response.status_code} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"[OLLAMA] Connection failed "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
        except requests.exceptions.Timeout:
            logger.warning(
                f"[OLLAMA] Request timed out "
                f"(attempt {attempt + 1}/{max_retries}) — "
                "Ollama may still be loading the model"
            )
        except Exception as e:
            logger.error(f"[OLLAMA] Unexpected error: {e}")
            break  # Non-retriable

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)   # 2s → 4s → 8s
            logger.info(f"[OLLAMA] Retrying in {delay:.0f}s...")
            time.sleep(delay)

    logger.error(
        f"[OLLAMA] All {max_retries} attempts failed — "
        "returning empty JSON for deterministic fallback"
    )
    return "{}"


# =============================================================================
# JSON EXTRACTOR
# =============================================================================

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM (Ollama) response text.

    Handles:
    - Clean JSON
    - JSON inside markdown fences
    - JSON with pre/post text
    - Invalid escape sequences
    - Truncated JSON (trim-to-last-brace recovery)

    Returns {} on complete failure (never raises).
    """

    def sanitise_escapes(s: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu0-9])', r'\\\\', s)

    def try_parse(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return json.loads(sanitise_escapes(s))
            except json.JSONDecodeError:
                return None

    def extract_brace_block(s: str):
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(s[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    text = text.strip()

    # Attempt 1: direct parse
    result = try_parse(text)
    if result is not None:
        return result

    # Attempt 2: extract from markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        candidate = extract_brace_block(fence.group(1))
        if candidate:
            result = try_parse(candidate)
            if result is not None:
                return result

    # Attempt 3: extract first JSON object from raw text
    candidate = extract_brace_block(text)
    if candidate:
        result = try_parse(candidate)
        if result is not None:
            return result

    # Attempt 4: trim-to-last-brace recovery
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end]
        result = try_parse(candidate)
        if result is not None:
            logger.warning("[OLLAMA] Recovered JSON via trim-to-last-brace")
            return result

    logger.error(
        f"[OLLAMA] Failed to extract JSON after all attempts. "
        f"Response ({len(text)} chars): {text[:200]}..."
    )
    return {}


# =============================================================================
# PROFILER AGENT NODE
# =============================================================================

def profile_data(state: AgentState) -> AgentState:
    """
    LangGraph Node 1: Profile the incoming CSV and infer semantic column types.

    Steps:
      1. Load raw stats via MCP tools (deterministic)
      2. Infer semantic types deterministically (no LLM)
      3. Send stats to Ollama for human-readable analysis
      4. Apply fallback profile if LLM fails
      5. Return enriched state with profile + semantic_types
    """
    logger.info("=" * 60)
    logger.info("🔍 [PROFILER AGENT] Starting data profiling...")
    logger.info("=" * 60)

    csv_path = state.get("csv_path", RAW_DATA_PATH)

    # ── Step 1: Gather statistics via MCP tools ───────────────────────────────
    logger.info(f"[PROFILER] MCP: compute_column_profile({csv_path})")
    profile_stats = compute_column_profile(csv_path)

    logger.info(f"[PROFILER] MCP: read_csv_data({csv_path})")
    data_info = read_csv_data(csv_path)

    logger.info(
        f"[PROFILER] Dataset: {data_info['row_count']} rows, "
        f"{data_info['column_count']} columns"
    )

    # ── Step 2: Infer semantic types (deterministic — no LLM) ─────────────────
    semantic_types = infer_semantic_types(
        {"raw_stats": profile_stats}, data_info
    )
    logger.info(f"[PROFILER] Semantic types inferred: {semantic_types}")

    # ── Step 3: Ask LLM for human-readable quality analysis ──────────────────
    system_prompt = """\
You are a senior data engineer analyzing a dataset for quality issues.
You receive per-column statistics and must identify data quality problems.

CRITICAL: Respond ONLY with a valid JSON object. No text before or after. No markdown.

Use exactly this structure:
{
  "dataset_summary": "<one paragraph>",
  "total_rows": <number>,
  "total_columns": <number>,
  "column_analyses": {
    "<column_name>": {
      "health_status": "HEALTHY" | "WARNING" | "CRITICAL",
      "issues_found": ["..."],
      "recommended_validations": ["..."],
      "notes": "..."
    }
  },
  "overall_quality_score": <0–100>,
  "priority_issues": ["top 3–5 issues"]
}"""

    # Compact stats for LLM (avoid token explosion on wide datasets)
    compact_stats = {
        col: {
            k: v for k, v in stats.items()
            if k in ("dtype", "null_pct", "unique_pct", "min", "max",
                     "negative_count", "future_date_pct",
                     "valid_email_count", "invalid_email_count",
                     "top_values", "has_empty_strings")
        }
        for col, stats in profile_stats.items()
    }

    user_prompt = (
        f"Dataset: {data_info['row_count']} rows, {data_info['column_count']} columns\n"
        f"Columns: {data_info['columns']}\n"
        f"Semantic types (pre-inferred): {semantic_types}\n\n"
        f"Column statistics:\n{json.dumps(compact_stats, indent=2)}\n\n"
        "Return the JSON analysis object only."
    )

    logger.info(f"[PROFILER] Calling Ollama ({OLLAMA_MODEL})...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm_response = call_ollama(messages)
    profile_analysis = extract_json(llm_response)

    # ── Step 4: Fallback if LLM failed ───────────────────────────────────────
    if not profile_analysis:
        logger.warning("[PROFILER] LLM parsing failed — applying deterministic fallback profile")
        profile_analysis = {
            "dataset_summary": (
                f"Deterministic fallback profile. "
                f"{data_info['row_count']} rows, {data_info['column_count']} columns."
            ),
            "total_rows": data_info.get("row_count", 0),
            "total_columns": data_info.get("column_count", 0),
            "column_analyses": {
                col: {
                    "health_status": _heuristic_health(profile_stats.get(col, {})),
                    "issues_found": _heuristic_issues(profile_stats.get(col, {}), col),
                    "recommended_validations": [],
                    "notes": "Deterministic analysis (LLM unavailable)",
                }
                for col in data_info.get("columns", [])
            },
            "overall_quality_score": _heuristic_quality_score(profile_stats),
            "priority_issues": ["LLM unavailable — proceeding with deterministic rules"],
        }

    # ── Step 5: Log results ───────────────────────────────────────────────────
    logger.info(f"[PROFILER] Quality score: {profile_analysis.get('overall_quality_score')}/100")
    for issue in profile_analysis.get("priority_issues", []):
        logger.info(f"  ⚠ {issue}")
    for col_name, col_analysis in profile_analysis.get("column_analyses", {}).items():
        status = col_analysis.get("health_status", "UNKNOWN")
        emoji = {"HEALTHY": "✅", "WARNING": "⚠️", "CRITICAL": "🔴"}.get(status, "❓")
        logger.info(f"  {emoji} {col_name} [{semantic_types.get(col_name, '?')}]: {status}")
        for issue in col_analysis.get("issues_found", []):
            logger.info(f"      → {issue}")

    # ── Step 6: Build suite name from filename ────────────────────────────────
    import os
    import re as re_mod
    base = os.path.splitext(os.path.basename(csv_path))[0]
    suite_name = re_mod.sub(r"[^a-z0-9_]", "_", base.lower()) + "_quality_suite"
    logger.info(f"[PROFILER] Dynamic suite name: '{suite_name}'")

    messages_log = state.get("messages", [])
    messages_log.append({
        "agent": "profiler",
        "step": "profile_complete",
        "content": (
            f"Profiled {data_info['row_count']} rows, "
            f"{data_info['column_count']} columns via Ollama ({OLLAMA_MODEL}). "
            f"Quality score: {profile_analysis.get('overall_quality_score')}/100. "
            f"Semantic types: {semantic_types}."
        ),
    })

    logger.info("[PROFILER] ✅ Profiling complete. Passing to Rule Generator.")

    return {
        **state,
        "profile": {
            "raw_stats": profile_stats,
            "analysis": profile_analysis,
            "data_info": data_info,
            "semantic_types": semantic_types,
        },
        "profile_summary": profile_analysis.get("dataset_summary", ""),
        "semantic_types": semantic_types,
        "suite_name": suite_name,
        "messages": messages_log,
    }


# =============================================================================
# DETERMINISTIC FALLBACK HELPERS
# =============================================================================

def _heuristic_health(stats: dict) -> str:
    """Assign a health status without LLM based on null% and negative counts."""
    null_pct = stats.get("null_pct", 0)
    neg = stats.get("negative_count", 0)
    fut = stats.get("future_date_pct", 0)
    invalid_email = stats.get("invalid_email_count", 0)

    if null_pct > 20 or fut > 5 or invalid_email > 0:
        return "CRITICAL"
    if null_pct > 5 or neg > 0:
        return "WARNING"
    return "HEALTHY"


def _heuristic_issues(stats: dict, col: str) -> list:
    """List detected issues without LLM."""
    issues = []
    if stats.get("null_pct", 0) > 5:
        issues.append(f"{stats['null_pct']}% null values")
    if stats.get("negative_count", 0) > 0:
        issues.append(f"{stats['negative_count']} negative values")
    if stats.get("future_date_pct", 0) > 0:
        issues.append(f"{stats['future_date_pct']}% future dates")
    if stats.get("invalid_email_count", 0) > 0:
        issues.append(f"{stats['invalid_email_count']} invalid email addresses")
    if stats.get("has_empty_strings", 0) > 0:
        issues.append(f"{stats['has_empty_strings']} empty strings")
    return issues


def _heuristic_quality_score(profile_stats: dict) -> int:
    """Compute a simple 0–100 quality score without LLM."""
    if not profile_stats:
        return 50
    total_cols = len(profile_stats)
    penalty = 0
    for stats in profile_stats.values():
        null_pct = stats.get("null_pct", 0)
        penalty += min(null_pct / 100, 0.3)  # cap at 30% per column
        if stats.get("negative_count", 0) > 0:
            penalty += 0.1
        if stats.get("future_date_pct", 0) > 0:
            penalty += 0.2
    avg_penalty = penalty / total_cols if total_cols > 0 else 0
    return max(0, int(100 - avg_penalty * 100))