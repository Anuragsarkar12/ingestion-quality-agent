# app.py
# Streamlit UI — Universal Edition
#
# Changes from original:
#   - Removed import / display of VALID_ORDER_STATUSES (domain-specific).
#     Sidebar now shows inferred semantic types after a run.
#   - initial_state uses build_initial_state() — all new fields included.
#   - Removed redundant shutil.copy after graph.invoke() — mark_success()
#     handles it; the copy here was also writing to the wrong path.
#   - "Final Rows" metric fixed: now shows actual clean-row count instead of
#     evaluated_expectations (rule count), which was factually wrong.
#   - Added "Quarantined" as a separate metric column.
#   - Healing Ledger: replaced st.table(pd.DataFrame(healing_actions)) with
#     a flattened, readable dataframe — the nested-dict version crashed on
#     some pandas versions and displayed unreadable Python repr strings.
#   - Upload preview: replaced bare pd.read_csv with _robust_read_csv so
#     non-UTF-8 uploads preview correctly.
#   - Session state: initialise all required keys at startup so widgets
#     that reference them never get KeyError on first load.
#   - Added run timestamp + filename display to sidebar.
#   - Agent Intelligence tab: added semantic types panel and repair confidence.

import os
import sys
import logging
import tempfile
import threading
import queue
from datetime import datetime

import pandas as pd
import sqlite3
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    MAX_HEALING_ITERATIONS,
)
from src.graph.workflow import build_workflow, build_initial_state
from src.mcp_tools import _robust_read_csv
import src.config as _cfg


# =============================================================================
# CSS
# =============================================================================

def inject_custom_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Round');

        /* ── Material 3 Root Tokens ─────────────────────────────── */
        :root {
            --md-primary: #4F46E5; --md-primary-dark: #4338CA;
            --md-primary-light: #818CF8; --md-primary-container: #E0E7FF;
            --md-on-primary: #FFF; --md-on-primary-container: #312E81;
            --md-secondary: #7C3AED; --md-secondary-container: #EDE9FE;
            --md-tertiary: #0891B2; --md-tertiary-container: #CFFAFE;
            --md-surface: #FFFFFF; --md-surface-dim: #F8FAFC;
            --md-surface-container: #F1F5F9; --md-surface-container-high: #E2E8F0;
            --md-on-surface: #0F172A; --md-on-surface-variant: #475569;
            --md-outline: #94A3B8; --md-outline-variant: #E2E8F0;
            --md-error: #DC2626; --md-error-container: #FEE2E2;
            --md-success: #059669; --md-success-container: #D1FAE5;
            --md-warning: #D97706; --md-warning-container: #FEF3C7;
            --md-elev1: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
            --md-elev2: 0 4px 6px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.05);
            --md-elev3: 0 10px 15px rgba(0,0,0,.06), 0 4px 6px rgba(0,0,0,.04);
            --md-radius-sm: 8px; --md-radius-md: 12px;
            --md-radius-lg: 16px; --md-radius-xl: 28px;
        }

        /* ── Global Base ────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .stApp { background: linear-gradient(180deg, #F0F2FF 0%, var(--md-surface-dim) 30%) !important; }
        .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }
        h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--md-on-surface) !important; }
        h1 { font-weight: 800 !important; letter-spacing: -0.025em !important; }
        h2, h3 { font-weight: 700 !important; letter-spacing: -0.015em !important; }

        /* ── App Bar / Header ───────────────────────────────────── */
        .app-bar {
            background: linear-gradient(135deg, #4338CA 0%, #6D28D9 50%, #7C3AED 100%);
            border-radius: var(--md-radius-lg); padding: 28px 36px;
            margin-bottom: 28px; position: relative; overflow: hidden;
            box-shadow: var(--md-elev3);
        }
        .app-bar::before {
            content: ''; position: absolute; top: -50%; right: -20%;
            width: 500px; height: 500px; border-radius: 50%;
            background: rgba(255,255,255,0.05);
        }
        .app-bar::after {
            content: ''; position: absolute; bottom: -60%; left: 30%;
            width: 400px; height: 400px; border-radius: 50%;
            background: rgba(255,255,255,0.03);
        }
        .app-bar-content { display: flex; align-items: center; gap: 20px; position: relative; z-index: 1; }
        .app-bar-icon {
            width: 56px; height: 56px; border-radius: 16px;
            background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
            display: flex; align-items: center; justify-content: center;
            font-size: 28px; border: 1px solid rgba(255,255,255,0.2);
        }
        .app-bar-title { font-size: 26px; font-weight: 800; color: #FFF; letter-spacing: -0.02em; }
        .app-bar-subtitle { font-size: 14px; color: rgba(255,255,255,0.8); margin-top: 2px; font-weight: 400; }
        .app-bar-badge {
            margin-left: auto; background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
            padding: 6px 16px; font-size: 12px; font-weight: 600;
            color: #FFF; backdrop-filter: blur(10px);
        }

        /* ── Sidebar ────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E1B4B 0%, #1E293B 100%) !important;
            border-right: none !important;
        }
        [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
        [data-testid="stSidebar"] .stMetric {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: var(--md-radius-md) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #FFF !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: #94A3B8 !important; font-size: 11px !important;
            text-transform: uppercase !important; letter-spacing: 0.08em !important;
        }
        [data-testid="stSidebar"] .stAlert {
            background: rgba(79,70,229,0.15) !important;
            border: 1px solid rgba(79,70,229,0.3) !important;
            border-radius: var(--md-radius-sm) !important;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 13px !important; text-transform: uppercase !important;
            letter-spacing: 0.1em !important; color: #94A3B8 !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }

        /* ── Tabs ───────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0; background: var(--md-surface); border-radius: var(--md-radius-md);
            padding: 4px; box-shadow: var(--md-elev1);
            border: 1px solid var(--md-outline-variant);
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px; border-radius: var(--md-radius-sm); color: var(--md-on-surface-variant);
            font-weight: 500; font-size: 13px; padding: 0 16px;
            transition: all 200ms cubic-bezier(.4,0,.2,1);
            border-bottom: none !important; white-space: nowrap;
        }
        .stTabs [data-baseweb="tab"]:hover { background: var(--md-primary-container); color: var(--md-primary); }
        .stTabs [aria-selected="true"] {
            background: var(--md-primary) !important; color: var(--md-on-primary) !important;
            font-weight: 600 !important; box-shadow: var(--md-elev2);
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

        /* ── Cards (containers with border) ──────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--md-radius-lg) !important;
            border: 1px solid var(--md-outline-variant) !important;
            box-shadow: var(--md-elev1) !important;
            background: var(--md-surface) !important;
            transition: box-shadow 200ms ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: var(--md-elev2) !important;
        }

        /* ── Metrics ────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--md-surface) !important;
            border: 1px solid var(--md-outline-variant) !important;
            border-radius: var(--md-radius-lg) !important;
            box-shadow: var(--md-elev1) !important; padding: 20px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 28px !important; font-weight: 700 !important;
            color: var(--md-on-surface) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 11px !important; font-weight: 600 !important;
            text-transform: uppercase !important; letter-spacing: 0.06em !important;
            color: var(--md-on-surface-variant) !important;
        }

        /* ── Buttons ────────────────────────────────────────────── */
        .stButton > button {
            border-radius: var(--md-radius-sm) !important; font-weight: 600 !important;
            font-size: 14px !important; padding: 10px 24px !important;
            transition: all 200ms cubic-bezier(.4,0,.2,1) !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--md-elev2) !important;
        }
        .stButton > button:active { transform: translateY(0) !important; }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--md-primary) 0%, var(--md-secondary) 100%) !important;
            border: none !important; color: #FFF !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 15px rgba(79,70,229,0.4) !important;
        }
        .stDownloadButton > button {
            border-radius: var(--md-radius-sm) !important; font-weight: 600 !important;
            transition: all 200ms ease !important;
        }
        .stDownloadButton > button:hover {
            transform: translateY(-1px) !important; box-shadow: var(--md-elev2) !important;
        }

        /* ── File Uploader (Dropzone) ───────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--md-primary-light) !important;
            border-radius: var(--md-radius-lg) !important;
            background: linear-gradient(135deg, rgba(79,70,229,0.03) 0%, rgba(124,58,237,0.03) 100%) !important;
            padding: 24px !important;
            transition: all 250ms ease !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--md-primary) !important;
            background: linear-gradient(135deg, rgba(79,70,229,0.06) 0%, rgba(124,58,237,0.06) 100%) !important;
            box-shadow: 0 0 0 4px rgba(79,70,229,0.1) !important;
        }
        [data-testid="stFileUploader"] section { padding: 0 !important; }
        [data-testid="stFileUploader"] button {
            background: var(--md-primary) !important; color: #FFF !important;
            border-radius: var(--md-radius-sm) !important; font-weight: 600 !important;
            border: none !important;
        }

        /* ── DataFrames ─────────────────────────────────────────── */
        .stDataFrame {
            border-radius: var(--md-radius-md) !important;
            border: 1px solid var(--md-outline-variant) !important;
            box-shadow: var(--md-elev1) !important; overflow: hidden !important;
        }

        /* ── Expanders (Accordions) ──────────────────────────── */
        [data-testid="stExpander"] {
            border: 1px solid var(--md-outline-variant) !important;
            border-radius: var(--md-radius-md) !important;
            box-shadow: var(--md-elev1) !important;
            background: var(--md-surface) !important;
            transition: box-shadow 200ms ease !important;
        }
        [data-testid="stExpander"]:hover { box-shadow: var(--md-elev2) !important; }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important; font-size: 14px !important;
        }

        /* ── Text Area (SQL Editor etc.) ────────────────────── */
        .stTextArea textarea {
            border-radius: var(--md-radius-sm) !important;
            border: 1px solid var(--md-outline-variant) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important; line-height: 1.6 !important;
            background: var(--md-surface-dim) !important;
            transition: border-color 200ms ease, box-shadow 200ms ease !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--md-primary) !important;
            box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
        }

        /* ── Selectbox ──────────────────────────────────────────── */
        [data-baseweb="select"] > div {
            border-radius: var(--md-radius-sm) !important;
            border-color: var(--md-outline-variant) !important;
        }

        /* ── Log Console ────────────────────────────────────────── */
        .log-container {
            background: linear-gradient(180deg, #0C1222 0%, #0F172A 100%);
            color: #E2E8F0; padding: 20px; border-radius: var(--md-radius-md);
            font-family: 'JetBrains Mono', monospace; font-size: 12px;
            line-height: 1.7; height: 420px; overflow-y: auto;
            border: 1px solid #1E293B; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }
        .log-container::-webkit-scrollbar { width: 6px; }
        .log-container::-webkit-scrollbar-track { background: transparent; }
        .log-container::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

        /* ── Stat Cards ─────────────────────────────────────────── */
        .stat-card {
            border-radius: var(--md-radius-lg); padding: 24px 20px;
            position: relative; overflow: hidden;
            transition: transform 200ms ease, box-shadow 200ms ease;
        }
        .stat-card:hover { transform: translateY(-2px); }
        .stat-card-icon { font-size: 24px; margin-bottom: 12px; }
        .stat-card-label {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.08em; margin-bottom: 6px;
        }
        .stat-card-value { font-size: 28px; font-weight: 800; color: var(--md-on-surface); }

        /* ── Chips ───────────────────────────────────────────────── */
        .mui-chip {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 14px; border-radius: 999px; font-size: 12px;
            font-weight: 600; letter-spacing: 0.02em;
        }
        .mui-chip-primary { background: var(--md-primary-container); color: var(--md-on-primary-container); }
        .mui-chip-success { background: var(--md-success-container); color: #065F46; }
        .mui-chip-warning { background: var(--md-warning-container); color: #92400E; }
        .mui-chip-error { background: var(--md-error-container); color: #991B1B; }
        .mui-chip-info { background: #DBEAFE; color: #1E40AF; }

        /* ── Timeline / Stepper ──────────────────────────────── */
        .timeline-item {
            display: flex; gap: 16px; padding: 16px 0;
            position: relative;
        }
        .timeline-item:not(:last-child)::after {
            content: ''; position: absolute; left: 19px; top: 52px;
            width: 2px; bottom: 0; background: var(--md-outline-variant);
        }
        .timeline-dot {
            width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; box-shadow: var(--md-elev1);
        }
        .timeline-content { flex: 1; min-width: 0; }
        .timeline-agent {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; margin-bottom: 4px;
        }
        .timeline-step { font-size: 12px; color: var(--md-on-surface-variant); font-weight: 500; }
        .timeline-text {
            font-size: 13px; color: var(--md-on-surface-variant);
            margin-top: 6px; line-height: 1.5;
        }

        /* ── Section Header ─────────────────────────────────── */
        .section-header {
            display: flex; align-items: center; gap: 12px;
            margin-bottom: 20px; padding-bottom: 12px;
            border-bottom: 2px solid var(--md-outline-variant);
        }
        .section-icon {
            width: 40px; height: 40px; border-radius: var(--md-radius-sm);
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
        }
        .section-title { font-size: 20px; font-weight: 700; color: var(--md-on-surface); }
        .section-subtitle { font-size: 13px; color: var(--md-on-surface-variant); }

        /* ── Risk Banner ────────────────────────────────────── */
        .risk-banner {
            padding: 20px 24px; border-radius: var(--md-radius-lg);
            display: flex; align-items: center; gap: 16px;
            box-shadow: var(--md-elev1); margin-bottom: 24px;
        }
        .risk-icon { font-size: 36px; }
        .risk-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
        .risk-value { font-size: 24px; font-weight: 800; }

        /* ── Empty State ────────────────────────────────────── */
        .empty-state {
            text-align: center; padding: 60px 40px;
            background: var(--md-surface); border-radius: var(--md-radius-lg);
            border: 2px dashed var(--md-outline-variant);
        }
        .empty-state-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.6; }
        .empty-state-title { font-size: 18px; font-weight: 600; color: var(--md-on-surface); margin-bottom: 8px; }
        .empty-state-text { font-size: 14px; color: var(--md-on-surface-variant); }

        /* ── Divider ────────────────────────────────────────── */
        hr { border: none !important; border-top: 1px solid var(--md-outline-variant) !important; margin: 24px 0 !important; }

        /* ── Alerts ─────────────────────────────────────────── */
        .stAlert { border-radius: var(--md-radius-sm) !important; }

        /* ── Scrollbar ──────────────────────────────────────── */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

        /* ── Animation ──────────────────────────────────────── */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-in { animation: fadeInUp 0.4s ease-out; }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# LOGGING INFRASTRUCTURE
# =============================================================================

class QueueLogHandler(logging.Handler):
    """Push formatted log records into a queue for real-time display."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        self.log_queue.put(self.format(record))


def setup_queue_logging(log_queue: queue.Queue) -> None:
    handler = QueueLogHandler(log_queue)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def format_log_line(line: str) -> str:
    """Colour-code a single log line for the terminal display."""
    if any(t in line for t in ("[ERROR]", "[CRITICAL]", "❌")):
        color = "#F87171"
    elif any(t in line for t in ("[WARNING]", "⚠️")):
        color = "#FB923C"
    elif any(t in line for t in ("✅", "PASS", "🎉", "SUCCESS")):
        color = "#34D399"
    elif any(t in line for t in ("🔍", "📋", "🔧", "🔨")):
        color = "#818CF8"
    else:
        color = "#94A3B8"
    safe = line.replace("<", "&lt;").replace(">", "&gt;")
    return f'<div style="color:{color};">{safe}</div>'


# =============================================================================
# PIPELINE RUNNER (background thread)
# =============================================================================

def run_pipeline(
    csv_path: str,
    result_container: dict,
    log_queue: queue.Queue,
) -> None:
    """
    Run the full LangGraph pipeline in a background thread.

    Sets up dynamic output paths inside the same temp directory as the
    uploaded CSV so multiple concurrent sessions don't collide.
    """
    setup_queue_logging(log_queue)

    base_dir       = os.path.dirname(csv_path)
    clean_path     = os.path.join(base_dir, "clean_output.csv")
    quarantine_path = os.path.join(base_dir, "quarantine_output.csv")

    # Clear stale outputs from any previous run in this temp dir
    for f in (clean_path, quarantine_path):
        if os.path.exists(f):
            os.remove(f)

    # Override config paths so mark_success() and apply_healing_action()
    # write to the per-session temp directory, not the global config path.
    _cfg.QUARANTINE_DATA_PATH = quarantine_path
    _cfg.PROCESSED_DATA_PATH  = clean_path

    try:
        graph         = build_workflow()
        initial_state = build_initial_state(csv_path)

        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 50},
        )
        # mark_success() writes clean_path; no shutil.copy needed here.

        result_container.update({
            "final_state":     final_state,
            "clean_path":      clean_path,
            "quarantine_path": quarantine_path,
            "success":         True,
        })

    except Exception as e:
        logging.getLogger("pipeline").error(f"Pipeline error: {e}", exc_info=True)
        result_container.update({"success": False, "error": str(e)})

    finally:
        log_queue.put("__DONE__")


# =============================================================================
# HELPERS
# =============================================================================

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV with robust encoding detection; return empty DataFrame on error."""
    try:
        return _robust_read_csv(path)
    except Exception as e:
        logging.getLogger("app").warning(f"Could not read {path}: {e}")
        return pd.DataFrame()


def _flatten_healing_actions(healing_actions: list) -> pd.DataFrame:
    """
    Convert the nested list-of-dicts healing_actions into a flat DataFrame
    safe for st.dataframe().

    Original format per item:
        {"action": {"action_type": ..., "column": ..., ...},
         "result": {"rows_affected": ..., "remaining_rows": ..., ...},
         "reasoning": "...",
         "confidence": 0.82}
    """
    rows = []
    for ha in healing_actions:
        action  = ha.get("action", {})
        result  = ha.get("result", {})
        conf    = ha.get("confidence")
        rows.append({
            "Column":         action.get("column", "N/A"),
            "Action":         action.get("action_type", "N/A"),
            "Detail":         (action.get("operation")
                               or action.get("condition", "—")),
            "Rows Affected":  result.get("rows_affected", 0),
            "Rows Remaining": result.get("remaining_rows", "N/A"),
            "Confidence":     f"{conf:.0%}" if conf is not None else "N/A",
            "Reasoning":      (ha.get("reasoning") or "")[:90],
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# MATERIAL UI HTML HELPERS
# =============================================================================

def _html_stat_card(icon: str, label: str, value: str, bg: str, fg: str) -> str:
    """Return an HTML stat card with gradient background."""
    return (
        f'<div class="stat-card" style="background:linear-gradient(135deg,{bg} 0%,#FFF 100%);'
        f'border:1px solid {fg}20;box-shadow:0 1px 3px {fg}15;">'
        f'<div class="stat-card-icon">{icon}</div>'
        f'<div class="stat-card-label" style="color:{fg};">{label}</div>'
        f'<div class="stat-card-value">{value}</div>'
        f'</div>'
    )


def _html_chip(text: str, variant: str = "primary") -> str:
    """Return an inline HTML chip span."""
    return f'<span class="mui-chip mui-chip-{variant}">{text}</span>'


def _html_section(title: str, icon: str, subtitle: str = "", bg: str = "#E0E7FF") -> str:
    """Return a Material section header."""
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="section-header">'
        f'<div class="section-icon" style="background:{bg};">{icon}</div>'
        f'<div><div class="section-title">{title}</div>{sub}</div>'
        f'</div>'
    )


def _html_timeline_step(agent: str, step: str, content: str, bg: str, emoji: str) -> str:
    """Return one timeline step for the reasoning trace."""
    safe_content = (content or "").replace("<", "&lt;").replace(">", "&gt;")[:200]
    return (
        f'<div class="timeline-item">'
        f'<div class="timeline-dot" style="background:{bg};">{emoji}</div>'
        f'<div class="timeline-content">'
        f'<div class="timeline-agent">{agent}</div>'
        f'<div class="timeline-step">{step}</div>'
        f'<div class="timeline-text">{safe_content}</div>'
        f'</div></div>'
    )


def _html_risk_banner(level: str, score: float, summary: str) -> str:
    """Return a risk level banner."""
    styles = {
        "LOW":      ("#D1FAE5", "#059669", "🟢"),
        "MEDIUM":   ("#FEF3C7", "#D97706", "🟡"),
        "HIGH":     ("#FED7AA", "#EA580C", "🟠"),
        "CRITICAL": ("#FEE2E2", "#DC2626", "🔴"),
    }
    bg, fg, dot = styles.get(level, ("#F1F5F9", "#475569", "⚪"))
    return (
        f'<div class="risk-banner" style="background:{bg};border:1px solid {fg}30;">'
        f'<div class="risk-icon">{dot}</div>'
        f'<div>'
        f'<div class="risk-label" style="color:{fg};">Overall Risk</div>'
        f'<div class="risk-value" style="color:{fg};">{level} ({score:.0%})</div>'
        f'<div style="font-size:13px;color:{fg};opacity:0.8;margin-top:4px;">{summary}</div>'
        f'</div></div>'
    )


def _html_empty_state(icon: str, title: str, text: str) -> str:
    """Return an empty state placeholder."""
    return (
        f'<div class="empty-state">'
        f'<div class="empty-state-icon">{icon}</div>'
        f'<div class="empty-state-title">{title}</div>'
        f'<div class="empty-state-text">{text}</div>'
        f'</div>'
    )


# =============================================================================
# MAIN UI
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="DataArmor AI | Agentic Quality",
        page_icon="🛡️",
        layout="wide",
    )
    inject_custom_css()

    # ── Session state initialisation ─────────────────────────────────────────
    # All keys declared upfront — prevents KeyError on first load and after
    # page navigation.
    _ss_defaults = {
        "pipeline_results":  None,
        "run_timestamp":     None,
        "uploaded_filename": None,
    }
    for key, default in _ss_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="app-bar animate-in">
            <div class="app-bar-content">
                <div class="app-bar-icon">🛡️</div>
                <div>
                    <div class="app-bar-title">DataArmor AI</div>
                    <div class="app-bar-subtitle">Autonomous agentic data quality &amp; self-healing — any CSV, any schema</div>
                </div>
                <div class="app-bar-badge">✨ Agentic Pipeline</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_upload, tab_intel, tab_results, tab_db, tab_lineage, tab_pii, tab_catalog, tab_graph = st.tabs([
        "📤 Ingestion",
        "🧠 Agent Intelligence",
        "💎 Final Clean Room",
        "🗄️ DB Explorer",
        "🧬 Lineage Analyzer",
        "🔐 PII & Governance",
        "📚 Metadata Catalog",
        "📈 Lineage Graph",
    ])

    # =========================================================================
    # TAB 1 — INGESTION
    # =========================================================================
    with tab_upload:
        st.markdown(_html_section("Data Ingestion", "📤", "Upload any CSV to begin autonomous quality analysis", "#E0E7FF"), unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop any CSV here — orders, healthcare, sensor data, HR, finance…",
            type=["csv"],
        )

        if uploaded_file:
            # Preview using robust loader (handles non-UTF-8 uploads)
            try:
                import io
                raw_bytes  = uploaded_file.read()
                uploaded_file.seek(0)
                tmp_preview = tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False
                )
                tmp_preview.write(raw_bytes)
                tmp_preview.flush()
                tmp_preview.close()
                preview_df = _safe_read_csv(tmp_preview.name)
                os.unlink(tmp_preview.name)
            except Exception:
                preview_df = pd.DataFrame()

            if not preview_df.empty:
                total_cells = preview_df.size or 1
                null_pct = preview_df.isna().sum().sum() / total_cells * 100
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(_html_stat_card("📊", "Total Rows", f"{len(preview_df):,}", "#E0E7FF", "#4F46E5"), unsafe_allow_html=True)
                with c2:
                    st.markdown(_html_stat_card("📋", "Columns", str(len(preview_df.columns)), "#EDE9FE", "#7C3AED"), unsafe_allow_html=True)
                with c3:
                    st.markdown(_html_stat_card("⚠️", "Null Rate", f"{null_pct:.1f}%", "#FEF3C7", "#D97706"), unsafe_allow_html=True)
                with c4:
                    file_size_kb = len(raw_bytes) / 1024
                    size_str = f"{file_size_kb:.0f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.1f} MB"
                    st.markdown(_html_stat_card("💾", "File Size", size_str, "#CFFAFE", "#0891B2"), unsafe_allow_html=True)


                with st.expander("🔍 Raw Data Preview (first 10 rows)", expanded=True):
                    st.dataframe(preview_df.head(10), use_container_width=True)
            else:
                st.warning(
                    "Could not preview the uploaded file. "
                    "The agent will still attempt to process it."
                )

            if st.button(
                "🚀 Execute Quality Agent",
                type="primary",
                use_container_width=True,
            ):
                # Write upload to a temp file the pipeline can access
                tmp_dir  = tempfile.mkdtemp(prefix="iqagent_")
                csv_path = os.path.join(tmp_dir, uploaded_file.name)
                uploaded_file.seek(0)
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                log_queue        = queue.Queue()
                result_container = {}

                thread = threading.Thread(
                    target=run_pipeline,
                    args=(csv_path, result_container, log_queue),
                    daemon=True,
                )
                thread.start()

                PROGRESS_LABELS = {
                    "starting data profiling":    "🔍 Profiling Dataset…",
                    "generating validation rules": "📋 Generating Rules…",
                    "running validation":          "✅ Validating Data…",
                    "applying":                    "🔧 Self-Healing…",
                    "pipeline completed":          "🎉 All Done!",
                }

                with st.status(
                    "🤖 Agent processing data…", expanded=True
                ) as status_box:
                    log_placeholder = st.empty()
                    log_lines: list[str] = []

                    while True:
                        try:
                            line = log_queue.get(timeout=0.1)
                            if line == "__DONE__":
                                break
                            log_lines.append(line)

                            for key, label in PROGRESS_LABELS.items():
                                if key in line.lower():
                                    status_box.update(label=label)

                            visible  = log_lines[-120:]
                            html_log = "".join(format_log_line(l) for l in visible)
                            log_placeholder.markdown(
                                f'<div class="log-container">{html_log}</div>',
                                unsafe_allow_html=True,
                            )
                        except queue.Empty:
                            continue

                    status_box.update(
                        label="✅ Pipeline Complete",
                        state="complete",
                        expanded=False,
                    )

                thread.join()

                if result_container.get("success"):
                    st.session_state.pipeline_results  = result_container
                    st.session_state.run_timestamp     = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.toast("Data is clean!", icon="✅")
                    st.balloons()
                else:
                    st.error(
                        f"Pipeline Error: {result_container.get('error', 'Unknown error')}"
                    )

    # =========================================================================
    # TAB 2 — AGENT INTELLIGENCE
    # =========================================================================
    with tab_intel:
        if st.session_state.pipeline_results:
            final_state = st.session_state.pipeline_results["final_state"]

            st.markdown(_html_section("Agent Intelligence", "🧠", "Validation rules, reasoning trace, and semantic analysis"), unsafe_allow_html=True)

            col_rules, col_trace = st.columns(2)

            with col_rules:
                st.markdown("#### 📋 Validation Suite")
                expectations = (
                    final_state.get("expectation_suite", {}).get("expectations", [])
                )
                if expectations:
                    st.markdown(
                        f'<div style="margin-bottom:12px;">{_html_chip(f"{len(expectations)} rules generated", "success")}</div>',
                        unsafe_allow_html=True,
                    )
                    for exp in expectations:
                        col_name = exp.get("kwargs", {}).get("column", "?")
                        label    = f"`{col_name}` — {exp['expectation_type']}"
                        with st.expander(label):
                            st.json(exp["kwargs"])
                            if exp.get("reasoning"):
                                st.caption(f"💡 {exp['reasoning']}")
                else:
                    st.info("No expectations generated.")

            with col_trace:
                st.markdown("#### 🧠 Reasoning Trace")
                agent_colors = {
                    "PROFILER": ("#E0E7FF", "🔍"), "RULE_GENERATOR": ("#EDE9FE", "📋"),
                    "VALIDATOR": ("#D1FAE5", "✅"), "SELF_HEALER": ("#FEF3C7", "🔧"),
                    "SYSTEM": ("#DBEAFE", "🎯"),
                }
                timeline_html = ""
                for msg in final_state.get("messages", []):
                    agent = msg.get("agent", "?").upper()
                    step  = msg.get("step", "")
                    bg, emoji = agent_colors.get(agent, ("#F1F5F9", "•"))
                    timeline_html += _html_timeline_step(agent, step, msg.get("content", ""), bg, emoji)
                if timeline_html:
                    st.markdown(f'<div class="animate-in">{timeline_html}</div>', unsafe_allow_html=True)

            # Semantic types + confidence panel
            sem_types       = final_state.get("semantic_types", {})
            repair_confidence = final_state.get("repair_confidence", {})
            if sem_types:
                st.markdown(_html_section("Inferred Semantic Types", "🔍", "Column types and repair confidence scores", "#CFFAFE"), unsafe_allow_html=True)
                type_rows = []
                for col, stype in sem_types.items():
                    conf = repair_confidence.get(col)
                    type_rows.append({
                        "Column":           col,
                        "Semantic Type":    stype,
                        "Repair Confidence": f"{conf:.0%}" if conf else "—",
                    })
                st.dataframe(
                    pd.DataFrame(type_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.markdown(_html_empty_state("🧠", "No Intelligence Data Yet", "Run the agent on the Ingestion tab to see the intelligence trace."), unsafe_allow_html=True)

    # =========================================================================
    # TAB 3 — FINAL CLEAN ROOM
    # =========================================================================
    with tab_results:
        if st.session_state.pipeline_results:
            res         = st.session_state.pipeline_results
            final_state = res["final_state"]
            stats       = (
                final_state.get("validation_result", {}).get("statistics", {})
            )
            pass_rate   = stats.get("success_percent", 0)

            # ── Read output files ─────────────────────────────────────────────
            clean_path     = res.get("clean_path", "")
            quarantine_path = res.get("quarantine_path", "")

            clean_df = (
                _safe_read_csv(clean_path)
                if clean_path and os.path.exists(clean_path)
                else pd.DataFrame()
            )
            q_df = (
                _safe_read_csv(quarantine_path)
                if quarantine_path and os.path.exists(quarantine_path)
                else pd.DataFrame()
            )

            st.markdown(_html_section("Final Clean Room", "💎", "Quality results dashboard"), unsafe_allow_html=True)

            # ── Metrics row ───────────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(_html_stat_card("✅", "Clean Rows", f"{len(clean_df):,}", "#D1FAE5", "#059669"), unsafe_allow_html=True)
            with m2:
                st.markdown(_html_stat_card("🚫", "Quarantined", f"{len(q_df):,}", "#FEE2E2", "#DC2626"), unsafe_allow_html=True)
            with m3:
                st.markdown(_html_stat_card("📊", "Quality Score", f"{int(pass_rate)}/100", "#E0E7FF", "#4F46E5"), unsafe_allow_html=True)
            with m4:
                st.markdown(_html_stat_card("🔄", "Healing Cycles", str(final_state.get("iteration", 0)), "#EDE9FE", "#7C3AED"), unsafe_allow_html=True)
            with m5:
                status = final_state.get("final_status", "UNKNOWN")
                s_bg = "#D1FAE5" if status == "SUCCESS" else "#FEF3C7"
                s_fg = "#059669" if status == "SUCCESS" else "#D97706"
                st.markdown(_html_stat_card("🏁", "Status", status, s_bg, s_fg), unsafe_allow_html=True)

            st.divider()

            # ── Data panels ───────────────────────────────────────────────────
            left, right = st.columns(2)

            with left:
                st.markdown("#### 💎 Clean Output")
                if not clean_df.empty:
                    st.dataframe(clean_df.head(25), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Clean CSV",
                        clean_df.to_csv(index=False).encode("utf-8"),
                        file_name="clean_data.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                        key="dl_clean",
                    )
                else:
                    st.warning("Clean output file not found or empty.")

            with right:
                st.markdown("#### 🗑️ Quarantine")
                if not q_df.empty:
                    st.dataframe(q_df.head(25), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Quarantine CSV",
                        q_df.to_csv(index=False).encode("utf-8"),
                        file_name="quarantine.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_quarantine",
                    )
                else:
                    st.success("✅ No rows required quarantine.")

            # ── Healing ledger ────────────────────────────────────────────────
            healing_actions = final_state.get("healing_actions", [])
            if healing_actions:
                st.divider()
                st.markdown(_html_section("Healing Ledger", "🛠️", "Applied data repairs and confidence scores", "#FEF3C7"), unsafe_allow_html=True)
                ledger_df = _flatten_healing_actions(healing_actions)
                if not ledger_df.empty:
                    st.dataframe(ledger_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No healing actions were applied.")

            # ── Per-iteration history (multi-cycle runs) ──────────────────────
            history = final_state.get("healing_history", [])
            if history and final_state.get("iteration", 0) > 1:
                with st.expander(
                    f"📜 Full Healing History ({len(history)} actions across "
                    f"{final_state.get('iteration')} iterations)"
                ):
                    history_rows = []
                    for h in history:
                        history_rows.append({
                            "Iteration":    h.get("iteration"),
                            "Column":       h.get("column"),
                            "Action":       h.get("action_type"),
                            "Detail":       h.get("operation_or_condition", "—"),
                            "Rows Affected": h.get("rows_affected", 0),
                            "Success":      "✅" if h.get("success") else "❌",
                            "Confidence":   (
                                f"{h['confidence']:.0%}"
                                if h.get("confidence") else "—"
                            ),
                        })
                    st.dataframe(
                        pd.DataFrame(history_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        else:
            st.markdown(_html_empty_state("💎", "No Results Yet", "Results will appear here after the agent completes its run on the Ingestion tab."), unsafe_allow_html=True)

    # =========================================================================
    # TAB 4 — DB EXPLORER
    # =========================================================================
    with tab_db:
        st.markdown(_html_section("Database Explorer", "🗄️", "Browse tables, run SQL queries, and manage persisted data", "#DBEAFE"), unsafe_allow_html=True)

        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

        # ── Persist button — write pipeline results to DB on demand ────────
        st.markdown("#### 💾 Persist Pipeline Results")

        pipeline_res = st.session_state.get("pipeline_results")
        if not pipeline_res:
            st.info(
                "No pipeline results available yet. "
                "Run the pipeline first, then come back here to persist."
            )
        else:
            clean_path     = pipeline_res.get("clean_path", "")
            quarantine_path = pipeline_res.get("quarantine_path", "")

            c1, c2 = st.columns(2)
            c1.caption(f"**Clean:** `{os.path.basename(clean_path) if clean_path else '—'}`")
            c2.caption(
                f"**Quarantine:** `{os.path.basename(quarantine_path) if quarantine_path else '—'}`"
            )

            if st.button(
                "💾 Persist to Database",
                type="primary",
                use_container_width=True,
                key="db_persist_btn",
            ):
                from src.mcp_tools import save_df_to_db

                # Derive table names from uploaded filename
                uploaded_name = st.session_state.get("uploaded_filename", "data.csv")
                base_name = os.path.splitext(uploaded_name)[0]
                # Sanitize: lowercase, replace spaces/hyphens with underscores
                base_name = base_name.lower().replace(" ", "_").replace("-", "_")

                tbl_clean      = f"{base_name}_clean"
                tbl_quarantine = f"{base_name}_quarantine"

                persisted = []
                errors = []

                # Clean
                if clean_path and os.path.exists(clean_path):
                    try:
                        clean_df = _safe_read_csv(clean_path)
                        if not clean_df.empty:
                            save_df_to_db(clean_df, tbl_clean, if_exists="replace")
                            persisted.append(f"✅ **{tbl_clean}**: {len(clean_df):,} rows")
                    except Exception as e:
                        errors.append(f"Clean: {e}")

                # Quarantine
                if quarantine_path and os.path.exists(quarantine_path):
                    try:
                        q_df = _safe_read_csv(quarantine_path)
                        if not q_df.empty:
                            save_df_to_db(q_df, tbl_quarantine, if_exists="replace")
                            persisted.append(f"✅ **{tbl_quarantine}**: {len(q_df):,} rows")
                    except Exception as e:
                        errors.append(f"Quarantine: {e}")

                if persisted:
                    st.success("Persisted to database:")
                    for p in persisted:
                        st.markdown(p)
                if errors:
                    for err in errors:
                        st.error(err)
                if not persisted and not errors:
                    st.warning("No CSV files found to persist.")

        # ── Table browser & SQL ────────────────────────────────────────────
        st.markdown("---")

        if not os.path.exists(db_path):
            st.warning(
                f"Database not found at `{db_path}`. "
                "Persist pipeline results first."
            )
        else:
            try:
                conn = sqlite3.connect(db_path)

                # Discover tables
                tables_df = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "ORDER BY name;", conn
                )
                table_names = [
                    t for t in tables_df["name"].tolist()
                    if not t.startswith("sqlite_")
                ]

                if not table_names:
                    st.info("Database exists but has no tables. Persist pipeline results first.")
                else:
                    st.caption(f"📁 **Database:** `{db_path}` — {len(table_names)} table(s)")

                    # ── Table Management ──────────────────────────────────
                    st.markdown("#### 🗂️ All Tables")
                    st.markdown(
                        f'<div style="margin-bottom:12px;">{_html_chip(f"{len(table_names)} tables", "primary")}</div>',
                        unsafe_allow_html=True,
                    )

                    for tbl_name in table_names:
                        cnt = pd.read_sql_query(
                            f'SELECT COUNT(*) as cnt FROM "{tbl_name}"', conn
                        )["cnt"].iloc[0]

                        col_left, col_right = st.columns([4, 1])
                        col_left.markdown(f"**{tbl_name}** — `{int(cnt):,}` rows")
                        if col_right.button(
                            "🗑️ Drop",
                            key=f"dropmgr_{tbl_name}",
                        ):
                            conn.execute(f'DROP TABLE IF EXISTS "{tbl_name}"')
                            conn.commit()
                            st.toast(f"Dropped `{tbl_name}`", icon="🗑️")
                            st.rerun()

                    st.markdown("---")
                    # ── Table browser ─────────────────────────────────────
                    st.markdown("#### 📋 Table Browser")

                    selected_table = st.selectbox(
                        "Select a table to preview",
                        table_names,
                        key="db_table_select",
                    )

                    if selected_table:
                        count_row = pd.read_sql_query(
                            f'SELECT COUNT(*) as cnt FROM "{selected_table}"', conn
                        )
                        row_count = int(count_row["cnt"].iloc[0])

                        col_info = pd.read_sql_query(
                            f'PRAGMA table_info("{selected_table}")', conn
                        )

                        col1, col2, col3 = st.columns([2, 2, 3])
                        col1.metric("Rows", f"{row_count:,}")
                        col2.metric("Columns", len(col_info))

                        with col3:
                            if st.button(
                                f"🗑️ Drop {selected_table}",
                                key=f"drop_{selected_table}",
                                type="secondary",
                            ):
                                try:
                                    conn.execute(f'DROP TABLE IF EXISTS "{selected_table}"')
                                    conn.commit()
                                    st.success(f"Dropped `{selected_table}`")
                                    st.rerun()
                                except Exception as drop_err:
                                    st.error(f"Drop failed: {drop_err}")

                        # Schema
                        with st.expander("📐 Schema", expanded=False):
                            schema_df = col_info[["name", "type", "notnull"]].rename(
                                columns={"name": "Column", "type": "Type", "notnull": "Not Null"}
                            )
                            schema_df["Not Null"] = schema_df["Not Null"].map(
                                {1: "✅", 0: "—"}
                            )
                            st.dataframe(schema_df, use_container_width=True, hide_index=True)

                        # Preview
                        preview_limit = st.slider(
                            "Preview rows", 5, 200, 25, key="db_preview_limit"
                        )
                        preview_df = pd.read_sql_query(
                            f'SELECT * FROM "{selected_table}" LIMIT {preview_limit}', conn
                        )
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)

                        # Download
                        full_df = pd.read_sql_query(
                            f'SELECT * FROM "{selected_table}"', conn
                        )
                        csv_data = full_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"⬇️ Download {selected_table} ({row_count:,} rows)",
                            csv_data,
                            file_name=f"{selected_table}.csv",
                            mime="text/csv",
                            key=f"dl_{selected_table}",
                        )

                    # ── Custom SQL query ──────────────────────────────────
                    st.markdown("---")
                    st.markdown("#### 🔍 Custom SQL Query")
                    st.caption(
                        "Supports all SQL: `SELECT`, `DROP TABLE`, `DELETE`, "
                        "`UPDATE`, `INSERT`, `ALTER TABLE`, etc."
                    )

                    default_sql = (
                        f'SELECT * FROM {table_names[0]} LIMIT 10;'
                        if table_names else "SELECT 1;"
                    )
                    sql_query = st.text_area(
                        "Enter SQL query",
                        value=default_sql,
                        height=100,
                        key="db_sql_input",
                    )

                    if st.button("▶️ Execute Query", key="db_run_sql"):
                        try:
                            sql_upper = sql_query.strip().upper()
                            is_select = sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")

                            if is_select:
                                result_df = pd.read_sql_query(sql_query, conn)
                                st.success(f"✅ {len(result_df)} row(s) returned")
                                st.dataframe(
                                    result_df, use_container_width=True, hide_index=True
                                )
                            else:
                                cursor = conn.execute(sql_query)
                                conn.commit()
                                st.success(
                                    f"✅ Query executed successfully. "
                                    f"Rows affected: {cursor.rowcount}"
                                )
                                st.rerun()
                        except Exception as sql_err:
                            st.error(f"❌ SQL Error: {sql_err}")

                conn.close()

            except Exception as db_err:
                st.error(f"Failed to connect to database: {db_err}")


    # =========================================================================
    # TAB 5 — LINEAGE ANALYZER
    # =========================================================================
    with tab_lineage:
        st.markdown(_html_section("SQL Lineage Analyzer", "🧬", "Paste SQL transformations to extract lineage, detect PII, and analyze governance risk", "#D1FAE5"), unsafe_allow_html=True)

        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

        sql_input = st.text_area(
            "Enter SQL transformation(s)",
            height=180,
            key="gov_sql_input",
            placeholder="CREATE TABLE premium_customers AS\nSELECT customer_id, email, order_amount\nFROM stress_test_orders_clean\nWHERE order_amount > 1000;",
        )

        if st.button("🧬 Analyze Lineage & Governance", type="primary", use_container_width=True, key="gov_run"):
            from src.graph.governance_workflow import build_governance_workflow, build_governance_initial_state
            from src.governance.catalog import persist_all

            with st.spinner("Running Governance Agent..."):
                try:
                    workflow = build_governance_workflow()
                    initial = build_governance_initial_state(sql_input, db_path)
                    result = workflow.invoke(initial, config={"recursion_limit": 20})

                    st.session_state.gov_result = result

                    # Persist metadata catalogue
                    persist_all(
                        result.get("lineage", {}),
                        result.get("pii_detections", []),
                        result.get("governance_report", {}),
                        db_path,
                    )

                    st.toast("Governance analysis complete!", icon="✅")
                except Exception as e:
                    st.error(f"Governance agent error: {e}")

        # Display results
        gov_result = st.session_state.get("gov_result")
        if gov_result:
            lineage = gov_result.get("lineage", {})

            if lineage.get("error"):
                st.error(f"Lineage error: {lineage['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(_html_stat_card("📥", "Source Tables", str(len(lineage.get("source_tables", []))), "#D1FAE5", "#059669"), unsafe_allow_html=True)
                with col2:
                    st.markdown(_html_stat_card("📤", "Target Tables", str(len(lineage.get("target_tables", []))), "#DBEAFE", "#2563EB"), unsafe_allow_html=True)
                with col3:
                    st.markdown(_html_stat_card("🔗", "Column Edges", str(len(lineage.get("edges", []))), "#EDE9FE", "#7C3AED"), unsafe_allow_html=True)

                # Source / Target
                st1, st2 = st.columns(2)
                with st1:
                    st.markdown("**📥 Source Tables**")
                    for t in lineage.get("source_tables", []):
                        st.code(t)
                with st2:
                    st.markdown("**📤 Target Tables**")
                    for t in lineage.get("target_tables", []):
                        st.code(t)

                # Column lineage edges
                edges = lineage.get("edges", [])
                if edges:
                    st.markdown("**🔗 Column Lineage**")
                    edge_rows = []
                    for e in edges:
                        edge_rows.append({
                            "Source Table": e["src_table"],
                            "Source Column": e["src_col"],
                            "→": "→",
                            "Target Table": e["tgt_table"],
                            "Target Column": e["tgt_col"],
                            "Transform": e.get("transform", "direct"),
                        })
                    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

                # Conditions
                conditions = lineage.get("conditions", [])
                if conditions:
                    with st.expander("🔍 WHERE Conditions"):
                        for c in conditions:
                            st.code(c, language="sql")

                # Reasoning trace
                with st.expander("🧠 Agent Reasoning Trace"):
                    for msg in gov_result.get("messages", []):
                        agent = msg.get("agent", "?").upper()
                        st.markdown(f"**{agent}** — *{msg.get('step', '')}*")
                        st.caption(msg.get("content", ""))

    # =========================================================================
    # TAB 6 — PII & GOVERNANCE
    # =========================================================================
    with tab_pii:
        st.markdown(_html_section("PII Detection & Governance", "🔐", "Privacy risk analysis and data masking", "#FEE2E2"), unsafe_allow_html=True)

        gov_result = st.session_state.get("gov_result")
        if not gov_result:
            st.markdown(_html_empty_state("🔐", "No Governance Data", "Run the Lineage Analyzer first to detect PII and assess governance risk."), unsafe_allow_html=True)
        else:
            pii_detections = gov_result.get("pii_detections", [])
            report = gov_result.get("governance_report", {})
            db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

            # Risk banner
            risk_level = report.get("risk_level", "LOW")
            st.markdown(
                _html_risk_banner(risk_level, report.get("risk_score", 0), report.get("summary", "")),
                unsafe_allow_html=True,
            )

            # PII detections
            if pii_detections:
                st.markdown("#### 🏷️ Detected PII Columns")
                pii_rows = []
                for d in pii_detections:
                    pii_rows.append({
                        "Table": d["table"],
                        "Column": d["column"],
                        "PII Type": d["pii_type"],
                        "Confidence": f"{d['confidence']:.0%}",
                        "Method": d["detection_method"],
                        "Samples": ", ".join(str(s) for s in d.get("sample_values", [])[:3]),
                    })
                st.dataframe(pd.DataFrame(pii_rows), use_container_width=True, hide_index=True)
            else:
                st.success("No PII detected in the analyzed tables.")

            # Recommendations
            recs = report.get("recommendations", [])
            if recs:
                st.markdown("#### 📋 Governance Recommendations")
                for rec in recs:
                    priority = rec.get("priority", "LOW")
                    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(priority, "🟢")
                    st.markdown(
                        f"{icon} **{rec['action'].upper()}** — `{rec.get('column', '')}` "
                        f"({rec.get('table', '')}) — {rec.get('detail', '')}"
                    )

            # Masking controls
            if pii_detections:
                st.markdown("---")
                st.markdown("#### 🎭 Active Governance Enforcement")

                if st.button(
                    "🎭 Apply Recommended Masking",
                    type="primary",
                    use_container_width=True,
                    key="apply_masking",
                ):
                    from src.agents.masking_engine import apply_masking

                    with st.spinner("Applying masking..."):
                        result = apply_masking(pii_detections, db_path)
                        st.session_state.masking_result = result

                    if result.get("columns_masked"):
                        st.success(
                            f"Masked {len(result['columns_masked'])} column(s) "
                            f"across {len(result['masked_tables'])} table(s)"
                        )
                    else:
                        st.warning("No columns were masked — tables may not exist in DB yet.")

                # Show masked preview
                masking_result = st.session_state.get("masking_result")
                if masking_result and masking_result.get("details"):
                    st.markdown("**Masking Preview**")
                    for d in masking_result["details"]:
                        with st.expander(
                            f"{d['column']} ({d['pii_type']}) — {d['table']}"
                        ):
                            c1, c2 = st.columns(2)
                            c1.markdown("**Before**")
                            for v in d["samples_before"]:
                                c1.code(v)
                            c2.markdown("**After**")
                            for v in d["samples_after"]:
                                c2.code(v)

                    # Preview full masked table
                    for tbl_name, mdf in masking_result.get("masked_tables", {}).items():
                        with st.expander(f"📄 Full preview: {tbl_name}_masked ({len(mdf)} rows)"):
                            st.dataframe(mdf.head(25), use_container_width=True, hide_index=True)

                    # Persist button
                    if st.button(
                        "💾 Persist Masked Tables to Database",
                        type="secondary",
                        use_container_width=True,
                        key="persist_masked",
                    ):
                        from src.agents.masking_engine import persist_masked_tables

                        results = persist_masked_tables(
                            masking_result["masked_tables"], db_path
                        )
                        for name, ok in results.items():
                            if ok:
                                st.success(f"✅ Persisted `{name}`")
                            else:
                                st.error(f"❌ Failed: `{name}`")

    # =========================================================================
    # TAB 7 — METADATA CATALOG
    # =========================================================================
    with tab_catalog:
        st.markdown(_html_section("Metadata Catalog", "📚", "Governance metadata, lineage edges, and PII tags", "#FEF3C7"), unsafe_allow_html=True)

        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

        if not os.path.exists(db_path):
            st.markdown(_html_empty_state("📚", "No Catalog Data", "Run the pipeline and governance agent first to populate the metadata catalog."), unsafe_allow_html=True)
        else:
            try:
                conn = sqlite3.connect(db_path)

                catalog_tables = [
                    ("catalog_tables", "Table-level metadata"),
                    ("catalog_columns", "Column-level metadata & PII tags"),
                    ("lineage_edges", "Column lineage graph edges"),
                    ("pii_tags", "Detected PII with confidence"),
                    ("governance_reports", "Risk assessments & recommendations"),
                ]

                for tbl_name, description in catalog_tables:
                    try:
                        df = pd.read_sql_query(f'SELECT * FROM "{tbl_name}"', conn)
                        with st.expander(
                            f"📋 {tbl_name} — {description} ({len(df)} rows)",
                            expanded=(len(df) > 0),
                        ):
                            if df.empty:
                                st.caption("No data yet.")
                            else:
                                st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception:
                        with st.expander(f"📋 {tbl_name} — {description}"):
                            st.caption("Table not created yet. Run the governance agent first.")

                conn.close()
            except Exception as e:
                st.error(f"Database error: {e}")

    # =========================================================================
    # TAB 8 — LINEAGE GRAPH
    # =========================================================================
    with tab_graph:
        st.markdown(_html_section("Lineage Visualization", "📈", "Column-level data flow with PII indicators", "#CFFAFE"), unsafe_allow_html=True)

        gov_result = st.session_state.get("gov_result")
        if not gov_result:
            st.markdown(_html_empty_state("📈", "No Lineage Data", "Run the Lineage Analyzer first to generate lineage graph."), unsafe_allow_html=True)
        else:
            lineage = gov_result.get("lineage", {})
            edges = lineage.get("edges", [])
            pii_detections = gov_result.get("pii_detections", [])

            if not edges:
                st.warning("No lineage edges to visualize.")
            else:
                pii_cols = {
                    (d["table"], d["column"]): d["pii_type"]
                    for d in pii_detections
                }

                # ── Visual flow: Source → Target ──────────────────────
                st.markdown("#### Data Flow")

                # Group edges by source table → target table
                flow_groups = {}
                for e in edges:
                    key = (e["src_table"], e["tgt_table"])
                    flow_groups.setdefault(key, []).append(e)

                for (src_tbl, tgt_tbl), group_edges in flow_groups.items():
                    st.markdown(f"**{src_tbl}** → **{tgt_tbl}**")
                    flow_rows = []
                    for e in group_edges:
                        src_pii = pii_cols.get((e["src_table"], e["src_col"]), "")
                        tgt_pii = pii_cols.get((e["tgt_table"], e["tgt_col"]), "")
                        flow_rows.append({
                            "Source Column": f"{'🔐 ' if src_pii else ''}{e['src_col']}",
                            "→": "→",
                            "Target Column": f"{'🔐 ' if tgt_pii else ''}{e['tgt_col']}",
                            "Transform": e.get("transform", "direct"),
                            "PII": src_pii or tgt_pii or "—",
                        })
                    st.dataframe(
                        pd.DataFrame(flow_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── Summary ──────────────────────────────────────────
                st.markdown("---")
                st.markdown(_html_section("Lineage Summary", "📊", "", "#CFFAFE"), unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(_html_stat_card("📥", "Sources", ", ".join(lineage.get("source_tables", [])) or "—", "#D1FAE5", "#059669"), unsafe_allow_html=True)
                with s2:
                    st.markdown(_html_stat_card("📤", "Targets", ", ".join(lineage.get("target_tables", [])) or "—", "#DBEAFE", "#2563EB"), unsafe_allow_html=True)
                with s3:
                    st.markdown(_html_stat_card("🔗", "Total Edges", str(len(edges)), "#EDE9FE", "#7C3AED"), unsafe_allow_html=True)
                with s4:
                    st.markdown(_html_stat_card("🔐", "PII Columns", str(len(pii_cols)), "#FEE2E2", "#DC2626"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()