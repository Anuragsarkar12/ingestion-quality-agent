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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F9FAFB;
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        .stMetric, .stDataFrame, div[data-testid="stExpander"] {
            background-color: white !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            padding: 1rem !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; background-color: transparent;
            border-radius: 4px; color: #6B7280; font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #4F46E5 !important;
            border-bottom: 2px solid #4F46E5 !important;
        }
        .stButton>button {
            border-radius: 8px !important; font-weight: 600 !important;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .log-container {
            background-color: #0F172A; color: #E2E8F0;
            padding: 16px; border-radius: 8px;
            font-family: 'IBM Plex Mono', monospace; font-size: 13px;
            line-height: 1.6; height: 400px; overflow-y: auto;
            border: 1px solid #1E293B;
        }
        div[data-testid="stMetricLabel"] {
            color: #4B5563 !important; font-weight: 500 !important;
            text-transform: uppercase; letter-spacing: 0.025em;
        }
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
    col_icon, col_title = st.columns([1, 5])
    with col_icon:
        st.write("")
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2092/2092063.png",
            width=72,
        )
    with col_title:
        st.title("DataArmor AI")
        st.markdown(
            "<p style='font-size:18px;color:#6B7280;'>"
            "Autonomous agentic data quality &amp; self-healing — "
            "any CSV, any schema."
            "</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ System")
        st.metric("Model", OLLAMA_MODEL)
        st.metric("Max Healing Cycles", MAX_HEALING_ITERATIONS)
        st.info(f"Ollama: {OLLAMA_BASE_URL}")

        # Last run info
        if st.session_state.run_timestamp:
            st.divider()
            st.success(f"Last run: {st.session_state.run_timestamp}")
            st.caption(f"File: {st.session_state.uploaded_filename}")

        # Inferred column types from last run
        if st.session_state.pipeline_results:
            final_state = st.session_state.pipeline_results["final_state"]
            sem_types   = final_state.get("semantic_types", {})
            if sem_types:
                with st.expander("🔍 Inferred Column Types"):
                    for col, stype in sem_types.items():
                        st.caption(f"`{col}` → **{stype}**")

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
        st.subheader("Data Ingestion")
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
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rows",    f"{len(preview_df):,}")
                c2.metric("Total Columns", len(preview_df.columns))
                c3.metric(
                    "Null Rate",
                    f"{preview_df.isna().sum().sum() / total_cells * 100:.1f}%",
                )

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

            col_rules, col_trace = st.columns(2)

            with col_rules:
                st.subheader("📋 Validation Suite")
                expectations = (
                    final_state.get("expectation_suite", {}).get("expectations", [])
                )
                if expectations:
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
                st.subheader("🧠 Reasoning Trace")
                for msg in final_state.get("messages", []):
                    agent = msg.get("agent", "?").upper()
                    step  = msg.get("step", "")
                    emoji = {
                        "PROFILER": "🔍", "RULE_GENERATOR": "📋",
                        "VALIDATOR": "✅", "SELF_HEALER": "🔧",
                        "SYSTEM": "🎯",
                    }.get(agent, "•")
                    st.markdown(f"**{emoji} {agent}** — *{step}*")
                    st.caption(msg["content"])
                    st.divider()

            # Semantic types + confidence panel
            sem_types       = final_state.get("semantic_types", {})
            repair_confidence = final_state.get("repair_confidence", {})
            if sem_types:
                st.subheader("🔍 Inferred Semantic Types & Repair Confidence")
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
            st.info("Run the agent on the Ingestion tab to see the intelligence trace.")

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

            # ── Metrics row ───────────────────────────────────────────────────
            # m1: actual clean row count (NOT evaluated_expectations)
            # m2: quarantined row count
            # m3: rule pass rate
            # m4: healing cycles used
            # m5: pipeline status
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Clean Rows",      f"{len(clean_df):,}")
            m2.metric("Quarantined",     f"{len(q_df):,}")
            m3.metric("Quality Score",   f"{int(pass_rate)}/100")
            m4.metric("Healing Cycles",  final_state.get("iteration", 0))
            m5.metric("Status",          final_state.get("final_status", "UNKNOWN"))

            st.divider()

            # ── Data panels ───────────────────────────────────────────────────
            left, right = st.columns(2)

            with left:
                st.markdown("### 💎 Clean Output")
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
                st.markdown("### 🗑️ Quarantine")
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
                st.subheader("🛠️ Healing Ledger")
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
            st.info("Results will appear here after the agent completes its run.")

    # =========================================================================
    # TAB 4 — DB EXPLORER
    # =========================================================================
    with tab_db:
        st.subheader("🗄️ Mock Database Explorer")

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
        st.subheader("🧬 SQL Lineage Analyzer")
        st.caption("Paste SQL transformations to extract lineage, detect PII, and analyze governance risk.")

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
                col1.metric("Source Tables", len(lineage.get("source_tables", [])))
                col2.metric("Target Tables", len(lineage.get("target_tables", [])))
                col3.metric("Column Edges", len(lineage.get("edges", [])))

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
        st.subheader("🔐 PII Detection & Governance")

        gov_result = st.session_state.get("gov_result")
        if not gov_result:
            st.info("Run the Lineage Analyzer first to detect PII and assess governance risk.")
        else:
            pii_detections = gov_result.get("pii_detections", [])
            report = gov_result.get("governance_report", {})
            db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

            # Risk banner
            risk_level = report.get("risk_level", "LOW")
            risk_colors = {
                "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"
            }
            st.markdown(
                f"### {risk_colors.get(risk_level, '⚪')} Overall Risk: **{risk_level}** "
                f"({report.get('risk_score', 0):.0%})"
            )
            st.caption(report.get("summary", ""))

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
        st.subheader("📚 Governance Metadata Catalog")

        db_path = getattr(_cfg, "DATABASE_PATH", "database/final.db")

        if not os.path.exists(db_path):
            st.info("No database found. Run the pipeline and governance agent first.")
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
        st.subheader("📈 Lineage Visualization")

        gov_result = st.session_state.get("gov_result")
        if not gov_result:
            st.info("Run the Lineage Analyzer first to generate lineage graph.")
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
                st.markdown("#### 📊 Lineage Summary")
                summary_data = {
                    "Source Tables": ", ".join(lineage.get("source_tables", [])),
                    "Target Tables": ", ".join(lineage.get("target_tables", [])),
                    "Total Edges": len(edges),
                    "PII Columns": len(pii_cols),
                    "Statements": lineage.get("statement_count", 0),
                }
                for k, v in summary_data.items():
                    st.markdown(f"**{k}:** {v}")


if __name__ == "__main__":
    main()