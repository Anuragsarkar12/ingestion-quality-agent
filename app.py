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
    tab_upload, tab_intel, tab_results = st.tabs([
        "📤 Ingestion",
        "🧠 Agent Intelligence",
        "💎 Final Clean Room",
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


if __name__ == "__main__":
    main()