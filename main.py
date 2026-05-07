# main.py
# CLI entry point for the Ingestion Quality Agent.
#
# Changes from original:
#   - Uses build_initial_state() from workflow.py — single source of truth
#     for AgentState initialisation; no more missing new fields.
#   - Removed redundant shutil.copy after graph.invoke(): mark_success()
#     in workflow.py now writes the clean file directly.
#   - print_final_report(): uses state["suite_name"] and config paths
#     instead of hardcoded "orders_rejected.csv" / "orders_quality_suite".
#   - Stale-file cleanup uses PROCESSED_DATA_PATH / QUARANTINE_DATA_PATH
#     from config instead of hardcoded paths.
#   - Added semantic_types and repair_confidence to the final report.
#   - Ollama health-check failure now exits with a clear remediation message.

import os
import sys
import subprocess
import logging
from datetime import datetime

import pandas as pd
import colorlog

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    QUARANTINE_DATA_PATH,
    GE_EXPECTATIONS_DIR,
    MAX_HEALING_ITERATIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.graph.workflow import build_workflow, build_initial_state
from src.agents.state import AgentState


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> str:
    """Configure colourised console logging + timestamped file logging."""
    os.makedirs("logs", exist_ok=True)

    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s "
        "%(blue)s%(name)s%(reset)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "red,bg_white",
        },
    )
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(color_formatter)
    console.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(file_formatter)
    fh.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_filename


# =============================================================================
# BANNER
# =============================================================================

def print_banner() -> None:
    print("\n" + "=" * 70)
    print("   🤖 INGESTION QUALITY AGENT — Universal Edition")
    print("   Profile → Rules → Validate → Self-Heal  (any CSV)")
    print("=" * 70 + "\n")


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(final_state: AgentState, start_time: datetime) -> None:
    """Print a comprehensive post-run summary."""
    duration    = (datetime.now() - start_time).total_seconds()
    status      = final_state.get("final_status", "UNKNOWN")
    status_emoji = {"PASS": "✅", "FAIL": "❌", "ALERT": "🚨"}.get(status, "❓")

    print("\n" + "=" * 70)
    print(f"   {status_emoji}  PIPELINE COMPLETE — {status}")
    print("=" * 70)

    print(f"\n⏱  Duration        : {duration:.1f}s")
    print(f"🔄 Healing cycles  : {final_state.get('iteration', 0)} / {MAX_HEALING_ITERATIONS}")
    print(f"📂 Suite           : {final_state.get('suite_name', 'N/A')}")

    # ── Profile ────────────────────────────────────────────────────────────────
    profile = final_state.get("profile", {})
    if profile:
        info  = profile.get("data_info", {})
        score = profile.get("analysis", {}).get("overall_quality_score", "N/A")
        print(f"\n📊 Dataset:")
        print(f"   Rows        : {info.get('row_count', 'N/A'):,}")
        print(f"   Columns     : {info.get('column_count', 'N/A')}")
        print(f"   Quality score (initial): {score}/100")

    # ── Semantic types ─────────────────────────────────────────────────────────
    semantic_types = final_state.get("semantic_types", {})
    if semantic_types:
        print(f"\n🔍 Inferred Column Types:")
        for col, stype in semantic_types.items():
            confidence = final_state.get("repair_confidence", {}).get(col)
            conf_str   = f"  (repair confidence: {confidence:.0%})" if confidence else ""
            print(f"   {col:<30} {stype}{conf_str}")

    # ── Validation ─────────────────────────────────────────────────────────────
    val = final_state.get("validation_result", {})
    if val:
        s = val.get("statistics", {})
        print(f"\n📋 Final Validation:")
        print(f"   Rules checked : {s.get('evaluated_expectations', 0)}")
        print(f"   Passed        : {s.get('successful_expectations', 0)}")
        print(f"   Failed        : {s.get('unsuccessful_expectations', 0)}")
        print(f"   Success rate  : {s.get('success_percent', 0):.1f}%")

    # ── Healing ledger ──────────────────────────────────────────────────────────
    healing = final_state.get("healing_actions", [])
    if healing:
        print(f"\n🔧 Healing Actions Applied ({len(healing)}):")
        for ha in healing:
            action = ha.get("action", {})
            result = ha.get("result", {})
            conf   = ha.get("confidence")
            conf_str = f"  [confidence: {conf:.0%}]" if conf else ""
            print(
                f"   • [{action.get('column', 'N/A')}] "
                f"{action.get('action_type')} / "
                f"{action.get('operation') or action.get('condition', '')} "
                f"→ {result.get('rows_affected', 0)} rows affected{conf_str}"
            )
            print(f"     {ha.get('reasoning', '')[:90]}")

    # ── Healing history across all iterations ──────────────────────────────────
    history = final_state.get("healing_history", [])
    if history and len(history) > len(healing):
        # Only show if there were multiple iterations
        print(f"\n📜 Full Healing History ({len(history)} actions across all iterations):")
        for h in history:
            ok = "✅" if h.get("success") else "❌"
            print(
                f"   {ok} Iter {h['iteration']}: [{h['column']}] "
                f"{h['action_type']} → {h.get('rows_affected', 0)} rows"
            )

    # ── Agent reasoning trace ──────────────────────────────────────────────────
    messages = final_state.get("messages", [])
    if messages:
        print(f"\n🧠 Reasoning Trace ({len(messages)} steps):")
        for msg in messages:
            preview = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
            print(f"   [{msg['agent'].upper()}] {msg['step']}: {preview}")

    # ── Output files ───────────────────────────────────────────────────────────
    print(f"\n📁 Output Files:")

    try:
        if os.path.exists(PROCESSED_DATA_PATH):
            clean_df = pd.read_csv(PROCESSED_DATA_PATH)
            print(f"   ✅ {PROCESSED_DATA_PATH}  ({len(clean_df):,} clean rows)")
        else:
            print(f"   ℹ️  {PROCESSED_DATA_PATH}  — not written (pipeline may have ended in ALERT)")
    except Exception as e:
        print(f"   ⚠️  Could not read processed file: {e}")

    try:
        if os.path.exists(QUARANTINE_DATA_PATH):
            q_df = pd.read_csv(QUARANTINE_DATA_PATH)
            print(f"   🗑️  {QUARANTINE_DATA_PATH}  ({len(q_df):,} quarantined rows)")
        else:
            print(f"   ℹ️  {QUARANTINE_DATA_PATH}  — no rows quarantined")
    except Exception as e:
        print(f"   ⚠️  Could not read quarantine file: {e}")

    suite_name = final_state.get("suite_name", "")
    if suite_name:
        suite_path = os.path.join(GE_EXPECTATIONS_DIR, f"{suite_name}.json")
        if os.path.exists(suite_path):
            print(f"   📋 {suite_path}")

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_banner()
    log_filename = setup_logging()
    logger = logging.getLogger("main")

    start_time = datetime.now()
    logger.info(f"📝 Log file : {log_filename}")
    logger.info(f"📂 Input    : {RAW_DATA_PATH}")

    # ── Prerequisite: Ollama reachable ────────────────────────────────────────
    try:
        import requests as _req
        _req.get("http://localhost:11434", timeout=3)
        logger.info(f"✅ Ollama reachable ({OLLAMA_BASE_URL}, model: {OLLAMA_MODEL})")
    except Exception:
        logger.error("❌ Cannot reach Ollama at http://localhost:11434")
        logger.error("   Start it with:  ollama serve")
        logger.error("   Then pull the model:  ollama pull llama3")
        sys.exit(1)

    # ── Reset raw data (regenerate the dirty CSV for a clean baseline) ────────
    if os.path.exists("generate_data.py"):
        logger.info("🔄 Regenerating raw data for a clean run baseline...")
        try:
            subprocess.run([sys.executable, "generate_data.py"], check=True)
            logger.info("✅ Raw data reset complete")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  generate_data.py failed: {e} — using existing raw file")

    # ── Clear stale output files ──────────────────────────────────────────────
    # Quarantine file uses append mode in apply_healing_action, so it MUST
    # be cleared each run to prevent accumulation from prior runs.
    for stale in [QUARANTINE_DATA_PATH, PROCESSED_DATA_PATH]:
        if os.path.exists(stale):
            os.remove(stale)
            logger.info(f"🗑️  Cleared stale output: {stale}")

    # ── Build workflow ────────────────────────────────────────────────────────
    logger.info("🔨 Building LangGraph workflow...")
    graph = build_workflow()

    # ── Build initial state (all fields, including new ones) ──────────────────
    initial_state = build_initial_state(RAW_DATA_PATH)

    logger.info("🚀 Starting agent workflow...")

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 50},
        )
        # NOTE: mark_success() in workflow.py now writes PROCESSED_DATA_PATH.
        # No shutil.copy needed here.

        print_final_report(final_state, start_time)
        sys.exit(0 if final_state.get("final_status") == "PASS" else 1)

    except Exception as e:
        logger.critical(f"❌ Workflow failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()