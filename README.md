<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2.28-4F46E5?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-llama3-000000?style=for-the-badge&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/Great_Expectations-0.18-FF6F00?style=for-the-badge" />
</p>

# 🛡️ DataArmor AI — Autonomous Agentic Data Quality & Governance

**DataArmor AI** is a production-grade, multi-agent data quality and governance platform powered by LLMs. Upload any CSV — the system autonomously profiles it, generates validation rules, detects anomalies, self-heals corrupted data, quarantines unfixable rows, and returns clean datasets. A separate governance pipeline handles SQL lineage extraction, PII detection, risk assessment, and data masking.

> **Zero manual configuration. Any CSV. Any schema.**

---

## 🏗️ Architecture Overview

The system implements two independent LangGraph-orchestrated agent workflows:

### Agent B1 — Ingestion Quality Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  📊 Profiler │───▶│ 📋 Rule Gen  │───▶│  ✅ Validator  │───▶│  Decision   │
│   Agent      │    │   Agent      │    │   Agent       │    │  Router     │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬──────┘
                                                                   │
                                              ┌────────────────────┼────────────────────┐
                                              │                    │                    │
                                        ┌─────▼─────┐       ┌─────▼─────┐        ┌─────▼─────┐
                                        │ ✅ PASS    │       │ 🔧 HEAL   │        │ 🚨 ALERT  │
                                        │ mark_      │       │ self_heal │        │ alert_    │
                                        │ success    │       │ (retry)   │        │ and_end   │
                                        └───────────┘       └───────────┘        └───────────┘
```

| Node | Agent | Responsibility |
|---|---|---|
| `profile_data` | Profiler Agent | Statistical profiling, null analysis, semantic type inference, anomaly detection |
| `generate_rules` | Rule Generator | LLM-powered validation rule generation via Great Expectations |
| `validate_data` | Validator | Execute expectation suite, compute pass/fail statistics |
| `self_heal` | Self-Healer | LLM-driven data repair with confidence scoring |
| `mark_success` | System | Write clean CSV output |
| `alert_and_end` | System | Emit structured CRITICAL alert after max retries |

The heal→validate loop runs up to **5 iterations** (configurable) before escalating.

### Agent B2 — Lineage & Governance Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ 🧬 Lineage   │───▶│ 🔐 PII       │───▶│ 📋 Governance    │───▶ END
│  Parser      │    │  Detector    │    │   Analyzer       │
└──────────────┘    └──────────────┘    └──────────────────┘
```

| Node | Responsibility |
|---|---|
| `parse_lineage` | SQL parsing via sqlglot — extracts source/target tables, column-level edges, transforms, WHERE conditions |
| `detect_pii` | Multi-layer PII detection (regex, column name heuristics, sample value analysis) |
| `analyze_governance` | LLM-powered risk scoring, recommendation generation, compliance assessment |

---

## ✨ Features

### Data Quality (B1)
- **Universal CSV Support** — orders, healthcare, sensor, HR, finance — any schema
- **LLM-Powered Profiling** — statistical analysis + semantic type inference
- **Intelligent Rule Generation** — Great Expectations suite generated from data profile
- **Self-Healing Pipeline** — autonomous data repair with confidence scores
- **Quarantine System** — unfixable rows isolated with audit trail
- **Healing Ledger** — full history of repairs across iterations

### Governance (B2)
- **SQL Lineage Extraction** — column-level lineage from CREATE/INSERT/SELECT statements
- **PII Detection** — email, phone, SSN, credit card, IP address, and more
- **Risk Assessment** — LOW / MEDIUM / HIGH / CRITICAL with actionable recommendations
- **Data Masking** — automated masking of detected PII columns
- **Metadata Catalog** — persistent governance metadata in SQLite

### UI
- **Material Design 3** — enterprise-grade Streamlit UI with gradient app bar, stat cards, timeline, chips
- **8 Tabs** — Ingestion, Agent Intelligence, Clean Room, DB Explorer, Lineage, PII/Governance, Catalog, Graph
- **Real-Time Logs** — live execution console during pipeline runs
- **One-Click Downloads** — clean CSV, quarantine CSV, database exports

---

## 📁 Project Structure

```
ingestion_quality_agent/
├── app.py                          # Streamlit UI (Material Design)
├── main.py                         # CLI entry point
├── generate_data.py                # Synthetic dirty data generator
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── config.py                   # Central configuration
│   ├── mcp_tools.py                # Shared utilities (CSV I/O, DB, alerts)
│   │
│   ├── agents/
│   │   ├── state.py                # AgentState TypedDict (B1)
│   │   ├── governance_state.py     # GovernanceState TypedDict (B2)
│   │   ├── profiler_agent.py       # Data profiling + semantic inference
│   │   ├── rule_generator.py       # LLM-powered expectation generation
│   │   ├── self_healer.py          # Autonomous data repair
│   │   ├── lineage_agent.py        # SQL lineage extraction (sqlglot)
│   │   ├── pii_detector.py         # Multi-layer PII detection
│   │   ├── governance_analyzer.py  # Risk scoring + recommendations
│   │   └── masking_engine.py       # PII masking + persistence
│   │
│   ├── graph/
│   │   ├── workflow.py             # B1 LangGraph workflow
│   │   └── governance_workflow.py  # B2 LangGraph workflow
│   │
│   ├── governance/
│   │   └── catalog.py              # Metadata catalog persistence
│   │
│   └── validation/
│       └── validator.py            # Great Expectations runner
│
├── data/
│   └── raw/                        # Raw CSV input
├── database/                       # SQLite database
├── great_expectations/             # GE config + generated suites
├── test_data/                      # Sample test datasets
└── logs/                           # Timestamped run logs
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** ([download](https://ollama.com/download))

### 1. Clone & Setup

```bash
git clone https://github.com/Anuragsarkar12/ingestion-quality-agent.git
cd ingestion-quality-agent

python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull llama3
```

### 3. Run

**Terminal 1 — Start Ollama:**
```bash
ollama serve
```

**Terminal 2 — Start the Web App:**
```bash
source venv/bin/activate
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

**Alternative — CLI Mode:**
```bash
source venv/bin/activate
python main.py
```

---

## 🖥️ Usage Guide

### Ingestion Tab
1. Upload any CSV file via the drag-and-drop zone
2. Review the dataset stats (rows, columns, null rate, file size)
3. Click **Execute Quality Agent** to start the autonomous pipeline
4. Watch the real-time log console as agents work

### Agent Intelligence Tab
- View the generated validation rules (Great Expectations suite)
- Follow the agent reasoning trace as a visual timeline
- Review inferred semantic types and repair confidence

### Final Clean Room Tab
- Dashboard of quality metrics (clean rows, quarantined, quality score)
- Side-by-side clean vs quarantine data preview
- Download clean and quarantine CSVs
- Full healing ledger and history

### DB Explorer Tab
- Persist pipeline results to SQLite with one click
- Browse, preview, and drop tables
- Run custom SQL queries (SELECT, DROP, INSERT, UPDATE, etc.)

### Lineage Analyzer Tab
- Paste SQL transformations (CREATE TABLE AS SELECT, etc.)
- Extract column-level lineage with source/target mappings
- View WHERE conditions and transformation types

### PII & Governance Tab
- Color-coded risk banner (LOW → CRITICAL)
- Detected PII columns with type, confidence, and detection method
- Apply automated masking with before/after preview
- Persist masked tables to database

### Metadata Catalog Tab
- Browse table-level and column-level metadata
- View lineage edges, PII tags, and governance reports

---

## ⚙️ Configuration

All configuration is centralized in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/api/chat` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3` | LLM model (alternatives: `llama3:70b`, `mistral`, `gemma2`) |
| `MAX_HEALING_ITERATIONS` | `5` | Max heal→validate retry cycles |
| `MAX_PROFILE_ROWS` | `100,000` | Sampling threshold for large datasets |
| `DATABASE_PATH` | `database/final.db` | SQLite database location |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) 0.2.28 |
| **LLM Runtime** | [Ollama](https://ollama.com) (local, privacy-preserving) |
| **Validation Engine** | [Great Expectations](https://greatexpectations.io) 0.18.15 |
| **SQL Parsing** | [sqlglot](https://github.com/tobymao/sqlglot) 25+ |
| **Data Processing** | [pandas](https://pandas.pydata.org) 2.1.4 |
| **Web UI** | [Streamlit](https://streamlit.io) 1.30+ with Material Design 3 CSS |
| **Database** | SQLite 3 |
| **Language** | Python 3.11+ |

---

## 📄 License

This project is for educational and portfolio purposes.

---

<p align="center">
  Built with ❤️ using LangGraph + Ollama + Great Expectations + Streamlit
</p>
