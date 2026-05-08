# src/config.py
# Central configuration for the system.
#
# Changes from original:
#   - Added MAX_PROFILE_ROWS: sampling guard for large datasets (used by mcp_tools)
#   - Added MAX_VALIDATION_ROWS: cap for validation pass on very large files
#   - Added GE_DEFAULT_SUITE_NAME: renamed from GE_SUITE_NAME for clarity
#     (GE_SUITE_NAME kept as alias for backward-compatibility)
#   - VALID_ORDER_STATUSES, MIN_ORDER_AMOUNT, MAX_ORDER_AMOUNT kept as
#     DEFAULT FALLBACK VALUES only — they are no longer injected into
#     validation rules for every CSV. The healer/rule_gen now derives
#     bounds and valid sets from the actual data profile.

import os

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"

# Which local model to use. Alternatives: "llama3:70b", "mistral", "gemma2"
OLLAMA_MODEL = "llama3"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_PATH = "database/final.db"
TABLE_STAGING = "orders_staging"
TABLE_CLEAN = "orders_clean"
TABLE_QUARANTINE = "orders_quarantine"

# =============================================================================
# FILE PATHS
# =============================================================================

RAW_DATA_PATH = "data/raw/orders_raw.csv"
PROCESSED_DATA_PATH = "data/processed/orders_clean.csv"
QUARANTINE_DATA_PATH = "data/quarantine/orders_rejected.csv"

# Great Expectations
GE_SUITE_NAME = "dynamic_quality_suite"          # default — overridden at runtime
GE_DEFAULT_SUITE_NAME = GE_SUITE_NAME            # explicit alias
GE_EXPECTATIONS_DIR = "great_expectations/expectations"

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Maximum heal→validate retry cycles before declaring ALERT
MAX_HEALING_ITERATIONS = 5

# =============================================================================
# PERFORMANCE / SCALABILITY GUARDS
# =============================================================================

# Datasets larger than this are profiled on a random sample.
# Prevents OOM on files with millions of rows.
# Profiling quality degrades gracefully — sampling is logged as a warning.
MAX_PROFILE_ROWS = 100_000

# Datasets larger than this trigger chunked validation (future enhancement).
# Currently used as a logging threshold only.
MAX_VALIDATION_ROWS = 500_000

# =============================================================================
# DOMAIN-SPECIFIC DEFAULTS (orders CSV)
# =============================================================================
# These values are kept as FALLBACK DEFAULTS for the sample orders dataset.
#
# ⚠️  IMPORTANT — they are NOT injected into rules for arbitrary CSVs.
#    - VALID_ORDER_STATUSES: used only if semantic inference identifies a
#      "status" column AND the column name is literally "status" AND the
#      profile top_values are empty (extremely rare edge case).
#    - MIN/MAX_ORDER_AMOUNT: no longer referenced anywhere in the pipeline.
#      Bounds are now derived from the actual data profile.

VALID_ORDER_STATUSES = ["pending", "processing", "shipped", "delivered"]
MIN_ORDER_AMOUNT = 0.0
MAX_ORDER_AMOUNT = 50_000.0

# Temp directory for Streamlit uploads
UPLOAD_TEMP_DIR = "data/uploads"