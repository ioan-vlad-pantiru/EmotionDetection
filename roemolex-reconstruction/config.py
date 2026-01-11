"""
Configuration for RoEmoLex reconstruction pipeline.
"""
import os
import multiprocessing
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUT_DIR = DATA_DIR / "out"

# Work directories
WORK_DIR = BASE_DIR / "work"
CHECKPOINTS_DIR = WORK_DIR / "checkpoints"
SHARDS_DIR = WORK_DIR / "shards"

# Logs
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [RAW_DIR, INTERIM_DIR, OUT_DIR, CHECKPOINTS_DIR, SHARDS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Performance settings
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(4, max(2, CPU_COUNT // 2))  # Default: 2-4 workers
CHUNK_SIZE = 20000  # Rows per shard/chunk

# Memory limits (approximate)
MAX_MEMORY_MB = 1500  # Target: stay under 1.5GB

# File paths
NRC_EMOLEX_RO_RAW = RAW_DIR / "nrc_emolex_ro.txt"
BASE_CLEANED = INTERIM_DIR / "base_cleaned.csv"
MAPPED = INTERIM_DIR / "mapped.csv"
EXPANDED = INTERIM_DIR / "expanded.csv"
AFFECT_ADDED = INTERIM_DIR / "affect_added.csv"
FINAL_CSV = OUT_DIR / "roemolex_recon.csv"
FINAL_JSONL = OUT_DIR / "roemolex_recon.jsonl"
STATS_JSON = OUT_DIR / "stats.json"
README_OUT = OUT_DIR / "README.md"
LOG_FILE = LOGS_DIR / "run.log"

# Checkpoint files
CHECKPOINT_STEP1 = CHECKPOINTS_DIR / "step1_acquire.done"
CHECKPOINT_STEP2 = CHECKPOINTS_DIR / "step2_normalize.done"
CHECKPOINT_STEP3 = CHECKPOINTS_DIR / "step3_map.done"
CHECKPOINT_STEP4 = CHECKPOINTS_DIR / "step4_expand.done"
CHECKPOINT_STEP5 = CHECKPOINTS_DIR / "step5_affect.done"
CHECKPOINT_STEP6 = CHECKPOINTS_DIR / "step6_derived.done"
CHECKPOINT_STEP7 = CHECKPOINTS_DIR / "step7_validate.done"
CHECKPOINT_STEP8 = CHECKPOINTS_DIR / "step8_readme.done"

# Emotion columns (Plutchik 8 + polarity)
EMOTION_COLUMNS = [
    "anger", "anticipation", "disgust", "fear", 
    "joy", "sadness", "surprise", "trust"
]
POLARITY_COLUMNS = ["positive", "negative"]

# Derived emotion mappings (dyads)
DERIVED_EMOTIONS = {
    "love": ["joy", "trust"],
    "submission": ["trust", "fear"],
    "awe": ["fear", "surprise"],
    "disapproval": ["surprise", "sadness"],
    "remorse": ["sadness", "disgust"],
    "contempt": ["anger", "disgust"],
    "aggressiveness": ["anger", "anticipation"],
    "optimism": ["anticipation", "joy"],
}



