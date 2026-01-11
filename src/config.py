"""
Configuration file for emotion detection system.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
ERRORS_DIR = REPORTS_DIR / "errors"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, ERRORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Plutchik 8 emotions
PLUTCHIK_8 = [
    "anger",
    "fear",
    "anticipation",
    "trust",
    "surprise",
    "sadness",
    "joy",
    "disgust",
]

# Neutral label
NEUTRAL_LABEL = "neutral"

# All labels including neutral
ALL_LABELS = PLUTCHIK_8 + [NEUTRAL_LABEL]

# Model parameters
TFIDF_MAX_FEATURES = 50000
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
CLASSIFIER_MAX_ITER = 5000  # Increased to avoid convergence warnings
CLASSIFIER_SOLVER = "saga"

# Train/test split
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Negation window
NEGATION_WINDOW = 4

# Languages
LANGUAGES = ["en", "ro"]

# Model types
MODEL_TYPES = ["lexicon", "ml", "hybrid"]


