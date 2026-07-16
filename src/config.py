"""
config.py

Central configuration file for the Used Cars Price Prediction project.
"""

from pathlib import Path

# ============================================================
# Project Directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ============================================================
# Data Files
# ============================================================

RAW_DATA_PATH = RAW_DATA_DIR / "used_cars_data.csv"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"

# ============================================================
# Model Artifacts
# ============================================================

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"

# ============================================================
# Training Configuration
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "Price"

TARGET_LOG_COLUMN = "Price_log"

# ============================================================
# Feature Engineering
# ============================================================

LUXURY_BRANDS = {
    "Audi",
    "BMW",
    "Jaguar",
    "Land",
    "Mercedes-Benz",
    "Mini",
    "Porsche",
    "Volvo",
}