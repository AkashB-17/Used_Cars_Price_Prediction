

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import clean_numeric_column
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


# ============================================================
# Configuration
# ============================================================

CURRENT_YEAR = datetime.now().year

NUMERIC_COLUMNS = [
    "Mileage",
    "Engine",
    "Power",
    "Seats",
]

DROP_COLUMNS = [
    "S.No.",
    "New_Price",
]


# ============================================================
# Logger
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw dataset.

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Loading dataset...")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    logger.info(f"Original Shape : {df.shape}")

    df = df.drop_duplicates()

    logger.info(f"After Removing Duplicates : {df.shape}")

    return df


# ============================================================
# Remove Invalid Rows
# ============================================================

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where target price is missing.
    """

    logger.info("Removing rows with missing target...")

    df = df[df["Price"].notna()].copy()

    logger.info(f"Remaining Rows : {len(df)}")

    return df


# ============================================================
# Remove Unused Columns
# ============================================================

def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns.
    """

    logger.info("Dropping unused columns...")

    existing_cols = [
        col for col in DROP_COLUMNS
        if col in df.columns
    ]

    df = df.drop(columns=existing_cols)

    return df


# ============================================================
# Brand Extraction
# ============================================================

def extract_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract brand from Name column.
    """

    logger.info("Extracting Brand feature...")

    df["Brand"] = (
    df["Name"]
    .str.strip()
    .str.split()
    .str[0]
    )

    df = df.drop(columns=["Name"])

    return df


# ============================================================
# Numeric Cleaning
# ============================================================

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns by removing units and converting to numeric types.
    """

    logger.info("Cleaning numeric columns...")

    df["Mileage"] = clean_numeric_column(
        df["Mileage"],
        ["kmpl", "km/kg"],
    )

    df["Engine"] = clean_numeric_column(
        df["Engine"],
        "CC",
    )

    df["Power"] = clean_numeric_column(
        df["Power"],
        "bhp",
    )

    df["Seats"] = pd.to_numeric(
        df["Seats"],
        errors="coerce",
    )

    return df


# ============================================================
# Feature Engineering
# ============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features.
    """

    logger.info("Creating engineered features...")

    df["Car_Age"] = CURRENT_YEAR - df["Year"]

    return df


# ============================================================
# Missing Values
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.
    """

    logger.info("Handling missing values...")

    for col in NUMERIC_COLUMNS:

        if df[col].isna().sum() > 0:

            median = df[col].median()

            df[col] = df[col].fillna(median)

            logger.info(
                f"{col} -> filled with median ({median:.2f})"
            )

    return df


# ============================================================
# Target Transformation
# ============================================================

def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform target variable.
    """

    logger.info("Applying log transformation...")

    df["Price_log"] = np.log1p(df["Price"])

    return df


# ============================================================
# Save Dataset
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    filepath: Path,
) -> None:
    """
    Save processed dataset.
    """

    logger.info("Saving cleaned dataset...")

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(filepath, index=False)

    logger.info(f"Saved -> {filepath}")


# ============================================================
# Pipeline
# ============================================================

def main():

    df = load_data(RAW_DATA_PATH)

    df = remove_invalid_rows(df)

    df = remove_unused_columns(df)

    df = extract_brand(df)

    df = clean_numeric_columns(df)

    df = feature_engineering(df)

    df = handle_missing_values(df)

    df = transform_target(df)

    save_dataset(
        df,
        PROCESSED_DATA_PATH,
    )

    logger.info("Preprocessing completed successfully!")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()