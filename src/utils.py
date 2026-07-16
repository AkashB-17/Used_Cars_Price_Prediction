"""
utils.py

Reusable utility functions for the Used Cars Price Prediction project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


# ============================================================
# Logger
# ============================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if not logger.handlers:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    return logger


# ============================================================
# Numeric Cleaning
# ============================================================

def clean_numeric_column(
    series: pd.Series,
    units: str | list[str],
) -> pd.Series:
    """
    Remove one or more unit suffixes and convert to numeric.
    """

    if isinstance(units, str):
        units = [units]

    cleaned = series.astype(str)

    for unit in units:
        cleaned = cleaned.str.replace(unit, "", regex=False)

    return (
        cleaned.str.strip()
        .replace("", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )


# ============================================================
# JSON Helpers
# ============================================================

def save_json(
    data: dict,
    filepath: Path,
) -> None:
    """
    Save dictionary as JSON.
    """

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(filepath, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def load_json(
    filepath: Path,
) -> dict:
    """
    Load JSON file.
    """

    with open(filepath, "r", encoding="utf-8") as file:

        return json.load(file)