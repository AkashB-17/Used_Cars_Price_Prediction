"""
inference.py

Load trained model and perform inference.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_model():
    """
    Load trained model pipeline.
    """

    logger.info("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")

    return model


def predict_price(
    input_data: dict,
) -> float:
    """
    Predict used car price.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    float
        Predicted price in Lakhs.
    """

    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction_log = model.predict(input_df)[0]

    prediction = np.expm1(prediction_log)

    return round(float(prediction), 2)