"""
training.py

Train and evaluate multiple machine learning models for
Used Cars Price Prediction.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.utils import (
    setup_logger,
    save_json,
)

logger = setup_logger(__name__)

def load_dataset() -> pd.DataFrame:
    """
    Load the processed dataset.
    """

    logger.info("Loading processed dataset...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    logger.info(f"Dataset Shape : {df.shape}")

    return df

def split_features_target(
    df: pd.DataFrame,
):
    """
    Split dataframe into features and target.
    """

    X = df.drop(
        columns=[
            "Price",
            "Price_log",
        ]
    )

    y = df["Price_log"]

    return X, y

def get_feature_types(
    X: pd.DataFrame,
):
    """
    Identify numerical and categorical columns.
    """

    numerical_features = X.select_dtypes(
        include=[
            "int64",
            "float64",
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "string"
        ],
    ).columns.tolist()

    logger.info(f"Numerical Features : {len(numerical_features)}")

    logger.info(f"Categorical Features : {len(categorical_features)}")

    return numerical_features, categorical_features

#if __name__ == "__main__":
#
#    df = load_dataset()
#
#    X, y = split_features_target(df)
#
#    numerical_features, categorical_features = get_feature_types(X)
#
#    print("\nTraining setup successful!")


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.
    """

    logger.info("Building preprocessing pipeline...")

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    return preprocessor

def get_models() -> dict:
    """
    Define regression models for comparison.
    """

    logger.info("Initializing models...")

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        ),

        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }

    return models

def create_pipeline(
    preprocessor: ColumnTransformer,
    model,
) -> Pipeline:
    """
    Combine preprocessing and model into a single pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_raw_test: pd.Series,
    preprocessor: ColumnTransformer,
):
    """
    Train all models, evaluate them, and return the best pipeline.

    Models are trained on the log-transformed target (y_train/y_test), but
    metrics are computed on the actual price scale (y_raw_test, in Lakh INR)
    since that's the unit that's meaningful to report and compare models on.
    """

    models = get_models()

    results = []

    best_model = None
    best_pipeline = None
    best_score = float("-inf")

    logger.info("Starting model training...\n")

    for model_name, model in models.items():

        logger.info(f"Training {model_name}...")

        pipeline = create_pipeline(
            preprocessor,
            model,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions_log = pipeline.predict(X_test)

        # Invert the log1p transform to get predictions back on the actual
        # Lakh-INR price scale before scoring — this is the fix for the bug
        # where metrics were previously reported on the log scale.
        predictions = np.expm1(predictions_log)

        r2 = r2_score(
            y_raw_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_raw_test,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_raw_test,
            predictions,
        )

        logger.info(
            f"{model_name} | "
            f"R²: {r2:.4f} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f}"
        )

        results.append(
            {
                "Model": model_name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

        if r2 > best_score:

            best_score = r2
            best_model = model_name
            best_pipeline = pipeline

    logger.info(f"\nBest Model : {best_model}")
    logger.info(f"Best R²     : {best_score:.4f}")

    return (
        best_pipeline,
        best_model,
        pd.DataFrame(results),
    )


def save_artifacts(
    best_pipeline: Pipeline,
    best_model_name: str,
    results_df: pd.DataFrame,
) -> None:
    """
    Save trained model and evaluation artifacts.
    """

    logger.info("Saving artifacts...")

    # Create artifacts directory if it doesn't exist
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save trained pipeline
    joblib.dump(
        best_pipeline,
        MODEL_PATH,
    )

    logger.info(f"Model saved to: {MODEL_PATH}")

    # Save comparison table
    results_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    logger.info(f"Comparison saved to: {MODEL_COMPARISON_PATH}")

    # Save metrics
    best_metrics = (
        results_df
        .sort_values("R2", ascending=False)
        .iloc[0]
        .to_dict()
    )

    metrics = {
        "Best Model": best_model_name,
        "R2": float(best_metrics["R2"]),
        "MAE_lakh": float(best_metrics["MAE"]),
        "RMSE_lakh": float(best_metrics["RMSE"]),
    }

    save_json(
        metrics,
        METRICS_PATH,
    )

    logger.info(f"Metrics saved to: {METRICS_PATH}")


def main():
    """
    Main training pipeline.
    """

    logger.info("=" * 60)
    logger.info("USED CARS PRICE PREDICTION MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    df = load_dataset()

    # Features & target
    X, y = split_features_target(df)

    # Feature types
    numerical_features, categorical_features = get_feature_types(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # y_test is log1p(Price); invert it to get the actual Lakh-INR price for
    # scoring, since MAE/RMSE/R2 should be reported in real, human units.
    y_raw_test = np.expm1(y_test)

    logger.info(f"Training Samples : {len(X_train)}")
    logger.info(f"Testing Samples  : {len(X_test)}")

    # Preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    # Train models
    best_pipeline, best_model_name, results_df = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        y_raw_test,
        preprocessor,
    )

    # Save outputs
    save_artifacts(
        best_pipeline,
        best_model_name,
        results_df,
    )

    logger.info("=" * 60)
    logger.info("Training Completed Successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()