"""
train.py
Trains an AI Job Salary Prediction model.
Tracks experiments with MLflow and saves the final pipeline.
"""

import os
import pickle
import logging
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    AIJobPreprocessor,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    get_target,
    load_data,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.pkl"

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    ),
    "ridge": Ridge(alpha=10.0),
}


# ── Build full sklearn Pipeline ────────────────────────────────────────────────
def build_pipeline(model_name: str = "gradient_boosting") -> Pipeline:
    preprocessor = AIJobPreprocessor(
        categorical_cols=CATEGORICAL_COLS,
        numerical_cols=NUMERICAL_COLS,
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", StandardScaler(), NUMERICAL_COLS),
        ]
    )

    regressor = MODELS[model_name]

    pipeline = Pipeline(
        steps=[
            ("clean", preprocessor),
            ("encode", column_transformer),
            ("model", regressor),
        ]
    )
    return pipeline


# ── Evaluation helpers ─────────────────────────────────────────────────────────
def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


# ── Main training loop ─────────────────────────────────────────────────────────
def train(
    data_path: str,
    model_name: str = "gradient_boosting",
    test_size: float = 0.2,
    random_state: int = 42,
    experiment_name: str = "AI_Job_Salary_Prediction",
):
    # Set up MLflow with SQLite backend (works on all platforms/versions)
    db_path = (ROOT / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)

    logger.info("=== Starting training: %s ===", model_name)

    # Load data
    df = load_data(data_path)
    y = get_target(df)
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    with mlflow.start_run(run_name=model_name) as run:
        # Log hyper-parameters
        params = {
            "model_name": model_name,
            "test_size": test_size,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        # Build & fit
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        # Evaluate
        train_metrics = evaluate(y_train, pipeline.predict(X_train))
        test_metrics = evaluate(y_test, pipeline.predict(X_test))

        logger.info("Train → MAE: %.2f | RMSE: %.2f | R²: %.4f", *train_metrics.values())
        logger.info("Test  → MAE: %.2f | RMSE: %.2f | R²: %.4f", *test_metrics.values())

        # Cross-validation R²
        cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        logger.info("CV R² (5-fold): %.4f ± %.4f", cv_r2.mean(), cv_r2.std())

        # Log metrics to MLflow
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("cv_r2_mean", cv_r2.mean())
        mlflow.log_metric("cv_r2_std", cv_r2.std())

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Persist locally
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(MODEL_PATH))

        logger.info("Model saved to %s (run_id=%s)", MODEL_PATH, run.info.run_id)

    return pipeline, test_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Job Salary model")
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "cleaned_ai_job_dataset.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--model",
        default="gradient_boosting",
        choices=list(MODELS.keys()),
        help="Model to train",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--experiment", default="AI_Job_Salary_Prediction")
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        test_size=args.test_size,
        experiment_name=args.experiment,
    )
