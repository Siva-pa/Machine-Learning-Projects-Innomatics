"""
predict.py
Loads the trained pipeline and returns salary predictions.
Logs every prediction for monitoring purposes.
"""

import csv
import json
import logging
import os
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(path: str = None):
    path = path or MODEL_PATH
    logger.info("Loading model from %s ...", path)
    pipeline = joblib.load(path)
    logger.info("Model loaded successfully.")
    return pipeline

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(input_data: dict, pipeline=None) -> dict:
    """
    Accepts a dictionary of feature values and returns a prediction dict.
    Logs each call to logs/predictions.csv.

    Parameters
    ----------
    input_data : dict
        Keys match the feature columns used during training.
    pipeline   : sklearn Pipeline (optional)
        Re-used across calls in the Streamlit app for efficiency.

    Returns
    -------
    dict with keys: 'predicted_salary', 'timestamp', 'input'
    """
    if pipeline is None:
        pipeline = load_model()

    df = pd.DataFrame([input_data])
    prediction = float(pipeline.predict(df)[0])

    result = {
        "predicted_salary": round(prediction, 2),
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
    }

    _log_prediction(result)
    return result


# ── Batch prediction ───────────────────────────────────────────────────────────
def predict_batch(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """Predict salaries for an entire DataFrame."""
    if pipeline is None:
        pipeline = load_model()

    df = df.copy()
    df["predicted_salary"] = pipeline.predict(df)
    logger.info("Batch prediction completed for %d rows.", len(df))
    return df


# ── Prediction logging ─────────────────────────────────────────────────────────
def _log_prediction(result: dict):
    """Append one prediction record to predictions.csv."""
    flat = {
        "timestamp": result["timestamp"],
        "predicted_salary": result["predicted_salary"],
        **result["input"],
    }

    file_exists = PRED_LOG.exists()
    with open(PRED_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)


# ── Performance monitoring helper ──────────────────────────────────────────────
def load_prediction_logs() -> pd.DataFrame:
    """Return all logged predictions as a DataFrame (used in monitoring tab)."""
    if not PRED_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(PRED_LOG, parse_dates=["timestamp"])


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "company_location": "US",
        "employee_residence": "US",
        "company_size": "M",
        "industry": "Technology",
        "education_required": "Bachelor's",
        "remote_ratio": 100,
        "years_experience": 5,
        "job_description_length": 800,
        "benefits_score": 4.0,
    }

    model = load_model()
    result = predict(sample, pipeline=model)
    print(f"\nPredicted Salary: ${result['predicted_salary']:,.2f}")
    print(f"Logged at       : {result['timestamp']}")
