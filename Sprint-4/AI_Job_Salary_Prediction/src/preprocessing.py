"""
preprocessing.py
Handles all data cleaning and feature engineering for AI Job Salary Prediction.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Column name constants ──────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_location",
    "employee_residence",
    "company_size",
    "industry",
    "education_required",
]

NUMERICAL_COLS = [
    "remote_ratio",
    "years_experience",
    "job_description_length",
    "benefits_score",
]

TARGET_COL = "salary_usd"


# ── Custom transformer (keeps it reusable in pipelines) ───────────────────────
class AIJobPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer.
    Cleans raw data and returns a feature DataFrame ready for encoding.
    """

    def __init__(
        self,
        categorical_cols: list = None,
        numerical_cols: list = None,
    ):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.numerical_cols = numerical_cols or NUMERICAL_COLS
        self.feature_names_out_ = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting preprocessor …")
        self._validate_columns(X)
        self.feature_names_out_ = self.categorical_cols + self.numerical_cols
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info("Transforming data …")
        df = X.copy()

        # 1. Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows.", before - len(df))

        # 2. Drop rows with missing target (only during training path)
        if TARGET_COL in df.columns:
            df = df.dropna(subset=[TARGET_COL])

        # 3. Fill missing categoricals with mode
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        # 4. Fill missing numericals with median
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        # 5. Strip whitespace in string columns
        for col in self.categorical_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.strip()

        # 6. Clip extreme salary values (IQR-based, only when target is present)
        if TARGET_COL in df.columns:
            q1, q3 = df[TARGET_COL].quantile([0.01, 0.99])
            df = df[(df[TARGET_COL] >= q1) & (df[TARGET_COL] <= q3)]

        return df[self.categorical_cols + self.numerical_cols]

    # ------------------------------------------------------------------
    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in self.categorical_cols + self.numerical_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input: {missing}")


# ── Standalone helper ─────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s …", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def get_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL]
