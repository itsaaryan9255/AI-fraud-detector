import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build a feature scaling and classification pipeline."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight="balanced")),
        ]
    )
    return pipeline


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Pipeline:
    """Train the fraud detection model pipeline."""
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate the trained model and return metrics."""
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    metrics = {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": report,
    }
    return metrics


def save_model(model: Pipeline, output_path: str) -> None:
    """Persist the trained model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: str) -> Pipeline:
    """Load a saved model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict(model: Pipeline, features: np.ndarray) -> int:
    """Predict whether a transaction is fraudulent."""
    return int(model.predict(features.reshape(1, -1))[0])


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Split data for training and validation."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
