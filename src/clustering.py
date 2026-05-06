import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_clustering_pipeline(n_clusters: int = 2, random_state: int = 42) -> Pipeline:
    """Build a feature scaling and K-means clustering pipeline."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clusterer", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ]
    )
    return pipeline


def fit_clusters(X: pd.DataFrame, n_clusters: int = 2, random_state: int = 42) -> Pipeline:
    """Fit the clustering pipeline to the data."""
    pipeline = build_clustering_pipeline(n_clusters=n_clusters, random_state=random_state)
    pipeline.fit(X)
    return pipeline


def evaluate_clusters(model: Pipeline, X: pd.DataFrame) -> dict:
    """Evaluate clustering quality with silhouette score."""
    labels = model.predict(X)
    score = silhouette_score(X, labels)
    metrics = {
        "silhouette_score": score,
        "n_clusters": model.named_steps["clusterer"].n_clusters,
        "cluster_centers": model.named_steps["clusterer"].cluster_centers_.tolist(),
    }
    return metrics


def save_clustering_model(model: Pipeline, output_path: str) -> None:
    """Persist the trained clustering model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def load_clustering_model(model_path: str) -> Pipeline:
    """Load a saved clustering model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_cluster(model: Pipeline, features: np.ndarray) -> int:
    """Predict the cluster for a single sample."""
    return int(model.predict(features.reshape(1, -1))[0])


def analyze_cluster_composition(X: pd.DataFrame, y: pd.Series, model: Pipeline) -> dict:
    """Analyze what each cluster contains in terms of fraud vs legitimate."""
    labels = model.predict(X)
    composition = {}
    for cluster in np.unique(labels):
        cluster_mask = labels == cluster
        fraud_count = y[cluster_mask].sum()
        total_count = cluster_mask.sum()
        legitimate_count = total_count - fraud_count
        composition[f"cluster_{cluster}"] = {
            "total_samples": int(total_count),
            "fraud_samples": int(fraud_count),
            "legitimate_samples": int(legitimate_count),
            "fraud_ratio": fraud_count / total_count if total_count > 0 else 0,
        }
    return composition
