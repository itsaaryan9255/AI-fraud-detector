import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_synthetic_fraud_data(
    n_samples: int = 10000,
    n_features: int = 10,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a synthetic fraud dataset with an imbalanced binary target."""
    n_informative = max(2, n_features // 3)
    n_redundant = max(1, n_features // 5)
    n_repeated = 0
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=2,
        weights=[1.0 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state,
    )

    columns = [f"feature_{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["is_fraud"] = y

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """Load transaction data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame):
    """Split a dataset into input features and the target label."""
    if "is_fraud" not in df.columns:
        raise KeyError("Expected column 'is_fraud' in dataset.")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    return X, y
