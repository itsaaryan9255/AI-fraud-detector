import os
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

from src.data import generate_synthetic_fraud_data, load_data, split_features_target
from src.clustering import (
    analyze_cluster_composition,
    evaluate_clusters,
    fit_clusters,
    load_clustering_model,
    predict_cluster,
    save_clustering_model,
)
from src.model import (evaluate_model, load_model, save_model, train_model,
                       train_test_split_data)


MODEL_PATH = "models/fraud_detector.joblib"
DATA_PATH = "data/fraud_transactions.csv"


def load_dataset(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    st.warning("No dataset found. Generate synthetic data or upload a CSV file.")
    return pd.DataFrame()


def display_dataset_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Class distribution")
    if "is_fraud" in df.columns:
        counts = df["is_fraud"].value_counts().sort_index()
        st.bar_chart(counts)
        st.write(counts)
    else:
        st.info("Upload a dataset containing an 'is_fraud' target column for training/evaluation.")

    st.write(f"Rows: {df.shape[0]}, Features: {df.shape[1]}")


def build_feature_inputs(df: pd.DataFrame) -> list[float]:
    feature_columns = [col for col in df.columns if col != "is_fraud"]
    if not feature_columns:
        raise ValueError("No feature columns available for prediction.")

    st.subheader("Predict a single transaction")
    values = [st.number_input(col, value=0.0, format="%.4f") for col in feature_columns]
    return values


def main() -> None:
    st.set_page_config(page_title="Fraud Detection", layout="wide")
    st.title("Detecting Financial Fraud Using Machine Learning")
    st.write(
        "Use the app to generate synthetic fraud data, train a classifier, evaluate model performance, and predict whether a transaction is fraudulent."
    )

    st.sidebar.header("Workflow")
    mode = st.sidebar.radio("Select mode", ["Generate Data", "Train Model", "Evaluate Model", "Predict", "Cluster Analysis"])

    default_model_path = st.sidebar.text_input("Model path", MODEL_PATH)
    default_data_path = st.sidebar.text_input("Data path", DATA_PATH)

    if mode == "Generate Data":
        st.sidebar.subheader("Synthetic data settings")
        n_samples = st.sidebar.number_input("Samples", min_value=1000, max_value=200000, value=10000, step=1000)
        n_features = st.sidebar.number_input("Features", min_value=5, max_value=50, value=10, step=1)
        fraud_ratio = st.sidebar.slider("Fraud ratio", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
        random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

        if st.button("Generate synthetic fraud data"):
            df = generate_synthetic_fraud_data(
                n_samples=n_samples,
                n_features=n_features,
                fraud_ratio=fraud_ratio,
                random_state=random_state,
                output_path=default_data_path,
            )
            st.success(f"Saved synthetic dataset to {default_data_path}")
            display_dataset_summary(df)

    dataset = None
    if mode in ["Train Model", "Evaluate Model", "Predict"]:
        st.sidebar.subheader("Dataset input")
        data_source = st.sidebar.radio("Data source", ["Load default dataset", "Upload CSV file"])
        uploaded_file = None
        if data_source == "Upload CSV file":
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

        dataset = load_dataset(uploaded_file)
        display_dataset_summary(dataset)

    if mode == "Train Model":
        if dataset is None or dataset.empty:
            st.warning("Provide a dataset to train the model.")
        elif "is_fraud" not in dataset.columns:
            st.error("Dataset must contain an 'is_fraud' target column.")
        else:
            test_size = st.sidebar.slider("Validation split", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
            if st.button("Train model"):
                X, y = split_features_target(dataset)
                X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=test_size)
                model = train_model(X_train, y_train)
                save_model(model, default_model_path)
                st.success(f"Model saved to {default_model_path}")
                st.subheader("Validation results")
                metrics = evaluate_model(model, X_test, y_test)
                st.json(metrics)

    if mode == "Evaluate Model":
        if not os.path.exists(default_model_path):
            st.error(f"Model file not found: {default_model_path}")
        elif dataset is None or dataset.empty:
            st.warning("Provide a dataset for evaluation.")
        elif "is_fraud" not in dataset.columns:
            st.error("Dataset must contain an 'is_fraud' target column.")
        else:
            if st.button("Evaluate model"):
                model = load_model(default_model_path)
                X, y = split_features_target(dataset)
                metrics = evaluate_model(model, X, y)
                st.success("Evaluation complete")
                st.json(metrics)

    if mode == "Predict":
        if not os.path.exists(default_model_path):
            st.error(f"Model file not found: {default_model_path}")
        elif dataset is None or dataset.empty:
            st.warning("Load a dataset to define the feature schema for prediction.")
        else:
            if "is_fraud" in dataset.columns:
                features = dataset.drop(columns=["is_fraud"]).columns.tolist()
            else:
                features = dataset.columns.tolist()

            if not features:
                st.error("No feature columns available.")
            else:
                st.subheader("Predict with current model")
                feature_values = [st.number_input(col, value=0.0, format="%.4f") for col in features]
                if st.button("Run prediction"):
                    model = load_model(default_model_path)
                    prediction = model.predict([feature_values])[0]
                    label = "Fraud" if int(prediction) == 1 else "Legitimate"
                    st.write(f"### Prediction: {label} ({int(prediction)})")

    if mode == "Cluster Analysis":
        if dataset is None or dataset.empty:
            st.warning("Provide a dataset for clustering analysis.")
        elif "is_fraud" not in dataset.columns:
            st.error("Dataset must contain an 'is_fraud' target column for cluster composition analysis.")
        else:
            n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=2, step=1)
            clustering_model_path = st.sidebar.text_input("Clustering model path", "models/clustering_model.joblib")
            if st.button("Fit clusters"):
                X, y = split_features_target(dataset)
                clustering_model = fit_clusters(X, n_clusters=n_clusters)
                save_clustering_model(clustering_model, clustering_model_path)
                st.success(f"Clustering model saved to {clustering_model_path}")
                metrics = evaluate_clusters(clustering_model, X)
                st.subheader("Clustering evaluation")
                st.json(metrics)
                composition = analyze_cluster_composition(X, y, clustering_model)
                st.subheader("Cluster composition")
                for cluster, info in composition.items():
                    st.write(f"**{cluster}**: {info['total_samples']} samples, {info['fraud_samples']} fraud ({info['fraud_ratio']:.2%})")

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit for rapid fraud model prototyping.")


if __name__ == "__main__":
    main()
