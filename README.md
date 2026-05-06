# Fraud Detection Clusters Using Machine Learning

This repository contains a complete end-to-end fraud detection project built with Python and Streamlit. It demonstrates how to generate synthetic transaction data, train and evaluate a machine learning model for fraud detection, and deploy an interactive application for end users.

## Project Summary

The goal of this project is to simulate a realistic financial fraud detection workflow by:

- Generating imbalanced synthetic transaction datasets.
- Training a robust classification model using `RandomForestClassifier`.
- Evaluating model performance with precision, recall, F1 score, and confusion matrix metrics.
- Providing a Streamlit-based user interface for data generation, training, evaluation, and prediction.

This project is ideal for learning, prototyping, and deploying a simple fraud detection service with minimal setup.

## Features

- Synthetic fraud data generation with adjustable sample size, feature count, and fraud ratio.
- Automated train/evaluate workflow for fraud classification.
- Model persistence using `joblib`.
- Interactive Streamlit app for non-technical users.
- Command-line interface for batch workflows.
- K-means clustering for anomaly detection and cluster analysis.
- Ready to deploy on Streamlit Cloud.

## Repository Structure

- `README.md` - project documentation and usage guide.
- `requirements.txt` - all required Python dependencies.
- `.gitignore` - ignored files and folders.
- `streamlit_app.py` - Streamlit frontend and deployment entrypoint.
- `src/__init__.py` - Python package marker.
- `src/data.py` - data generation and dataset utilities.
- `src/clustering.py` - K-means clustering for anomaly detection and cluster composition analysis.
- `src/app.py` - command-line interface for generate/train/evaluate/predict.
- `notebooks/Fraud_Detection_Workflow.ipynb` - exploratory notebook for experimentation.

## Detailed Component Overview

### `src/data.py`

This module handles dataset creation and loading:

- `generate_synthetic_fraud_data(...)` generates a synthetic dataset with a configurable fraud fraction.
- `load_data(csv_path)` loads a CSV file into a pandas DataFrame.
- `split_features_target(df)` splits the dataset into feature matrix `X` and label vector `y`.

### `src/clustering.py`

This module provides clustering functionality:

- `build_clustering_pipeline(...)` creates a `StandardScaler` + `KMeans` pipeline.
- `fit_clusters(X, n_clusters)` fits the clustering model to the data.
- `evaluate_clusters(model, X)` computes silhouette score and cluster centers.
- `analyze_cluster_composition(X, y, model)` analyzes fraud vs legitimate distribution in each cluster.

### `src/app.py`

This file provides a command-line interface for common tasks:

- `generate` - create synthetic datasets.
- `train` - train the model from a dataset.
- `evaluate` - evaluate a saved model against a dataset.
- `predict` - run prediction on a new single transaction.

### `streamlit_app.py`

This is the main Streamlit application. It allows users to:

- Generate synthetic fraud data from the browser.
- Upload and preview datasets.
- Train and save a fraud detection model.
- Evaluate a saved model.
- Predict whether a single transaction is fraudulent.
- Perform K-means clustering and analyze cluster compositions.

## Installation

1. Clone the repository or download the project files.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Command-Line Usage

### Generate synthetic data

```powershell
python src\app.py generate --output data\fraud_transactions.csv
```

### Train the fraud detection model

```powershell
python src\app.py train --input data\fraud_transactions.csv --model models\fraud_detector.joblib
```

### Evaluate the trained model

```powershell
python src\app.py evaluate --input data\fraud_transactions.csv --model models\fraud_detector.joblib
```

### Predict a single transaction

```powershell
python src\app.py predict --model models\fraud_detector.joblib --features 0.1 0.3 -0.2 1.5 0.0 0.6 0.0 1.8 0.5 0.2
```

## Streamlit Deployment

### Run locally

```powershell
python -m streamlit run streamlit_app.py
```

Then visit:

- `http://localhost:8501`

### Main file path for deployment

- `streamlit_app.py`

This is the correct entrypoint for Streamlit Cloud or any hosted Streamlit deployment.

## GitHub Deployment

This project has been pushed to GitHub and can be deployed directly from the repository.

### Repository URL

- `https://github.com/itsaaryan9255/AI-fraud-detector.git`

### Branch

- `main`

## Notes

- The dataset is synthetic and intended for demonstration and learning.
- For production, integrate real financial transaction features, advanced feature engineering, and stronger validation.
- Consider adding explainability and monitoring for a production-grade fraud detection pipeline.

## Troubleshooting

- If `streamlit` is not recognized, use:

```powershell
python -m streamlit run streamlit_app.py
```

- Make sure your virtual environment is activated before installing dependencies.
- Ensure `models\fraud_detector.joblib` exists before running prediction.
