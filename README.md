# Detecting Financial Fraud Using Machine Learning

This project demonstrates a complete fraud detection workflow using Python and machine learning.

## Project Overview

- Generate or load synthetic financial transaction data.
- Train an imbalanced classification model for fraud detection.
- Evaluate model performance with classification metrics.
- Use a command-line interface to run training, evaluation, and prediction.

## Structure

- `src/data.py` - data generation and loading utilities.
- `src/model.py` - model training, evaluation, persistence, and prediction.
- `src/app.py` - command-line interface for the end-to-end workflow.
- `requirements.txt` - Python dependencies.
- `.gitignore` - ignored files.

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Generate synthetic data and save it locally:

```powershell
python src\app.py generate --output data\fraud_transactions.csv
```

Train the fraud detection model:

```powershell
python src\app.py train --input data\fraud_transactions.csv --model models\fraud_detector.joblib
```

Evaluate the trained model:

```powershell
python src\app.py evaluate --input data\fraud_transactions.csv --model models\fraud_detector.joblib
```

Run prediction for a new sample:

```powershell
python src\app.py predict --model models\fraud_detector.joblib --features 0.1 0.3 -0.2 1.5 0.0 0.6 0.0 1.8 0.5 0.2
```

Run the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

## Notes

- This example uses synthetic generated data with class imbalance to mimic fraud detection.
- The model pipeline is built with standard scaling and `RandomForestClassifier`.
- Extend the workflow by adding real transaction data and feature engineering.
