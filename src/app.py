import argparse
import json
import os
from typing import List

import numpy as np

from src.data import generate_synthetic_fraud_data, load_data, split_features_target
from src.model import (evaluate_model, load_model, save_model, train_model,
                       train_test_split_data)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud detection workflow tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic fraud data.")
    generate_parser.add_argument("--output", required=True, help="Output CSV path for generated data.")
    generate_parser.add_argument("--samples", type=int, default=10000, help="Number of samples.")
    generate_parser.add_argument("--features", type=int, default=10, help="Number of features.")
    generate_parser.add_argument("--fraud_ratio", type=float, default=0.02, help="Fraction of fraud cases.")

    train_parser = subparsers.add_parser("train", help="Train the fraud detection model.")
    train_parser.add_argument("--input", required=True, help="CSV file containing training data.")
    train_parser.add_argument("--model", required=True, help="Path to save the trained model.")
    train_parser.add_argument("--test_size", type=float, default=0.25, help="Test set ratio.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    evaluate_parser.add_argument("--input", required=True, help="CSV file containing evaluation data.")
    evaluate_parser.add_argument("--model", required=True, help="Path to the trained model.")

    predict_parser = subparsers.add_parser("predict", help="Predict fraud for a single transaction.")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model.")
    predict_parser.add_argument("--features", type=float, nargs="+", required=True, help="Feature values for prediction.")

    return parser.parse_args()


def command_generate(args: argparse.Namespace) -> None:
    print(f"Generating synthetic data to {args.output}...")
    df = generate_synthetic_fraud_data(
        n_samples=args.samples,
        n_features=args.features,
        fraud_ratio=args.fraud_ratio,
        output_path=args.output,
    )
    print(f"Generated {len(df)} rows with fraud ratio {args.fraud_ratio:.2%}.")


def command_train(args: argparse.Namespace) -> None:
    print(f"Loading training data from {args.input}...")
    df = load_data(args.input)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=args.test_size)

    print("Training model...")
    model = train_model(X_train, y_train)
    save_model(model, args.model)
    print(f"Saved trained model to {args.model}.")

    print("Evaluating on hold-out test set...")
    metrics = evaluate_model(model, X_test, y_test)
    print(json.dumps(metrics, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    print(f"Loading evaluation data from {args.input}...")
    df = load_data(args.input)
    X, y = split_features_target(df)
    model = load_model(args.model)
    metrics = evaluate_model(model, X, y)
    print(json.dumps(metrics, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    features = np.array(args.features, dtype=float)

    if features.ndim != 1:
        raise ValueError("Feature vector must be one-dimensional.")

    prediction = model.predict(features.reshape(1, -1))[0]
    label = "Fraud" if int(prediction) == 1 else "Legitimate"
    print(f"Prediction: {label} ({int(prediction)})")


def main() -> None:
    args = parse_arguments()

    if args.command == "generate":
        command_generate(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "predict":
        command_predict(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
