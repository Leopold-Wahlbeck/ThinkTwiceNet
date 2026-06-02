from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.data.loaders import load_cifar10_datasets, dataset_to_numpy
from src.evaluation.confidence import (
    evaluate_confidence_thresholds,
    print_confidence_threshold_results,
)


def train_logistic_regression(random_state: int = 1337) -> None:
    print("Loading CIFAR-10 data...")

    train_dataset, val_dataset, _ = load_cifar10_datasets(
        data_dir="data/raw",
        val_size=0.15,
        random_state=random_state,
        flatten_for_stage1=True,
    )

    class_names = train_dataset.dataset.classes


    print("Converting datasets to NumPy...")

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)

    print("Training logistic regression...")

    model = LogisticRegression(
        max_iter=100,
        tol=1e-3,
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    print("Evaluating...")

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    confidence = np.max(y_proba, axis=1)

    acc = accuracy_score(y_val, y_pred)

    print(f"\nValidation accuracy: {acc:.4f}")


    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    confidence_results = evaluate_confidence_thresholds(
        y_true=y_val,
        y_pred=y_pred,
        confidence=confidence,
        thresholds=thresholds,
    )

    print_confidence_threshold_results(confidence_results)

    output_model_dir = Path("outputs/models")
    output_model_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_model_dir / "stage1_logistic_regression.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")

    output_preds_dir = Path("outputs/preds")
    output_preds_dir.mkdir(parents=True, exist_ok=True)

    preds_path = output_preds_dir / "stage1_val_predictions.npz"

    np.savez(
        preds_path,
        y_true=y_val,
        y_pred=y_pred,
        y_proba=y_proba,
        confidence=confidence,
    )

    print(f"Saved validation predictions to: {preds_path}")

def save_results():
    pass

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=1337)
    args = parser.parse_args()

    train_logistic_regression(random_state=args.random_state)


if __name__ == "__main__":
    main()