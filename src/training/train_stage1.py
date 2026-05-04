from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.data.loaders import load_cifar10_datasets, dataset_to_numpy


def train_logistic_regression(random_state: int = 1337) -> None:
    print("Loading CIFAR-10 data...")

    train_dataset, val_dataset, _ = load_cifar10_datasets(
        data_dir="data/raw",
        val_size=0.15,
        random_state=random_state,
        flatten_for_stage1=True,
    )

    print("Converting datasets to NumPy...")

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)

    print("Training logistic regression...")

    model = LogisticRegression(
        max_iter=100,
        tol=1e-3,
        solver="lbfgs",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Evaluating...")

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nValidation accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "stage1_logistic_regression.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=1337)
    args = parser.parse_args()

    train_logistic_regression(random_state=args.random_state)


if __name__ == "__main__":
    main()