from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.data.loaders import DataConfig, create_dataloaders, load_cifar10_datasets, dataset_to_numpy
from src.models.stage2_pretrained_cnn import load_pretrained_cnn_model


def predict_stage1(
    model_path: Path,
    split: str,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the trained logistic regression model and predict on val or test."""
    train_dataset, val_dataset, test_dataset = load_cifar10_datasets(
        data_dir="data/raw",
        val_size=val_size,
        random_state=random_state,
        flatten_for_stage1=True,
        normalize=False,
    )

    if split == "val":
        dataset = val_dataset
    elif split == "test":
        dataset = test_dataset
    else:
        raise ValueError("split must be either 'val' or 'test'")

    X, y_true = dataset_to_numpy(dataset)

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    confidence = np.max(y_proba, axis=1)

    return y_true, y_pred, y_proba, confidence


def predict_stage2(
    split: str,
    val_size: float,
    random_state: int,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pretrained ResNet20 and predict on val or test."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device for ResNet20: {device}")

    config = DataConfig(
        data_dir="data/raw",
        batch_size=batch_size,
        num_workers=num_workers,
        val_size=val_size,
        random_state=random_state,
        flatten_for_stage1=False,
        normalize=True,
    )

    _, val_loader, test_loader = create_dataloaders(config)

    if split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    else:
        raise ValueError("split must be either 'val' or 'test'")

    model = load_pretrained_cnn_model()
    model.to(device)
    model.eval()

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_y_proba: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_y_true.append(labels.cpu().numpy())
            all_y_pred.append(predictions.cpu().numpy())
            all_y_proba.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_y_proba)

    return y_true, y_pred, y_proba


def evaluate_multistage(
    y_true: np.ndarray,
    stage1_pred: np.ndarray,
    stage1_confidence: np.ndarray,
    stage2_pred: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Evaluate final multi-stage predictions for each confidence threshold."""
    stage1_accuracy = accuracy_score(y_true, stage1_pred)
    stage2_accuracy = accuracy_score(y_true, stage2_pred)

    results: list[dict[str, float]] = []

    for threshold in thresholds:
        use_stage1 = stage1_confidence >= threshold
        use_stage2 = ~use_stage1

        final_pred = np.where(use_stage1, stage1_pred, stage2_pred)
        multistage_accuracy = accuracy_score(y_true, final_pred)

        if use_stage1.sum() > 0:
            accepted_accuracy = accuracy_score(y_true[use_stage1], stage1_pred[use_stage1])
        else:
            accepted_accuracy = 0.0

        if use_stage2.sum() > 0:
            deferred_accuracy = accuracy_score(y_true[use_stage2], stage2_pred[use_stage2])
        else:
            deferred_accuracy = 0.0

        coverage = float(use_stage1.mean())
        deferred_rate = float(use_stage2.mean())

        results.append(
            {
                "threshold": threshold,
                "stage1_accuracy": stage1_accuracy,
                "resnet20_accuracy": stage2_accuracy,
                "multistage_accuracy": multistage_accuracy,
                "accuracy_diff_vs_resnet20": multistage_accuracy - stage2_accuracy,
                "stage1_coverage": coverage,
                "resnet20_usage": deferred_rate,
                "stage1_accepted_accuracy": accepted_accuracy,
                "resnet20_deferred_accuracy": deferred_accuracy,
                "num_stage1": int(use_stage1.sum()),
                "num_resnet20": int(use_stage2.sum()),
            }
        )

    return results


def print_results(results: list[dict[str, float]]) -> None:
    print("\nMulti-stage evaluation:")
    print(
        "Threshold | Accuracy | ResNet20 acc | Diff vs ResNet20 | "
        "Stage1 cov | ResNet20 usage | Stage1 acc accepted"
    )
    print("-" * 103)
    for row in results:
        print(
            f"{row['threshold']:9.2f} | "
            f"{row['multistage_accuracy']:8.4f} | "
            f"{row['resnet20_accuracy']:11.4f} | "
            f"{row['accuracy_diff_vs_resnet20']:16.4f} | "
            f"{row['stage1_coverage']:10.4f} | "
            f"{row['resnet20_usage']:14.4f} | "
            f"{row['stage1_accepted_accuracy']:19.4f}"
        )


def save_results(results: list[dict[str, float]], output_path: Path) -> None:
    if not results:
        raise ValueError("No results to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--stage1-model", type=Path, default=Path("outputs/models/stage1_logistic_regression.joblib"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/multistage_threshold_results.csv"),
    )
    parser.add_argument(
        "--output-preds",
        type=Path,
        default=Path("outputs/preds/multistage_predictions.npz"),
    )
    args = parser.parse_args()

    print(f"Evaluating multi-stage pipeline on split: {args.split}")

    y_true_stage1, stage1_pred, stage1_proba, stage1_confidence = predict_stage1(
        model_path=args.stage1_model,
        split=args.split,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    y_true_stage2, stage2_pred, stage2_proba = predict_stage2(
        split=args.split,
        val_size=args.val_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if not np.array_equal(y_true_stage1, y_true_stage2):
        raise RuntimeError(
            "Stage 1 and Stage 2 labels do not match. "
            "Ensure both models are evaluated on the same split in the same order."
        )

    results = evaluate_multistage(
        y_true=y_true_stage1,
        stage1_pred=stage1_pred,
        stage1_confidence=stage1_confidence,
        stage2_pred=stage2_pred,
        thresholds=args.thresholds,
    )

    print_results(results)
    save_results(results, args.output_csv)
    print(f"Saved multi-stage results to: {args.output_csv}")

    args.output_preds.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_preds,
        y_true=y_true_stage1,
        stage1_pred=stage1_pred,
        stage1_proba=stage1_proba,
        stage1_confidence=stage1_confidence,
        stage2_pred=stage2_pred,
        stage2_proba=stage2_proba,
        thresholds=np.array(args.thresholds),
    )
    print(f"Saved multi-stage prediction inputs to: {args.output_preds}")


if __name__ == "__main__":
    main()
