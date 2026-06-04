from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from src.data.loaders import DataConfig, create_dataloaders
from src.models.stage2_pretrained_cnn import load_pretrained_cnn_model


def evaluate_pretrained_cnn(split: str = "val"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    config = DataConfig(
        data_dir="data/raw",
        batch_size=64,
        num_workers=2,
        val_size=0.15,
        random_state=1337,
        flatten_for_stage1=False,
        normalize=True,
    )

    _, val_loader, test_loader = create_dataloaders(config)

    model = load_pretrained_cnn_model()
    model.to(device)
    model.eval()

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    with torch.no_grad():
        for images, labels in val_loader:
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

    acc = accuracy_score(y_true, y_pred)
    print(f"Stage 2 pretrained CNN validation accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred))

    output_dir = Path("outputs/preds")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "stage2_pretrained_cnn_val_predictions.npz"  

    np.savez(
        output_path,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    print(f"Saved validation predictions to: {output_path}")

def main() -> None:
    evaluate_pretrained_cnn()

if __name__ == "__main__":    
    main()