"""Train a compact CNN baseline on cached log-mel spectrograms."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .logmel_cache import load_logmel_cache
from .train_sklearn import save_confusion_matrix, score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", required=True, help="Training log-mel .npz cache.")
    parser.add_argument("--eval-features-path", required=True, help="External eval log-mel .npz cache.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. auto uses CUDA when available.",
    )
    return parser.parse_args()


class CompactLogMelCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device(name)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = torch.from_numpy(features[:, None, :, :].astype(np.float32, copy=False))
    targets = torch.from_numpy(labels.astype(np.int64, copy=False))
    return DataLoader(TensorDataset(tensors, targets), batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()) * len(labels))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(len(labels))

    return sum(losses) / total, correct / total


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(batch_probabilities)
            predictions.append(batch_probabilities.argmax(axis=1))
    return np.concatenate(predictions), np.vstack(probabilities)


def save_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    frame.to_csv(output_path.with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    artifacts_dir = Path(args.artifacts_dir) / args.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features, labels, rows, train_metadata = load_logmel_cache(args.features_path)
    eval_features, eval_labels, eval_rows, eval_metadata = load_logmel_cache(args.eval_features_path)
    train_x, val_x, train_y, val_y = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)
    eval_loader = make_loader(eval_features, eval_labels, args.batch_size, shuffle=False)

    model = CompactLogMelCNN(num_classes=len(np.unique(labels)), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state = None
    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_predictions, eval_probabilities = predict(model, eval_loader, device)
    eval_scores = score_predictions(eval_labels, eval_predictions)
    report = classification_report(eval_labels, eval_predictions, digits=4)
    class_names = [str(value) for value in sorted(np.unique(np.concatenate([labels, eval_labels])))]

    metrics = {
        "run_name": args.run_name,
        "model": "CompactLogMelCNN",
        "features_path": args.features_path,
        "eval_features_path": args.eval_features_path,
        "train_cache_metadata": train_metadata,
        "eval_cache_metadata": eval_metadata,
        "rows_total": int(len(rows)),
        "rows_used": int(len(labels)),
        "eval_rows_used": int(len(eval_labels)),
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "seed": args.seed,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "device": str(device),
        "best_validation_accuracy": best_val_accuracy,
        "history": history,
        **eval_scores,
    }

    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_history(history, artifacts_dir / "training_curves.png")
    save_confusion_matrix(
        eval_labels,
        eval_predictions,
        class_names,
        artifacts_dir / "confusion_matrix.png",
    )
    torch.save(model.state_dict(), artifacts_dir / "model.pt")

    predictions_df = eval_rows.copy()
    predictions_df["predicted_label"] = eval_predictions
    predictions_df["confidence"] = eval_probabilities.max(axis=1)
    predictions_df["positive_probability"] = eval_probabilities[:, 1]
    predictions_df.to_csv(artifacts_dir / "predictions.csv", index=False)

    print(report)
    print(f"metrics: {artifacts_dir / 'metrics.json'}")
    print(f"model: {artifacts_dir / 'model.pt'}")
    print(f"curves: {artifacts_dir / 'training_curves.png'}")
    print(f"confusion_matrix: {artifacts_dir / 'confusion_matrix.png'}")
    print(f"predictions: {artifacts_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
