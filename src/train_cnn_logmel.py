"""Train a compact CNN baseline on cached log-mel spectrograms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
import time
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
from .tune_sklearn import predictions_at_threshold, threshold_values


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
    parser.add_argument(
        "--channels",
        default="16,32,64",
        help="Comma-separated CNN channel sizes, for example 16,32,64 or 32,64,128.",
    )
    parser.add_argument(
        "--conv-kernel-size",
        default="3",
        help="CNN kernel size as K or H,W. For example 3 keeps 3x3 kernels; 3,5 uses wider temporal kernels.",
    )
    parser.add_argument(
        "--pool-output-size",
        type=int,
        default=1,
        help="Final adaptive pooling output size. Use 1 for global average pooling, 4 for 4x4, or 0 to disable.",
    )
    parser.add_argument(
        "--pool-kernel-size",
        type=int,
        default=2,
        help="Kernel size for intermediate max-pooling layers.",
    )
    parser.add_argument(
        "--pool-strides",
        default="2",
        help="Comma-separated strides for intermediate max-pooling layers. A single value is broadcast to all pools.",
    )
    parser.add_argument(
        "--classifier-hidden-size",
        type=int,
        default=0,
        help="Optional hidden dense layer size before the output layer. Use 0 for a direct linear head.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument(
        "--scheduler",
        choices=("none", "reduce_on_plateau"),
        default="none",
        help="Optional learning-rate scheduler.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience by validation macro F1. Use 0 to disable.",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. auto uses CUDA when available.",
    )
    return parser.parse_args()


class CompactLogMelCNN(nn.Module):
    def __init__(
        self,
        channels: list[int],
        input_shape: tuple[int, int],
        pool_output_size: int,
        conv_kernel_size: tuple[int, int] = (3, 3),
        pool_kernel_size: int = 2,
        pool_strides: list[int] | None = None,
        classifier_hidden_size: int = 0,
        num_classes: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty.")
        if any(size < 1 for size in conv_kernel_size):
            raise ValueError("conv_kernel_size must contain positive integers.")
        if pool_output_size < 0:
            raise ValueError("pool_output_size must be non-negative.")
        if pool_kernel_size < 1:
            raise ValueError("pool_kernel_size must be positive.")
        if classifier_hidden_size < 0:
            raise ValueError("classifier_hidden_size must be non-negative.")
        expected_pool_layers = max(len(channels) - 1, 0)
        if pool_strides is None:
            pool_strides = [pool_kernel_size] * expected_pool_layers
        if len(pool_strides) != expected_pool_layers:
            raise ValueError(
                "pool_strides must have one entry per intermediate pooling layer."
            )
        if any(stride < 1 for stride in pool_strides):
            raise ValueError("pool_strides must contain positive integers.")
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=conv_kernel_size,
                        padding=(conv_kernel_size[0] // 2, conv_kernel_size[1] // 2),
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            if index < len(channels) - 1:
                layers.extend(
                    [
                        nn.MaxPool2d(pool_kernel_size, stride=pool_strides[index]),
                        nn.Dropout2d(dropout / 2 if index == 0 else dropout),
                    ]
                )
            in_channels = out_channels
        if pool_output_size > 0:
            layers.append(nn.AdaptiveAvgPool2d((pool_output_size, pool_output_size)))
        self.features = nn.Sequential(*layers)
        self.conv_kernel_size = conv_kernel_size
        self.pool_output_size = pool_output_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_strides = pool_strides
        self.classifier_input_features = compute_classifier_input_features(
            input_shape=input_shape,
            channels=channels,
            pool_output_size=pool_output_size,
            pool_kernel_size=pool_kernel_size,
            pool_strides=pool_strides,
        )
        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Dropout(dropout),
        ]
        if classifier_hidden_size > 0:
            classifier_layers.extend(
                [
                    nn.Linear(self.classifier_input_features, classifier_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden_size, num_classes),
                ]
            )
        else:
            classifier_layers.append(nn.Linear(self.classifier_input_features, num_classes))
        self.classifier_hidden_size = classifier_hidden_size
        self.classifier = nn.Sequential(*classifier_layers)

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


def parse_channels(value: str) -> list[int]:
    try:
        channels = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --channels value: {value}") from exc
    if not channels or any(channel <= 0 for channel in channels):
        raise SystemExit(f"--channels must contain positive integers: {value}")
    return channels


def parse_kernel_size(value: str) -> tuple[int, int]:
    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --conv-kernel-size value: {value}") from exc
    if len(sizes) == 1:
        sizes = sizes * 2
    if len(sizes) != 2 or any(size <= 0 for size in sizes):
        raise SystemExit(f"--conv-kernel-size must be K or H,W with positive integers: {value}")
    if any(size % 2 == 0 for size in sizes):
        raise SystemExit(f"--conv-kernel-size must use odd sizes to preserve feature-map shape: {value}")
    return sizes[0], sizes[1]


def parse_pool_strides(value: str, expected_pool_layers: int) -> list[int]:
    try:
        strides = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --pool-strides value: {value}") from exc
    if not strides or any(stride <= 0 for stride in strides):
        raise SystemExit(f"--pool-strides must contain positive integers: {value}")
    if expected_pool_layers == 0:
        return []
    if len(strides) == 1:
        return strides * expected_pool_layers
    if len(strides) != expected_pool_layers:
        raise SystemExit(
            f"--pool-strides must have 1 or {expected_pool_layers} values for channels={expected_pool_layers + 1}: {value}"
        )
    return strides


def compute_classifier_input_features(
    input_shape: tuple[int, int],
    channels: list[int],
    pool_output_size: int,
    pool_kernel_size: int = 2,
    pool_strides: list[int] | None = None,
) -> int:
    if pool_output_size > 0:
        return channels[-1] * pool_output_size * pool_output_size

    expected_pool_layers = max(len(channels) - 1, 0)
    if pool_strides is None:
        pool_strides = [pool_kernel_size] * expected_pool_layers
    if len(pool_strides) != expected_pool_layers:
        raise ValueError("pool_strides must have one entry per intermediate pooling layer.")

    height, width = input_shape
    for stride in pool_strides:
        height = ((height - pool_kernel_size) // stride) + 1
        width = ((width - pool_kernel_size) // stride) + 1
    if height <= 0 or width <= 0:
        raise ValueError(f"Input shape is too small after pooling: {input_shape}")
    return channels[-1] * height * width


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


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def tune_threshold(
    labels: np.ndarray,
    positive_probabilities: np.ndarray,
    thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    best = None
    for threshold in thresholds:
        predictions = predictions_at_threshold(positive_probabilities, threshold)
        scores = score_predictions(labels, predictions)
        row = {
            "threshold": threshold,
            "validation_accuracy": scores["accuracy"],
            "validation_precision_macro": scores["precision_macro"],
            "validation_recall_macro": scores["recall_macro"],
            "validation_f1_macro": scores["f1_macro"],
        }
        rows.append(row)
        if best is None or (
            row["validation_f1_macro"],
            row["validation_accuracy"],
        ) > (
            best["validation_f1_macro"],
            best["validation_accuracy"],
        ):
            best = row

    if best is None:
        raise SystemExit("No threshold candidates were evaluated.")
    return best, rows


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
    channels = parse_channels(args.channels)
    conv_kernel_size = parse_kernel_size(args.conv_kernel_size)
    if args.pool_output_size < 0:
        raise SystemExit("--pool-output-size must be non-negative.")
    if args.pool_kernel_size < 1:
        raise SystemExit("--pool-kernel-size must be positive.")
    if args.classifier_hidden_size < 0:
        raise SystemExit("--classifier-hidden-size must be non-negative.")
    pool_strides = parse_pool_strides(args.pool_strides, max(len(channels) - 1, 0))
    artifacts_dir = Path(args.artifacts_dir) / args.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    run_start_time = time.perf_counter()

    features, labels, rows, train_metadata = load_logmel_cache(args.features_path)
    eval_features, eval_labels, eval_rows, eval_metadata = load_logmel_cache(args.eval_features_path)
    train_x, val_x, train_y, val_y, train_rows, val_rows = train_test_split(
        features,
        labels,
        rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)
    eval_loader = make_loader(eval_features, eval_labels, args.batch_size, shuffle=False)

    model = CompactLogMelCNN(
        channels=channels,
        input_shape=(features.shape[1], features.shape[2]),
        conv_kernel_size=conv_kernel_size,
        pool_output_size=args.pool_output_size,
        pool_kernel_size=args.pool_kernel_size,
        pool_strides=pool_strides,
        classifier_hidden_size=args.classifier_hidden_size,
        num_classes=len(np.unique(labels)),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    history = []
    best_state = None
    best_val_accuracy = -1.0
    best_val_f1_macro = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        val_predictions, _ = predict(model, val_loader, device)
        val_scores = score_predictions(val_y, val_predictions)
        val_f1_macro = float(val_scores["f1_macro"])
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision_macro": val_scores["precision_macro"],
                "val_recall_macro": val_scores["recall_macro"],
                "val_f1_macro": val_f1_macro,
                "learning_rate": current_learning_rate(optimizer),
                "epoch_duration_seconds": epoch_duration_seconds,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        if val_f1_macro > best_val_f1_macro + args.min_delta:
            best_val_f1_macro = val_f1_macro
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"val_f1={val_f1_macro:.4f} lr={current_learning_rate(optimizer):.6f} "
            f"epoch_sec={epoch_duration_seconds:.1f}"
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_early = True
            print(f"early stopping at epoch {epoch:03d}; best_epoch={best_epoch:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_predictions, val_probabilities = predict(model, val_loader, device)
    thresholds = threshold_values(args.threshold_min, args.threshold_max, args.threshold_step)
    best_threshold, threshold_rows = tune_threshold(
        val_y,
        val_probabilities[:, 1],
        thresholds,
    )

    eval_argmax_predictions, eval_probabilities = predict(model, eval_loader, device)
    eval_predictions = predictions_at_threshold(
        eval_probabilities[:, 1],
        best_threshold["threshold"],
    )
    eval_scores = score_predictions(eval_labels, eval_predictions)
    report = classification_report(eval_labels, eval_predictions, digits=4)
    class_names = [str(value) for value in sorted(np.unique(np.concatenate([labels, eval_labels])))]
    finished_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - run_start_time
    epoch_durations = [row["epoch_duration_seconds"] for row in history]
    mean_epoch_seconds = float(np.mean(epoch_durations)) if epoch_durations else None

    metrics = {
        "run_name": args.run_name,
        "model": "CompactLogMelCNN",
        "features_path": args.features_path,
        "eval_features_path": args.eval_features_path,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "mean_epoch_seconds": mean_epoch_seconds,
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
        "channels": channels,
        "conv_kernel_size": list(conv_kernel_size),
        "pool_output_size": args.pool_output_size,
        "pool_kernel_size": args.pool_kernel_size,
        "pool_strides": pool_strides,
        "classifier_hidden_size": args.classifier_hidden_size,
        "classifier_input_features": model.classifier_input_features,
        "device": str(device),
        "scheduler": args.scheduler,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_f1_macro": best_val_f1_macro,
        "threshold_grid": thresholds,
        "best_threshold": best_threshold,
        "history": history,
        **eval_scores,
    }

    pd.DataFrame(threshold_rows).to_csv(artifacts_dir / "threshold_results.csv", index=False)
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
    predictions_df["argmax_predicted_label"] = eval_argmax_predictions
    predictions_df["confidence"] = eval_probabilities.max(axis=1)
    predictions_df["positive_probability"] = eval_probabilities[:, 1]
    predictions_df.to_csv(artifacts_dir / "predictions.csv", index=False)

    validation_predictions_df = val_rows.copy()
    validation_predictions_df["predicted_label"] = predictions_at_threshold(
        val_probabilities[:, 1],
        best_threshold["threshold"],
    )
    validation_predictions_df["argmax_predicted_label"] = val_predictions
    validation_predictions_df["confidence"] = val_probabilities.max(axis=1)
    validation_predictions_df["positive_probability"] = val_probabilities[:, 1]
    validation_predictions_df.to_csv(artifacts_dir / "validation_predictions.csv", index=False)

    print(report)
    print(f"duration_seconds: {duration_seconds:.1f}")
    if mean_epoch_seconds is not None:
        print(f"mean_epoch_seconds: {mean_epoch_seconds:.1f}")
    print(f"metrics: {artifacts_dir / 'metrics.json'}")
    print(f"threshold_results: {artifacts_dir / 'threshold_results.csv'}")
    print(f"model: {artifacts_dir / 'model.pt'}")
    print(f"curves: {artifacts_dir / 'training_curves.png'}")
    print(f"confusion_matrix: {artifacts_dir / 'confusion_matrix.png'}")
    print(f"predictions: {artifacts_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
