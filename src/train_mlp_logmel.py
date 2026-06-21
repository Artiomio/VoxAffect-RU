"""Train MLP baselines on cached log-mel spectrograms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn

from .logmel_cache import load_logmel_cache
from .train_cnn_logmel import (
    choose_device,
    current_learning_rate,
    make_loader,
    predict,
    run_epoch,
    save_history,
    set_seed,
    tune_threshold,
)
from .train_sklearn import save_confusion_matrix, score_predictions
from .tune_sklearn import predictions_at_threshold, threshold_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", required=True, help="Training log-mel .npz cache.")
    parser.add_argument("--eval-features-path", required=True, help="External eval log-mel .npz cache.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--activation",
        choices=("relu", "tanh", "sigmoid"),
        default="relu",
        help="Activation function for the hidden dense layer.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="Hidden dense layer size. Use 0 for Flatten -> Linear(num_features, 2).",
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


class LogMelMLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_size: int = 0,
        num_classes: int = 2,
        dropout: float = 0.25,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_size < 0:
            raise ValueError("hidden_size must be non-negative.")
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.input_features = input_shape[0] * input_shape[1]
        self.hidden_size = hidden_size
        self.activation = activation
        layers: list[nn.Module] = [nn.Flatten()]
        if hidden_size > 0:
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(self.input_features, hidden_size),
                    activations[activation](),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes),
                ]
            )
        else:
            layers.append(nn.Linear(self.input_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if args.hidden_size < 0:
        raise SystemExit("--hidden-size must be non-negative.")
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

    model = LogMelMLP(
        input_shape=(features.shape[1], features.shape[2]),
        hidden_size=args.hidden_size,
        num_classes=len(np.unique(labels)),
        dropout=args.dropout,
        activation=args.activation,
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
        "model": "LogMelMLP",
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
        "activation": args.activation,
        "input_shape": [int(features.shape[1]), int(features.shape[2])],
        "input_features": int(model.input_features),
        "hidden_size": args.hidden_size,
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
