"""Train a simple sklearn baseline on a processed DUSHA subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from .features import extract_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-path",
        default="data/processed/subset_binary.csv",
        help="CSV created by src.make_subset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Base directory where metrics and reports will be written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run subdirectory under --artifacts-dir.",
    )
    parser.add_argument(
        "--model",
        choices=("logreg", "random_forest", "svm_rbf"),
        default="logreg",
        help="Sklearn model to train.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum seconds to load from each audio file. Use 0 for full audio.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def load_subset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Subset CSV does not exist: {path}")
    df = pd.read_csv(path)
    required = {"audio_path", "target_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    sample_rate: int,
    max_duration: float | None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    vectors = []
    labels = []
    rows = []
    feature_names: list[str] | None = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            print(f"Skipping missing audio: {audio_path}")
            continue
        try:
            vector, names = extract_feature_vector(
                audio_path,
                sample_rate=sample_rate,
                max_duration=max_duration,
            )
        except Exception as exc:  # noqa: BLE001 - keep a baseline run moving.
            print(f"Skipping unreadable audio {audio_path}: {exc}")
            continue
        if feature_names is None:
            feature_names = names
        vectors.append(vector)
        labels.append(row["target_label"])
        rows.append(row)

    if not vectors:
        raise SystemExit("No usable audio rows found.")

    return (
        np.vstack(vectors),
        np.asarray(labels),
        feature_names or [],
        pd.DataFrame(rows).reset_index(drop=True),
    )


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def prediction_confidence(model: object, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return probabilities.max(axis=1)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        if scores.ndim == 1:
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
    return np.full(features.shape[0], np.nan)


def build_model(args: argparse.Namespace) -> tuple[object, str]:
    if args.model == "logreg":
        return (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
            "StandardScaler + LogisticRegression",
        )
    if args.model == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                class_weight="balanced",
                random_state=args.seed,
                n_jobs=-1,
            ),
            "RandomForestClassifier",
        )
    if args.model == "svm_rbf":
        return (
            make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=args.seed,
                ),
            ),
            "StandardScaler + SVC(kernel='rbf')",
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset_path)
    artifacts_dir = Path(args.artifacts_dir)
    if args.run_name:
        artifacts_dir = artifacts_dir / args.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    max_duration = args.max_duration if args.max_duration > 0 else None
    df = load_subset(subset_path)
    features, labels, feature_names, used_rows = build_feature_matrix(
        df,
        sample_rate=args.sample_rate,
        max_duration=max_duration,
    )

    train_x, val_x, train_y, val_y, train_rows, val_rows = train_test_split(
        features,
        labels,
        used_rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model, model_name = build_model(args)
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    confidences = prediction_confidence(model, val_x)

    class_names = [str(value) for value in sorted(pd.Series(labels).unique())]
    report = classification_report(val_y, predictions, digits=4)
    metrics = {
        "subset_path": str(subset_path),
        "rows_total": int(len(df)),
        "rows_used": int(len(used_rows)),
        "rows_train": int(len(train_y)),
        "rows_val": int(len(val_y)),
        "sample_rate": args.sample_rate,
        "max_duration": max_duration,
        "test_size": args.test_size,
        "seed": args.seed,
        "model": model_name,
        "model_key": args.model,
        "run_name": args.run_name,
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "accuracy": float(accuracy_score(val_y, predictions)),
        "precision_macro": float(
            precision_score(val_y, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(val_y, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(val_y, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            val_y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(
        val_y,
        predictions,
        class_names,
        artifacts_dir / "confusion_matrix.png",
    )

    predictions_df = val_rows.copy()
    predictions_df["predicted_label"] = predictions
    predictions_df["confidence"] = confidences
    predictions_df.to_csv(artifacts_dir / "predictions.csv", index=False)

    print(report)
    print(f"metrics: {artifacts_dir / 'metrics.json'}")
    print(f"report: {artifacts_dir / 'classification_report.txt'}")
    print(f"confusion_matrix: {artifacts_dir / 'confusion_matrix.png'}")
    print(f"predictions: {artifacts_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
