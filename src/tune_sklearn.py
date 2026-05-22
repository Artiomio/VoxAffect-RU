"""Tune sklearn baselines on cached features and evaluate once on test."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .train_sklearn import (
    load_features_from_args,
    prediction_confidence,
    save_confusion_matrix,
    score_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        required=True,
        help="Training .npz feature cache created by src.build_feature_cache.",
    )
    parser.add_argument(
        "--eval-features-path",
        required=True,
        help="External evaluation .npz feature cache.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Base directory where tuning artifacts will be written.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run subdirectory under --artifacts-dir.",
    )
    parser.add_argument(
        "--model",
        choices=("logreg", "svm_rbf"),
        required=True,
        help="Sklearn model family to tune.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.30,
        help="Minimum positive-class probability threshold.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
        help="Maximum positive-class probability threshold.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
        help="Positive-class probability threshold step.",
    )
    return parser.parse_args()


def candidate_params(model_key: str) -> list[dict[str, Any]]:
    if model_key == "logreg":
        return [
            {"C": c, "class_weight": class_weight}
            for c, class_weight in product(
                [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                ["balanced", None],
            )
        ]

    if model_key == "svm_rbf":
        return [
            {"C": c, "gamma": gamma, "class_weight": class_weight}
            for c, gamma, class_weight in product(
                [0.3, 1.0, 3.0, 10.0],
                ["scale", 0.003, 0.01, 0.03],
                ["balanced", None],
            )
        ]

    raise ValueError(f"Unsupported model: {model_key}")


def build_model(model_key: str, params: dict[str, Any], seed: int, max_iter: int) -> tuple[object, str]:
    if model_key == "logreg":
        return (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=params["C"],
                    class_weight=params["class_weight"],
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
            "StandardScaler + LogisticRegression",
        )

    if model_key == "svm_rbf":
        return (
            make_pipeline(
                StandardScaler(),
                SVC(
                    C=params["C"],
                    gamma=params["gamma"],
                    kernel="rbf",
                    class_weight=params["class_weight"],
                    probability=True,
                    random_state=seed,
                ),
            ),
            "StandardScaler + SVC(kernel='rbf')",
        )

    raise ValueError(f"Unsupported model: {model_key}")


def positive_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Tuning requires models with predict_proba support.")
    probabilities = model.predict_proba(features)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    if 1 not in classes:
        raise SystemExit(f"Positive class 1 not found in model classes: {classes}")
    return probabilities[:, classes.index(1)]


def threshold_values(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        raise SystemExit("--threshold-step must be positive.")
    values = np.arange(min_value, max_value + step / 2, step)
    return [round(float(value), 6) for value in values]


def predictions_at_threshold(positive_probs: np.ndarray, threshold: float) -> np.ndarray:
    return (positive_probs >= threshold).astype(int)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir) / args.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        features,
        labels,
        feature_names,
        used_rows,
        train_cache_metadata,
    ) = load_features_from_args(
        Path("unknown_train_subset.csv"),
        Path(args.features_path),
        sample_rate=16_000,
        max_duration=None,
    )
    (
        eval_df,
        eval_features,
        eval_labels,
        _,
        eval_rows,
        eval_cache_metadata,
    ) = load_features_from_args(
        Path("unknown_eval_subset.csv"),
        Path(args.eval_features_path),
        sample_rate=16_000,
        max_duration=None,
    )

    train_x, val_x, train_y, val_y, _, _ = train_test_split(
        features,
        labels,
        used_rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    thresholds = threshold_values(args.threshold_min, args.threshold_max, args.threshold_step)
    tuning_rows = []
    best: dict[str, Any] | None = None

    for index, params in enumerate(candidate_params(args.model), start=1):
        model, model_name = build_model(args.model, params, args.seed, args.max_iter)
        model.fit(train_x, train_y)
        validation_probs = positive_probabilities(model, val_x)

        for threshold in thresholds:
            validation_predictions = predictions_at_threshold(validation_probs, threshold)
            scores = score_predictions(val_y, validation_predictions)
            row = {
                "candidate": index,
                "model_key": args.model,
                "model": model_name,
                "params": params,
                "threshold": threshold,
                "validation_accuracy": scores["accuracy"],
                "validation_precision_macro": scores["precision_macro"],
                "validation_recall_macro": scores["recall_macro"],
                "validation_f1_macro": scores["f1_macro"],
            }
            tuning_rows.append(row)
            if best is None or (
                row["validation_f1_macro"],
                row["validation_accuracy"],
            ) > (
                best["validation_f1_macro"],
                best["validation_accuracy"],
            ):
                best = row

    if best is None:
        raise SystemExit("No tuning candidates were evaluated.")

    final_model, model_name = build_model(args.model, best["params"], args.seed, args.max_iter)
    final_model.fit(features, labels)
    eval_probs = positive_probabilities(final_model, eval_features)
    eval_predictions = predictions_at_threshold(eval_probs, best["threshold"])
    eval_scores = score_predictions(eval_labels, eval_predictions)
    confidences = prediction_confidence(final_model, eval_features)

    class_names = [str(value) for value in sorted(pd.Series(np.concatenate([labels, eval_labels])).unique())]
    report = classification_report(eval_labels, eval_predictions, digits=4)
    metrics = {
        "run_name": args.run_name,
        "model_key": args.model,
        "model": model_name,
        "features_path": args.features_path,
        "eval_features_path": args.eval_features_path,
        "train_cache_metadata": train_cache_metadata,
        "eval_cache_metadata": eval_cache_metadata,
        "rows_total": int(len(train_df)),
        "rows_used": int(len(used_rows)),
        "eval_rows_total": int(len(eval_df)),
        "eval_rows_used": int(len(eval_rows)),
        "validation_rows": int(len(val_y)),
        "final_train_rows": int(len(labels)),
        "seed": args.seed,
        "test_size": args.test_size,
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "threshold_grid": thresholds,
        "candidate_count": len(candidate_params(args.model)),
        "best_validation": best,
        **eval_scores,
    }

    pd.DataFrame(tuning_rows).to_csv(artifacts_dir / "tuning_results.csv", index=False)
    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(
        eval_labels,
        eval_predictions,
        class_names,
        artifacts_dir / "confusion_matrix.png",
    )

    predictions_df = eval_rows.copy()
    predictions_df["predicted_label"] = eval_predictions
    predictions_df["positive_probability"] = eval_probs
    predictions_df["confidence"] = confidences
    predictions_df.to_csv(artifacts_dir / "predictions.csv", index=False)

    print(f"best_validation: {best}")
    print(report)
    print(f"metrics: {artifacts_dir / 'metrics.json'}")
    print(f"tuning_results: {artifacts_dir / 'tuning_results.csv'}")
    print(f"report: {artifacts_dir / 'classification_report.txt'}")
    print(f"confusion_matrix: {artifacts_dir / 'confusion_matrix.png'}")
    print(f"predictions: {artifacts_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
