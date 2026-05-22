"""Create a human validation CSV from model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "audio_path",
    "source_label",
    "target_label",
    "predicted_label",
    "confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-path",
        default="artifacts/sklearn_logreg_binary_300/predictions.csv",
        help="Predictions CSV produced by src.train_sklearn.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/validation_sample_logreg_binary_300.csv",
        help="Output CSV for human validation.",
    )
    parser.add_argument("--n", type=int, default=30, help="Target number of rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=("mixed", "errors_first", "random"),
        default="mixed",
        help="Sampling strategy.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")


def sample_group(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if group.empty or n <= 0:
        return group.head(0)
    return group.sample(n=min(n, len(group)), random_state=seed)


def mixed_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample correct/error examples across true labels."""

    groups = []
    group_keys = sorted(df.groupby(["is_correct", "target_label"]).groups.keys())
    base = max(n // max(len(group_keys), 1), 1)

    for index, key in enumerate(group_keys):
        is_correct, target_label = key
        group = df[(df["is_correct"] == is_correct) & (df["target_label"] == target_label)]
        groups.append(sample_group(group, base, seed + index))

    selected = pd.concat(groups, ignore_index=False) if groups else df.head(0)
    if len(selected) < n:
        remaining = df.drop(index=selected.index, errors="ignore")
        selected = pd.concat(
            [selected, sample_group(remaining, n - len(selected), seed + 10_000)],
            ignore_index=False,
        )

    return selected.sample(frac=1.0, random_state=seed).head(n)


def errors_first_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    errors = df[~df["is_correct"]]
    correct = df[df["is_correct"]]
    error_target = min(len(errors), max(round(n * 0.6), 1))
    selected_errors = sample_group(errors, error_target, seed)
    selected_correct = sample_group(correct, n - len(selected_errors), seed + 1)
    selected = pd.concat([selected_errors, selected_correct], ignore_index=False)
    return selected.sample(frac=1.0, random_state=seed).head(n)


def build_validation_sample(args: argparse.Namespace) -> Path:
    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        raise SystemExit(f"Predictions CSV does not exist: {predictions_path}")

    df = pd.read_csv(predictions_path)
    require_columns(df, predictions_path)
    df["is_correct"] = df["target_label"].astype(str) == df["predicted_label"].astype(str)

    if args.strategy == "random":
        sample = sample_group(df, args.n, args.seed)
    elif args.strategy == "errors_first":
        sample = errors_first_sample(df, args.n, args.seed)
    else:
        sample = mixed_sample(df, args.n, args.seed)

    sample = sample.copy().reset_index(drop=True)
    sample["human_label"] = ""
    sample["human_comment"] = ""
    sample["review_status"] = "todo"
    sample["notes"] = ""

    columns = [
        "audio_path",
        "source_label",
        "target_label",
        "predicted_label",
        "confidence",
        "is_correct",
        "human_label",
        "human_comment",
        "review_status",
        "notes",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.loc[:, columns].to_csv(output_path, index=False)

    print(f"predictions_path: {predictions_path}")
    print(f"input_rows: {len(df)}")
    print(f"output: {output_path}")
    print(f"output_rows: {len(sample)}")
    print("\n== Correctness ==")
    print(sample["is_correct"].value_counts(dropna=False).to_string())
    print("\n== Target Labels ==")
    print(sample["target_label"].value_counts(dropna=False).to_string())
    return output_path


def main() -> None:
    build_validation_sample(parse_args())


if __name__ == "__main__":
    main()
