"""Compare two prediction CSV files on the same evaluation subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="First predictions.csv path.")
    parser.add_argument("--left-name", required=True, help="Short name for first model.")
    parser.add_argument("--right", required=True, help="Second predictions.csv path.")
    parser.add_argument("--right-name", required=True, help="Short name for second model.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison artifacts.")
    return parser.parse_args()


def load_predictions(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"audio_path", "target_label", "predicted_label", "positive_probability"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")

    columns = [
        "audio_path",
        "file_name",
        "source_label",
        "target_label",
        "predicted_label",
        "positive_probability",
        "confidence",
    ]
    available = [column for column in columns if column in df.columns]
    result = df[available].copy()
    result = result.rename(
        columns={
            "predicted_label": f"{name}_predicted_label",
            "positive_probability": f"{name}_positive_probability",
            "confidence": f"{name}_confidence",
        }
    )
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = load_predictions(args.left, args.left_name)
    right = load_predictions(args.right, args.right_name)
    merged = left.merge(
        right.drop(columns=["target_label", "file_name", "source_label"], errors="ignore"),
        on="audio_path",
        how="inner",
    )
    if merged.empty:
        raise SystemExit("Prediction files have no overlapping audio_path values.")

    left_correct = f"{args.left_name}_correct"
    right_correct = f"{args.right_name}_correct"
    merged[left_correct] = merged[f"{args.left_name}_predicted_label"] == merged["target_label"]
    merged[right_correct] = merged[f"{args.right_name}_predicted_label"] == merged["target_label"]

    def category(row: pd.Series) -> str:
        if row[left_correct] and row[right_correct]:
            return "both_correct"
        if row[left_correct] and not row[right_correct]:
            return f"{args.left_name}_only_correct"
        if not row[left_correct] and row[right_correct]:
            return f"{args.right_name}_only_correct"
        return "both_wrong"

    merged["comparison_category"] = merged.apply(category, axis=1)
    merged.to_csv(output_dir / "comparison.csv", index=False)

    counts = merged["comparison_category"].value_counts().sort_index()
    by_label = (
        merged.groupby(["target_label", "comparison_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["target_label", "comparison_category"])
    )
    by_label.to_csv(output_dir / "comparison_by_label.csv", index=False)

    summary = {
        "left": args.left_name,
        "right": args.right_name,
        "left_path": args.left,
        "right_path": args.right,
        "rows": int(len(merged)),
        "left_correct": int(merged[left_correct].sum()),
        "right_correct": int(merged[right_correct].sum()),
        "left_accuracy": float(merged[left_correct].mean()),
        "right_accuracy": float(merged[right_correct].mean()),
        "category_counts": {key: int(value) for key, value in counts.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for category_name, group in merged.groupby("comparison_category"):
        group.to_csv(output_dir / f"{category_name}.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"comparison: {output_dir / 'comparison.csv'}")
    print(f"by_label: {output_dir / 'comparison_by_label.csv'}")


if __name__ == "__main__":
    main()
