"""Build a reusable feature cache from a processed subset CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from .feature_cache import save_feature_cache
from .train_sklearn import build_feature_matrix, load_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-path",
        required=True,
        help="Processed subset CSV created by src.make_subset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz feature cache path.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum seconds to load from each audio file. Use 0 for full audio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset_path)
    max_duration = args.max_duration if args.max_duration > 0 else None

    df = load_subset(subset_path)
    features, labels, feature_names, rows = build_feature_matrix(
        df,
        sample_rate=args.sample_rate,
        max_duration=max_duration,
    )
    output_path = save_feature_cache(
        args.output,
        features=features,
        labels=labels,
        rows=rows,
        feature_names=feature_names,
        metadata={
            "subset_path": str(subset_path),
            "rows_total": len(df),
            "rows_used": len(rows),
            "sample_rate": args.sample_rate,
            "max_duration": max_duration,
            "feature_count": int(features.shape[1]),
        },
    )

    print(f"subset_path: {subset_path}")
    print(f"output: {output_path}")
    print(f"rows_used: {len(rows)}")
    print(f"feature_count: {features.shape[1]}")


if __name__ == "__main__":
    main()
