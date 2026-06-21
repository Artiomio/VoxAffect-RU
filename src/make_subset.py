"""Create a small balanced DUSHA subset for baseline experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BINARY_LABELS = {
    "positive": 1,
    "happiness": 1,
    "happy": 1,
    "angry": 0,
    "anger": 0,
    "sad": 0,
    "sadness": 0,
}

THREE_CLASS_LABELS = {
    "positive": "positive",
    "happiness": "positive",
    "happy": "positive",
    "angry": "negative",
    "anger": "negative",
    "sad": "negative",
    "sadness": "negative",
    "neutral": "neutral",
}

NEUTRAL_VS_EMOTIONAL_LABELS = {
    "neutral": 0,
    "positive": 1,
    "happiness": 1,
    "happy": 1,
    "angry": 1,
    "anger": 1,
    "sad": 1,
    "sadness": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/dusha_emotion_audio/data",
        help="Local DUSHA data directory containing split CSVs and audio folders.",
    )
    parser.add_argument("--split", default="train", help="Split CSV to read.")
    parser.add_argument(
        "--task",
        choices=("binary", "3class", "neutral_vs_emotional"),
        default="binary",
        help="Label mapping task.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=300,
        help="Maximum examples to keep per mapped class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to data/processed/subset_<task>.csv.",
    )
    parser.add_argument(
        "--keep-missing-audio",
        action="store_true",
        help="Do not drop rows whose audio file does not exist locally.",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def map_label(value: object, task: str) -> int | str | None:
    normalized = normalize_label(value)
    if task == "binary":
        return BINARY_LABELS.get(normalized)
    if task == "neutral_vs_emotional":
        return NEUTRAL_VS_EMOTIONAL_LABELS.get(normalized)
    return THREE_CLASS_LABELS.get(normalized)


def default_output_path(task: str) -> Path:
    return Path("data/processed") / f"subset_{task}.csv"


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], csv_path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {missing}")


def add_audio_path(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    result = df.copy()
    result["audio_path"] = result["file_name"].map(lambda value: str(data_dir / str(value)))
    return result


def print_counts(title: str, series: pd.Series) -> None:
    print(f"\n== {title} ==")
    print(series.value_counts(dropna=False).to_string())


def make_subset(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    csv_path = data_dir / f"{args.split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Split CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    require_columns(df, ("file_name", "label"), csv_path)

    print(f"input_csv: {csv_path}")
    print(f"input_rows: {len(df)}")
    print_counts("Original Labels", df["label"])

    df = add_audio_path(df, data_dir)
    if not args.keep_missing_audio:
        exists_mask = df["audio_path"].map(lambda value: Path(value).exists())
        missing_count = int((~exists_mask).sum())
        if missing_count:
            print(f"dropping_missing_audio: {missing_count}")
        df = df.loc[exists_mask].copy()

    df["source_label"] = df["label"]
    df["target_label"] = df["label"].map(lambda value: map_label(value, args.task))
    unknown = sorted(df.loc[df["target_label"].isna(), "label"].map(normalize_label).unique())
    if unknown:
        print(f"skipped_unknown_or_unmapped_labels: {unknown}")

    mapped = df.dropna(subset=["target_label"]).copy()
    if mapped.empty:
        raise SystemExit("No rows left after label mapping.")
    if args.task in {"binary", "neutral_vs_emotional"}:
        mapped["target_label"] = mapped["target_label"].astype(int)

    print_counts("Mapped Labels Before Balancing", mapped["target_label"])

    groups = []
    for label, group in mapped.groupby("target_label", sort=True):
        if len(group) < args.samples_per_class:
            print(
                f"warning: class {label!r} has only {len(group)} rows, "
                f"requested {args.samples_per_class}"
            )
        groups.append(
            group.sample(n=min(args.samples_per_class, len(group)), random_state=args.seed)
        )

    subset = pd.concat(groups, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    subset["task"] = args.task
    subset["split"] = args.split

    columns = ["audio_path", "file_name", "source_label", "target_label", "task", "split"]
    output_path = Path(args.output) if args.output else default_output_path(args.task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.loc[:, columns].to_csv(output_path, index=False)

    print_counts("Mapped Labels After Balancing", subset["target_label"])
    print(f"\noutput_csv: {output_path}")
    print(f"output_rows: {len(subset)}")
    return output_path


def main() -> None:
    make_subset(parse_args())


if __name__ == "__main__":
    main()
