"""Inspect local or Hugging Face DUSHA dataset structure.

This script is intentionally read-only. It prints dataset metadata, likely
label columns, label values, and cheap duration statistics for a small sample.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LIKELY_LABEL_COLUMNS = (
    "label",
    "emotion",
    "emotion_label",
    "class",
    "target",
    "sentiment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional Hugging Face dataset name, e.g. KELONMYOSA/dusha_emotion_audio.",
    )
    parser.add_argument(
        "--data-dir",
        "--local-path",
        dest="data_dir",
        default="data/dusha_emotion_audio/data",
        help="Local dataset data directory containing split CSVs and audio folders.",
    )
    parser.add_argument("--split", default="train", help="Split to inspect.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of examples to inspect for paths and durations.",
    )
    parser.add_argument(
        "--label-values-limit",
        type=int,
        default=30,
        help="Maximum number of label values to print.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def read_csv_head(path: Path, limit: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
        return list(reader.fieldnames or []), rows


def iter_csv_column(path: Path, column: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get(column)
            if value is not None:
                yield value


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def likely_label_columns(columns: list[str]) -> list[str]:
    normalized = {column.lower(): column for column in columns}
    direct = [normalized[name] for name in LIKELY_LABEL_COLUMNS if name in normalized]
    fuzzy = [
        column
        for column in columns
        if column not in direct
        and any(token in column.lower() for token in ("label", "emotion", "class"))
    ]
    return direct + fuzzy


def duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile is not installed; skipping duration stats.")
        return None

    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001 - inspection should not fail on one bad file.
        print(f"Could not read duration for {path}: {exc}")
        return None

    if not info.samplerate:
        return None
    return float(info.frames) / float(info.samplerate)


def print_duration_stats(paths: list[Path]) -> None:
    try:
        import soundfile  # noqa: F401
    except ImportError:
        print("soundfile is not installed; skipping duration stats.")
        return

    durations = [duration_seconds(path) for path in paths]
    values = sorted(value for value in durations if value is not None)
    if not values:
        print("No duration stats available.")
        return

    def percentile(p: float) -> float:
        index = min(round((len(values) - 1) * p), len(values) - 1)
        return values[index]

    print(
        compact_json(
            {
                "count": len(values),
                "min_sec": round(values[0], 3),
                "p50_sec": round(percentile(0.5), 3),
                "p90_sec": round(percentile(0.9), 3),
                "max_sec": round(values[-1], 3),
            }
        )
    )


def inspect_local(data_dir: Path, split: str, sample_size: int, label_values_limit: int) -> None:
    print_section("Local Dataset")
    print(f"data_dir: {data_dir}")
    if not data_dir.exists():
        raise SystemExit(f"Local data directory does not exist: {data_dir}")

    split_csvs = sorted(data_dir.glob("*.csv"))
    splits = [path.stem for path in split_csvs]
    print(f"available_splits: {splits}")

    split_csv = data_dir / f"{split}.csv"
    if not split_csv.exists():
        raise SystemExit(f"Split CSV not found: {split_csv}. Available splits: {splits}")

    columns, sample_rows = read_csv_head(split_csv, sample_size)

    print_section("Columns")
    print(compact_json(columns))

    print_section("Rows")
    print(f"row_count: {count_csv_rows(split_csv)}")
    if sample_rows:
        print("first_row:")
        print(compact_json(sample_rows[0]))

    label_columns = likely_label_columns(columns)
    print_section("Likely Label Columns")
    print(compact_json(label_columns))
    for column in label_columns:
        counts = Counter(iter_csv_column(split_csv, column))
        print(f"{column}:")
        print(compact_json(dict(counts.most_common(label_values_limit))))

    path_column = "file_name" if "file_name" in columns else None
    if path_column is None:
        path_candidates = [
            column
            for column in columns
            if any(token in column.lower() for token in ("path", "file", "audio", "wav"))
        ]
        path_column = path_candidates[0] if path_candidates else None

    print_section("Audio Field")
    if path_column is None:
        print("No likely audio path column found.")
        return

    print(f"path_column: {path_column}")
    audio_paths = [data_dir / row[path_column] for row in sample_rows if row.get(path_column)]
    existing = [path for path in audio_paths if path.exists()]
    missing = [path for path in audio_paths if not path.exists()]
    print(
        compact_json(
            {
                "sample_checked": len(audio_paths),
                "existing": len(existing),
                "missing": len(missing),
                "first_existing": str(existing[0]) if existing else None,
                "first_missing": str(missing[0]) if missing else None,
            }
        )
    )

    print_section("Duration Stats")
    print_duration_stats(existing)


def describe_hf_audio_field(example: dict[str, Any]) -> None:
    audio_keys = [
        key
        for key, value in example.items()
        if key.lower() in ("audio", "wav", "file", "path", "file_name")
        or isinstance(value, dict)
        and {"array", "path", "sampling_rate"}.intersection(value.keys())
    ]
    if not audio_keys:
        print("No obvious audio field found in first example.")
        return
    for key in audio_keys:
        value = example[key]
        if isinstance(value, dict):
            print(f"{key}:")
            print(compact_json({k: type(v).__name__ for k, v in value.items()}))
        else:
            print(f"{key}: {value!r}")


def inspect_hf(dataset_name: str, split: str, sample_size: int, label_values_limit: int) -> None:
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    print_section("Hugging Face Dataset")
    print(f"dataset_name: {dataset_name}")
    splits = get_dataset_split_names(dataset_name)
    print(f"available_splits: {splits}")
    if split not in splits:
        raise SystemExit(f"Split {split!r} is not available. Available splits: {splits}")

    dataset = load_dataset(dataset_name, split=f"{split}[:{sample_size}]")
    columns = list(dataset.column_names)

    print_section("Columns")
    print(compact_json(columns))

    first = dict(dataset[0]) if len(dataset) else {}
    print_section("First Example")
    print(compact_json({key: type(value).__name__ for key, value in first.items()}))
    print(compact_json(first))

    print_section("Audio Field")
    describe_hf_audio_field(first)

    label_columns = likely_label_columns(columns)
    print_section("Likely Label Columns")
    print(compact_json(label_columns))
    for column in label_columns:
        counts = Counter(str(row[column]) for row in dataset if column in row)
        print(f"{column}:")
        print(compact_json(dict(counts.most_common(label_values_limit))))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if data_dir.exists():
        inspect_local(data_dir, args.split, args.sample_size, args.label_values_limit)
    elif args.dataset_name:
        inspect_hf(args.dataset_name, args.split, args.sample_size, args.label_values_limit)
    else:
        raise SystemExit(
            f"Local data directory does not exist: {data_dir}. "
            "Pass --dataset-name to inspect Hugging Face instead."
        )


if __name__ == "__main__":
    main()
