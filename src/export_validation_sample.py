"""Export validation sample audio files and review sheet for human annotation."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "audio_path",
    "source_label",
    "target_label",
    "predicted_label",
    "confidence",
    "is_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-path",
        default="artifacts/validation_sample_logreg_binary_300.csv",
        help="Validation sample CSV produced by src.make_validation_sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/validation_sample_logreg_binary_300_audio",
        help="Directory for copied audio files and review_sheet.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory contents if it already exists.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")


def safe_label(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9а-яё_-]+", "-", text)
    return text.strip("-") or "unknown"


def audio_filename(index: int, row: pd.Series) -> str:
    true_label = safe_label(row["target_label"])
    pred_label = safe_label(row["predicted_label"])
    source_label = safe_label(row.get("source_label", "source"))
    correctness = "ok" if bool(row["is_correct"]) else "err"
    confidence = float(row["confidence"])
    return (
        f"{index:03d}_{correctness}_"
        f"true-{true_label}_pred-{pred_label}_src-{source_label}_conf-{confidence:.2f}.wav"
    )


def write_readme(output_dir: Path, sample_path: Path, row_count: int) -> None:
    readme = f"""# Validation Sample Audio

This folder contains copied audio files for human validation.

Source CSV:

```text
{sample_path}
```

Files:

```text
review_sheet.csv
*.wav
```

Rows: {row_count}

Recommended `human_comment` values:

```text
ok
wrong
unclear
garbage
good example for thesis
```

Suggested workflow:

1. Listen to each `audio_file`.
2. Compare `source_label`, `target_label`, `predicted_label`, and `confidence`.
3. Fill `human_label`, `human_comment`, `review_status`, and `notes`.
4. Keep the edited `review_sheet.csv` with this folder.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def export_validation_sample(args: argparse.Namespace) -> Path:
    sample_path = Path(args.sample_path)
    output_dir = Path(args.output_dir)

    if not sample_path.exists():
        raise SystemExit(f"Validation sample CSV does not exist: {sample_path}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sample_path)
    require_columns(df, sample_path)

    review_rows = []
    missing_audio = []

    for zero_index, row in df.iterrows():
        index = zero_index + 1
        source_path = Path(row["audio_path"])
        if not source_path.exists():
            missing_audio.append(str(source_path))
            continue

        file_name = audio_filename(index, row)
        target_path = output_dir / file_name
        shutil.copy2(source_path, target_path)

        review_rows.append(
            {
                "audio_file": file_name,
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "predicted_label": row["predicted_label"],
                "confidence": row["confidence"],
                "is_correct": row["is_correct"],
                "human_label": row.get("human_label", ""),
                "human_comment": row.get("human_comment", ""),
                "review_status": row.get("review_status", "todo"),
                "notes": row.get("notes", ""),
                "original_audio_path": row["audio_path"],
            }
        )

    if missing_audio:
        print("Missing audio files:")
        for path in missing_audio:
            print(f"- {path}")

    review_df = pd.DataFrame(review_rows)
    review_path = output_dir / "review_sheet.csv"
    review_df.to_csv(review_path, index=False)
    write_readme(output_dir, sample_path, len(review_df))

    print(f"sample_path: {sample_path}")
    print(f"output_dir: {output_dir}")
    print(f"copied_audio: {len(review_df)}")
    print(f"missing_audio: {len(missing_audio)}")
    print(f"review_sheet: {review_path}")
    return output_dir


def main() -> None:
    export_validation_sample(parse_args())


if __name__ == "__main__":
    main()
