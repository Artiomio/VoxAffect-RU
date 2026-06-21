"""Build a reusable log-mel tensor cache from a processed subset CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import extract_log_mel
from .train_sklearn import load_subset


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
        help="Output .npz log-mel cache path.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument(
        "--n-fft",
        type=int,
        default=None,
        help="FFT window size for librosa.feature.melspectrogram. Omit to use librosa default.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="Hop length for librosa.feature.melspectrogram. Omit to use librosa default.",
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=None,
        help="Window length for librosa.feature.melspectrogram. Omit to use librosa default.",
    )
    return parser.parse_args()


def save_logmel_cache(
    output_path: str | Path,
    tensors: np.ndarray,
    labels: np.ndarray,
    rows: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=tensors,
        y=labels,
        rows_json=rows.to_json(orient="records", force_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    return path


def main() -> None:
    args = parse_args()
    subset_path = Path(args.subset_path)
    df = load_subset(subset_path)

    tensors = []
    labels = []
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting log-mel"):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            print(f"Skipping missing audio: {audio_path}")
            continue
        try:
            tensor = extract_log_mel(
                audio_path,
                sample_rate=args.sample_rate,
                duration=args.duration,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
            )
        except Exception as exc:  # noqa: BLE001 - keep a baseline run moving.
            print(f"Skipping unreadable audio {audio_path}: {exc}")
            continue
        tensors.append(tensor)
        labels.append(row["target_label"])
        rows.append(row)

    if not tensors:
        raise SystemExit("No usable audio rows found.")

    tensor_array = np.stack(tensors).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    used_rows = pd.DataFrame(rows).reset_index(drop=True)
    output_path = save_logmel_cache(
        args.output,
        tensors=tensor_array,
        labels=label_array,
        rows=used_rows,
        metadata={
            "subset_path": str(subset_path),
            "rows_total": len(df),
            "rows_used": len(used_rows),
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "shape": list(tensor_array.shape),
        },
    )

    print(f"subset_path: {subset_path}")
    print(f"output: {output_path}")
    print(f"rows_used: {len(used_rows)}")
    print(f"shape: {tensor_array.shape}")


if __name__ == "__main__":
    main()
