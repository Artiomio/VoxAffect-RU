"""Helpers for saving and loading extracted feature matrices."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_feature_cache(
    output_path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    rows: pd.DataFrame,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=features,
        y=labels,
        rows_json=rows.to_json(orient="records", force_ascii=False),
        feature_names_json=json.dumps(feature_names, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    return path


def load_feature_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        raise SystemExit(f"Feature cache does not exist: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as data:
        features = data["X"]
        labels = data["y"]
        rows = pd.read_json(StringIO(str(data["rows_json"])), orient="records")
        feature_names = json.loads(str(data["feature_names_json"]))
        metadata = json.loads(str(data["metadata_json"]))

    return features, labels, feature_names, rows, metadata
