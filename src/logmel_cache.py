"""Helpers for loading log-mel tensor caches."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_logmel_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        raise SystemExit(f"Log-mel cache does not exist: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as data:
        tensors = data["X"]
        labels = data["y"]
        rows = pd.read_json(StringIO(str(data["rows_json"])), orient="records")
        metadata = json.loads(str(data["metadata_json"]))

    return tensors, labels, rows, metadata
