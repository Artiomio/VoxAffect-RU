"""Audio feature extraction for CPU-friendly baselines."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 16_000,
    max_duration: float | None = 6.0,
) -> np.ndarray:
    """Load mono audio at a fixed sample rate, optionally truncating duration."""

    duration = max_duration if max_duration and max_duration > 0 else None
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
    return audio.astype(np.float32, copy=False)


def _mean_std(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    if values.ndim == 1:
        values = values.reshape(1, -1)
    means = values.mean(axis=1)
    stds = values.std(axis=1)
    features = []
    names = []
    for index, value in enumerate(means):
        suffix = f"_{index:02d}" if len(means) > 1 else ""
        features.append(float(value))
        names.append(f"{prefix}{suffix}_mean")
    for index, value in enumerate(stds):
        suffix = f"_{index:02d}" if len(stds) > 1 else ""
        features.append(float(value))
        names.append(f"{prefix}{suffix}_std")
    return features, names


def extract_feature_vector(
    audio_path: str | Path,
    sample_rate: int = 16_000,
    max_duration: float | None = 6.0,
    n_mfcc: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """Extract compact summary features from one audio file."""

    audio = load_audio(audio_path, sample_rate=sample_rate, max_duration=max_duration)
    if audio.size == 0:
        raise ValueError(f"Empty audio after loading: {audio_path}")

    feature_values: list[float] = []
    feature_names: list[str] = []

    feature_blocks = [
        _mean_std(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc), "mfcc"),
        _mean_std(librosa.feature.rms(y=audio), "rms"),
        _mean_std(librosa.feature.spectral_centroid(y=audio, sr=sample_rate), "centroid"),
        _mean_std(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate), "bandwidth"),
        _mean_std(librosa.feature.zero_crossing_rate(y=audio), "zcr"),
    ]

    for values, names in feature_blocks:
        feature_values.extend(values)
        feature_names.extend(names)

    return np.asarray(feature_values, dtype=np.float32), feature_names


def extract_log_mel(
    audio_path: str | Path,
    sample_rate: int = 16_000,
    duration: float = 3.0,
    n_mels: int = 64,
) -> np.ndarray:
    """Extract a normalized log-mel spectrogram for later CNN work."""

    audio = load_audio(audio_path, sample_rate=sample_rate, max_duration=duration)
    target_len = int(sample_rate * duration)
    if audio.size < target_len:
        audio = np.pad(audio, (0, target_len - audio.size))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    if std < 1e-6:
        return (log_mel - mean).astype(np.float32)
    return ((log_mel - mean) / std).astype(np.float32)
