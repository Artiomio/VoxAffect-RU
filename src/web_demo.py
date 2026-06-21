"""Local browser microphone demo for the best binary CNN checkpoint."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .features import extract_log_mel

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from .train_cnn_logmel import CompactLogMelCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu"
    / "model.pt"
)

SAMPLE_RATE = 16_000
DURATION_SECONDS = 6.0
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = None
CHANNELS = [32, 64, 128]
CONV_KERNEL_SIZE = (3, 3)
POOL_OUTPUT_SIZE = 6
POOL_KERNEL_SIZE = 2
POOL_STRIDES = [2, 2]
CLASSIFIER_HIDDEN_SIZE = 512
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

LABELS = {
    0: "negative",
    1: "positive",
}


app = FastAPI(title="Vox Games Emotion Demo")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model() -> tuple[CompactLogMelCNN, torch.device]:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = _device()
    model = CompactLogMelCNN(
        channels=CHANNELS,
        input_shape=(N_MELS, 376),
        conv_kernel_size=CONV_KERNEL_SIZE,
        pool_output_size=POOL_OUTPUT_SIZE,
        pool_kernel_size=POOL_KERNEL_SIZE,
        pool_strides=POOL_STRIDES,
        classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE,
        num_classes=2,
        dropout=0.25,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict_wav_bytes(audio_bytes: bytes) -> dict[str, object]:
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio body is too large.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            log_mel = extract_log_mel(
                audio_file.name,
                sample_rate=SAMPLE_RATE,
                duration=DURATION_SECONDS,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
            )
    except Exception as exc:  # noqa: BLE001 - return a clear API error.
        raise HTTPException(status_code=400, detail=f"Could not read audio as WAV: {exc}") from exc

    model, device = load_model()
    features = torch.from_numpy(log_mel[None, None, :, :].astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    positive_probability = float(probabilities[1])
    predicted_index = int(positive_probability >= THRESHOLD)
    confidence = float(max(positive_probability, 1.0 - positive_probability))

    return {
        "label": LABELS[predicted_index],
        "predicted_index": predicted_index,
        "positive_probability": positive_probability,
        "negative_probability": float(probabilities[0]),
        "confidence": confidence,
        "threshold": THRESHOLD,
        "model": CHECKPOINT_PATH.parent.name,
        "input": {
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": DURATION_SECONDS,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    model_ready = CHECKPOINT_PATH.exists()
    return {
        "ok": model_ready,
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(_device()),
    }


@app.post("/api/predict")
async def predict(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if "audio/wav" not in content_type and "audio/x-wav" not in content_type:
        raise HTTPException(status_code=415, detail="Send audio as WAV.")
    return predict_wav_bytes(await request.body())
