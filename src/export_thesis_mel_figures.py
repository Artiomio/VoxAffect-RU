"""Generate thesis-ready log-mel figures from evaluated audio examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DPI = 300
VMIN_DB = -80.0
VMAX_DB = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cnn-predictions",
        default="artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/predictions.csv",
        help="Predictions CSV for the best CNN run by macro F1.",
    )
    parser.add_argument(
        "--svm-predictions",
        default="artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv",
        help="Predictions CSV for the tuned SVM run.",
    )
    parser.add_argument("--output-dir", default="artifacts/thesis_mel_figures")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-mels", type=int, default=64)
    return parser.parse_args()


def load_predictions(cnn_path: Path, svm_path: Path) -> pd.DataFrame:
    cnn = pd.read_csv(cnn_path)
    svm = pd.read_csv(svm_path)
    required = {"audio_path", "source_label", "target_label", "predicted_label", "positive_probability"}
    for path, frame in ((cnn_path, cnn), (svm_path, svm)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise SystemExit(f"{path} is missing required columns: {missing}")

    cnn = cnn.rename(
        columns={
            "predicted_label": "cnn_predicted_label",
            "positive_probability": "cnn_positive_probability",
            "confidence": "cnn_confidence",
        }
    )
    svm = svm[
        ["audio_path", "predicted_label", "positive_probability", "confidence"]
    ].rename(
        columns={
            "predicted_label": "svm_predicted_label",
            "positive_probability": "svm_positive_probability",
            "confidence": "svm_confidence",
        }
    )
    merged = cnn.merge(svm, on="audio_path", how="inner")
    merged["cnn_correct"] = merged["cnn_predicted_label"] == merged["target_label"]
    merged["svm_correct"] = merged["svm_predicted_label"] == merged["target_label"]
    merged["model_probability_gap"] = (
        merged["cnn_positive_probability"] - merged["svm_positive_probability"]
    ).abs()
    return merged


def choose_first(frame: pd.DataFrame, description: str, sort_columns: list[str]) -> pd.Series:
    if frame.empty:
        raise SystemExit(f"No available example for: {description}")
    return frame.sort_values(sort_columns, ascending=False).iloc[0]


def select_examples(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    correct = predictions[predictions["cnn_correct"]]
    positive = correct[correct["source_label"] == "positive"]
    negative = correct[correct["target_label"] == 0]
    sad = correct[correct["source_label"] == "sad"]
    angry = correct[correct["source_label"] == "angry"]

    examples = {
        "input_example": choose_first(positive, "input example", ["cnn_confidence"]),
        "class_positive": choose_first(positive, "positive class", ["cnn_confidence"]),
        "class_negative": choose_first(negative, "negative class", ["cnn_confidence"]),
        "emotion_positive": choose_first(positive, "positive emotion", ["cnn_confidence"]),
        "emotion_sad": choose_first(sad, "sad emotion", ["cnn_confidence"]),
        "emotion_angry": choose_first(angry, "angry emotion", ["cnn_confidence"]),
    }

    categories = {
        "both_correct": predictions[predictions["cnn_correct"] & predictions["svm_correct"]],
        "svm_only_correct": predictions[~predictions["cnn_correct"] & predictions["svm_correct"]],
        "cnn_only_correct": predictions[predictions["cnn_correct"] & ~predictions["svm_correct"]],
        "both_wrong": predictions[~predictions["cnn_correct"] & ~predictions["svm_correct"]],
    }
    for name, frame in categories.items():
        examples[f"error_{name}"] = choose_first(
            frame,
            name,
            ["model_probability_gap", "cnn_confidence"],
        )
    return examples


def extract_log_mel(path: Path, sample_rate: int, duration: float, n_mels: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, duration=duration)
    target_length = int(sample_rate * duration)
    if audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size))
    else:
        audio = audio[:target_length]
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def plot_panel(
    axis: plt.Axes,
    row: pd.Series,
    title: str,
    sample_rate: int,
    duration: float,
    n_mels: int,
) -> object:
    log_mel = extract_log_mel(Path(row["audio_path"]), sample_rate, duration, n_mels)
    image = librosa.display.specshow(
        log_mel,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axis,
        cmap="magma",
        vmin=VMIN_DB,
        vmax=VMAX_DB,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Время, с")
    axis.set_ylabel("Частота, Гц (mel)")
    return image


def save_input_example(output_dir: Path, row: pd.Series, args: argparse.Namespace) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    image = plot_panel(
        axis,
        row,
        "Пример входа CNN: log-mel спектрограмма",
        args.sample_rate,
        args.duration,
        args.n_mels,
    )
    fig.colorbar(image, ax=axis, format="%+2.0f dB", label="Относительная энергия, dB")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_01_logmel_input_example.png", dpi=FIGURE_DPI)
    plt.close(fig)


def save_class_examples(output_dir: Path, examples: dict[str, pd.Series], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    panels = [
        ("class_positive", "Положительный класс (positive)"),
        ("class_negative", "Неположительный класс (negative)"),
    ]
    image = None
    for axis, (key, title) in zip(axes, panels):
        image = plot_panel(axis, examples[key], title, args.sample_rate, args.duration, args.n_mels)
    fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Относительная энергия, dB")
    fig.suptitle("Примеры целевых классов бинарной классификации", fontsize=12)
    fig.savefig(output_dir / "figure_02_class_examples_positive_vs_nonpositive.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_emotion_examples(output_dir: Path, examples: dict[str, pd.Series], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), sharey=True)
    panels = [
        ("emotion_positive", "positive (target=1)"),
        ("emotion_sad", "sad (target=0)"),
        ("emotion_angry", "angry (target=0)"),
    ]
    image = None
    for axis, (key, title) in zip(axes, panels):
        image = plot_panel(axis, examples[key], title, args.sample_rate, args.duration, args.n_mels)
    fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Относительная энергия, dB")
    fig.suptitle("Исходные эмоции внутри бинарной постановки", fontsize=12)
    fig.savefig(output_dir / "figure_03_emotion_examples_positive_sad_angry.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def error_title(label: str, row: pd.Series) -> str:
    return (
        f"{label}\n"
        f"истина={int(row['target_label'])}; CNN={int(row['cnn_predicted_label'])}; "
        f"SVM={int(row['svm_predicted_label'])}\n"
        f"P(+): CNN={float(row['cnn_positive_probability']):.2f}; "
        f"SVM={float(row['svm_positive_probability']):.2f}"
    )


def save_error_examples(output_dir: Path, examples: dict[str, pd.Series], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharey=True)
    panels = [
        ("error_both_correct", "Обе модели верны"),
        ("error_svm_only_correct", "Верна только SVM"),
        ("error_cnn_only_correct", "Верна только CNN"),
        ("error_both_wrong", "Обе модели ошиблись"),
    ]
    image = None
    for axis, (key, title) in zip(axes.flat, panels):
        image = plot_panel(
            axis,
            examples[key],
            error_title(title, examples[key]),
            args.sample_rate,
            args.duration,
            args.n_mels,
        )
    fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Относительная энергия, dB")
    fig.suptitle("Анализ согласия и ошибок CNN и SVM", fontsize=12)
    fig.savefig(output_dir / "figure_04_error_analysis_best_cnn_vs_svm.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_provenance(output_dir: Path, examples: dict[str, pd.Series], args: argparse.Namespace) -> None:
    figure_keys = {
        "figure_01_logmel_input_example.png": ["input_example"],
        "figure_02_class_examples_positive_vs_nonpositive.png": ["class_positive", "class_negative"],
        "figure_03_emotion_examples_positive_sad_angry.png": [
            "emotion_positive",
            "emotion_sad",
            "emotion_angry",
        ],
        "figure_04_error_analysis_best_cnn_vs_svm.png": [
            "error_both_correct",
            "error_svm_only_correct",
            "error_cnn_only_correct",
            "error_both_wrong",
        ],
    }
    rows = []
    for figure, keys in figure_keys.items():
        for panel, key in enumerate(keys, start=1):
            row = examples[key]
            rows.append(
                {
                    "figure": figure,
                    "panel": panel,
                    "role": key,
                    "audio_path": row["audio_path"],
                    "source_label": row["source_label"],
                    "target_label": int(row["target_label"]),
                    "cnn_predicted_label": int(row["cnn_predicted_label"]),
                    "cnn_positive_probability": float(row["cnn_positive_probability"]),
                    "svm_predicted_label": int(row["svm_predicted_label"]),
                    "svm_positive_probability": float(row["svm_positive_probability"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "provenance.csv", index=False)
    readme = f"""# Thesis Mel Figures

Figures generated from the external test subset using the best CNN macro-F1 run
and the tuned SVM baseline.

Parameters:

```text
sample_rate: {args.sample_rate}
duration: {args.duration}
n_mels: {args.n_mels}
color range: {VMIN_DB:.0f} to {VMAX_DB:.0f} dB relative to each clip maximum
```

Files:

```text
figure_01_logmel_input_example.png
figure_02_class_examples_positive_vs_nonpositive.png
figure_03_emotion_examples_positive_sad_angry.png
figure_04_error_analysis_best_cnn_vs_svm.png
provenance.csv
```

The images are intended for methodology, dataset description, and model error
analysis sections. `provenance.csv` records the source and model outputs for
every displayed panel.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(Path(args.cnn_predictions), Path(args.svm_predictions))
    examples = select_examples(predictions)
    save_input_example(output_dir, examples["input_example"], args)
    save_class_examples(output_dir, examples, args)
    save_emotion_examples(output_dir, examples, args)
    save_error_examples(output_dir, examples, args)
    save_provenance(output_dir, examples, args)
    print(f"output_dir: {output_dir}")
    print(f"figures: 4")
    print(f"provenance: {output_dir / 'provenance.csv'}")


if __name__ == "__main__":
    main()
