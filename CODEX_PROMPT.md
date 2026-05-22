# Codex task: DUSHA Emotion Baseline MVP

You are working in an already initialized Python project.

The project goal is to build a minimal reproducible MVP for a thesis on Russian speech emotion recognition using the DUSHA dataset.

Important: do not over-engineer. First make a small working baseline, not a large research framework.

## Context

There is a legacy notebook by Diana at:

```text
legacy/diana_legacy.json.ipynb
```

or, on the user's machine:

```text
/home/art/projects/vox_games/legacy/diana_legacy.json.ipynb
```

It contains useful ideas:

- librosa audio loading;
- log-mel spectrogram extraction;
- PyTorch Dataset/DataLoader;
- compact CNN;
- train/eval loop.

But it currently trains on dummy white-noise audio with random labels. Treat it as a smoke-test/reference artifact, not a final experiment.

DUSHA already has emotion labels, so do not start with Whisper or text sentiment weak labeling. That can be future work.

## Main goal

Create a small working pipeline:

1. inspect local or Hugging Face DUSHA dataset structure;
2. create a small balanced subset;
3. map labels into either binary or 3-class task;
4. train a simple baseline;
5. save metrics and a small human validation CSV.

## First task

Please inspect the current repository and create/modify the project files needed for the MVP.

Suggested structure:

```text
src/
  __init__.py
  inspect_dataset.py
  make_subset.py
  features.py
  dataset.py
  models.py
  train_cnn.py
  train_sklearn.py
  evaluate.py
  make_validation_sample.py
artifacts/
data/
notebooks/
legacy/
requirements.txt
README.md
```

Do not assume the exact DUSHA column names. Write inspection code first.

## Requirements

Use a minimal stack first:

- datasets
- soundfile
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm

For PyTorch:
- do not blindly install a random CUDA build;
- prepare instructions for CPU and GPU separately;
- use official PyTorch selector logic for CUDA machine.

Avoid transformers/Wav2Vec2 in the first MVP unless the simple baseline is already working.

## Dataset handling

Support two modes:

1. load from Hugging Face dataset name:

```text
KELONMYOSA/dusha_emotion_audio
```

2. load from local cache/path if available.

Write `src/inspect_dataset.py` that prints:

- dataset splits;
- column names;
- first example keys;
- audio field structure;
- possible label/emotion values;
- duration statistics for a small sample if cheap.

Command example:

```bash
python -m src.inspect_dataset --dataset-name KELONMYOSA/dusha_emotion_audio --split train --sample-size 20
```

Make the script defensive:
- if split names differ, print available splits;
- if `emotion` column is not present, search for likely label columns;
- if audio is not decoded, print the raw structure;
- do not download/process the full dataset unless explicitly requested.

## Label mapping

Implement mapping config as simple functions.

Binary task:

```text
positive / happiness -> 1
anger + sadness      -> 0
neutral              -> skip
```

3-class task:

```text
positive / happiness -> positive
anger + sadness      -> negative
neutral              -> neutral
```

Be defensive about label names:
- normalize lowercase;
- strip spaces;
- print unknown labels;
- skip unknown labels instead of crashing.

## Baseline #0: sklearn fallback

Implement `src/train_sklearn.py` first or early.

It should:

- read processed subset metadata;
- extract simple vector features:
  - MFCC means/stds;
  - RMS mean/std;
  - spectral centroid mean/std;
- train LogisticRegression or RandomForest;
- save `artifacts/classification_report.txt`;
- save `artifacts/metrics.json`;
- save `artifacts/confusion_matrix.png`.

This gives us a fast CPU-friendly baseline if CNN training is slow.

## Baseline #1: CNN

Implement:

### `src/features.py`

- load audio at 16 kHz;
- pad/truncate to fixed duration, e.g. 3 seconds;
- compute log-mel spectrogram with librosa;
- normalize per sample.

### `src/dataset.py`

- PyTorch Dataset from metadata DataFrame;
- return `(feature_tensor, label_tensor)`.

### `src/models.py`

- compact CNN for binary classification;
- for binary classification prefer raw logits and `BCEWithLogitsLoss`, not `Sigmoid + BCELoss`.

### `src/train_cnn.py`

- train/val split;
- train loop;
- eval loop;
- save classification report and metrics.

## Human validation sample

Implement `src/make_validation_sample.py`.

Create:

```text
artifacts/validation_sample.csv
```

Columns:

```text
audio_path,true_label,predicted_label,confidence,human_label,human_comment
```

The goal is to give Diana 20-30 examples to inspect manually. Her task is not to label the whole dataset, but to check quality:

- ok
- wrong
- unclear
- garbage
- good example for thesis

## Artifacts

Save outputs to `artifacts/`:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `validation_sample.csv`
- `run_notes.md`

## Constraints

Keep it runnable on CPU for a tiny subset.

Avoid downloading the full dataset unless explicitly requested.

Use command-line arguments where practical:

```bash
python -m src.inspect_dataset
python -m src.make_subset --samples-per-class 300 --task binary
python -m src.train_sklearn --subset-path data/processed/subset_binary.csv
python -m src.train_cnn --subset-path data/processed/subset_binary.csv --epochs 5
python -m src.make_validation_sample --n 30
```

If exact dataset paths or columns are unknown, implement inspection and ask for the printed output rather than guessing.

## Expected first deliverable

A first runnable version that can do:

```bash
python -m src.inspect_dataset
```

and then, after we know the columns:

```bash
python -m src.make_subset --samples-per-class 200 --task binary
python -m src.train_sklearn --subset-path data/processed/subset_binary.csv
```

Do the simplest correct thing first.
