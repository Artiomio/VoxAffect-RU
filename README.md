# VoxAffect-RU

Russian speech emotion classification experiments on the DUSHA dataset.

This repository is a compact, reproducible MVP for audio emotion recognition:
dataset inspection, balanced subset construction, CPU-friendly sklearn
baselines, log-mel CNN training, evaluation artifacts, and a local microphone
demo.

The project originated in Diana Peisakhovskaya's thesis work and in ideas
developed together with Artem Peisakhovsky. Diana's contribution is foundational:
she provided the original thesis context and motivation, invested substantial
time in the 2021 data-collection experiments, built part of the early
YouTube-derived dataset, and later participated in human review and
interpretation. Artem led much of the ML engineering, experimentation,
reproducibility work, and the current codebase. The repository is published with
Diana's collaboration and consent.

## Project History and Collaboration

### 2021: custom weakly supervised YouTube dataset

The first version of the project did not start from a ready-made emotional
speech dataset. We experimented with building our own dataset from spoken
YouTube material.

The approximate pipeline was:

```text
YouTube videos -> audio extraction / segmentation -> STT -> text sentiment -> weak label for the corresponding audio fragment
```

This was a weak-supervision / proxy-labeling approach: the emotional polarity of
the recognized text was used as an indirect label for the corresponding audio.
The labels were therefore noisy by design. Text sentiment is not the same thing
as acoustic emotion or stress, STT introduces its own errors, and prosody can
disagree with the literal text.

Despite those limitations, the pipeline was a practical way to explore how a
larger Russian speech dataset could be assembled without manually labeling every
fragment. Diana spent a substantial amount of time on this stage: processing
YouTube material, working through the collection pipeline, and building a real
part of the early dataset.

Some notebooks and experimental materials from this stage have been preserved
outside the current repository and may be added later under `legacy/` as
historical material.

### 2025: reproducible thesis version on DUSHA

Several years later, when the project was revived for the thesis, we deliberately
chose a simpler and more reliable path instead of rebuilding the fragile YouTube
collection pipeline under deadline pressure.

The current project therefore uses DUSHA, a ready-made Russian emotional speech
dataset with existing labels, and focuses on a reproducible ML pipeline:

```text
DUSHA audio -> label mapping -> balanced subsets -> features/log-mels -> models -> metrics -> validation samples
```

The goal was not to erase the earlier work. It was to turn the research idea
into a version that could be completed, reproduced, evaluated, and explained
clearly.

## Highlights

```text
Best external result, binary valence task:
accuracy: 0.8475
macro F1: 0.8475

Task:
0 = negative = angry + sad
1 = positive
neutral and other labels excluded

Model:
CompactLogMelCNN
6s audio, 16 kHz
80 log-mel bands
n_fft=1024, hop_length=256
CNN channels 32,64,128
adaptive pooling 6x6
classifier head 4608 -> 512 -> 2
```

The best validation check for the same checkpoint reproduced `0.8572` accuracy
and macro F1 on the internal validation split. The external test score is the
number to quote for generalization.

## Why This Project Exists

The goal is not to claim a state-of-the-art speech emotion model. The goal is a
clear engineering pipeline for an applied thesis project:

```text
DUSHA audio -> label mapping -> balanced subsets -> features/log-mels -> models -> metrics -> validation samples
```

The project deliberately starts with simple, inspectable baselines before moving
to a compact CNN. This makes the results easier to reproduce and explain.

## Dataset

The experiments use DUSHA emotional speech data:

```text
KELONMYOSA/dusha_emotion_audio
```

Local data is not committed. Put the extracted dataset under:

```text
data/dusha_emotion_audio/data
```

The repository keeps only code, documentation, and reproducibility notes. Large
generated files are ignored:

```text
data/
artifacts/
.venv/
```

Important experiment results are summarized in `EXPERIMENT_LOG.md`.

## Supported Tasks

### Binary Valence

The main historical benchmark in this repo:

```text
0 = negative = angry + sad
1 = positive
excluded = neutral, other
```

Best external result:

```text
run: cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu
accuracy: 0.8475
macro F1: 0.8475
eval rows: 4800, balanced 2400/class
```

### Neutral vs Emotional

An arousal/proxy experiment:

```text
0 = neutral
1 = emotional = positive + angry + sad
excluded = other
```

Results on 1000/class train and 1000/class external eval:

```text
SVM RBF macro F1: 0.6619
CNN log-mel macro F1: 0.6938
```

### Neutral vs Active Emotional

A stricter arousal/proxy experiment that excludes sadness:

```text
0 = neutral
1 = active_emotional = positive + angry
excluded = sad, other
```

Results on 1000/class train and 1000/class external eval:

```text
SVM RBF macro F1: 0.7140
CNN log-mel macro F1: 0.7720
```

This result suggests that `sad` behaves differently from active emotional speech
and blurs the merged emotional class.

## Result Summary

```text
run | model | task | data | accuracy | macro F1
----|-------|------|------|----------|---------
sklearn_svm_rbf_binary_1000_tuned | StandardScaler + RBF SVM | negative vs positive | 1000/class train, 1000/class eval | 0.7200 | 0.7195
cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu | CompactLogMelCNN | negative vs positive | 1000/class train, 1000/class eval | 0.7575 | 0.7575
cnn_logmel_binary_9000_6s80mels_wide_pool4_1200ep_gpu | CompactLogMelCNN | negative vs positive | 9000/class train, 2400/class eval | 0.8338 | 0.8337
cnn_logmel_binary_9000_6s80mels_wide_pool4_hidden512_1200ep_gpu | CompactLogMelCNN | negative vs positive | 9000/class train, 2400/class eval | 0.8471 | 0.8470
cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu | CompactLogMelCNN | negative vs positive | 9000/class train, 2400/class eval | 0.8475 | 0.8475
sklearn_svm_rbf_neutral_vs_active_emotional_1000_tuned | StandardScaler + RBF SVM | neutral vs positive+angry | 1000/class train, 1000/class eval | 0.7145 | 0.7140
cnn_logmel_neutral_vs_active_emotional_1000_6s80mels_fft1024_hop256_pool6_hidden512_300ep_gpu | CompactLogMelCNN | neutral vs positive+angry | 1000/class train, 1000/class eval | 0.7720 | 0.7720
```

## Repository Layout

```text
src/
  inspect_dataset.py          Inspect local DUSHA metadata and label columns.
  make_subset.py              Build balanced CSV subsets for supported tasks.
  build_feature_cache.py      Extract compact sklearn features.
  build_logmel_cache.py       Build fixed-size log-mel .npz caches.
  train_sklearn.py            Train simple sklearn baselines.
  tune_sklearn.py             Tune sklearn hyperparameters and thresholds.
  train_cnn_logmel.py         Train compact PyTorch CNNs on log-mel caches.
  train_mlp_logmel.py         Train flattened log-mel MLP baselines.
  compare_predictions.py      Compare prediction CSVs between runs.
  make_validation_sample.py   Sample examples for human review.
  export_validation_sample.py Export audio snippets for review.
  web_demo.py                 FastAPI backend for local microphone demo.

web/
  index.html
  styles.css
  app.js

legacy/
  diana_legacy.json.ipynb     Original prototype notebook.

EXPERIMENT_LOG.md             Canonical experiment journal.
PROJECT_PLAN.md               Project plan and scope.
SETUP_NOTES.md                Environment setup notes.
SOURCES.md                    External references.
```

## Setup

Create a virtual environment and install the CPU-friendly dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch separately for the target machine. For CUDA, use the official
PyTorch selector and match the command to your driver. The local GPU experiments
were run with a CUDA-enabled PyTorch build on an NVIDIA GeForce RTX 3060.

## Reproduce The Main CNN Result

Assuming DUSHA is available at `data/dusha_emotion_audio/data`:

```bash
.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task binary \
  --samples-per-class 9000 \
  --seed 42 \
  --output data/processed/subset_binary_train_9000.csv

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --task binary \
  --samples-per-class 2400 \
  --seed 42 \
  --output data/processed/subset_binary_test_2400.csv

.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_train_9000.csv \
  --output data/features/subset_binary_train_9000_logmel_6s_80mels_fft1024_hop256.npz \
  --sample-rate 16000 \
  --duration 6.0 \
  --n-mels 80 \
  --n-fft 1024 \
  --hop-length 256

.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_test_2400.csv \
  --output data/features/subset_binary_test_2400_logmel_6s_80mels_fft1024_hop256.npz \
  --sample-rate 16000 \
  --duration 6.0 \
  --n-mels 80 \
  --n-fft 1024 \
  --hop-length 256

.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_9000_logmel_6s_80mels_fft1024_hop256.npz \
  --eval-features-path data/features/subset_binary_test_2400_logmel_6s_80mels_fft1024_hop256.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu \
  --epochs 1200 \
  --batch-size 128 \
  --channels 32,64,128 \
  --pool-output-size 6 \
  --classifier-hidden-size 512 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 10 \
  --patience 120 \
  --device cuda
```

For CPU-only smoke tests, use smaller subsets and sklearn first.

## Local Demo

The demo records audio in the browser, sends WAV bytes to a local FastAPI
backend, and returns the CNN prediction.

```bash
.venv/bin/python -m uvicorn src.web_demo:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The checkpoint is not committed, so the demo requires the best model artifact to
exist locally at:

```text
artifacts/cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu/model.pt
```

## Human Validation

The current DUSHA-based phase includes utilities for exporting compact review
samples. Diana participates in this human quality-validation step:

```text
20-30 fragments -> ok / wrong / unclear / garbage / good example for thesis
```

This small review sample is only one part of Diana's contribution to the overall
project. Her earlier work included substantial hands-on dataset collection and
curation for the original 2021 YouTube-based experiments described above.

Example:

```bash
.venv/bin/python -m src.make_validation_sample \
  --predictions-path artifacts/sklearn_logreg_binary_300/predictions.csv \
  --output artifacts/validation_sample_logreg_binary_300.csv \
  --n 30 \
  --strategy mixed \
  --seed 42

.venv/bin/python -m src.export_validation_sample \
  --sample-path artifacts/validation_sample_logreg_binary_300.csv \
  --output-dir artifacts/validation_sample_logreg_binary_300_audio \
  --overwrite
```

## Notes For Public Review

- Data and generated artifacts are intentionally excluded from Git.
- Metrics are recorded in `EXPERIMENT_LOG.md` with run names and artifact paths.
- The best quoted number for the public README is external macro F1, not
  internal validation macro F1.
- Transformer models such as Wav2Vec2 or HuBERT are future work, not part of the
  current MVP.
