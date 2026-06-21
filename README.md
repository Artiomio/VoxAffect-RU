# Vox Games / DUSHA Emotion MVP

Минимальный проект для дипломного MVP по анализу эмоциональной окраски русской речи.

## Идея

Сначала делаем маленький воспроизводимый baseline:

```text
DUSHA subset -> features -> sklearn/CNN baseline -> metrics -> validation sample
```

Не начинаем с тяжёлого Wav2Vec2/HuBERT fine-tuning. Это можно оставить на второй этап.

## Что уже есть

Legacy notebook Дианы:

```text
legacy/diana_legacy.json.ipynb
```

Он полезен как исходный каркас:

- `librosa`;
- log-mel spectrogram;
- PyTorch Dataset/DataLoader;
- compact CNN;
- train/eval loop.

Но он использует dummy white-noise data и случайные метки, поэтому это smoke test, а не содержательный эксперимент.

## Первый шаг

Попросить Codex прочитать:

```text
AGENTS.md
PROJECT_PLAN.md
CODEX_PROMPT.md
START_CODEX_PROMPT.txt
```

и продолжать работу с обязательным ведением `EXPERIMENT_LOG.md`.

## Минимальный стек

Сначала:

```bash
pip install -r requirements.txt
```

PyTorch/torchaudio лучше ставить отдельно по `SETUP_NOTES.md`, особенно если машина с CUDA.

## Локальное веб-демо с микрофоном

Демо использует браузерный микрофон, локальный FastAPI backend и текущий лучший
CNN checkpoint:

```text
artifacts/cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu/model.pt
```

Запуск:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn src.web_demo:app --host 127.0.0.1 --port 8000
```

Открыть:

```text
http://127.0.0.1:8000
```

На `localhost` браузер разрешает доступ к микрофону без HTTPS. Для удаленного
доступа через интернет потребуется HTTPS.

## Задачи классификации

Основная ранняя задача `binary` измеряет valence внутри эмоциональной речи:

```text
0 = negative = angry + sad
1 = positive
excluded = neutral, other
```

Отдельная задача `neutral_vs_emotional` измеряет наличие эмоциональной окраски
как proxy для эмоциональной напряженности:

```text
0 = neutral
1 = emotional = positive + angry + sad
excluded = other
```

Вариант `neutral_vs_active_emotional` исключает `sad` и проверяет более активную
эмоциональную окраску:

```text
0 = neutral
1 = active_emotional = positive + angry
excluded = sad, other
```

Метрики этих задач нельзя напрямую сравнивать как "лучше/хуже": первая
разделяет направление эмоции, вторая отделяет нейтральную речь от эмоциональной.

## Первые команды, которые должен дать Codex

```bash
python -m src.inspect_dataset --dataset-name KELONMYOSA/dusha_emotion_audio --split train --sample-size 20
python -m src.make_subset --samples-per-class 200 --task binary
python -m src.train_sklearn --subset-path data/processed/subset_binary.csv
```

## Текущий локальный запуск

Если DUSHA уже скачан и распакован в `data/dusha_emotion_audio/data`:

```bash
.venv/bin/python -m src.inspect_dataset --data-dir data/dusha_emotion_audio/data --split train --sample-size 20

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task binary \
  --samples-per-class 300 \
  --output data/processed/subset_binary.csv

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --task binary \
  --samples-per-class 300 \
  --output data/processed/subset_binary_test.csv

.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_300 \
  --model logreg \
  --test-size 0.2 \
  --seed 42

.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --eval-subset-path data/processed/subset_binary_test.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_300_test_eval \
  --model logreg \
  --seed 42

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task binary \
  --samples-per-class 1000 \
  --output data/processed/subset_binary_train_1000.csv

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --task binary \
  --samples-per-class 1000 \
  --output data/processed/subset_binary_test_1000.csv

.venv/bin/python -m src.build_feature_cache \
  --subset-path data/processed/subset_binary_train_1000.csv \
  --output data/features/subset_binary_train_1000_features.npz

.venv/bin/python -m src.build_feature_cache \
  --subset-path data/processed/subset_binary_test_1000.csv \
  --output data/features/subset_binary_test_1000_features.npz

.venv/bin/python -m src.train_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_1000_test_eval \
  --model logreg \
  --seed 42

.venv/bin/python -m src.tune_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_svm_rbf_binary_1000_tuned \
  --model svm_rbf \
  --seed 42

.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_train_1000.csv \
  --output data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --sample-rate 16000 \
  --duration 3.0 \
  --n-mels 64

.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_test_1000.csv \
  --output data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --sample-rate 16000 \
  --duration 3.0 \
  --n-mels 64

.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_60ep_f1_scheduler \
  --epochs 60 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 5 \
  --patience 12 \
  --seed 42

.venv/bin/python -m src.compare_predictions \
  --left artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv \
  --left-name svm \
  --right artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/predictions.csv \
  --right-name cnn60 \
  --output-dir artifacts/compare_svm_tuned_vs_cnn_logmel_60ep_f1_scheduler

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

Первый sklearn baseline на `subset_binary.csv`:

```text
accuracy: 0.7333
macro F1: 0.7327
validation rows: 120
```

Честная оценка на отдельном balanced subset из DUSHA `test` split:

```text
accuracy: 0.6967
macro F1: 0.6966
test rows: 600
```

Larger cached run на `1000/1000` train и `1000/1000` test:

```text
LogisticRegression: accuracy 0.7165, macro F1 0.7164
RBF-SVM:            accuracy 0.7180, macro F1 0.7179
```

Tuned cached runs с подбором hyperparameters и probability threshold на
internal validation split, затем финальной оценкой на отдельном test cache:

```text
LogisticRegression tuned: accuracy 0.7165, macro F1 0.7157
  best validation: C=0.03, class_weight=balanced, threshold=0.52

RBF-SVM tuned:            accuracy 0.7200, macro F1 0.7195
  best validation: C=10.0, gamma=0.003, class_weight=balanced, threshold=0.46
```

Первый compact CNN на log-mel spectrograms:

```text
input: 1 x 64 x 94
architecture: Conv2d 16 -> 32 -> 64, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool
accuracy: 0.6155
macro F1: 0.6107
best threshold: 0.50
comparison vs tuned SVM: both_correct 1003, svm_only_correct 437, cnn_only_correct 228, both_wrong 332
```

CNN с best checkpoint по validation macro F1, `ReduceLROnPlateau` и early
stopping:

```text
run: cnn_logmel_binary_1000_60ep_f1_scheduler
best epoch: 25
epochs completed: 37 / 60
accuracy: 0.6260
macro F1: 0.6253
best threshold: 0.50
comparison vs tuned SVM: both_correct 1025, svm_only_correct 415, cnn60_only_correct 227, both_wrong 333
```

Разные подходы нужно сохранять в отдельные run directories через `--run-name`,
чтобы результаты не перетирали друг друга:

```bash
.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_random_forest_binary_300 \
  --model random_forest \
  --n-estimators 300 \
  --test-size 0.2 \
  --seed 42
```

## Роль Дианы

Не ручная каторжная разметка, а человеческая проверка качества:

```text
20-30 фрагментов -> ok / wrong / unclear / garbage / good example for thesis
```

Результат:

```text
artifacts/validation_sample_logreg_binary_300.csv
```

Поля для ручной проверки:

```text
human_label,human_comment,review_status,notes
```

Для удобной ручной проверки аудио экспортируется в отдельную папку:

```text
artifacts/validation_sample_logreg_binary_300_audio/
```

Внутри лежат 30 `.wav` с читаемыми именами и `review_sheet.csv`.
