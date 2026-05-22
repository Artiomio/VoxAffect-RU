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
PROJECT_PLAN.md
CODEX_PROMPT.md
START_CODEX_PROMPT.txt
```

и начать с `src/inspect_dataset.py`.

## Минимальный стек

Сначала:

```bash
pip install -r requirements.txt
```

PyTorch/torchaudio лучше ставить отдельно по `SETUP_NOTES.md`, особенно если машина с CUDA.

## Первые команды, которые должен дать Codex

```bash
python -m src.inspect_dataset --dataset-name KELONMYOSA/dusha_emotion_audio --split train --sample-size 20
python -m src.make_subset --samples-per-class 200 --task binary
python -m src.train_sklearn --subset-path data/processed/subset_binary.csv
```

## Роль Дианы

Не ручная каторжная разметка, а человеческая проверка качества:

```text
20-30 фрагментов -> ok / wrong / unclear / garbage / good example for thesis
```

Результат:

```text
artifacts/validation_sample.csv
```
