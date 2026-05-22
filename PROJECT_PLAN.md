# Vox Games / DUSHA Emotion Baseline MVP

## Цель проекта

Собрать минимальный рабочий пайплайн для дипломной работы по анализу эмоциональной окраски русской речи.

Главная цель первого этапа — не построить идеальную state-of-the-art модель, а быстро получить воспроизводимый MVP:

```text
audio -> labels -> features -> baseline model -> metrics -> small human validation sample
```

Проект должен быть понятным, запускаемым и пригодным для описания в дипломе.

## Контекст

Есть датасет DUSHA с русской эмоциональной речью и готовыми эмоциональными метками. Поэтому на первом этапе не нужно делать weak labeling через Whisper/sentiment analysis.

Есть legacy notebook Дианы:

```text
/home/art/projects/vox_games/legacy/diana_legacy.json.ipynb
```

В нём уже есть полезный каркас:

- загрузка аудио через `librosa`;
- построение log-mel spectrogram;
- PyTorch `Dataset/DataLoader`;
- compact CNN;
- train/eval loop.

Но текущий notebook использует `dummy_audio`: белый шум и случайные метки. Это не содержательный эксперимент, а smoke test, который проверяет, что код технически запускается.

Наша задача — сохранить полезную идею каркаса, заменить dummy data на реальный subset DUSHA и получить первые осмысленные метрики.

## Основной подход MVP

### 1. Проверить структуру DUSHA

Первым делом не обучаем модель, а смотрим структуру данных:

- какие есть split'ы;
- какие реальные имена колонок;
- как представлен `audio`;
- какие значения у emotion/label;
- есть ли speaker/source/domain;
- какие длительности у записей на небольшой выборке.

Важно: не предполагать имена колонок заранее.

### 2. Собрать маленький balanced subset

Для первого запуска:

- 200-500 примеров на класс;
- ограничение длительности аудио, если нужно;
- привести audio к 16 kHz;
- сохранить metadata в `data/processed/subset_binary.csv` или `data/processed/subset_3class.csv`.

### 3. Mapping меток

Первый вариант — бинарная задача:

```text
positive / happiness -> 1
anger + sadness      -> 0
neutral              -> skip
```

Второй вариант — 3 класса:

```text
positive / happiness -> positive
anger + sadness      -> negative
neutral              -> neutral
```

Если встречаются неизвестные labels — вывести их и пропустить, не падать.

### 4. Baseline #0: sklearn, CPU-friendly

Сначала быстрый baseline:

```text
audio -> MFCC/log-mel statistics -> LogisticRegression/SVM/RandomForest
```

Фичи:

- MFCC mean/std;
- RMS mean/std;
- spectral centroid mean/std;
- optional: zero crossing rate, spectral bandwidth.

Метрики:

- accuracy;
- precision;
- recall;
- F1;
- classification_report;
- confusion_matrix.

Это нужно как быстрый sanity check, особенно на CPU-ноуте.

### 5. Baseline #1: compact CNN

После sklearn baseline:

```text
audio -> log-mel spectrogram -> compact CNN
```

Технически:

- fixed duration, например 3 sec;
- 16 kHz;
- log-mel spectrogram;
- per-sample normalization;
- PyTorch Dataset/DataLoader;
- compact CNN.

Для binary classification использовать raw logits + `BCEWithLogitsLoss`, а не `Sigmoid + BCELoss`.

### 6. Human validation sample для Дианы

Диане не нужно вручную размечать весь датасет. Её часть — человеческий контроль качества:

```text
audio_path,true_label,predicted_label,confidence,human_label,human_comment
```

Размер первой выборки: 20-30 фрагментов.

Возможные human comments:

- ok
- wrong
- unclear
- garbage
- good example for thesis

Это позволит сказать в дипломе, что автоматическая/датасетная разметка была проверена на небольшой экспертной выборке.

### 7. Artifacts

Сохранять в `artifacts/`:

```text
metrics.json
classification_report.txt
confusion_matrix.png
validation_sample.csv
run_notes.md
```

## Что не делаем на первом этапе

Не начинаем с Wav2Vec2/HuBERT fine-tuning.

Не делаем full-scale training на всём DUSHA.

Не делаем YouTube scraping.

Не делаем автоматическую нарезку длинных файлов, если DUSHA уже содержит готовые короткие аудиофрагменты.

Не делаем speaker clustering в первом MVP, если это тормозит прогресс.

Не делаем weak labeling через Whisper/sentiment, пока используем DUSHA с готовыми labels.

Все эти вещи можно оставить в future work.

## Wav2Vec2 / HuBERT как future work

Wav2Vec2/HuBERT fine-tuning — красивый и современный вариант, но это второй этап, а не стартовая точка.

Плюсы:

- выглядит солидно;
- можно сравнить с простым baseline;
- хорошо подходит для GPU на `existo`.

Минусы:

- тяжелее по зависимостям;
- дольше настраивать;
- больнее на CPU;
- больше риск зависнуть в инфраструктуре вместо дипломного MVP.

## Legacy notebook от Дианы

Файл:

```text
legacy/diana_legacy.json.ipynb
```

Это не основной рабочий pipeline, а исходный прототип.

Важно зафиксировать это бережно:

- Диана уже сделала хорошую часть работы — каркас;
- каркас не выбрасываем;
- переносим полезные идеи в `src/`;
- заменяем synthetic/dummy data на DUSHA subset;
- добавляем нормальную оценку качества.

## Технологический стек

Минимально:

- Python 3.10+
- datasets
- soundfile
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
- torch
- torchaudio

Опционально позже:

- transformers
- accelerate
- evaluate
- pyannote/speechbrain for speaker embeddings
- faster-whisper for future weak labeling

## Важное по PyTorch

PyTorch и torchaudio лучше ставить отдельно под конкретную машину.

На CPU-ноуте:
- можно начать с CPU-only версии;
- сначала достаточно `inspect_dataset.py` и sklearn baseline.

На `existo` с RTX 3060:
- проверить `nvidia-smi`;
- выбрать официальную команду установки PyTorch под Linux/pip/Python/CUDA;
- проверить:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Предлагаемая структура проекта

```text
.
├── README.md
├── PROJECT_PLAN.md
├── CODEX_PROMPT.md
├── START_CODEX_PROMPT.txt
├── SETUP_NOTES.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── legacy/
│   └── diana_legacy.json.ipynb
├── notebooks/
│   └── 01_explore_dusha.ipynb
├── src/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── make_subset.py
│   ├── features.py
│   ├── dataset.py
│   ├── models.py
│   ├── train_cnn.py
│   ├── train_sklearn.py
│   ├── evaluate.py
│   └── make_validation_sample.py
└── artifacts/
    ├── metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── validation_sample.csv
```

## Первый рабочий результат

Первый результат считается успешным, если есть:

1. скрипт, который показывает структуру DUSHA;
2. скрипт, который собирает маленький balanced subset;
3. скрипт, который обучает простой baseline;
4. `classification_report.txt`;
5. `validation_sample.csv` для ручной проверки Дианой;
6. короткий текст, объясняющий ограничения подхода.

## Черновая формулировка для диплома

На первом этапе был реализован baseline-подход к распознаванию эмоциональной окраски русской речи на основе акустических признаков. Аудиосигнал приводился к единой частоте дискретизации, затем преобразовывался в log-mel spectrogram. Полученные частотно-временные представления использовались для обучения компактной сверточной нейронной сети.

В качестве источника данных использовался датасет DUSHA с готовыми эмоциональными метками. Для минимизации сложности первого эксперимента исходные классы были сведены к бинарной или трехклассовой постановке задачи. Качество оценивалось по accuracy, precision, recall и F1-score. Дополнительно была подготовлена небольшая выборка для ручной проверки адекватности меток и предсказаний.
