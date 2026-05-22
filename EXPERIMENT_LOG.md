# Experiment Log

Рабочий журнал проекта Vox Games / DUSHA Emotion MVP.

Цель журнала — фиксировать ход экспериментов так, чтобы из него можно было
потом собрать фрагменты дипломной работы: описание данных, методики,
экспериментальных условий, результатов и ограничений.

## Как вести записи

Для каждого существенного шага фиксируем:

- дату;
- цель;
- входные данные;
- команды или скрипты;
- параметры;
- результат;
- выводы;
- ограничения;
- следующий шаг.

Generated files в `data/` и `artifacts/` локальные и не коммитятся. В журнале
фиксируются пути, параметры и численные результаты.

Это правило также вынесено в `AGENTS.md`, чтобы Codex видел его как проектную
директиву при следующих запусках.

Для экспериментов с метриками каждый подход должен иметь отдельный `run-name` и
отдельный подкаталог в `artifacts/`. Результаты разных подходов не должны
перетирать друг друга. В журнале для серии экспериментов нужно добавлять
таблицу сравнения:

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
```

## 2026-05-22 — Инициализация проекта

### Цель

Подготовить минимальный воспроизводимый проект для дипломного MVP по анализу
эмоциональной окраски русской речи.

### Входные материалы

- Датасет DUSHA: `KELONMYOSA/dusha_emotion_audio`.
- Legacy notebook Дианы: `legacy/diana_legacy.json.ipynb`.
- План проекта: `PROJECT_PLAN.md`.

### Решения

- Не начинать с Wav2Vec2/HuBERT fine-tuning.
- Сначала построить CPU-friendly baseline.
- Использовать DUSHA labels напрямую, без weak labeling через Whisper или text sentiment.
- Сохранять большие локальные данные в `data/`, а generated results в `artifacts/`.
- Не коммитить `.venv/`, `data/`, generated `artifacts/`.

### Git

Создан репозиторий, ветка `main`.

Коммиты:

```text
882a158 Initial DUSHA emotion baseline project
```

## 2026-05-22 — Загрузка и распаковка DUSHA

### Цель

Получить локальную копию DUSHA, пригодную для быстрого эксперимента без
повторного скачивания из Hugging Face.

### Команды

```bash
.venv/bin/hf download KELONMYOSA/dusha_emotion_audio \
  --repo-type dataset \
  --local-dir data/dusha_emotion_audio

tar -xzf data/dusha_emotion_audio/data/train.tar.gz \
  -C data/dusha_emotion_audio/data

tar -xzf data/dusha_emotion_audio/data/test.tar.gz \
  -C data/dusha_emotion_audio/data
```

### Результат

Локальная структура:

```text
data/dusha_emotion_audio/data/train.csv
data/dusha_emotion_audio/data/test.csv
data/dusha_emotion_audio/data/train/
data/dusha_emotion_audio/data/test/
data/dusha_emotion_audio/data/train.tar.gz
data/dusha_emotion_audio/data/test.tar.gz
```

Количество аудиофайлов после распаковки:

```text
train: 96,680 wav-файлов
test:  24,171 wav-файлов
```

Полный размер локальной папки датасета после распаковки:

```text
data/dusha_emotion_audio: 28G
```

### Ограничения

Архивы оставлены на месте. Это увеличивает локальный размер, но даёт быстрый
fallback на случай повреждения распакованных файлов.

## 2026-05-22 — Inspection датасета

### Цель

Проверить реальную структуру DUSHA перед обучением модели.

### Скрипт

```text
src/inspect_dataset.py
```

### Команды

```bash
.venv/bin/python -m src.inspect_dataset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --sample-size 20

.venv/bin/python -m src.inspect_dataset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --sample-size 20
```

### Результат

Доступные split'ы:

```text
train
test
```

Колонки:

```text
file_name
label
```

Распределение labels в `train`:

```text
neutral:  63,701
angry:    12,298
sad:      10,256
positive:  9,541
other:       884
```

Распределение labels в `test`:

```text
neutral:  15,886
angry:     3,072
sad:       2,506
positive:  2,481
other:       226
```

Проверка путей:

```text
20/20 sample train audio paths exist
20/20 sample test audio paths exist
```

Оценка длительностей на 20 train examples:

```text
min: 2.390 sec
p50: 4.280 sec
p90: 5.280 sec
max: 5.660 sec
```

Оценка длительностей на 20 test examples:

```text
min: 0.799 sec
p50: 4.040 sec
p90: 5.920 sec
max: 6.380 sec
```

### Вывод

DUSHA уже содержит готовые короткие аудиофрагменты и emotion labels. Для первого
MVP можно не делать нарезку длинного аудио и не применять weak labeling.

### Git

```text
882a158 Initial DUSHA emotion baseline project
```

## 2026-05-22 — Balanced subset

### Цель

Собрать маленький сбалансированный subset для быстрого baseline.

### Скрипт

```text
src/make_subset.py
```

### Binary mapping

```text
positive -> 1
angry    -> 0
sad      -> 0
neutral  -> skip
other    -> skip
```

### 3-class mapping

```text
positive -> positive
angry    -> negative
sad      -> negative
neutral  -> neutral
other    -> skip
```

### Команды

```bash
.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task binary \
  --samples-per-class 300 \
  --output data/processed/subset_binary.csv

.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task 3class \
  --samples-per-class 300 \
  --output data/processed/subset_3class.csv
```

### Результат

Binary subset:

```text
target_label 0: 300
target_label 1: 300
total:          600
```

3-class subset:

```text
negative: 300
neutral:  300
positive: 300
total:    900
```

### Вывод

Binary subset подходит для первого sanity-check baseline. Нейтральные и `other`
примеры на первом этапе исключены, чтобы задача была проще и интерпретируемее.

### Git

```text
da25f6a Add DUSHA subset builder
```

## 2026-05-22 — Sklearn audio baseline

### Цель

Получить первые осмысленные метрики без PyTorch/CUDA на маленьком binary subset.

### Скрипты

```text
src/features.py
src/train_sklearn.py
```

### Данные

```text
subset: data/processed/subset_binary.csv
rows_total: 600
rows_used: 600
train rows: 480
validation rows: 120
```

### Признаки

Извлечено 48 числовых признаков:

```text
MFCC mean/std, n_mfcc=20
RMS mean/std
spectral centroid mean/std
spectral bandwidth mean/std
zero crossing rate mean/std
```

Аудио загружалось в mono при `sample_rate=16000`, `max_duration=6.0 sec`.

### Модель

```text
StandardScaler + LogisticRegression
class_weight=balanced
test_size=0.2
seed=42
```

### Команда

```bash
.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --test-size 0.2 \
  --seed 42
```

### Результаты

Validation set:

```text
accuracy:        0.7333
precision_macro: 0.7357
recall_macro:    0.7333
f1_macro:        0.7327
```

Classification report:

```text
              precision    recall  f1-score   support

           0     0.7121    0.7833    0.7460        60
           1     0.7593    0.6833    0.7193        60

    accuracy                         0.7333       120
   macro avg     0.7357    0.7333    0.7327       120
weighted avg     0.7357    0.7333    0.7327       120
```

Generated artifacts:

```text
artifacts/metrics.json
artifacts/classification_report.txt
artifacts/confusion_matrix.png
artifacts/predictions.csv
```

### Предварительный вывод

Даже простой baseline на hand-crafted audio features показывает качество выше
случайного угадывания на сбалансированной бинарной задаче. Это хороший sanity
check: labels, пути к аудио, feature extraction и train/eval loop работают.

### Ограничения

- Используется маленький subset: 600 examples.
- Validation split получен из того же train split DUSHA.
- Пока нет проверки на официальном `test.csv`.
- Нет speaker/source-aware split, поэтому возможна утечка похожих голосов между train и validation.
- Нейтральные и `other` примеры не участвуют в binary baseline.

### Git

```text
9c16e46 Add sklearn audio baseline
```

## 2026-05-22 — Сравнение sklearn-подходов

### Цель

Проверить, даёт ли более сложный классический алгоритм прирост качества по
сравнению с линейной логистической регрессией.

### Организация артефактов

Чтобы разные подходы не перетирали результаты друг друга, `src/train_sklearn.py`
получил параметры:

```text
--model
--run-name
```

Каждый запуск теперь можно сохранять в отдельный подкаталог:

```text
artifacts/sklearn_logreg_binary_300/
artifacts/sklearn_random_forest_binary_300/
artifacts/sklearn_svm_rbf_binary_300/
```

В каждом run directory сохраняются:

```text
metrics.json
classification_report.txt
confusion_matrix.png
predictions.csv
```

### Данные и признаки

Во всех запусках использовался один и тот же subset и один и тот же feature
extractor:

```text
subset: data/processed/subset_binary.csv
rows_total: 600
train rows: 480
validation rows: 120
features: 48 hand-crafted audio features
sample_rate: 16000
max_duration: 6.0 sec
test_size: 0.2
seed: 42
```

### Команды

```bash
.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_300 \
  --model logreg \
  --test-size 0.2 \
  --seed 42

.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_random_forest_binary_300 \
  --model random_forest \
  --n-estimators 300 \
  --test-size 0.2 \
  --seed 42

.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_svm_rbf_binary_300 \
  --model svm_rbf \
  --test-size 0.2 \
  --seed 42
```

### Результаты

```text
run                              model                              accuracy  macro F1
sklearn_logreg_binary_300         StandardScaler + LogisticRegression  0.7333    0.7327
sklearn_random_forest_binary_300  RandomForestClassifier               0.6667    0.6652
sklearn_svm_rbf_binary_300        StandardScaler + SVC(kernel='rbf')   0.7333    0.7315
```

### Вывод

На текущем маленьком binary subset более сложные алгоритмы не дали улучшения.
Random Forest заметно хуже линейной логистической регрессии. RBF-SVM совпал по
accuracy, но немного уступил по macro F1.

На этом этапе логистическая регрессия остаётся основным sklearn baseline:
она проще, быстрее интерпретируется и показывает не худшее качество среди
проверенных классических моделей.

### Ограничения

- Все сравнения выполнены на одном маленьком subset `300/300`.
- Feature extraction каждый раз повторяется; позже можно добавить feature cache.
- Не проводился подбор гиперпараметров.
- Пока нет оценки на отдельном DUSHA `test` split.

### Следующий шаг

Сформировать human validation sample из predictions лучшего/основного baseline:

```text
artifacts/sklearn_logreg_binary_300/predictions.csv
```

## 2026-05-22 — Human validation sample

### Цель

Сформировать небольшую выборку предсказаний для ручной проверки Дианой. Это
нужно не для полной переразметки датасета, а для экспертной проверки качества
модели и поиска показательных примеров для диплома.

### Скрипт

```text
src/make_validation_sample.py
```

### Входные данные

```text
predictions: artifacts/sklearn_logreg_binary_300/predictions.csv
input rows: 120
model: StandardScaler + LogisticRegression
task: binary
```

### Команда

```bash
.venv/bin/python -m src.make_validation_sample \
  --predictions-path artifacts/sklearn_logreg_binary_300/predictions.csv \
  --output artifacts/validation_sample_logreg_binary_300.csv \
  --n 30 \
  --strategy mixed \
  --seed 42
```

### Стратегия выборки

Использована стратегия `mixed`: выбрать смесь правильных и ошибочных
предсказаний по разным true labels. Это полезнее для ручной проверки, чем
случайная выборка только из уверенных правильных ответов.

### Результат

```text
output: artifacts/validation_sample_logreg_binary_300.csv
output rows: 30
correct predictions: 16
wrong predictions: 14
target_label 0: 16
target_label 1: 14
```

Поля CSV:

```text
audio_path
source_label
target_label
predicted_label
confidence
is_correct
human_label
human_comment
review_status
notes
```

### Инструкция для ручной проверки

Диана слушает `audio_path` и заполняет:

```text
human_label
human_comment
review_status
notes
```

Рекомендуемые значения `human_comment`:

```text
ok
wrong
unclear
garbage
good example for thesis
```

### Вывод

Подготовлен первый human validation artifact. Его можно использовать как
материал для раздела диплома об экспертной проверке и как список примеров для
качественного анализа ошибок.

### Ограничения

- Выборка сделана из validation split маленького binary subset.
- Объём 30 examples подходит для первичной проверки, но не для статистически
  строгой экспертной оценки.
- Проверка пока не выполнена человеком; поля `human_*` пустые.

## 2026-05-22 — Export validation sample for Diana

### Цель

Сделать human validation sample удобным для ручного прослушивания: не заставлять
Диану искать длинные исходные пути в `data/dusha_emotion_audio`, а дать одну
папку с WAV-файлами и таблицей для заполнения.

### Скрипт

```text
src/export_validation_sample.py
```

### Входные данные

```text
sample: artifacts/validation_sample_logreg_binary_300.csv
rows: 30
```

### Команда

```bash
.venv/bin/python -m src.export_validation_sample \
  --sample-path artifacts/validation_sample_logreg_binary_300.csv \
  --output-dir artifacts/validation_sample_logreg_binary_300_audio \
  --overwrite
```

### Результат

```text
output_dir: artifacts/validation_sample_logreg_binary_300_audio
copied_audio: 30
missing_audio: 0
size: 4.5M
```

В папке созданы:

```text
review_sheet.csv
README.md
001_ok_true-1_pred-1_src-positive_conf-0.77.wav
002_ok_true-0_pred-0_src-sad_conf-0.62.wav
...
```

`review_sheet.csv` содержит относительное имя аудиофайла и поля для заполнения:

```text
audio_file
source_label
target_label
predicted_label
confidence
is_correct
human_label
human_comment
review_status
notes
original_audio_path
```

### Вывод

Подготовлен компактный пакет для ручной экспертной проверки. Его можно передать
Диане как одну папку: она слушает WAV-файлы и заполняет `review_sheet.csv`.

### Ограничения

- Экспортная папка находится в ignored `artifacts/`, поэтому в git фиксируются
  только скрипт, команда и описание результата.
- После ручной проверки нужно отдельно сохранить агрегированные итоги в журнале.

## 2026-05-22 — External test split evaluation

### Цель

Получить более честную оценку основного sklearn baseline: обучить модель на
balanced subset из DUSHA `train` split и оценить на отдельном balanced subset из
DUSHA `test` split.

Это важнее, чем метрика на internal validation split, потому что internal
validation был получен из того же `train.csv`.

### Изменения в коде

`src/train_sklearn.py` получил параметр:

```text
--eval-subset-path
```

Если он указан, модель обучается на всех строках `--subset-path`, а оценивается
на отдельном eval CSV.

### Подготовка test subset

Команда:

```bash
.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --task binary \
  --samples-per-class 300 \
  --output data/processed/subset_binary_test.csv
```

Результат:

```text
input: data/dusha_emotion_audio/data/test.csv
target_label 0: 300
target_label 1: 300
total: 600
```

### External eval command

```bash
.venv/bin/python -m src.train_sklearn \
  --subset-path data/processed/subset_binary.csv \
  --eval-subset-path data/processed/subset_binary_test.csv \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_300_test_eval \
  --model logreg \
  --seed 42
```

### Данные

```text
train subset: data/processed/subset_binary.csv
train rows: 600
eval subset: data/processed/subset_binary_test.csv
eval rows: 600
features: 48 hand-crafted audio features
sample_rate: 16000
max_duration: 6.0 sec
model: StandardScaler + LogisticRegression
```

### Результаты

```text
accuracy:        0.6967
precision_macro: 0.6967
recall_macro:    0.6967
f1_macro:        0.6966
```

Classification report:

```text
              precision    recall  f1-score   support

           0     0.6928    0.7067    0.6997       300
           1     0.7007    0.6867    0.6936       300

    accuracy                         0.6967       600
   macro avg     0.6967    0.6967    0.6966       600
weighted avg     0.6967    0.6967    0.6966       600
```

Artifacts:

```text
artifacts/sklearn_logreg_binary_300_test_eval/metrics.json
artifacts/sklearn_logreg_binary_300_test_eval/classification_report.txt
artifacts/sklearn_logreg_binary_300_test_eval/confusion_matrix.png
artifacts/sklearn_logreg_binary_300_test_eval/predictions.csv
```

### Вывод

На отдельном DUSHA `test` split качество ниже, чем на internal validation:

```text
internal validation macro F1: 0.7327
external test macro F1:      0.6966
```

Это ожидаемое и важное уточнение: internal validation был менее строгим
измерением. Для диплома external test split следует считать более честной
оценкой текущего sklearn baseline.

### Ограничения

- Используется balanced test subset `300/300`, а не весь test split.
- Нет проверки speaker/source leakage.
- Нет подбора гиперпараметров.
- Feature extraction повторяется при каждом запуске; нужен feature cache для
  больших экспериментов.

## Следующие шаги

1. Передать `artifacts/validation_sample_logreg_binary_300.csv` Диане на ручную проверку.
2. Передать Диане папку `artifacts/validation_sample_logreg_binary_300_audio/`.
3. После проверки внести агрегированные результаты в журнал.
4. Добавить `run_notes.md` или генерировать краткий Markdown-отчёт по запуску.
5. Прогнать baseline на большем subset, например 1000 examples/class.
6. Добавить feature cache, чтобы не пересчитывать librosa features при каждом запуске.
7. После sklearn baseline перейти к compact CNN на log-mel spectrogram.
