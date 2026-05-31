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

## 2026-05-22 — Feature cache and larger binary baseline

### Цель

Ускорить повторные эксперименты и проверить baseline на более крупном subset.
До этого `librosa` features пересчитывались при каждом запуске, что мешает
сравнивать модели на больших выборках.

### Изменения в коде

Добавлены:

```text
src/feature_cache.py
src/build_feature_cache.py
```

`src/train_sklearn.py` получил параметры:

```text
--features-path
--eval-features-path
```

Если переданы `.npz` cache-файлы, обучение и оценка идут без повторного чтения
WAV и без повторного извлечения признаков.

### Larger subsets

Train subset:

```bash
.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split train \
  --task binary \
  --samples-per-class 1000 \
  --output data/processed/subset_binary_train_1000.csv
```

```text
target_label 0: 1000
target_label 1: 1000
total: 2000
```

Test subset:

```bash
.venv/bin/python -m src.make_subset \
  --data-dir data/dusha_emotion_audio/data \
  --split test \
  --task binary \
  --samples-per-class 1000 \
  --output data/processed/subset_binary_test_1000.csv
```

```text
target_label 0: 1000
target_label 1: 1000
total: 2000
```

### Feature cache

```bash
.venv/bin/python -m src.build_feature_cache \
  --subset-path data/processed/subset_binary_train_1000.csv \
  --output data/features/subset_binary_train_1000_features.npz \
  --sample-rate 16000 \
  --max-duration 6.0

.venv/bin/python -m src.build_feature_cache \
  --subset-path data/processed/subset_binary_test_1000.csv \
  --output data/features/subset_binary_test_1000_features.npz \
  --sample-rate 16000 \
  --max-duration 6.0
```

Результат:

```text
data/features/subset_binary_train_1000_features.npz: 2000 rows, 48 features
data/features/subset_binary_test_1000_features.npz:  2000 rows, 48 features
data/features total size: 884K
```

### Cached model runs

```bash
.venv/bin/python -m src.train_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_1000_test_eval \
  --model logreg \
  --seed 42

.venv/bin/python -m src.train_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_svm_rbf_binary_1000_test_eval \
  --model svm_rbf \
  --seed 42
```

### Results

```text
run                                   model                                  accuracy  macro F1
sklearn_logreg_binary_1000_test_eval  StandardScaler + LogisticRegression    0.7165    0.7164
sklearn_svm_rbf_binary_1000_test_eval StandardScaler + SVC(kernel='rbf')     0.7180    0.7179
```

### Вывод

На более крупном и отдельном test subset результат стал стабильнее и выше, чем
у предыдущего `300/300` external test run:

```text
300/300 external test logreg macro F1:   0.6966
1000/1000 external test logreg macro F1: 0.7164
1000/1000 external test SVM macro F1:    0.7179
```

Feature cache сработал: после построения `.npz` новые sklearn-запуски занимают
секунды, а не минуты.

### Ограничения

- Это всё ещё hand-crafted feature baseline, не neural model.
- Hyperparameter tuning не проводился.
- Feature cache сейчас хранит summary features, а не log-mel tensors для CNN.

## 2026-05-22 — Hyperparameter and threshold tuning on cached features

### Цель

Проверить, можно ли улучшить sklearn baseline без перехода к neural model:

- подобрать hyperparameters на internal validation split;
- подобрать probability threshold для positive class;
- финально оценить выбранную конфигурацию один раз на отдельном DUSHA `test`
  cache.

Важно: threshold и hyperparameters не подбирались на test split. Test
использовался только для финальной оценки выбранной конфигурации.

### Изменения в коде

Добавлен:

```text
src/tune_sklearn.py
```

Скрипт:

1. читает `.npz` feature cache;
2. делит training cache на train/validation;
3. перебирает сетку hyperparameters;
4. для каждой конфигурации перебирает probability thresholds;
5. выбирает лучший вариант по validation macro F1;
6. переобучает выбранную модель на всём training cache;
7. оценивает один раз на external test cache.

Artifacts сохраняются в отдельные run directories:

```text
artifacts/sklearn_logreg_binary_1000_tuned/
artifacts/sklearn_svm_rbf_binary_1000_tuned/
```

### Commands

LogisticRegression:

```bash
.venv/bin/python -m src.tune_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_logreg_binary_1000_tuned \
  --model logreg \
  --seed 42
```

RBF-SVM:

```bash
.venv/bin/python -m src.tune_sklearn \
  --features-path data/features/subset_binary_train_1000_features.npz \
  --eval-features-path data/features/subset_binary_test_1000_features.npz \
  --artifacts-dir artifacts \
  --run-name sklearn_svm_rbf_binary_1000_tuned \
  --model svm_rbf \
  --seed 42
```

### Search spaces

LogisticRegression:

```text
C:            0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
class_weight: balanced, None
threshold:    0.30..0.70, step 0.02
```

RBF-SVM:

```text
C:            0.3, 1.0, 3.0, 10.0
gamma:        scale, 0.003, 0.01, 0.03
class_weight: balanced, None
threshold:    0.30..0.70, step 0.02
```

### Results

```text
run                                best validation params                                  validation macro F1  test accuracy  test macro F1
sklearn_logreg_binary_1000_tuned   C=0.03, class_weight=balanced, threshold=0.52            0.7220               0.7165         0.7157
sklearn_svm_rbf_binary_1000_tuned  C=10.0, gamma=0.003, class_weight=balanced, threshold=0.46 0.7249               0.7200         0.7195
```

Baseline comparison:

```text
untuned LogisticRegression macro F1: 0.7164
tuned LogisticRegression macro F1:   0.7157

untuned RBF-SVM macro F1:            0.7179
tuned RBF-SVM macro F1:              0.7195
```

### Вывод

Tuning дал небольшой выигрыш только для RBF-SVM:

```text
macro F1 +0.0016
accuracy +0.0020
```

Для LogisticRegression tuning не улучшил external test score, несмотря на
лучший validation вариант. Это полезный результат: validation selection может
не переноситься на test идеально, особенно при малом feature set и небольшом
dataset subset.

Текущий лучший sklearn baseline:

```text
StandardScaler + SVC(kernel='rbf')
C=10.0
gamma=0.003
class_weight=balanced
positive threshold=0.46
test accuracy=0.7200
test macro F1=0.7195
```

### Artifacts

```text
artifacts/sklearn_logreg_binary_1000_tuned/metrics.json
artifacts/sklearn_logreg_binary_1000_tuned/tuning_results.csv
artifacts/sklearn_logreg_binary_1000_tuned/classification_report.txt
artifacts/sklearn_logreg_binary_1000_tuned/confusion_matrix.png
artifacts/sklearn_logreg_binary_1000_tuned/predictions.csv

artifacts/sklearn_svm_rbf_binary_1000_tuned/metrics.json
artifacts/sklearn_svm_rbf_binary_1000_tuned/tuning_results.csv
artifacts/sklearn_svm_rbf_binary_1000_tuned/classification_report.txt
artifacts/sklearn_svm_rbf_binary_1000_tuned/confusion_matrix.png
artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv
```

### Ограничения

- Search space небольшой и ручной.
- Threshold оптимизировался по probability estimates; для SVM эти вероятности
  калибруются внутренним механизмом sklearn.
- Test subset всё ещё balanced `1000/1000`, не полное естественное
  распределение DUSHA.
- Улучшение SVM маленькое; его стоит трактовать как incremental baseline, а не
  как качественный прорыв.

## 2026-05-22 — First compact CNN on log-mel spectrograms

### Цель

Перейти от classical ML baseline на summary audio features к neural baseline,
который получает на вход спектральное представление звука:

```text
audio -> log-mel spectrogram -> compact CNN -> binary emotion prediction
```

Это соответствует исходной идее проекта: сверточная сеть работает не с
ручными средними/стандартными отклонениями, а с двумерной time-frequency
картой.

### Environment

PyTorch был установлен отдельно в проектное `.venv`:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Проверка:

```text
torch 2.12.0+cpu
cuda False
```

Обучение выполнялось на CPU.

### Изменения в коде

Добавлены:

```text
src/build_logmel_cache.py
src/logmel_cache.py
src/train_cnn_logmel.py
```

`src/build_logmel_cache.py` строит `.npz` cache с log-mel tensors, чтобы CNN
запуски не перечитывали WAV каждый раз.

`src/train_cnn_logmel.py` обучает compact CNN, сохраняет:

```text
metrics.json
classification_report.txt
confusion_matrix.png
training_curves.png
training_curves.csv
predictions.csv
model.pt
```

### Log-mel cache

Train cache:

```bash
.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_train_1000.csv \
  --output data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --sample-rate 16000 \
  --duration 3.0 \
  --n-mels 64
```

Test cache:

```bash
.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_test_1000.csv \
  --output data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --sample-rate 16000 \
  --duration 3.0 \
  --n-mels 64
```

Result:

```text
train shape: (2000, 64, 94), float32
test shape:  (2000, 64, 94), float32
train cache size: 40M
test cache size:  40M
```

### Architecture

Input tensor:

```text
1 x 64 x 94
```

Model:

```text
Conv2d(1 -> 16, kernel=3x3, padding=1)
BatchNorm2d(16)
ReLU
MaxPool2d(2)
Dropout2d(0.125)

Conv2d(16 -> 32, kernel=3x3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(2)
Dropout2d(0.25)

Conv2d(32 -> 64, kernel=3x3, padding=1)
BatchNorm2d(64)
ReLU
AdaptiveAvgPool2d(1x1)

Flatten
Dropout(0.25)
Linear(64 -> 2)
```

### Training command

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_test_eval \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --seed 42
```

Training split:

```text
train rows: 1600
validation rows: 400
external test rows: 2000
```

Best validation accuracy:

```text
0.6225
```

### Results

External test:

```text
              precision    recall  f1-score   support

           0     0.5944    0.7270    0.6541      1000
           1     0.6486    0.5040    0.5672      1000

    accuracy                         0.6155      2000
   macro avg     0.6215    0.6155    0.6107      2000
weighted avg     0.6215    0.6155    0.6107      2000
```

Comparison with current best sklearn baseline:

```text
tuned RBF-SVM macro F1: 0.7195
first CNN macro F1:     0.6107
```

### Вывод

Первый CNN baseline работает end-to-end: есть log-mel cache, PyTorch training,
training curves, confusion matrix, predictions и сохраненный `model.pt`.

Качество пока ниже sklearn baseline. Это ожидаемо для первого neural run:
архитектура маленькая, обучение на CPU, только 2000 train examples, без
augmentation, scheduler, threshold tuning и подбора длительности/размера
mel-представления.

Ценность этого шага не в победе над SVM, а в том, что neural pipeline теперь
существует и его можно улучшать воспроизводимо.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_test_eval/metrics.json
artifacts/cnn_logmel_binary_1000_test_eval/classification_report.txt
artifacts/cnn_logmel_binary_1000_test_eval/confusion_matrix.png
artifacts/cnn_logmel_binary_1000_test_eval/training_curves.png
artifacts/cnn_logmel_binary_1000_test_eval/training_curves.csv
artifacts/cnn_logmel_binary_1000_test_eval/predictions.csv
artifacts/cnn_logmel_binary_1000_test_eval/model.pt
```

### Ограничения

- CPU-only training.
- Только 3 секунды аудио на пример.
- Нет data augmentation.
- Нет learning-rate scheduler.
- Нет early stopping по macro F1.
- Нет threshold tuning для CNN probabilities.
- CNN пока использует простой `argmax`, а не оптимизированный threshold.

## 2026-05-22 — CNN threshold tuning and SVM/CNN error comparison

### Цель

Проверить два вопроса без изменения dataset split:

1. Поможет ли CNN простой probability threshold tuning.
2. Насколько ошибки CNN и лучшего tuned SVM пересекаются на одном external test
   subset.

### Изменения в коде

Обновлен:

```text
src/train_cnn_logmel.py
```

Добавлено:

```text
src/compare_predictions.py
```

`src/train_cnn_logmel.py` теперь:

- сохраняет `validation_predictions.csv`;
- перебирает thresholds `0.30..0.70` на validation probabilities;
- выбирает threshold по validation macro F1;
- применяет выбранный threshold на external test;
- сохраняет `threshold_results.csv`.

`src/compare_predictions.py` объединяет два `predictions.csv` по `audio_path` и
сохраняет категории:

```text
both_correct
svm_only_correct
cnn_only_correct
both_wrong
```

### Commands

CNN with validation threshold tuning:

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_threshold_tuned \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --seed 42
```

SVM vs CNN comparison:

```bash
.venv/bin/python -m src.compare_predictions \
  --left artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv \
  --left-name svm \
  --right artifacts/cnn_logmel_binary_1000_threshold_tuned/predictions.csv \
  --right-name cnn \
  --output-dir artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned
```

### Threshold tuning result

Best validation threshold:

```text
threshold: 0.50
validation accuracy: 0.6225
validation macro F1: 0.6216
```

External test:

```text
accuracy: 0.6155
macro F1: 0.6107
```

Threshold tuning did not improve the first CNN run because the best validation
threshold was the default `0.50`.

### SVM/CNN comparison

Compared runs:

```text
svm: artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv
cnn: artifacts/cnn_logmel_binary_1000_threshold_tuned/predictions.csv
```

Overall:

```text
rows: 2000
svm correct: 1440 / 2000 = 0.7200
cnn correct: 1231 / 2000 = 0.6155

both_correct:     1003
svm_only_correct: 437
cnn_only_correct: 228
both_wrong:       332
```

By target label:

```text
target 0:
  both_correct:     556
  svm_only_correct: 123
  cnn_only_correct: 171
  both_wrong:       150

target 1:
  both_correct:     447
  svm_only_correct: 314
  cnn_only_correct: 57
  both_wrong:       182
```

### Вывод

CNN пока особенно проигрывает SVM на positive class (`target 1`):

```text
svm_only_correct on target 1: 314
cnn_only_correct on target 1: 57
```

При этом CNN не бесполезен: есть 228 примеров, где CNN прав, а SVM ошибается.
Это значит, что модель видит часть сигналов иначе, но текущая архитектура и
training setup пока не извлекают из log-mel представления достаточно устойчивые
признаки.

Практический следующий шаг для CNN:

- выбирать best epoch по validation macro F1, а не accuracy;
- добавить scheduler;
- попробовать более длинный input, например `duration=6.0`;
- затем попробовать чуть более широкую CNN `32 -> 64 -> 128`.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_threshold_tuned/metrics.json
artifacts/cnn_logmel_binary_1000_threshold_tuned/threshold_results.csv
artifacts/cnn_logmel_binary_1000_threshold_tuned/validation_predictions.csv
artifacts/cnn_logmel_binary_1000_threshold_tuned/predictions.csv

artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/summary.json
artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/comparison.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/comparison_by_label.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/svm_only_correct.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/cnn_only_correct.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_threshold_tuned/both_wrong.csv
```

## 2026-05-22 — Longer CNN training with validation macro F1 checkpointing

### Цель

Проверить, недообучается ли первый CNN baseline, и улучшить training loop без
изменения входного log-mel cache:

- больше эпох: до `60`;
- best checkpoint выбирать по validation macro F1, а не accuracy;
- добавить `ReduceLROnPlateau`;
- добавить early stopping по validation macro F1.

### Изменения в коде

Обновлен:

```text
src/train_cnn_logmel.py
```

Добавлено:

```text
--scheduler reduce_on_plateau
--lr-factor
--lr-patience
--patience
--min-delta
```

Training history теперь включает:

```text
val_precision_macro
val_recall_macro
val_f1_macro
learning_rate
```

Модель сохраняется из лучшей эпохи по `val_f1_macro`.

### Command

```bash
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
```

### Training result

```text
epochs completed: 37 / 60
stopped early: true
best epoch: 25
best validation accuracy: 0.6225
best validation macro F1: 0.6224
best threshold: 0.50
```

Learning rate did not reduce during this run:

```text
final lr: 0.001
```

This means validation loss kept improving often enough for
`ReduceLROnPlateau` not to trigger before early stopping by macro F1.

### External test result

```text
              precision    recall  f1-score   support

           0     0.6160    0.6690    0.6414      1000
           1     0.6379    0.5830    0.6092      1000

    accuracy                         0.6260      2000
   macro avg     0.6269    0.6260    0.6253      2000
weighted avg     0.6269    0.6260    0.6253      2000
```

Comparison to earlier CNN runs:

```text
first CNN macro F1:             0.6107
CNN threshold tuned macro F1:   0.6107
CNN 60ep scheduler macro F1:    0.6253
```

Improvement over first CNN:

```text
macro F1 +0.0146
accuracy +0.0105
```

Comparison to tuned SVM:

```text
tuned RBF-SVM macro F1:         0.7195
CNN 60ep scheduler macro F1:    0.6253
gap:                            0.0942
```

### SVM/CNN comparison

Command:

```bash
.venv/bin/python -m src.compare_predictions \
  --left artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv \
  --left-name svm \
  --right artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/predictions.csv \
  --right-name cnn60 \
  --output-dir artifacts/compare_svm_tuned_vs_cnn_logmel_60ep_f1_scheduler
```

Overall:

```text
rows: 2000
svm correct: 1440 / 2000 = 0.7200
cnn60 correct: 1252 / 2000 = 0.6260

both_correct:       1025
svm_only_correct:   415
cnn60_only_correct: 227
both_wrong:         333
```

By target label:

```text
target 0:
  both_correct:       511
  svm_only_correct:   168
  cnn60_only_correct: 158
  both_wrong:         163

target 1:
  both_correct:       514
  svm_only_correct:   247
  cnn60_only_correct: 69
  both_wrong:         170
```

### Вывод

Увеличение числа эпох и выбор checkpoint по validation macro F1 помогли:

```text
0.6107 -> 0.6253 macro F1
```

Но это всё ещё не догоняет tuned SVM. При этом результат стал более
сбалансированным по классам: recall positive class вырос с `0.5040` до
`0.5830`, что для нашей задачи важно.

CNN по-прежнему слабее всего там, где SVM хорошо ловит `target 1`. Следующий
наиболее рациональный шаг: изменить input/архитектуру, а не просто крутить
epochs дальше.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/metrics.json
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/training_curves.csv
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/training_curves.png
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/threshold_results.csv
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/validation_predictions.csv
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/predictions.csv
artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/model.pt

artifacts/compare_svm_tuned_vs_cnn_logmel_60ep_f1_scheduler/summary.json
artifacts/compare_svm_tuned_vs_cnn_logmel_60ep_f1_scheduler/comparison.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_60ep_f1_scheduler/comparison_by_label.csv
```

## Следующие шаги

1. Передать `artifacts/validation_sample_logreg_binary_300.csv` Диане на ручную проверку.
2. Передать Диане папку `artifacts/validation_sample_logreg_binary_300_audio/`.
3. После проверки внести агрегированные результаты в журнал.
4. Добавить компактный `run_notes.md` generator для каждого run directory.
5. Улучшить CNN input: `duration=6.0`, возможно `n_mels=80`.
6. Попробовать более широкую CNN `32 -> 64 -> 128`.
7. Подготовить для Дианы короткий набор `both_wrong` и `model_disagreement` аудио.


## 2026-05-22 - Wide CNN baseline on 3s/64-mel log-mel cache

### Goal / hypothesis

Check whether widening the compact CNN from `16,32,64` to `32,64,128` improves binary external-test performance while keeping the same 3-second, 64-mel input and the same validation macro-F1 checkpoint selection.

### Input data

```text
train cache: data/features/subset_binary_train_1000_logmel_3s_64mels.npz
eval cache:  data/features/subset_binary_test_1000_logmel_3s_64mels.npz
train rows: 2000 total, split into 1600 train / 400 validation
eval rows:  2000
classes:    binary target_label, balanced 1000/1000 external eval
```

### Code change

`src/train_cnn_logmel.py` now accepts `--channels`, for example `16,32,64` or `32,64,128`. The classifier input size is derived from the final channel count.

### Command

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_wide_60ep_f1_scheduler \
  --epochs 60 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --channels 32,64,128 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 5 \
  --patience 12 \
  --seed 42
```

### Important parameters

```text
model: CompactLogMelCNN
channels: 32,64,128
device: cpu
epochs requested: 60
epochs completed: 19
early stopping: yes
best epoch: 7
best validation macro F1: 0.6172
best threshold: 0.50
```

### External test result

```text
              precision    recall  f1-score   support

           0     0.6093    0.6440    0.6262      1000
           1     0.6225    0.5870    0.6042      1000

    accuracy                         0.6155      2000
   macro avg     0.6159    0.6155    0.6152      2000
weighted avg     0.6159    0.6155    0.6152      2000
```

### Comparison table

```text
run                                           | model            | data/subset            | key parameters                             | accuracy | macro F1 | interpretation
----------------------------------------------|------------------|------------------------|--------------------------------------------|----------|----------|----------------
sklearn_svm_rbf_binary_1000_tuned              | RBF-SVM          | binary train/test 1000 | tuned C/gamma, cached sklearn features     | 0.7200   | 0.7195   | strongest current baseline
cnn_logmel_binary_1000_60ep_f1_scheduler       | CompactLogMelCNN | 3s, 64-mel log-mel     | channels 16,32,64; best val-F1 checkpoint  | 0.6260   | 0.6253   | best CNN so far, still below SVM
cnn_logmel_binary_1000_wide_60ep_f1_scheduler  | CompactLogMelCNN | 3s, 64-mel log-mel     | channels 32,64,128; best val-F1 checkpoint | 0.6155   | 0.6152   | widening alone did not help
```

### SVM / wide-CNN comparison

Command:

```bash
.venv/bin/python -m src.compare_predictions \
  --left artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv \
  --left-name svm \
  --right artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/predictions.csv \
  --right-name cnn_wide \
  --output-dir artifacts/compare_svm_tuned_vs_cnn_logmel_wide_60ep_f1_scheduler
```

Overall:

```text
rows: 2000
svm correct:      1440 / 2000 = 0.7200
cnn_wide correct: 1231 / 2000 = 0.6155

both_correct:          994
svm_only_correct:      446
cnn_wide_only_correct: 237
both_wrong:            323
```

### Interpretation

The wider `32,64,128` CNN underperformed the previous `16,32,64` CNN:

```text
previous CNN macro F1: 0.6253
wide CNN macro F1:     0.6152
delta:                -0.0101
```

This suggests that model width alone is not the missing factor for the current 3-second, 64-mel input. The next local experiment should change input duration and frequency resolution, using the already available `6s_80mels` caches.

### Limitations

Only one random seed was tested. The run used the same simple CNN topology and only changed channel widths. No human label validation has been incorporated yet.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/metrics.json
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/training_curves.csv
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/training_curves.png
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/threshold_results.csv
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/validation_predictions.csv
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/predictions.csv
artifacts/cnn_logmel_binary_1000_wide_60ep_f1_scheduler/model.pt

artifacts/compare_svm_tuned_vs_cnn_logmel_wide_60ep_f1_scheduler/summary.json
artifacts/compare_svm_tuned_vs_cnn_logmel_wide_60ep_f1_scheduler/comparison.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_wide_60ep_f1_scheduler/comparison_by_label.csv
```

### Next step

Run the same CNN training protocol on:

```text
data/features/subset_binary_train_1000_logmel_6s_80mels.npz
data/features/subset_binary_test_1000_logmel_6s_80mels.npz
```


## 2026-05-22 - CNN on 6s/80-mel log-mel cache

### Goal / hypothesis

Check whether a longer input window and higher mel resolution improve CNN performance compared with the best previous 3-second, 64-mel CNN baseline.

### Input data

```text
train cache: data/features/subset_binary_train_1000_logmel_6s_80mels.npz
eval cache:  data/features/subset_binary_test_1000_logmel_6s_80mels.npz
train rows: 2000 total, split into 1600 train / 400 validation
eval rows:  2000
sample rate: 16000
duration: 6.0 seconds
n_mels: 80
feature shape: 80 x 188
```

### Command

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_6s_80mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_6s_80mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler \
  --epochs 60 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --channels 16,32,64 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 5 \
  --patience 12 \
  --seed 42
```

### Important parameters

```text
model: CompactLogMelCNN
channels: 16,32,64
device: cpu
epochs requested: 60
epochs completed: 31
early stopping: yes
best epoch: 19
best validation accuracy: 0.6225
best validation macro F1: 0.6196
best threshold: 0.50
```

### External test result

```text
              precision    recall  f1-score   support

           0     0.6047    0.7510    0.6699      1000
           1     0.6715    0.5090    0.5791      1000

    accuracy                         0.6300      2000
   macro avg     0.6381    0.6300    0.6245      2000
weighted avg     0.6381    0.6300    0.6245      2000
```

### Comparison table

```text
run                                           | model            | data/subset            | key parameters                             | accuracy | macro F1 | interpretation
----------------------------------------------|------------------|------------------------|--------------------------------------------|----------|----------|----------------
sklearn_svm_rbf_binary_1000_tuned              | RBF-SVM          | binary train/test 1000 | tuned C/gamma, cached sklearn features     | 0.7200   | 0.7195   | strongest current baseline
cnn_logmel_binary_1000_60ep_f1_scheduler       | CompactLogMelCNN | 3s, 64-mel log-mel     | channels 16,32,64; best val-F1 checkpoint  | 0.6260   | 0.6253   | best CNN macro F1 so far
cnn_logmel_binary_1000_wide_60ep_f1_scheduler  | CompactLogMelCNN | 3s, 64-mel log-mel     | channels 32,64,128; best val-F1 checkpoint | 0.6155   | 0.6152   | widening alone did not help
cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler | CompactLogMelCNN | 6s, 80-mel log-mel  | channels 16,32,64; best val-F1 checkpoint  | 0.6300   | 0.6245   | accuracy improved, macro F1 roughly tied with best CNN
```

### SVM / 6s80-CNN comparison

Command:

```bash
.venv/bin/python -m src.compare_predictions \
  --left artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv \
  --left-name svm \
  --right artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/predictions.csv \
  --right-name cnn_6s80 \
  --output-dir artifacts/compare_svm_tuned_vs_cnn_logmel_6s80mels_60ep_f1_scheduler
```

Overall:

```text
rows: 2000
svm correct:      1440 / 2000 = 0.7200
cnn_6s80 correct: 1260 / 2000 = 0.6300

both_correct:          1017
svm_only_correct:      423
cnn_6s80_only_correct: 243
both_wrong:            317
```

### Interpretation

The 6-second, 80-mel input improved accuracy over the previous 3-second CNN:

```text
3s/64mels CNN accuracy:  0.6260
6s/80mels CNN accuracy: 0.6300
```

Macro F1 did not improve materially:

```text
3s/64mels CNN macro F1:  0.6253
6s/80mels CNN macro F1: 0.6245
```

The model became more conservative for `target_label=1`: class 0 recall rose to `0.7510`, while class 1 recall fell to `0.5090`. For the thesis this is useful as evidence that longer context/finer mel resolution alone is not enough; it changes the error balance but does not close the gap to the tuned SVM.

### Limitations

Only one random seed was tested. The threshold grid selected `0.50`, so no threshold adjustment improved validation macro F1. No manual label validation has been incorporated yet.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/metrics.json
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/training_curves.csv
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/training_curves.png
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/threshold_results.csv
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/validation_predictions.csv
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/predictions.csv
artifacts/cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler/model.pt

artifacts/compare_svm_tuned_vs_cnn_logmel_6s80mels_60ep_f1_scheduler/summary.json
artifacts/compare_svm_tuned_vs_cnn_logmel_6s80mels_60ep_f1_scheduler/comparison.csv
artifacts/compare_svm_tuned_vs_cnn_logmel_6s80mels_60ep_f1_scheduler/comparison_by_label.csv
```

### Next step

Try a targeted change that addresses class-balance behavior instead of only input size, for example class-weighted loss or a threshold policy chosen for the external validation objective. Keep the tuned SVM as the current main baseline.


## 2026-05-23 - Add runtime timing metrics to CNN training

### Goal / hypothesis

Make future CNN experiment durations reproducible instead of estimating training time from memory.

### Input data

No dataset was processed in this step. This was a training-script instrumentation change.

### Code change

`src/train_cnn_logmel.py` now records timing fields in `metrics.json`:

```text
started_at
finished_at
duration_seconds
mean_epoch_seconds
```

Each training history row, and therefore `training_curves.csv`, now also includes:

```text
epoch_duration_seconds
```

The epoch progress printout includes `epoch_sec=...`, and the final output prints total duration and mean epoch duration.

### Commands used

```bash
.venv/bin/python -m py_compile src/train_cnn_logmel.py
.venv/bin/python -m src.train_cnn_logmel --help
```

### Metrics / artifacts

No model metrics were produced. Validation commands completed successfully.

### Interpretation

Future CNN runs will have exact wall-clock duration and per-epoch timing in their artifacts. Existing runs still only have approximate times unless re-run.

### Limitations

Timing is wall-clock timing on the local machine and depends on CPU/GPU availability and system load.

### Next step

Use these timing fields in the next CNN diagnostic run, likely an overfit test or a class-balance experiment.


## 2026-05-24 - CNN without final adaptive pooling

### Goal / hypothesis

Test whether removing the final `AdaptiveAvgPool2d((1, 1))` helps the CNN retain more time-frequency information before classification.

### Input data

```text
train cache: data/features/subset_binary_train_1000_logmel_3s_64mels.npz
eval cache:  data/features/subset_binary_test_1000_logmel_3s_64mels.npz
train rows: 1600
validation rows: 400
eval rows: 2000
sample_rate: 16000
duration: 3.0 seconds
n_mels: 64
feature shape: 64 x 94
```

### Command

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz \
  --artifacts-dir artifacts \
  --run-name cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler \
  --epochs 60 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --channels 16,32,64 \
  --pool-output-size 0 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 5 \
  --patience 12 \
  --seed 42
```

### Important parameters

```text
pool_output_size: 0
classifier_input_features: 23552
training device: cpu
epochs completed: 46 / 60
early stopping: yes
best epoch: 34
best validation macro F1: 0.6298
duration_seconds: 310.8
mean_epoch_seconds: 6.6
```

### External test result

```text
              precision    recall  f1-score   support

           0     0.6094    0.6850    0.6450      1000
           1     0.6404    0.5610    0.5981      1000

    accuracy                         0.6230      2000
   macro avg     0.6249    0.6230    0.6215      2000
weighted avg     0.6249    0.6230    0.6215      2000
```

### Comparison table

```text
run                                              | model            | data/subset        | key parameters                            | accuracy | macro F1 | interpretation
-------------------------------------------------|------------------|--------------------|-------------------------------------------|----------|----------|----------------
sklearn_svm_rbf_binary_1000_tuned                 | RBF-SVM          | binary test 1000   | tuned C/gamma, cached sklearn features    | 0.7200   | 0.7195   | strongest current baseline
cnn_logmel_binary_1000_60ep_f1_scheduler          | CompactLogMelCNN | 3s, 64-mel log-mel | pool 1x1, channels 16,32,64               | 0.6260   | 0.6253   | best CNN macro F1 so far
cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler | CompactLogMelCNN | 3s, 64-mel log-mel | no final adaptive pool, 23552 classifier inputs | 0.6230 | 0.6215 | retained more information but did not improve external generalization
```

### Interpretation

Removing final adaptive pooling let the model fit the training split much more strongly; train accuracy reached about `0.94`, while previous compact CNN runs looked closer to underfit. Validation best macro F1 also reached `0.6298`, slightly above the previous validation score. However, external macro F1 fell to `0.6215`, below the previous best CNN `0.6253`.

This suggests the old `AdaptiveAvgPool2d((1, 1))` was likely too aggressive, but fully removing it overfits or learns validation-specific details. A compromise such as `AdaptiveAvgPool2d((4, 4))` or a smaller MLP head is the next architectural test.

### Limitations

Only one seed was tested. No comparison CSV against SVM was generated before the session was interrupted. Generated artifacts are in `artifacts/` and are not included in the project handoff archive.

### Artifacts

```text
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/metrics.json
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/training_curves.csv
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/training_curves.png
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/threshold_results.csv
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/validation_predictions.csv
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/predictions.csv
artifacts/cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler/model.pt
```

### Next step

Try `--pool-output-size 4` on the same 3s/64-mel cache as a compromise between global average pooling and no final pooling.


## 2026-05-27 - Thesis-ready log-mel illustration package

### Goal / hypothesis

Generate a compact, reproducible set of log-mel figures for the thesis methodology, dataset description, and model error-analysis sections.

### Input data

```text
CNN predictions: artifacts/cnn_logmel_binary_1000_60ep_f1_scheduler/predictions.csv
SVM predictions: artifacts/sklearn_svm_rbf_binary_1000_tuned/predictions.csv
Audio source: external test subset paths referenced by the prediction files
```

### Code and command

Added:

```text
src/export_thesis_mel_figures.py
```

Command:

```bash
.venv/bin/python -m src.export_thesis_mel_figures \
  --output-dir artifacts/thesis_mel_figures
```

Validation:

```bash
.venv/bin/python -m py_compile src/export_thesis_mel_figures.py
```

### Important parameters

```text
sample_rate: 16000 Hz
duration: 3.0 s
n_mels: 64
color map: magma
color range: -80 to 0 dB relative to each clip maximum
export resolution: 300 dpi
```

Figures are regenerated from audio in log-mel dB space before the per-clip standardization used for CNN tensor input.

### Selection protocol

```text
figure 01: highest-confidence correctly classified positive example from the best CNN
figure 02: highest-confidence CNN-correct positive and negative class examples
figure 03: highest-confidence CNN-correct examples for positive, sad, and angry source labels
figure 04: one example each from both_correct, SVM_only_correct, CNN_only_correct, and both_wrong; prioritised by model probability disagreement
```

Each panel's audio path, source/target label, predicted labels and positive probabilities are recorded in `provenance.csv`.

### Comparison table for models illustrated

```text
run                                      | model            | data/subset            | key parameters                           | accuracy | macro F1 | interpretation
-----------------------------------------|------------------|------------------------|------------------------------------------|----------|----------|----------------
sklearn_svm_rbf_binary_1000_tuned        | RBF-SVM          | binary external test   | tuned C/gamma, summary features          | 0.7200   | 0.7195   | strongest baseline shown in error figure
cnn_logmel_binary_1000_60ep_f1_scheduler | CompactLogMelCNN | 3s, 64-mel log-mel     | channels 16,32,64; best val-F1 checkpoint | 0.6260   | 0.6253   | best CNN macro F1 shown in figures
```

### Generated artifacts

```text
artifacts/thesis_mel_figures/figure_01_logmel_input_example.png
artifacts/thesis_mel_figures/figure_02_class_examples_positive_vs_nonpositive.png
artifacts/thesis_mel_figures/figure_03_emotion_examples_positive_sad_angry.png
artifacts/thesis_mel_figures/figure_04_error_analysis_best_cnn_vs_svm.png
artifacts/thesis_mel_figures/provenance.csv
artifacts/thesis_mel_figures/README.md
```

Output validation:

```text
figure_01_logmel_input_example.png:                 2100 x 1200 px, 300 dpi
figure_02_class_examples_positive_vs_nonpositive.png: 2796 x 1233 px, 300 dpi
figure_03_emotion_examples_positive_sad_angry.png:  3488 x 1233 px, 300 dpi
figure_04_error_analysis_best_cnn_vs_svm.png:        2941 x 2277 px, 300 dpi
```

### Interpretation

The package gives thesis-ready visuals tied to recorded model results rather than arbitrary samples. The error-analysis figure exposes model agreement/disagreement together with labels and `P(+)`, while the three class-oriented figures illustrate input construction and class composition.

### Limitations

Selections are representative by deterministic confidence/disagreement rules, not by human qualitative judgement. Color values are dB relative to the maximum within each clip, so panels compare time-frequency structure rather than absolute recording loudness. Manual review can replace selected examples if any audio is noisy or unsuitable for publication.

### Next step

Let Diana review the four PNG figures and `provenance.csv`; revise figure captions or substitute samples only if needed for the final thesis layout.


## 2026-05-31 - Local DUSHA archive extraction on GPU workstation

### Goal / hypothesis

Prepare the transferred local DUSHA data directory for GPU-side CNN experiments by unpacking the audio archives under `data/dusha_emotion_audio/data`.

### Input data

```text
train archive: data/dusha_emotion_audio/data/train.tar.gz
test archive:  data/dusha_emotion_audio/data/test.tar.gz
train archive size: 8.5G
test archive size: 2.1G
train metadata CSV: data/dusha_emotion_audio/data/train.csv
test metadata CSV:  data/dusha_emotion_audio/data/test.csv
```

### Commands / checks

```bash
find data/dusha_emotion_audio/data -maxdepth 2 -type f | sort | head -n 40
tar -tzf data/dusha_emotion_audio/data/train.tar.gz
tar -tzf data/dusha_emotion_audio/data/test.tar.gz
tar -xzf data/dusha_emotion_audio/data/train.tar.gz -C data/dusha_emotion_audio/data
tar -xzf data/dusha_emotion_audio/data/test.tar.gz -C data/dusha_emotion_audio/data
gzip -t data/dusha_emotion_audio/data/train.tar.gz
gzip -t data/dusha_emotion_audio/data/test.tar.gz
df -h . data/dusha_emotion_audio/data
find data/dusha_emotion_audio/data/train -type f -name '*.wav' | wc -l
find data/dusha_emotion_audio/data/test -type f -name '*.wav' | wc -l
wc -l data/dusha_emotion_audio/data/train.csv data/dusha_emotion_audio/data/test.csv
```

### Results

```text
test extraction: completed successfully
test wav files: 24171
test.csv lines: 24172 including header
test archive integrity: gzip -t completed successfully

train extraction: failed
train wav files after partial extraction: 26011
train.csv lines: 96681 including header
train archive integrity: failed
gzip error: invalid compressed data--crc error; invalid compressed data--length error

free disk space on /mnt/data4: about 3.0T available
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
archive_extract_2026_05_31 | n/a | DUSHA local train/test archives | tar extraction and gzip integrity checks | n/a | n/a | test split is usable; train archive is corrupted and only partially extracted
```

### Interpretation

The local `test` split is ready for scripts that require audio files. The local `train` split is not ready because `train.tar.gz` is corrupted or incomplete. The failure is not caused by lack of disk space.

### Limitations

The partially extracted `data/dusha_emotion_audio/data/train/` directory contains only about 27% of the expected train wav files. Any experiment depending on raw train audio should wait until `train.tar.gz` is replaced and extracted cleanly.

### Next step

Replace or re-download `data/dusha_emotion_audio/data/train.tar.gz`, then re-run extraction and `gzip -t` validation before GPU CNN training from raw audio or rebuilding log-mel caches.


## 2026-05-31 - DUSHA train/test archive extraction completed locally

### Goal / hypothesis

Unpack the local DUSHA `train.tar.gz` and `test.tar.gz` archives into usable raw-audio directories for the next inspection and baseline steps.

### Input data

```text
train archive: data/dusha_emotion_audio/data/train.tar.gz
test archive:  data/dusha_emotion_audio/data/test.tar.gz
train archive size: 8.5G
test archive size: 2.1G
metadata: data/dusha_emotion_audio/data/train.csv, data/dusha_emotion_audio/data/test.csv
```

### Commands / checks

```bash
df -h .
find data/dusha_emotion_audio/data -maxdepth 1 -type d -print
tar -xzf data/dusha_emotion_audio/data/train.tar.gz -C data/dusha_emotion_audio/data --keep-old-files
tar -xzf data/dusha_emotion_audio/data/test.tar.gz -C data/dusha_emotion_audio/data --keep-old-files
find data/dusha_emotion_audio/data/train -type f -name '*.wav' | wc -l
find data/dusha_emotion_audio/data/test -type f -name '*.wav' | wc -l
du -sh data/dusha_emotion_audio/data/train data/dusha_emotion_audio/data/test
wc -l data/dusha_emotion_audio/data/train.csv data/dusha_emotion_audio/data/test.csv
```

### Results

```text
free disk space before extraction: 491G available
train extraction: completed successfully
test extraction: completed successfully
train wav files: 96680
test wav files: 24171
train directory size: 14G
test directory size: 3.4G
train.csv lines: 96681 including header
test.csv lines: 24172 including header
artifact paths:
  data/dusha_emotion_audio/data/train/
  data/dusha_emotion_audio/data/test/
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
archive_extract_2026_05_31_complete | n/a | DUSHA local train/test archives | tar extraction with --keep-old-files; wav counts checked against CSV row counts | n/a | n/a | train and test raw-audio directories are now complete by file-count check
```

### Interpretation

Both raw-audio splits are available locally. The `.wav` counts match the metadata row counts excluding headers, so the extracted directories are suitable for dataset inspection and CPU-friendly sklearn baseline preparation.

### Limitations

This step validates file counts and successful `tar` exit status only. It does not yet verify every `.wav` header, audio duration, label distribution, or checksum against an external manifest.

### Next step

Run the DUSHA inspection step on the extracted directories, then create the balanced subset for the sklearn baseline.


## 2026-05-31 - GPU readiness check for CNN training

### Goal / hypothesis

Check whether the local workstation can run `src/train_cnn_logmel.py` on GPU after the DUSHA audio archives were extracted.

### Input data

```text
training cache: data/features/subset_binary_train_1000_logmel_3s_64mels.npz
eval cache: data/features/subset_binary_test_1000_logmel_3s_64mels.npz
training script: src/train_cnn_logmel.py
python env: .venv
```

### Commands / checks

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count())"
.venv/bin/python -m pip show torch
.venv/bin/python -m src.train_cnn_logmel --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz --run-name cuda_smoke_check_2026_05_31 --epochs 1 --batch-size 16 --device cuda
find data/features -maxdepth 1 -type f -name '*logmel*.npz' -exec ls -lh {} \;
```

### Results

```text
GPU hardware outside sandbox: visible
GPU: NVIDIA GeForce RTX 3060
GPU memory: 12288 MiB
driver version: 580.159.03
nvidia-smi CUDA version: 13.0
current .venv torch: 2.12.0+cpu
torch.cuda.is_available(): False
torch.version.cuda: None
torch.cuda.device_count(): 0
CUDA smoke run result: failed before training with "CUDA was requested but is not available."
available log-mel caches:
  data/features/subset_binary_train_1000_logmel_3s_64mels.npz 40M
  data/features/subset_binary_test_1000_logmel_3s_64mels.npz 40M
  data/features/subset_binary_train_1000_logmel_6s_80mels.npz 78M
  data/features/subset_binary_test_1000_logmel_6s_80mels.npz 77M
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
cuda_smoke_check_2026_05_31 | CompactLogMelCNN | subset_binary_train/test_1000_logmel_3s_64mels | --epochs 1 --batch-size 16 --device cuda | n/a | n/a | not run; current .venv has CPU-only PyTorch despite GPU being visible through nvidia-smi
```

### Interpretation

The workstation hardware can expose an RTX 3060, but the project `.venv` cannot run CUDA workloads because it contains a CPU-only PyTorch build. GPU training should be possible after installing a CUDA-enabled PyTorch build into an environment that has access to `/dev/nvidia*`.

### Limitations

The smoke check did not install or modify PyTorch. It did not benchmark GPU throughput or validate memory limits. `nvidia-smi` required execution outside the default sandbox to see the GPU device.

### Next step

Install a CUDA-enabled PyTorch build in a separate GPU environment or replace the CPU-only torch package in `.venv`, then rerun the same one-epoch smoke command with `--device cuda`.


## 2026-05-31 - CUDA PyTorch install and GPU smoke training

### Goal / hypothesis

Replace the CPU-only PyTorch package in the project `.venv` with a CUDA-enabled build and verify that `src/train_cnn_logmel.py` can train on the RTX 3060.

### Input data

```text
training cache: data/features/subset_binary_train_1000_logmel_3s_64mels.npz
eval cache: data/features/subset_binary_test_1000_logmel_3s_64mels.npz
training script: src/train_cnn_logmel.py
python: .venv/bin/python, Python 3.12.3
GPU: NVIDIA GeForce RTX 3060, 12288 MiB
driver: 580.159.03
```

### Commands / checks

```bash
.venv/bin/python -m pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); x=torch.ones((2,3), device='cuda'); print(x.device); print(float(x.sum().item()))"
.venv/bin/python -m pip check
nvidia-smi
.venv/bin/python -m src.train_cnn_logmel --features-path data/features/subset_binary_train_1000_logmel_3s_64mels.npz --eval-features-path data/features/subset_binary_test_1000_logmel_3s_64mels.npz --run-name cuda_smoke_check_2026_05_31_rerun --epochs 1 --batch-size 64 --device cuda
du -sh artifacts/cuda_smoke_check_2026_05_31_rerun .venv
```

### Results

```text
installed torch: 2.11.0+cu128
torch.cuda.is_available(): True
torch.version.cuda: 12.8
torch.cuda.device_count(): 1
torch CUDA device: NVIDIA GeForce RTX 3060
test tensor device: cuda:0
test tensor sum: 6.0
.venv size after CUDA install: 7.5G
pip check warning:
  datasets 4.8.5 requires fsspec[http]<=2026.2.0,>=2023.1.0, but fsspec 2026.4.0 is installed
smoke artifact path: artifacts/cuda_smoke_check_2026_05_31_rerun/
smoke artifact size: 580K
smoke duration_seconds: 2.86
smoke mean_epoch_seconds: 0.65
smoke device: cuda
smoke eval accuracy: 0.5475
smoke eval macro F1: 0.5379
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
cuda_smoke_check_2026_05_31_rerun | CompactLogMelCNN | subset_binary_train/test_1000_logmel_3s_64mels | torch 2.11.0+cu128; --epochs 1 --batch-size 64 --device cuda | 0.5475 | 0.5379 | GPU training path works; metrics are only a smoke check, not a meaningful trained model
```

### Interpretation

The project environment is now GPU-capable. PyTorch can allocate tensors on `cuda:0`, and the CNN training script completed a one-epoch GPU run on the cached 1000-per-class log-mel subset.

### Limitations

The smoke run used one epoch and should not be interpreted as a final model. The CUDA-enabled PyTorch install introduced an `fsspec` version conflict with `datasets`; this is unlikely to affect cached CNN training, but dataset download/loading workflows should be checked before use. The project intentionally keeps `requirements.txt` CPU-friendly, so this GPU install is an environment-level change rather than a requirements change.

### Next step

Run the planned longer CNN experiment on GPU using a distinct `run-name`, then compare its metrics against the existing CPU/GPU baseline entries in this log.


## 2026-05-31 - Longer GPU CNN training on 6s/80-mel cache

### Goal / hypothesis

Test whether the previous compact CNN was undertrained by running a wider model for a longer schedule on GPU, using the more detailed 6-second / 80-mel cached features.

### Input data

```text
training cache: data/features/subset_binary_train_1000_logmel_6s_80mels.npz
eval cache: data/features/subset_binary_test_1000_logmel_6s_80mels.npz
train subset: data/processed/subset_binary_train_1000.csv
eval subset: data/processed/subset_binary_test_1000.csv
rows: 2000 train-subset rows split into 1600 train / 400 validation; 2000 external eval rows
```

### Commands / scripts used

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_6s_80mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_6s_80mels.npz \
  --run-name cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu \
  --epochs 120 \
  --batch-size 128 \
  --channels 32,64,128 \
  --pool-output-size 4 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 5 \
  --patience 30 \
  --device cuda
```

### Important parameters

```text
model: CompactLogMelCNN
channels: [32, 64, 128]
pool_output_size: 4
classifier_input_features: 2048
dropout: 0.25
learning_rate: 0.001
weight_decay: 0.0001
scheduler: reduce_on_plateau
lr_factor: 0.5
lr_patience: 5
patience: 30
device: cuda
torch: 2.11.0+cu128
```

### Metrics and artifacts

```text
run_name: cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu
artifact dir: artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/
artifact size: 972K
epochs completed: 120
best epoch: 96
best validation accuracy: 0.7775
best validation macro F1: 0.7773
best validation threshold: 0.60
best threshold validation macro F1: 0.7797
external eval accuracy: 0.7450
external eval precision macro: 0.7526
external eval recall macro: 0.7450
external eval macro F1: 0.7431
duration: 99.3 seconds
mean epoch time: 0.80 seconds
generated files:
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/metrics.json
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/model.pt
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/predictions.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/threshold_results.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/training_curves.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/training_curves.png
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/confusion_matrix.png
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/classification_report.txt
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/validation_predictions.csv
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler | CompactLogMelCNN | 1000/class, 6s/80mels | CPU; channels 16,32,64; pool 1; early stop at 31/60 | 0.6300 | 0.6245 | previous compact 6s CNN baseline; likely capacity/training-limited
cnn_logmel_binary_1000_no_final_pool_60ep_f1_scheduler | CompactLogMelCNN | 1000/class, 3s/64mels | CPU; channels 16,32,64; no final pool; early stop at 46/60 | 0.6230 | 0.6215 | stronger validation than default, but external eval did not improve
cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu | CompactLogMelCNN | 1000/class, 6s/80mels | GPU; channels 32,64,128; pool 4; 120 epochs; scheduler | 0.7450 | 0.7431 | substantial improvement; previous CNN setup was undertrained and/or under-capacity
```

### Interpretation

The wider GPU run improves external macro F1 from about 0.62 to 0.7431 on the same 1000-per-class binary evaluation design. Validation performance continued improving past the old 60-epoch budget and peaked at epoch 96, supporting the hypothesis that the earlier CNN runs were undertrained and too compact.

### Limitations

The experiment still uses the balanced 1000-per-class subset rather than all extracted DUSHA audio. The validation split is internal to the training subset, so the external test metrics are the primary result. The best validation threshold was recorded but the reported external metrics are from the default prediction path; a separate threshold-tuned eval can be run if needed for final comparison.

### Next step

Run a threshold-tuned evaluation and/or repeat the same GPU configuration on a larger balanced subset built from the now-complete extracted DUSHA train/test audio.


## 2026-05-31 - Extended 1200-epoch budget GPU CNN training

### Goal / hypothesis

Test whether increasing the training budget by 10x over the 120-epoch GPU run gives additional validation and external-eval improvement, while relying on early stopping to avoid wasting time after convergence.

### Input data

```text
training cache: data/features/subset_binary_train_1000_logmel_6s_80mels.npz
eval cache: data/features/subset_binary_test_1000_logmel_6s_80mels.npz
train subset: data/processed/subset_binary_train_1000.csv
eval subset: data/processed/subset_binary_test_1000.csv
rows: 2000 train-subset rows split into 1600 train / 400 validation; 2000 external eval rows
previous comparison run: artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu/
```

### Commands / scripts used

```bash
.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_6s_80mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_6s_80mels.npz \
  --run-name cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu \
  --epochs 1200 \
  --batch-size 128 \
  --channels 32,64,128 \
  --pool-output-size 4 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 10 \
  --patience 120 \
  --device cuda
```

### Important parameters

```text
model: CompactLogMelCNN
channels: [32, 64, 128]
pool_output_size: 4
classifier_input_features: 2048
dropout: 0.25
learning_rate: 0.001
weight_decay: 0.0001
scheduler: reduce_on_plateau
lr_factor: 0.5
lr_patience: 10
patience: 120
device: cuda
torch: 2.11.0+cu128
```

### Metrics and artifacts

```text
run_name: cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu
artifact dir: artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/
artifact size: 1.1M
epochs requested: 1200
epochs completed: 281
stopped early: true
best epoch: 161
best validation accuracy: 0.7875
best validation macro F1: 0.7870
best validation threshold: 0.50
external eval accuracy: 0.7575
external eval precision macro: 0.7575
external eval recall macro: 0.7575
external eval macro F1: 0.7575
duration: 229.6 seconds
mean epoch time: 0.81 seconds
generated files:
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/metrics.json
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/model.pt
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/predictions.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/threshold_results.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/training_curves.csv
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/training_curves.png
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/confusion_matrix.png
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/classification_report.txt
  artifacts/cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu/validation_predictions.csv
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
cnn_logmel_binary_1000_6s80mels_60ep_f1_scheduler | CompactLogMelCNN | 1000/class, 6s/80mels | CPU; channels 16,32,64; pool 1; early stop at 31/60 | 0.6300 | 0.6245 | compact earlier CNN baseline
cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu | CompactLogMelCNN | 1000/class, 6s/80mels | GPU; channels 32,64,128; pool 4; 120 epochs | 0.7450 | 0.7431 | large gain from wider model and GPU training
cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu | CompactLogMelCNN | 1000/class, 6s/80mels | GPU; channels 32,64,128; pool 4; 1200 epoch budget; early stop at 281 | 0.7575 | 0.7575 | modest additional gain; convergence around epoch 161, then plateau
```

### Interpretation

The 10x epoch budget improved the external macro F1 from 0.7431 to 0.7575. Validation macro F1 peaked at epoch 161, then stayed on a plateau while the scheduler drove the learning rate effectively to zero. This suggests that longer training helps up to about 160 epochs for this subset/configuration, but a full 1200 epochs is unnecessary with the current schedule.

### Limitations

The experiment still uses the balanced 1000-per-class cached subset, not the full extracted DUSHA train/test corpus. The run is deterministic for the existing seed and split, but only one seed was tested. Further gains probably require more data, augmentation, a different schedule, or architecture changes rather than simply more epochs.

### Next step

Build a larger balanced subset from the now-complete extracted data and repeat the best GPU configuration, or try a schedule with a lower starting learning rate and less aggressive decay around the 120-200 epoch window.


## 2026-05-31 - 7-second log-mel duration probe

### Goal / hypothesis

Check whether increasing log-mel input duration from 6 seconds to 7 seconds improves the current best CNN configuration, before spending more effort on model architecture changes.

### Input data

```text
train subset: data/processed/subset_binary_train_1000.csv
test subset: data/processed/subset_binary_test_1000.csv
train rows: 2000, balanced 1000 negative / 1000 positive
test rows: 2000, balanced 1000 negative / 1000 positive
duration distribution in train subset:
  min 1.416s; p10 3.000s; p25 3.780s; median 4.700s; p75 5.760s; p90 7.180s; max 18.380s
  >=3s: 1800; >=6s: 456; >=7s: 226; >=10s: 22
duration distribution in test subset:
  min 1.346s; p10 2.986s; p25 3.660s; median 4.640s; p75 5.600s; p90 6.934s; max 15.520s
  >=3s: 1797; >=6s: 400; >=7s: 195; >=10s: 24
```

### Commands / scripts used

```bash
.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_train_1000.csv \
  --output data/features/subset_binary_train_1000_logmel_7s_80mels.npz \
  --sample-rate 16000 \
  --duration 7.0 \
  --n-mels 80

.venv/bin/python -m src.build_logmel_cache \
  --subset-path data/processed/subset_binary_test_1000.csv \
  --output data/features/subset_binary_test_1000_logmel_7s_80mels.npz \
  --sample-rate 16000 \
  --duration 7.0 \
  --n-mels 80

.venv/bin/python -m src.train_cnn_logmel \
  --features-path data/features/subset_binary_train_1000_logmel_7s_80mels.npz \
  --eval-features-path data/features/subset_binary_test_1000_logmel_7s_80mels.npz \
  --run-name cnn_logmel_binary_1000_7s80mels_wide_pool4_1200ep_gpu \
  --epochs 1200 \
  --batch-size 128 \
  --channels 32,64,128 \
  --pool-output-size 4 \
  --scheduler reduce_on_plateau \
  --lr-factor 0.5 \
  --lr-patience 10 \
  --patience 120 \
  --device cuda
```

### Important parameters

```text
sample_rate: 16000
duration: 7.0
n_mels: 80
train cache shape: (2000, 80, 219)
test cache shape: (2000, 80, 219)
model: CompactLogMelCNN
channels: [32, 64, 128]
pool_output_size: 4
classifier_input_features: 2048
device: cuda
torch: 2.11.0+cu128
```

### Metrics and artifacts

```text
train cache: data/features/subset_binary_train_1000_logmel_7s_80mels.npz, 80M
test cache: data/features/subset_binary_test_1000_logmel_7s_80mels.npz, 79M
run_name: cnn_logmel_binary_1000_7s80mels_wide_pool4_1200ep_gpu
artifact dir: artifacts/cnn_logmel_binary_1000_7s80mels_wide_pool4_1200ep_gpu/
artifact size: 1.1M
epochs requested: 1200
epochs completed: 220
stopped early: true
best epoch: 100
best validation accuracy: 0.7800
best validation macro F1: 0.7799
best validation threshold: 0.60
best threshold validation macro F1: 0.7847
external eval accuracy: 0.7440
external eval precision macro: 0.7509
external eval recall macro: 0.7440
external eval macro F1: 0.7422
duration: 209.3 seconds
mean epoch time: 0.94 seconds
```

### Comparison table

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
----|-------|-------------|----------------|----------|----------|----------------
cnn_logmel_binary_1000_6s80mels_wide_pool4_120ep_gpu | CompactLogMelCNN | 1000/class, 6s/80mels | GPU; channels 32,64,128; pool 4; 120 epochs | 0.7450 | 0.7431 | strong 6s baseline
cnn_logmel_binary_1000_6s80mels_wide_pool4_1200ep_gpu | CompactLogMelCNN | 1000/class, 6s/80mels | GPU; channels 32,64,128; pool 4; early stop at 281 | 0.7575 | 0.7575 | current best; duration 6s with longer schedule
cnn_logmel_binary_1000_7s80mels_wide_pool4_1200ep_gpu | CompactLogMelCNN | 1000/class, 7s/80mels | GPU; channels 32,64,128; pool 4; early stop at 220 | 0.7440 | 0.7422 | 7s does not improve; likely adds padding/noise more than useful signal
```

### Interpretation

Increasing the duration from 6s to 7s did not help on the current balanced 1000-per-class subset. Only 226/2000 train files and 195/2000 test files are at least 7 seconds long, so most examples either contain no extra signal beyond 6 seconds or add mostly padding. The 7s run underperforms the best 6s run on external macro F1.

### Limitations

This tests only one seed, one balanced subset size, and one architecture. It uses first-`duration` truncation rather than random crops or center crops. A longer duration might behave differently with full-dataset training, crop augmentation, or attention/pooling over time.

### Next step

Prioritize model/data tweaks over simply extending fixed duration to 7s: larger balanced subset, crop augmentation, alternative pooling/time aggregation, or a lower-LR schedule around the 100-200 epoch convergence region.
