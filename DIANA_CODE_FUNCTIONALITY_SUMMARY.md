# Описание функционала кода и последних экспериментов

Коротко по текущему состоянию проекта.

Проект сейчас устроен как экспериментальный пайплайн для распознавания
эмоциональной окраски русской речи на датасете DUSHA. Основная текущая задача
бинарная: отличать `positive` от `negative`, где `negative` объединяет классы
`angry` и `sad`. Классы `neutral` и `other` в текущей бинарной постановке не
используются.

## Общий принцип работы

Пайплайн идет так:

1. Берем аудио и разметку из DUSHA.
2. Собираем сбалансированные подвыборки, чтобы классы были одинакового размера.
3. Из аудио извлекаем признаки.
4. Обучаем несколько моделей.
5. Сохраняем метрики, отчеты, confusion matrix и predictions в `artifacts/`.
6. Все эксперименты фиксируются в `EXPERIMENT_LOG.md`.

## Sklearn / SVM baseline

Это простая CPU-friendly ветка, которая нужна как базовый уровень качества.
Аудио здесь превращается не в спектрограмму-картинку, а в компактный числовой
вектор из 48 признаков.

Используемые признаки:

```text
20 MFCC mean
20 MFCC std
RMS mean/std
spectral centroid mean/std
spectral bandwidth mean/std
zero-crossing rate mean/std
```

То есть вектор описывает тембр, энергию и спектральные свойства речи.

На этих векторах обучались:

```text
Logistic Regression
Random Forest
SVM with RBF kernel
```

Лучший SVM baseline на подвыборке 1000 примеров на класс дал примерно:

```text
accuracy: 0.7200
macro F1: 0.7195
```

Вывод: sklearn/SVM работает как понятный baseline, но заметно уступает CNN.

## CNN на log-mel спектрограммах

Для CNN аудио преобразуется в log-mel spectrogram. Простыми словами, это
матрица "частота x время": одна ось показывает частотные диапазоны, другая -
время, а значения показывают энергию сигнала. CNN работает с этой матрицей
почти как с изображением.

Текущая лучшая версия использует:

```text
sample_rate: 16000
duration: 6 секунд
n_mels: 80
n_fft: 1024
hop_length: 256
input shape: 80 x 376
```

Архитектура лучшей CNN:

```text
model: CompactLogMelCNN
conv channels: 32 -> 64 -> 128
conv kernel: 3x3
adaptive pooling: 6x6
classifier head: 4608 -> 512 -> 2
dropout: 0.25
optimizer: AdamW
scheduler: ReduceLROnPlateau
early stopping: да
```

Обучение шло на сбалансированной выборке:

```text
train: 18000 аудио, 9000 positive / 9000 negative
test: 4800 аудио, 2400 positive / 2400 negative
```

Лучший текущий результат CNN:

```text
run_name: cnn_logmel_binary_9000_6s80mels_fft1024_hop256_wide_pool6_hidden512_1200ep_gpu
accuracy: 0.8475
macro F1: 0.8475
```

Это сейчас основной сильный результат.

## Что изменилось в последних экспериментах

Самый большой прирост качества дала не столько архитектура, сколько увеличение
объема обучающей выборки.

Переход с 1000 примеров на класс к 9000 примеров на класс поднял CNN примерно:

```text
macro F1: 0.7575 -> 0.8337
```

После этого проверялись архитектурные улучшения:

```text
hidden classifier head после CNN-признаков
увеличение hidden layer до 512
более высокое временное разрешение spectrogram через hop_length=256
adaptive pooling 6x6 вместо 4x4
```

Лучший вариант сейчас:

```text
CNN, 9000/class train, 2400/class eval
80 mel bands
6 секунд аудио
n_fft=1024
hop_length=256
pool 6x6
hidden classifier layer 512
accuracy: 0.8475
macro F1: 0.8475
```

Улучшение последнего варианта относительно предыдущего лучшего CNN маленькое
примерно `0.8470 -> 0.8475` macro F1, поэтому его лучше считать текущим
tentative best, а не окончательно доказанным стабильным результатом.

## MLP baseline

Дополнительно пробовалась MLP-модель на тех же log-mel спектрограммах. Она
просто разворачивает спектрограмму в длинный вектор и обучает dense layers,
без сверточной обработки локальных частотно-временных паттернов.

Один из лучших MLP-экспериментов:

```text
model: LogMelMLP
hidden_size: 1000
accuracy: 0.6883
macro F1: 0.6883
```

Вывод: MLP заметно слабее CNN. Это ожидаемо, потому что CNN лучше использует
локальную структуру спектрограммы.

## Проверка на non-neutral

Отдельно проверили, что текущий binary test уже не содержит `neutral`.

В test predictions:

```text
rows total: 4800
neutral rows: 0
target_label 0: 2400
target_label 1: 2400
```

Поэтому фильтрация `source_label != neutral` не меняет метрики:

```text
accuracy: 0.8475
macro F1: 0.8475
```

Важно: negative class внутри неоднородный, потому что он объединяет `angry` и
`sad`. Для текущей лучшей CNN:

```text
angry recall: 0.8351
sad recall: 0.8871
positive recall: 0.8371
```

То есть `sad` модель распознает внутри negative лучше, чем `angry`.

## Listening review set

Для качественной проверки был собран отдельный набор аудио, чтобы руками
послушать, что модель считает очевидным и где она уверенно ошибается.

Собраны 4 группы по 40 аудио:

```text
уверенно правильные positive
уверенно правильные negative
positive, которые модель уверенно приняла за negative
negative, которые модель уверенно приняла за positive
```

Путь:

```text
artifacts/listening_review/best_cnn_pool6_extreme_confidence_2026-06-01/
```

Это не новая метрика, а материал для ручного анализа ошибок: можно слушать и
отмечать, есть ли шум, неоднозначная эмоция, ошибки разметки, особенности
диктора, канал записи или другие артефакты.

## Где в коде что находится

Основные файлы:

```text
src/features.py
```

Извлечение признаков:

```text
extract_feature_vector() - 48 компактных признаков для sklearn/SVM
extract_log_mel() - log-mel спектрограммы для CNN/MLP
```

```text
src/build_feature_cache.py
```

Создание кеша компактных признаков для sklearn/SVM.

```text
src/build_logmel_cache.py
```

Создание кеша log-mel спектрограмм для CNN/MLP.

```text
src/train_sklearn.py
```

Обучение Logistic Regression, Random Forest и SVM.

```text
src/tune_sklearn.py
```

Подбор параметров и threshold для Logistic Regression / SVM.

```text
src/train_cnn_logmel.py
```

Обучение CNN на log-mel спектрограммах.

```text
src/train_mlp_logmel.py
```

Обучение MLP baseline на log-mel спектрограммах.

```text
EXPERIMENT_LOG.md
```

Главный журнал экспериментов: команды, параметры, метрики, пути к артефактам,
интерпретации и ограничения.

## Короткий вывод

Сейчас есть понятная экспериментальная цепочка:

```text
DUSHA -> balanced subset -> feature extraction -> sklearn baseline -> CNN -> error listening review
```

SVM и logistic regression нужны как простые baseline-модели на компактных
аудиопризнаках. CNN сейчас является основной моделью, потому что лучше работает
с log-mel спектрограммами и дает лучший результат: около `0.8475 macro F1` на
сбалансированном external test subset.

Главные ограничения текущей версии:

```text
задача пока бинарная, без neutral;
negative объединяет angry и sad;
лучший CNN результат проверен на одном seed/split;
нет финальной проверки устойчивости на нескольких random seeds;
qualitative listening review еще нужно разобрать вручную.
```
