# Setup notes

## 1. Базовые зависимости

Сначала можно поставить лёгкую часть:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. PyTorch / torchaudio

PyTorch лучше ставить отдельно под конкретную машину.

### CPU-ноут

Для первых шагов (`inspect_dataset.py`, `make_subset.py`, `train_sklearn.py`) PyTorch может быть не нужен.

Если нужен CPU-only PyTorch, возьми команду с официального PyTorch selector:

- OS: Linux
- Package: Pip
- Language: Python
- Compute Platform: CPU

В текущем CPU-окружении проекта использовалась команда:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Проверенный результат:

```text
torch 2.12.0+cpu
cuda False
```

### `existo` с RTX 3060

Перед установкой:

```bash
nvidia-smi
```

Потом открыть официальный PyTorch selector и выбрать:

- Linux
- Pip
- Python
- CUDA version, подходящую под драйвер

После установки проверить:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Проверенная локальная GPU-конфигурация на `existo` 2026-05-31:

```bash
.venv/bin/python -m pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Проверенный результат:

```text
torch 2.11.0+cu128
torch.cuda.is_available() True
torch.version.cuda 12.8
device NVIDIA GeForce RTX 3060
```

Замечание: эта установка подтягивает CUDA runtime packages и увеличивает `.venv`
примерно до `7.5G`. После установки `pip check` показывал конфликт
`datasets 4.8.5` с `fsspec 2026.4.0`; cached CNN training из `.npz` работал,
но workflows через Hugging Face `datasets` надо перепроверять отдельно.

## 3. Почему не фиксируем CUDA-команду прямо тут

CUDA/PyTorch-команда зависит от драйвера и доступной версии CUDA. Лучше не хардкодить её в проекте, чтобы не поставить не тот wheel.

## 4. DUSHA / audio loading

Hugging Face `datasets` audio column обычно содержит:

- `array`
- `path`
- `sampling_rate`

Для приведения к 16 kHz обычно используется:

```python
from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

Но сначала нужно посмотреть реальную структуру скачанного датасета через `inspect_dataset.py`.
