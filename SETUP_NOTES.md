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
