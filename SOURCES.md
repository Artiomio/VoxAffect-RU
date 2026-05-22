# External references

These are useful references for the project setup and implementation decisions.

## PyTorch install

Official PyTorch local install selector:
https://pytorch.org/get-started/locally/

Use it for choosing CPU vs CUDA builds.

## Hugging Face audio datasets

Audio loading docs:
https://huggingface.co/docs/datasets/v2.4.0/en/audio_load

Audio processing docs:
https://huggingface.co/docs/datasets/audio_process

Useful point: audio columns can include `array`, `path`, `sampling_rate`, and can be resampled with `cast_column("audio", Audio(sampling_rate=16000))`.

## Jupyter notebook format

Official nbformat docs:
https://nbformat.readthedocs.io/en/latest/format_description.html

Useful point: `.ipynb` files are JSON documents with cells and metadata. Diana's JSON notebook can be saved/opened as `.ipynb` if valid.
