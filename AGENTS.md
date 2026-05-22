# Codex Project Directives

These instructions are for Codex and other coding agents working in this
repository.

## Experiment Log Is Mandatory

Treat `EXPERIMENT_LOG.md` as the project laboratory journal and as a future
source for thesis text.

After every meaningful experiment or project step, update `EXPERIMENT_LOG.md`
with:

- date;
- goal or hypothesis;
- input data;
- commands/scripts used;
- important parameters;
- metrics and generated artifact paths;
- short interpretation;
- limitations;
- next step.

Do not rely on generated files in `data/` or `artifacts/` as the only record of
an experiment. Those directories are intentionally git-ignored. Important
results must be summarized in `EXPERIMENT_LOG.md`.

Prefer factual, reproducible entries over narrative prose. The log should make
it possible to reconstruct thesis sections such as dataset description,
methodology, baseline models, results, limitations, and future work.

## Experiment Run Naming

Every experiment that produces metrics must use a distinct `run-name` and write
artifacts into a dedicated subdirectory, for example:

```text
artifacts/sklearn_logreg_binary_300/
artifacts/sklearn_random_forest_binary_300/
artifacts/sklearn_svm_rbf_binary_300/
```

Do not let different approaches overwrite the same `metrics.json`,
`classification_report.txt`, `confusion_matrix.png`, or `predictions.csv`.

When adding an experiment to `EXPERIMENT_LOG.md`, include a compact comparison
table with at least:

```text
run | model | data/subset | key parameters | accuracy | macro F1 | interpretation
```

This table is the canonical source for later thesis text about experimental
comparison.

## Current Project Direction

The current MVP path is:

```text
DUSHA -> inspect -> balanced subset -> sklearn baseline -> validation sample -> CNN
```

Keep the first baseline CPU-friendly. Do not introduce PyTorch/CUDA,
transformers, Wav2Vec2, or HuBERT until the simple sklearn pipeline and human
validation sample are in good shape.

## Local Data Policy

Do not commit:

- `.venv/`
- `data/`
- generated `artifacts/`
- `src/__pycache__/`

If a generated artifact matters for the thesis or for reproducibility, record
its path, parameters, and key contents in `EXPERIMENT_LOG.md`.
