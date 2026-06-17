# Repository Guidelines

## Project Structure & Module Organization

Core packages live in `infinity/`: models in `infinity/models/`, datasets in `infinity/dataset/`, normal-estimation code in `infinity/normal_estimation/`, tokenizer finetuning in `infinity/tokenizer_finetune/`, and shared helpers in `infinity/utils/`. Root entrypoints include `train.py`, `trainer.py`, `predict.py`, and `tui.py`. Scripts are in `scripts/`, research and data-prep commands in `tools/`, evaluation code in `evaluation/`, docs in `docs/`, assets in `assets/`, and tests in `tests/`.

## Build, Test, and Development Commands

- `source .venv/bin/activate`: use the repository-local virtual environment.
- `python -m pip install -r requirements.txt`: install the full training/inference dependency set.
- `python -m pip install -r requirements-check.txt`: install lightweight check dependencies after installing CPU torch as documented in `CONTRIBUTING.md`.
- `scripts/check.sh`: run the repository gate: whitespace, metadata, unit tests, selected Ruff rules, shell syntax, secret checks, and artifact safeguards.
- `PYTHON_BIN=/path/to/python scripts/check.sh`: run the same gate with a specific interpreter.
- `python -m unittest discover -s tests`: run the unit tests directly.
- `python -m ruff check .` and `python -m ruff format .`: lint and format Python files.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code. Ruff uses a 120-character line length and checks Pyflakes, pycodestyle, import sorting, pyupgrade, bugbear, and simplification rules. `.editorconfig` requires UTF-8, LF endings, final newlines, and spaces: 4 for Python, shell, Markdown, TOML, and most config files; 2 for YAML. Name tests `test_*.py` and test functions `test_*`. Prefer shared helpers under `infinity/normal_estimation/` and `infinity/utils/`.

## Testing Guidelines

Tests use `unittest` discovery with pytest-compatible naming in `pyproject.toml`. Add focused regression tests in `tests/` for training defaults, resume/checkpoint behavior, data loading, command builders, shell entrypoints, and repository checks. Lightweight tests must not require GPUs, datasets, downloaded checkpoints, or full training dependencies.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects, for example `Update normal eval and training workflows` and `Remove legacy normal tokenizer default`; avoid vague `update`-only messages. PRs should complete the template: summary, validation, training impact, dataset/sampler changes, checkpoint compatibility, and expected GPU/runtime impact. Confirm `scripts/check.sh` ran and no generated artifacts, secrets, machine-local paths, or large files were added.

## Security & Configuration Tips

Keep credentials in environment variables or local secret stores. Do not commit `.env` files, API keys, Hugging Face tokens, cloud keys, private keys, checkpoints, datasets, outputs, local TUI state, virtual environments, or oversized files. Document new environment variables, data roots, checkpoint paths, or operational defaults in docs and PR descriptions.
