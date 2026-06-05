# Engineering Guide

This repository is research-oriented, but changes should still preserve predictable local development, review, and
training behavior.

## Checks

Use the lightweight check gate before changing normal training, TUI tasks, shell entrypoints, or repository metadata:

```bash
scripts/check.sh
```

The check gate:

- compiles repository Python sources without importing training entrypoints
- validates `pyproject.toml`
- validates YAML and CFF metadata files
- validates GitHub issue template front matter
- runs unit tests under `tests/`
- validates shell syntax for repository launch scripts
- validates that executable scripts have shebangs and shebang scripts have executable bits
- rejects executable bits on Python and shell files outside approved script entrypoint directories
- validates that core project governance files are present and non-empty
- validates that requirement files are deduplicated and lightweight checks avoid heavyweight training dependencies
- rejects interactive debuggers, unsafe dynamic execution, `os.system(...)`, and `shell=True`
- rejects high-confidence committed secret patterns such as API keys and private key blocks
- rejects machine-local paths in source, scripts, docs, and config files
- rejects changed text files with CRLF endings, trailing whitespace, merge conflict markers, or missing final newlines
- rejects changed or untracked source files larger than 10 MiB
- rejects source-tree `__pycache__` directories

Set `PYTHON_BIN=/path/to/python` to run checks with a specific interpreter.

## Dependency Boundaries

Keep the lightweight check environment separate from the full training environment:

- `requirements.txt` is for full training and inference installs.
- `requirements-check.txt` is for CI and local checks that should not require datasets, checkpoints, CUDA extensions, or
  GPUs.
- CI installs CPU torch explicitly before `requirements-check.txt` so normal-data tests can import torch without pulling
  in the full GPU stack.
- Do not add `torch`, `transformers`, `flash_attn`, or `opencv-python` to `requirements-check.txt`; keep those in the
  full training environment.

## Normal Training Architecture

Normal-related shared code lives under `infinity/normal_estimation/`.

- `defaults.py` owns repo-relative defaults and environment-overridable checkpoint paths.
- `data.py` owns Hypersim and VKITTI2 dataset readers.
- `sampling.py` owns mixed-dataset construction and DDP-safe grouped sampling.
- `tools/train_normal_estimation.py` and `tools/train_normal_tokenizer.py` should reuse shared helpers instead of
  duplicating dataset, sampler, or SwanLab behavior.

Mixed normal training uses positive-integer dataset step ratios. Per-rank batches and global DDP steps stay homogeneous
by dataset and target resolution.

## Training Safety

Code edits on disk do not change already-running Python training processes. They affect newly started runs only.

Do not change launch defaults, checkpoint paths, dataset roots, or resume behavior without documenting the operational
impact in the PR description and adding a lightweight regression test when feasible.

## Local State

Generated data, checkpoints, TUI job state, local virtual environments, and Python bytecode should remain outside
commits. The repository `.gitignore` covers common local state; add narrowly scoped ignore rules when introducing a new
generated artifact family.

Do not commit service credentials, API keys, Hugging Face tokens, GitHub tokens, cloud access keys, private keys, or
`.env` files with real secrets. Use environment variables or local secret stores for credentials.

Large artifacts should be hosted outside git and documented with reproduction or download instructions. The lightweight
check gate rejects changed or untracked files above 10 MiB to catch accidental checkpoint, dataset, or binary commits.

## Script Entrypoints

Shell or Python files under `scripts/` that are intended to run directly must start with a shebang and be executable.
Keep helper-only code non-executable unless it is a documented command entrypoint.
