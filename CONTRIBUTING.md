# Contributing

## Local Checks

Run the lightweight repository checks before submitting changes that touch normal training, TUI command builders, or
shell entrypoints:

```bash
scripts/check.sh
```

For a fresh lightweight check environment, install the CPU/check-only dependencies first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r requirements-check.txt
```

The check script compiles repository Python sources without importing training entrypoints, runs unit tests under
`tests/`, checks shell syntax for repository launch scripts, validates `pyproject.toml` plus YAML/CFF metadata, rejects
unsafe shell/dynamic execution patterns, checks text formatting, validates GitHub issue template metadata, and rejects
source-tree Python bytecode caches.
It also verifies that directly runnable scripts keep their shebang and executable-bit metadata in sync.
Python and shell helper files outside approved script entrypoint directories should not be executable.
Core project files such as `LICENSE`, `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, and CI metadata must remain
present and non-empty.
The check gate also rejects common committed secret patterns, including API keys, service tokens, cloud access keys, and
private key blocks.
Requirement files must stay deduplicated, and `requirements-check.txt` must not include heavyweight training
dependencies such as `torch`, `transformers`, `flash_attn`, or `opencv-python`.

Set `PYTHON_BIN=/path/to/python` when the desired interpreter is not the default `python` on `PATH`.

Use LF line endings, final newlines, and no trailing whitespace in changed text files. Keep generated files out of
commits. The repository-level `.editorconfig` captures the expected basic formatting for editors that support it.

Keep credentials in environment variables or local secret stores. Do not commit real `.env` files, API keys, Hugging
Face tokens, GitHub tokens, cloud access keys, or private keys.

Optional local pre-commit integration is available after installing `pre-commit` in your development environment:

```bash
pre-commit install
```

The hook reuses `scripts/check.sh`.

## Dependency Sets

`requirements.txt` is the full training and inference dependency set. It includes GPU- and platform-sensitive packages
such as `flash_attn`, so it is not installed by the lightweight CI gate.

`requirements-check.txt` is intentionally small and is only for repository checks that should run without datasets,
checkpoints, or GPUs.

## Running Training Jobs

Do not interrupt active training jobs while making code changes. Already-running Python training processes keep their
loaded modules in memory; edits to files on disk affect only newly started jobs.

For normal training, the default mixed dataset policy is documented in:

- `docs/normal_readme.md`
- `docs/normal_tokenizer_finetune.md`

Repository engineering conventions are documented in `docs/engineering.md`.

## Generated Files

Keep generated outputs, checkpoints, datasets, local TUI jobs, and Python bytecode out of commits. The repository
`.gitignore` already covers the common runtime outputs.

If a local experiment produces a new artifact that is useful for future contributors, document how to reproduce it
instead of committing the artifact itself.

Changed or untracked source files above 10 MiB are rejected by `scripts/check.sh`; publish large assets outside git and
link to them from documentation.

## Security

Report suspected vulnerabilities privately. See `SECURITY.md` for the current reporting policy.

## Conduct

Project communication is covered by `CODE_OF_CONDUCT.md`.
