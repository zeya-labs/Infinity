#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONDONTWRITEBYTECODE=1

git diff --check

"${PYTHON_BIN}" scripts/check_repo.py

"${PYTHON_BIN}" -m unittest discover -s tests

"${PYTHON_BIN}" -m ruff check --select F821,B018,B905 .

while IFS= read -r script; do
  bash -n "${script}"
done < <(find scripts -path 'scripts/download-data' -prune -o -type f -name '*.sh' -print | sort)
