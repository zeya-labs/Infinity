#!/usr/bin/env python
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


EXCLUDED_DIRS = {".git", ".tui", ".venv", "data", "external", "outputs"}
PATH_SCAN_EXCLUDED_DIRS = EXCLUDED_DIRS | {"tests"}
EXCLUDED_FILES = {"prompt.md"}
TEXT_EXTENSIONS = {".cff", ".md", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml"}
TEXT_FILE_NAMES = {".editorconfig", ".gitattributes"}
MAX_SOURCE_FILE_SIZE = 10 * 1024 * 1024
FORBIDDEN_MACHINE_PATHS = (
    "/root/" + "vepfs/Infinity",
    "/home/" + "tiger",
    "/mnt/" + "bn",
    "/opt/" + "tiger",
    "/Users/" + "bytedance",
)
CONFLICT_MARKERS = ("<<<<<<< ", "=======\n", ">>>>>>> ")
EXECUTABLE_SCRIPT_DIRS = {"scripts"}
SCRIPT_EXTENSIONS = {".py", ".sh"}
REQUIRED_PROJECT_FILES = {
    ".editorconfig",
    ".gitattributes",
    ".github/CODEOWNERS",
    ".github/workflows/ci.yml",
    ".pre-commit-config.yaml",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/engineering.md",
    "pyproject.toml",
    "requirements-check.txt",
    "scripts/check.sh",
    "scripts/check_repo.py",
}
HEAVY_CHECK_DEPENDENCIES = {"flash_attn", "opencv-python", "torch", "transformers"}
SECRET_PATTERNS = (
    ("AWS access key", re.compile("AKIA" + r"[0-9A-Z]{16}")),
    ("Google API key", re.compile("AIza" + r"[0-9A-Za-z_-]{35}")),
    ("GitHub personal access token", re.compile("ghp_" + r"[A-Za-z0-9]{36,}")),
    ("GitHub fine-grained token", re.compile("github_pat_" + r"[A-Za-z0-9_]{40,}")),
    ("Hugging Face token", re.compile("hf_" + r"[A-Za-z0-9]{30,}")),
    ("OpenAI API key", re.compile("sk-" + r"[A-Za-z0-9_-]{32,}")),
    ("private key block", re.compile("-----BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
)
ISSUE_TEMPLATE_REQUIRED_KEYS = {"about", "assignees", "labels", "name", "title"}


def iter_files(excluded_dirs: set[str] = EXCLUDED_DIRS) -> list[Path]:
    files = []
    for root, dirs, names in os.walk("."):
        dirs[:] = [name for name in dirs if name not in excluded_dirs]
        for name in names:
            files.append(Path(root) / name)
    return sorted(files)


def iter_dirs(excluded_dirs: set[str] = EXCLUDED_DIRS) -> list[Path]:
    found = []
    for root, dirs, _names in os.walk("."):
        dirs[:] = [name for name in dirs if name not in excluded_dirs]
        found.extend(Path(root) / name for name in dirs)
    return sorted(found)


def is_text_file(path: Path) -> bool:
    return path.suffix in TEXT_EXTENSIONS or path.name in TEXT_FILE_NAMES or path.name.startswith(".env")


def changed_and_untracked_files() -> list[Path]:
    changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    return [Path(raw_path) for raw_path in sorted(set(changed + untracked))]


def requirement_entries(path: Path) -> list[str]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def requirement_name(entry: str) -> str:
    name = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|;|\[)", entry, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def markdown_front_matter(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("missing YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("unterminated YAML front matter") from exc
    front_matter = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(front_matter, dict):
        raise ValueError("YAML front matter must be a mapping")
    return front_matter


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []

    def add(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"{self.path}:{getattr(node, 'lineno', 0)}: {message}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in {"eval", "exec"}:
                self.add(node, "unsafe dynamic execution call")
            elif func.id == "breakpoint":
                self.add(node, "interactive debugger call")
        elif isinstance(func, ast.Attribute):
            if func.attr == "set_trace":
                self.add(node, "interactive debugger call")
            if func.attr == "system" and isinstance(func.value, ast.Name) and func.value.id == "os":
                self.add(node, "os.system call")
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self.add(node, "subprocess shell=True call")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.add(node, "bare except handler")
        self.generic_visit(node)


def check_pyproject() -> None:
    with open("pyproject.toml", "rb") as config:
        tomllib.load(config)


def check_yaml() -> None:
    for path in iter_files():
        if path.suffix not in {".cff", ".yml", ".yaml"}:
            continue
        with path.open("r", encoding="utf-8") as config:
            yaml.safe_load(config)


def check_github_issue_templates() -> None:
    violations = []
    for path in sorted(Path(".github/ISSUE_TEMPLATE").glob("*.md")):
        try:
            front_matter = markdown_front_matter(path)
        except ValueError as exc:
            violations.append(f"{path}: {exc}")
            continue
        missing = sorted(ISSUE_TEMPLATE_REQUIRED_KEYS - set(front_matter))
        if missing:
            violations.append(f"{path}: missing issue template front matter keys: {', '.join(missing)}")
    if violations:
        raise SystemExit("\n".join(violations))


def check_python_sources() -> None:
    violations = []
    for path in iter_files():
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source, filename=str(path))
        visitor = SafetyVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    if violations:
        raise SystemExit("\n".join(violations))


def check_machine_local_paths() -> None:
    violations = []
    for path in iter_files(PATH_SCAN_EXCLUDED_DIRS):
        if path.name in EXCLUDED_FILES or not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in FORBIDDEN_MACHINE_PATHS:
            if marker in text:
                violations.append(f"{path}: contains forbidden machine-local path {marker}")
    if violations:
        raise SystemExit("\n".join(violations))


def check_large_changed_files() -> None:
    violations = []
    for path in changed_and_untracked_files():
        if not path.is_file() or any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        size = path.stat().st_size
        if size > MAX_SOURCE_FILE_SIZE:
            violations.append(f"{path}: file is {size} bytes, above the 10 MiB source limit")
    if violations:
        raise SystemExit("\n".join(violations))


def check_changed_text_format() -> None:
    violations = []
    for path in changed_and_untracked_files():
        if not path.is_file() or path.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if not is_text_file(path):
            continue
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            violations.append(f"{path}: missing final newline")
        for line_number, line in enumerate(data.decode("utf-8", errors="ignore").splitlines(True), start=1):
            if line.endswith("\r\n"):
                violations.append(f"{path}:{line_number}: CRLF line ending")
            if line.rstrip("\r\n").rstrip(" \t") != line.rstrip("\r\n"):
                violations.append(f"{path}:{line_number}: trailing whitespace")
            if line in CONFLICT_MARKERS or line.startswith(("<<<<<<< ", ">>>>>>> ")):
                violations.append(f"{path}:{line_number}: merge conflict marker")
    if violations:
        raise SystemExit("\n".join(violations))


def check_no_pycache() -> None:
    pycache_dirs = [path for path in iter_dirs() if path.name == "__pycache__"]
    if pycache_dirs:
        print("Python bytecode cache directories found. Remove __pycache__ before committing.", file=sys.stderr)
        for path in pycache_dirs:
            print(path, file=sys.stderr)
        raise SystemExit(1)


def check_executable_script_metadata() -> None:
    violations = []
    for path in iter_files():
        if not path.parts or path.suffix not in SCRIPT_EXTENSIONS:
            continue
        is_executable = os.access(path, os.X_OK)
        if path.parts[0] not in EXECUTABLE_SCRIPT_DIRS:
            if is_executable:
                violations.append(f"{path}: unexpected executable bit outside script entrypoint directories")
            continue
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        has_shebang = bool(first_line and first_line[0].startswith("#!"))
        if is_executable and not has_shebang:
            violations.append(f"{path}: executable script is missing a shebang")
        if has_shebang and not is_executable:
            violations.append(f"{path}: shebang script is not executable")
    if violations:
        raise SystemExit("\n".join(violations))


def check_required_project_files() -> None:
    violations = []
    for raw_path in sorted(REQUIRED_PROJECT_FILES):
        path = Path(raw_path)
        if not path.is_file():
            violations.append(f"{path}: required project file is missing")
        elif path.stat().st_size == 0:
            violations.append(f"{path}: required project file is empty")
    if violations:
        raise SystemExit("\n".join(violations))


def check_requirements_files() -> None:
    violations = []
    for raw_path in ("requirements.txt", "requirements-check.txt"):
        path = Path(raw_path)
        entries = requirement_entries(path)
        duplicates = sorted({entry for entry in entries if entries.count(entry) > 1})
        for duplicate in duplicates:
            violations.append(f"{path}: duplicate requirement {duplicate}")
    check_entries = requirement_entries(Path("requirements-check.txt"))
    check_names = {requirement_name(entry) for entry in check_entries}
    for package in sorted(HEAVY_CHECK_DEPENDENCIES & check_names):
        violations.append(f"requirements-check.txt: heavyweight dependency is not allowed in lightweight checks: {package}")
    if violations:
        raise SystemExit("\n".join(violations))


def check_no_committed_secrets() -> None:
    violations = []
    for path in iter_files(PATH_SCAN_EXCLUDED_DIRS):
        if path.name in EXCLUDED_FILES or not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{path}:{line_number}: possible committed secret ({label})")
    if violations:
        raise SystemExit("\n".join(violations))


def main() -> None:
    check_required_project_files()
    check_requirements_files()
    check_pyproject()
    check_yaml()
    check_github_issue_templates()
    check_python_sources()
    check_no_committed_secrets()
    check_machine_local_paths()
    check_large_changed_files()
    check_changed_text_format()
    check_no_pycache()
    check_executable_script_metadata()


if __name__ == "__main__":
    main()
