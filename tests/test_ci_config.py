from __future__ import annotations

import ast
import fnmatch
import os
import unittest
from pathlib import Path

import yaml

import scripts.check_repo as check_repo


class CiConfigTest(unittest.TestCase):
    def test_github_actions_uses_repository_check_script(self) -> None:
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        config = yaml.load(workflow, Loader=yaml.BaseLoader)
        self.assertIn("scripts/check.sh", workflow)
        self.assertIn("requirements-check.txt", workflow)
        self.assertEqual({"contents": "read"}, config["permissions"])
        self.assertEqual("true", config["concurrency"]["cancel-in-progress"])
        self.assertIn("pull_request", config["on"])
        self.assertEqual(["main", "master"], config["on"]["push"]["branches"])
        job = config["jobs"]["lightweight-tests"]
        self.assertEqual("ubuntu-latest", job["runs-on"])
        self.assertEqual("15", job["timeout-minutes"])
        checkout_step = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
        self.assertEqual("false", checkout_step["with"]["persist-credentials"])
        self.assertEqual("scripts/check.sh", job["steps"][-1]["run"])

    def test_contribution_templates_cover_training_safety(self) -> None:
        pr_template = Path(".github/pull_request_template.md").read_text(encoding="utf-8")
        training_template = Path(".github/ISSUE_TEMPLATE/training_change.md").read_text(encoding="utf-8")
        self.assertIn("Did not start or interrupt active training jobs", pr_template)
        self.assertIn("Does not interrupt currently running jobs", training_template)
        self.assertIn("Dataset or sampler changes", pr_template)
        self.assertIn("generated artifacts, checkpoints, datasets, or other large files", pr_template)
        self.assertIn("machine-local paths", pr_template)
        self.assertIn("unsafe shell execution", pr_template)

    def test_open_source_project_baseline_files_exist(self) -> None:
        for path in [
            ".editorconfig",
            ".gitattributes",
            "CITATION.cff",
            ".pre-commit-config.yaml",
            "pyproject.toml",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            "THIRD_PARTY_NOTICES.md",
            "docs/engineering.md",
            ".github/CODEOWNERS",
            ".github/dependabot.yml",
            ".github/pull_request_template.md",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/training_change.md",
            ".github/ISSUE_TEMPLATE/config.yml",
        ]:
            self.assertTrue(Path(path).is_file(), path)

    def test_issue_template_config_routes_security_privately(self) -> None:
        config = Path(".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8")
        self.assertIn("blank_issues_enabled: false", config)
        self.assertIn("Security vulnerability", config)
        self.assertIn("/security/advisories/new", config)
        self.assertIn("FoundationVision/Infinity", config)

    def test_dependabot_covers_actions_and_pip(self) -> None:
        config = yaml.safe_load(Path(".github/dependabot.yml").read_text(encoding="utf-8"))
        self.assertEqual(2, config["version"])
        updates = {entry["package-ecosystem"]: entry for entry in config["updates"]}
        self.assertEqual({"github-actions", "pip"}, set(updates))
        for ecosystem in ["github-actions", "pip"]:
            self.assertEqual("/", updates[ecosystem]["directory"])
            self.assertEqual("weekly", updates[ecosystem]["schedule"]["interval"])
            self.assertEqual(5, updates[ecosystem]["open-pull-requests-limit"])
        self.assertEqual(
            ["numpy", "pyyaml", "tomli", "textual"],
            updates["pip"]["groups"]["lightweight-checks"]["patterns"],
        )

    def test_contributing_links_security_and_conduct_policies(self) -> None:
        contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("SECURITY.md", contributing)
        self.assertIn("CODE_OF_CONDUCT.md", contributing)
        self.assertIn("docs/engineering.md", contributing)

    def test_security_policy_has_private_reporting_path(self) -> None:
        security = Path("SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("GitHub private vulnerability reporting", security)
        self.assertIn("avoid sharing exploit details in public", security)
        self.assertIn("acknowledge receipt", security)
        for placeholder in ["TODO", "TBD", "example.com", "REPLACE"]:
            self.assertNotIn(placeholder, security)

    def test_third_party_notices_document_external_assets(self) -> None:
        notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        for term in ["Model Weights", "Datasets", "Hypersim", "VKITTI2", "FLAN-T5"]:
            self.assertIn(term, notices)
        self.assertIn("THIRD_PARTY_NOTICES.md", readme)

    def test_readme_links_engineering_guide(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        engineering = Path("docs/engineering.md").read_text(encoding="utf-8")
        docs_link_test = Path("tests/test_docs_links.py").read_text(encoding="utf-8")
        self.assertIn("docs/engineering.md", readme)
        self.assertIn('-v "$PWD":/workspace', readme)
        self.assertNotIn("{your-local-path}", readme)
        self.assertIn("Normal Training Architecture", engineering)
        self.assertIn("Training Safety", engineering)
        self.assertIn("unsafe dynamic execution", engineering)
        self.assertIn("10 MiB", engineering)
        self.assertIn("YAML and CFF", engineering)
        self.assertIn("GitHub issue template front matter", engineering)
        self.assertIn("executable scripts have shebangs", engineering)
        self.assertIn("outside approved script entrypoint directories", engineering)
        self.assertIn("Script Entrypoints", engineering)
        self.assertIn("core project governance files", engineering)
        self.assertIn("committed secret patterns", engineering)
        self.assertIn("private key blocks", engineering)
        self.assertIn("requirement files are deduplicated", engineering)
        self.assertIn("heavyweight training dependencies", engineering)
        self.assertIn("CODE_OF_CONDUCT.md", docs_link_test)
        self.assertIn("CITATION.cff", docs_link_test)
        self.assertIn(".github/ISSUE_TEMPLATE/*.md", docs_link_test)
        self.assertIn(".github/ISSUE_TEMPLATE/*.yml", docs_link_test)

    def test_codeowners_uses_real_repository_owner(self) -> None:
        codeowners = Path(".github/CODEOWNERS").read_text(encoding="utf-8")
        self.assertIn("@XyeaOvO", codeowners)
        self.assertNotIn("@maintainers", codeowners)
        for path in [
            "/.github/workflows/ci.yml",
            "/.github/dependabot.yml",
            "/.pre-commit-config.yaml",
            "/pyproject.toml",
            "/requirements-check.txt",
            "/scripts/check.sh",
            "/scripts/check_repo.py",
            "/tests/test_check_repo.py",
            "/tests/test_ci_config.py",
            "/evaluation/gen_eval/rename.py",
            "/tests/test_eval_tools.py",
            "/README.md",
            "/CONTRIBUTING.md",
            "/SECURITY.md",
            "/CODE_OF_CONDUCT.md",
            "/THIRD_PARTY_NOTICES.md",
            "/CITATION.cff",
            "/LICENSE",
            "/docs/engineering.md",
            "/docs/normal_readme.md",
            "/docs/normal_tokenizer_finetune.md",
            "/.github/pull_request_template.md",
            "/.github/ISSUE_TEMPLATE/bug_report.md",
            "/.github/ISSUE_TEMPLATE/training_change.md",
            "/.github/ISSUE_TEMPLATE/config.yml",
        ]:
            self.assertIn(path, codeowners)

    def test_codeowners_paths_exist(self) -> None:
        missing = []
        for raw_line in Path(".github/CODEOWNERS").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pattern = line.split()[0]
            if pattern == "*":
                continue
            path_pattern = pattern.lstrip("/")
            has_glob = any(char in path_pattern for char in "*?[")
            if has_glob:
                exists = any(fnmatch.fnmatch(str(path), path_pattern) for path in Path(".").rglob("*"))
            else:
                exists = Path(path_pattern).exists()
            if not exists:
                missing.append(pattern)
        self.assertEqual([], missing)

    def test_required_project_files_have_explicit_codeowners(self) -> None:
        codeowners = Path(".github/CODEOWNERS").read_text(encoding="utf-8")
        missing = []
        for raw_path in sorted(check_repo.REQUIRED_PROJECT_FILES):
            owner_pattern = f"/{raw_path}"
            if owner_pattern not in codeowners:
                missing.append(owner_pattern)
        self.assertEqual([], missing)

    def test_check_script_rejects_python_bytecode_caches(self) -> None:
        check_script = Path("scripts/check.sh").read_text(encoding="utf-8")
        check_repo = Path("scripts/check_repo.py").read_text(encoding="utf-8")
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python}"', check_script)
        self.assertIn('"${PYTHON_BIN}" scripts/check_repo.py', check_script)
        self.assertIn('"${PYTHON_BIN}" -m unittest discover -s tests', check_script)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", check_script)
        self.assertIn("git diff --check", check_script)
        self.assertIn("tomllib.load", check_repo)
        self.assertIn("yaml.safe_load", check_repo)
        self.assertIn('{".cff", ".yml", ".yaml"}', check_repo)
        self.assertIn("os.walk", check_repo)
        self.assertIn("dirs[:]", check_repo)
        self.assertIn("interactive debugger call", check_repo)
        self.assertIn("unsafe dynamic execution call", check_repo)
        self.assertIn("subprocess shell=True call", check_repo)
        self.assertIn("os.system call", check_repo)
        self.assertIn("bare except handler", check_repo)
        self.assertIn("ast.NodeVisitor", check_repo)
        self.assertIn("bash -n", check_script)
        self.assertIn("scripts/download-data", check_script)
        self.assertIn("__pycache__", check_repo)
        self.assertIn('"external"', check_repo)
        self.assertIn("forbidden machine-local path", check_repo)
        self.assertIn('"/root/" + "vepfs/Infinity"', check_repo)
        self.assertIn('"/Users/" + "bytedance"', check_repo)
        self.assertIn('"/opt/" + "tiger"', check_repo)
        self.assertIn('EXCLUDED_FILES = {"prompt.md"}', check_repo)
        self.assertIn('".toml"', check_repo)
        self.assertIn('".cff"', check_repo)
        self.assertIn('TEXT_FILE_NAMES = {".editorconfig", ".gitattributes"}', check_repo)
        self.assertNotIn('".github", ".tui", ".venv"', check_repo)
        self.assertIn("trailing whitespace", check_repo)
        self.assertIn("merge conflict marker", check_repo)
        self.assertIn("CRLF line ending", check_repo)
        self.assertIn("missing final newline", check_repo)
        self.assertIn("10 MiB source limit", check_repo)
        self.assertIn("MAX_SOURCE_FILE_SIZE = 10 * 1024 * 1024", check_repo)
        self.assertIn("git\", \"diff\", \"--name-only", check_repo)
        self.assertIn("git\", \"ls-files\", \"--others\"", check_repo)
        self.assertIn("Python bytecode cache directories found", check_repo)
        self.assertIn("check_executable_script_metadata", check_repo)
        self.assertIn("executable script is missing a shebang", check_repo)
        self.assertIn("shebang script is not executable", check_repo)
        self.assertIn("unexpected executable bit outside script entrypoint directories", check_repo)
        self.assertIn("REQUIRED_PROJECT_FILES", check_repo)
        self.assertIn("required project file is missing", check_repo)
        self.assertIn("required project file is empty", check_repo)
        self.assertIn('"LICENSE"', check_repo)
        self.assertIn("SECRET_PATTERNS", check_repo)
        self.assertIn("check_no_committed_secrets", check_repo)
        self.assertIn("possible committed secret", check_repo)
        self.assertIn("check_requirements_files", check_repo)
        self.assertIn("HEAVY_CHECK_DEPENDENCIES", check_repo)
        self.assertIn("duplicate requirement", check_repo)
        self.assertIn("heavyweight dependency is not allowed", check_repo)
        self.assertIn("check_github_issue_templates", check_repo)
        self.assertIn("ISSUE_TEMPLATE_REQUIRED_KEYS", check_repo)
        self.assertIn("missing issue template front matter keys", check_repo)

    def test_check_script_is_directly_executable(self) -> None:
        path = Path("scripts/check.sh")
        self.assertTrue(path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash"))
        self.assertTrue(os.access(path, os.X_OK), "scripts/check.sh must be executable because CI runs it directly")

    def test_check_repo_script_is_directly_executable(self) -> None:
        path = Path("scripts/check_repo.py")
        self.assertTrue(path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python"))
        self.assertTrue(os.access(path, os.X_OK), "scripts/check_repo.py should be runnable as a standalone local check")

    def test_pre_commit_reuses_repository_check_script(self) -> None:
        config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
        contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("repo: local", config)
        self.assertIn("entry: scripts/check.sh", config)
        self.assertIn("pass_filenames: false", config)
        self.assertIn("pre-commit install", contributing)
        self.assertIn("after installing `pre-commit`", contributing)

    def test_source_tree_has_no_live_debugger_breakpoints(self) -> None:
        excluded_dirs = {".git", ".tui", ".venv", "data", "external", "outputs"}
        violations = []
        for path in Path(".").rglob("*.py"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "set_" + "trace(" in stripped or "break" + "point(" in stripped:
                    violations.append(f"{path}:{line_number}")
        self.assertEqual([], violations)

    def test_source_tree_has_no_unsafe_dynamic_or_shell_execution(self) -> None:
        excluded_dirs = {".git", ".tui", ".venv", "data", "external", "outputs"}
        violations = []
        for path in Path(".").rglob("*.py"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
                    violations.append(f"{path}:{node.lineno}: dynamic execution")
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "system"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    violations.append(f"{path}:{node.lineno}: os.system")
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                        violations.append(f"{path}:{node.lineno}: shell=True")
        self.assertEqual([], violations)

    def test_source_tree_has_no_bare_except_handlers(self) -> None:
        excluded_dirs = {".git", ".tui", ".venv", "data", "external", "outputs"}
        violations = []
        for path in Path(".").rglob("*.py"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    violations.append(f"{path}:{node.lineno}: bare except")
        self.assertEqual([], violations)

    def test_source_tree_only_passes_idempotent_file_exceptions(self) -> None:
        excluded_dirs = {".git", ".tui", ".venv", "data", "external", "outputs"}
        allowed = {"FileNotFoundError", "FileExistsError"}
        violations = []
        for path in Path(".").rglob("*.py"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
                    continue
                caught = ast.unparse(node.type) if node.type is not None else "bare"
                if caught not in allowed:
                    violations.append(f"{path}:{node.lineno}: pass-only except {caught}")
        self.assertEqual([], violations)

    def test_gitignore_covers_local_python_state(self) -> None:
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        for pattern in [".venv/", "*.py[cod]", ".pytest_cache/", ".ruff_cache/", ".mypy_cache/"]:
            self.assertIn(pattern, gitignore)
        self.assertIn("!requirements-check.txt", gitignore)
        self.assertIn("prompt.md", gitignore)

    def test_repository_tooling_metadata_is_present(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        gitattributes = Path(".gitattributes").read_text(encoding="utf-8")
        citation = Path("CITATION.cff").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("[tool.pytest.ini_options]", pyproject)
        self.assertIn("[tool.ruff]", pyproject)
        self.assertIn('testpaths = ["tests"]', pyproject)
        self.assertIn("line-length = 120", pyproject)
        self.assertIn("* text=auto eol=lf", gitattributes)
        self.assertIn("*.safetensors binary", gitattributes)
        self.assertIn("*.pth binary", gitattributes)
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn("10.48550/arXiv.2412.04431", citation)
        self.assertIn("CITATION.cff", readme)

    def test_requirement_files_are_deduplicated_and_documented(self) -> None:
        requirements = [
            line.strip()
            for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(requirements), len(set(requirements)))
        check_requirements = [
            line.strip()
            for line in Path("requirements-check.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        check_requirements_text = "\n".join(check_requirements)
        self.assertEqual(len(check_requirements), len(set(check_requirements)))
        contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("textual", check_requirements_text)
        self.assertIn("pyyaml", check_requirements_text)
        self.assertIn("tomli; python_version < \"3.11\"", check_requirements_text)
        for heavy_dependency in ["flash_attn", "torch", "opencv-python", "transformers"]:
            self.assertNotIn(heavy_dependency, check_requirements_text)
        self.assertIn("requirements-check.txt", contributing)
        self.assertIn("flash_attn", contributing)
        self.assertIn("Dependency Sets", contributing)
        self.assertIn("shebang and executable-bit metadata", contributing)
        self.assertIn("helper files outside approved script entrypoint directories", contributing)
        self.assertIn("present and non-empty", contributing)
        self.assertIn("common committed secret patterns", contributing)
        self.assertIn("local secret stores", contributing)
        self.assertIn("Requirement files must stay deduplicated", contributing)
        self.assertIn("opencv-python", contributing)
        self.assertIn("final newlines", contributing)
        self.assertIn("GitHub issue template metadata", contributing)

    def test_8b_helper_scripts_support_python_bin_override(self) -> None:
        for path in ["scripts/infer_8b.sh", "scripts/setup_8b_infer.sh"]:
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python}"', text)
            self.assertNotIn("\npython ", text)

    def test_legacy_inference_scripts_keep_interpreter_overrides(self) -> None:
        infer_script = Path("scripts/infer.sh").read_text(encoding="utf-8")
        eval_script = Path("scripts/eval.sh").read_text(encoding="utf-8")
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python3}"', infer_script)
        self.assertIn('"${PYTHON_BIN}" tools/run_infinity.py', infer_script)
        self.assertIn('python_ext="${PYTHON_BIN:-python3}"', eval_script)
        self.assertIn('pip_ext="${PIP_BIN:-pip3}"', eval_script)
        self.assertIn("HPSV2_OPEN_CLIP_DIR", eval_script)
        self.assertNotIn("/home/tiger", eval_script)
        self.assertNotIn("${pip_ext}install", eval_script)

    def test_inception_weights_path_is_not_machine_local(self) -> None:
        inception = Path("tools/inception.py").read_text(encoding="utf-8")
        rename = Path("evaluation/gen_eval/rename.py").read_text(encoding="utf-8")
        self.assertIn("FID_WEIGHTS_PATH", inception)
        self.assertIn("os.environ.get", inception)
        self.assertNotIn("/mnt/bn", inception)
        self.assertIn("--reference-cache", rename)
        self.assertNotIn("/Users/bytedance", rename)


if __name__ == "__main__":
    unittest.main()
