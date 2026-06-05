from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.check_repo as check_repo


class CheckRepoTest(unittest.TestCase):
    def test_required_project_files_reject_missing_and_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            present = root / "present.txt"
            present.write_text("ok\n", encoding="utf-8")
            empty = root / "empty.txt"
            empty.touch()
            with mock.patch.object(
                check_repo,
                "REQUIRED_PROJECT_FILES",
                {"present.txt", "empty.txt", "missing.txt"},
            ):
                try:
                    os.chdir(root)
                    with self.assertRaises(SystemExit) as raised:
                        check_repo.check_required_project_files()
                finally:
                    os.chdir(cwd)

        message = str(raised.exception)
        self.assertIn("empty.txt: required project file is empty", message)
        self.assertIn("missing.txt: required project file is missing", message)

    def test_secret_scanner_rejects_high_confidence_tokens_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            source = root / "source.py"
            source.write_text(
                "\n".join(
                    [
                        "api_key = openai_ak",
                        "aws = 'AKIA0123456789ABCDEF'",
                        "openai = 'sk-" + "a" * 32 + "'",
                        "hf = 'hf_" + "b" * 30 + "'",
                        "github = 'ghp_" + "c" * 36 + "'",
                        "private_key = '-----BEGIN PRIVATE KEY-----'",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                os.chdir(root)
                with self.assertRaises(SystemExit) as raised:
                    check_repo.check_no_committed_secrets()
            finally:
                os.chdir(cwd)

        message = str(raised.exception)
        self.assertNotIn("api_key = openai_ak", message)
        self.assertIn("possible committed secret (AWS access key)", message)
        self.assertIn("possible committed secret (OpenAI API key)", message)
        self.assertIn("possible committed secret (Hugging Face token)", message)
        self.assertIn("possible committed secret (GitHub personal access token)", message)
        self.assertIn("possible committed secret (private key block)", message)

    def test_requirements_check_rejects_duplicates_and_heavy_check_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("numpy\nnumpy\n", encoding="utf-8")
            (root / "requirements-check.txt").write_text("pyyaml\ntorch==2.6.0\ntransformers\n", encoding="utf-8")
            try:
                os.chdir(root)
                with self.assertRaises(SystemExit) as raised:
                    check_repo.check_requirements_files()
            finally:
                os.chdir(cwd)

        message = str(raised.exception)
        self.assertIn("requirements.txt: duplicate requirement numpy", message)
        self.assertIn("requirements-check.txt: heavyweight dependency is not allowed", message)
        self.assertIn("torch", message)
        self.assertIn("transformers", message)

    def test_changed_text_format_rejects_missing_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            source = root / "source.py"
            source.write_text("print('missing newline')", encoding="utf-8")
            try:
                os.chdir(root)
                with mock.patch.object(check_repo, "changed_and_untracked_files", return_value=[source]):
                    with self.assertRaises(SystemExit) as raised:
                        check_repo.check_changed_text_format()
            finally:
                os.chdir(cwd)

        self.assertIn("source.py: missing final newline", str(raised.exception))

    def test_github_issue_templates_require_valid_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            template_dir = root / ".github" / "ISSUE_TEMPLATE"
            template_dir.mkdir(parents=True)
            (template_dir / "missing_front_matter.md").write_text("## Body\n", encoding="utf-8")
            (template_dir / "missing_keys.md").write_text(
                "---\nname: Broken\nabout: Missing labels\n---\n\n## Body\n",
                encoding="utf-8",
            )
            try:
                os.chdir(root)
                with self.assertRaises(SystemExit) as raised:
                    check_repo.check_github_issue_templates()
            finally:
                os.chdir(cwd)

        message = str(raised.exception)
        self.assertIn("missing_front_matter.md: missing YAML front matter", message)
        self.assertIn("missing_keys.md: missing issue template front matter keys", message)
        self.assertIn("assignees", message)
        self.assertIn("labels", message)
        self.assertIn("title", message)

    def test_executable_metadata_rejects_executable_python_outside_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            root = Path(tmpdir)
            tool = root / "tools" / "helper.py"
            tool.parent.mkdir()
            tool.write_text("print('helper')\n", encoding="utf-8")
            tool.chmod(0o755)
            try:
                os.chdir(root)
                with self.assertRaises(SystemExit) as raised:
                    check_repo.check_executable_script_metadata()
            finally:
                os.chdir(cwd)

        self.assertIn("tools/helper.py: unexpected executable bit outside script entrypoint directories", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
