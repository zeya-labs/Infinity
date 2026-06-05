from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT_DOCS = [
    Path("README.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("CITATION.cff"),
]
DOC_GLOBS = [
    "docs/*.md",
    ".github/*.md",
    ".github/ISSUE_TEMPLATE/*.md",
    ".github/ISSUE_TEMPLATE/*.yml",
    ".github/ISSUE_TEMPLATE/*.yaml",
]


def documentation_files() -> list[Path]:
    docs = list(ROOT_DOCS)
    for pattern in DOC_GLOBS:
        docs.extend(sorted(Path(".").glob(pattern)))
    return sorted({path for path in docs if path.is_file()})


class DocsLinksTest(unittest.TestCase):
    def test_local_markdown_links_point_to_existing_files(self) -> None:
        patterns = [
            re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)"),
            re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
            re.compile(r"""(?:src|href)=["']([^"']+)["']"""),
            re.compile(r"^\s*url:\s*([^\s#]+)", re.MULTILINE),
        ]
        missing = []
        for source in documentation_files():
            text = source.read_text(encoding="utf-8")
            for pattern in patterns:
                for match in pattern.finditer(text):
                    target = match.group(1).split("#", 1)[0].strip().strip("\"'")
                    if not target or "://" in target or target.startswith("mailto:"):
                        continue
                    target_path = (source.parent / unquote(target)).resolve()
                    if not target_path.exists():
                        missing.append(f"{source}: {target}")
        self.assertEqual([], missing)

    def test_link_check_covers_governance_and_github_templates(self) -> None:
        docs = documentation_files()
        for path in [
            Path("CODE_OF_CONDUCT.md"),
            Path("CITATION.cff"),
            Path(".github/pull_request_template.md"),
            Path(".github/ISSUE_TEMPLATE/bug_report.md"),
            Path(".github/ISSUE_TEMPLATE/training_change.md"),
            Path(".github/ISSUE_TEMPLATE/config.yml"),
        ]:
            self.assertIn(path, docs)


if __name__ == "__main__":
    unittest.main()
