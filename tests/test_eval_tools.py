from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.gen_eval import rename


class EvalToolsTest(unittest.TestCase):
    def test_rename_remaps_source_values_to_reference_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.json"
            source = root / "source.json"
            output = root / "nested" / "prompt_rewrite_cache.json"
            reference.write_text(json.dumps({"prompt-a": "old", "prompt-b": "old"}), encoding="utf-8")
            source.write_text(json.dumps({"bad-a": "rewrite-a", "bad-b": "rewrite-b"}), encoding="utf-8")

            with mock.patch(
                "sys.argv",
                [
                    "rename.py",
                    "--reference-cache",
                    str(reference),
                    "--source-cache",
                    str(source),
                    "--output",
                    str(output),
                ],
            ):
                rename.main()

            self.assertEqual({"prompt-a": "rewrite-a", "prompt-b": "rewrite-b"}, json.loads(output.read_text()))
            self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_rename_rejects_mismatched_cache_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.json"
            source = root / "source.json"
            output = root / "prompt_rewrite_cache.json"
            reference.write_text(json.dumps({"prompt-a": "old", "prompt-b": "old"}), encoding="utf-8")
            source.write_text(json.dumps({"bad-a": "rewrite-a"}), encoding="utf-8")

            with mock.patch(
                "sys.argv",
                [
                    "rename.py",
                    "--reference-cache",
                    str(reference),
                    "--source-cache",
                    str(source),
                    "--output",
                    str(output),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "cache sizes differ"):
                    rename.main()

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
