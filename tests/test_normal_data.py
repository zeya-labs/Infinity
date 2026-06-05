from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infinity.normal_estimation.data import VKITTI2NormalDataset


class NormalDataTest(unittest.TestCase):
    def test_vkitti2_manifest_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifest is empty"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)

    def test_vkitti2_manifest_reports_bad_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                '{"rgb_path": "rgb.png", "normal_path": "normal.npy", "mask_path": "mask.png"}\n{bad-json}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "line 2"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)

    def test_vkitti2_manifest_reports_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text('{"rgb_path": "rgb.png"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)


if __name__ == "__main__":
    unittest.main()
