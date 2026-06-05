from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infinity.normal_estimation.checkpoints import atomic_torch_save, resolve_checkpoint_resume_path


class NormalCheckpointTest(unittest.TestCase):
    def test_explicit_missing_resume_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "--resume checkpoint not found"):
                resolve_checkpoint_resume_path(
                    output_dir=Path(tmpdir),
                    resume_arg=str(Path(tmpdir) / "missing.pth"),
                    auto_checkpoint_names=("last.pth",),
                )

    def test_auto_resume_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(
                resolve_checkpoint_resume_path(
                    output_dir=Path(tmpdir),
                    resume_arg="",
                    auto_checkpoint_names=("last.pth",),
                )
            )

    def test_atomic_torch_save_uses_temp_file_and_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints" / "last.pth"

            with (
                mock.patch("infinity.normal_estimation.checkpoints.torch.save") as torch_save,
                mock.patch("infinity.normal_estimation.checkpoints.os.replace") as os_replace,
            ):
                atomic_torch_save({"step": 1}, checkpoint_path)

            saved_path = torch_save.call_args.args[1]
            self.assertEqual(saved_path.parent, checkpoint_path.parent)
            self.assertTrue(saved_path.name.startswith(".last.pth.tmp."))
            os_replace.assert_called_once_with(saved_path, checkpoint_path)


if __name__ == "__main__":
    unittest.main()
