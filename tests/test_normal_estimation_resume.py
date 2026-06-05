from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infinity.normal_estimation.checkpoints import resolve_checkpoint_resume_path


class NormalEstimationResumeTest(unittest.TestCase):
    def test_explicit_last_prefers_step_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            last = ckpt_dir / "last.pth"
            last_step = ckpt_dir / "last_step.pth"
            last.write_bytes(b"last")
            last_step.write_bytes(b"last-step")

            self.assertEqual(
                resolve_checkpoint_resume_path(
                    output_dir=Path(tmpdir),
                    resume_arg=str(last),
                    auto_checkpoint_names=("last_step.pth", "last.pth"),
                    prefer_step_checkpoint_for_last=True,
                ),
                last_step,
            )

    def test_explicit_missing_resume_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "--resume checkpoint not found"):
                resolve_checkpoint_resume_path(
                    output_dir=Path(tmpdir),
                    resume_arg=str(Path(tmpdir) / "missing.pth"),
                    auto_checkpoint_names=("last_step.pth", "last.pth"),
                    prefer_step_checkpoint_for_last=True,
                )

    def test_auto_resume_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(
                resolve_checkpoint_resume_path(
                    output_dir=Path(tmpdir),
                    resume_arg="",
                    auto_checkpoint_names=("last_step.pth", "last.pth"),
                    prefer_step_checkpoint_for_last=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
