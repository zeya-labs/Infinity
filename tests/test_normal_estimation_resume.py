from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from tools.train_normal_estimation import resolve_resume_path


class NormalEstimationResumeTest(unittest.TestCase):
    def test_explicit_last_prefers_step_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir()
            last = ckpt_dir / "last.pth"
            last_step = ckpt_dir / "last_step.pth"
            last.write_bytes(b"last")
            last_step.write_bytes(b"last-step")

            args = argparse.Namespace(output_dir=tmpdir, resume=str(last))

            self.assertEqual(resolve_resume_path(args), last_step)

    def test_explicit_missing_resume_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(output_dir=tmpdir, resume=str(Path(tmpdir) / "missing.pth"))

            with self.assertRaisesRegex(FileNotFoundError, "--resume checkpoint not found"):
                resolve_resume_path(args)

    def test_auto_resume_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(output_dir=tmpdir, resume="")

            self.assertIsNone(resolve_resume_path(args))


if __name__ == "__main__":
    unittest.main()
