from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from infinity.normal_estimation.checkpoints import resolve_checkpoint_resume_path
from tools.train_normal_tokenizer import save_checkpoint


class NormalTokenizerCheckpointTest(unittest.TestCase):
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

    def test_save_checkpoint_uses_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            args = argparse.Namespace(example=True)

            with (
                mock.patch("tools.train_normal_tokenizer.torch.save") as torch_save,
                mock.patch("tools.train_normal_tokenizer.os.replace") as os_replace,
            ):
                ckpt_path = save_checkpoint(
                    Path(tmpdir),
                    model,
                    optimizer,
                    scheduler=None,
                    epoch=1,
                    step=2,
                    best_val_angle=3.0,
                    args=args,
                    tag="last",
                )

            expected_path = Path(tmpdir) / "checkpoints" / "last.pth"
            self.assertEqual(ckpt_path, expected_path)
            saved_path = torch_save.call_args.args[1]
            self.assertEqual(saved_path.parent, expected_path.parent)
            self.assertTrue(saved_path.name.startswith(".last.pth.tmp."))
            os_replace.assert_called_once_with(saved_path, expected_path)


if __name__ == "__main__":
    unittest.main()
