from __future__ import annotations

import unittest
from pathlib import Path

from infinity.normal_estimation.defaults import (
    DEFAULT_NORMAL_ESTIMATION_CKPT,
    DEFAULT_NORMAL_TOKENIZER_CKPT,
    DEFAULT_HYPERSIM_ROOT,
    DEFAULT_VKITTI2_ROOT,
    LEGACY_NORMAL_TOKENIZER_CKPT,
    REPO_ROOT,
)


class NormalDefaultsTest(unittest.TestCase):
    def test_data_roots_are_repo_relative_absolute_paths(self) -> None:
        self.assertEqual(Path(DEFAULT_HYPERSIM_ROOT), REPO_ROOT / "data" / "hypersim" / "processed" / "hypersim")
        self.assertEqual(Path(DEFAULT_VKITTI2_ROOT), REPO_ROOT / "data" / "VKITTI2")
        self.assertTrue(Path(DEFAULT_HYPERSIM_ROOT).is_absolute())
        self.assertTrue(Path(DEFAULT_VKITTI2_ROOT).is_absolute())

    def test_checkpoint_defaults_are_centralized_repo_relative_paths(self) -> None:
        self.assertEqual(
            Path(DEFAULT_NORMAL_TOKENIZER_CKPT),
            REPO_ROOT / "outputs" / "normal_tokenizer" / "2026-06-03" / "00-39-35" / "checkpoints" / "best_angle_3.5732.pth",
        )
        self.assertEqual(
            Path(LEGACY_NORMAL_TOKENIZER_CKPT),
            REPO_ROOT / "outputs" / "normal_tokenizer" / "2026-05-31" / "15-08-43" / "checkpoints" / "best_angle_6.7867.pth",
        )
        self.assertEqual(
            Path(DEFAULT_NORMAL_ESTIMATION_CKPT),
            REPO_ROOT / "outputs" / "normal_estimation" / "2026-06-01" / "09-27-05" / "checkpoints" / "best_angle_18.5532.pth",
        )


if __name__ == "__main__":
    unittest.main()
