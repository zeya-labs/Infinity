from __future__ import annotations

import unittest
from pathlib import Path

from infinity.normal_estimation.defaults import (
    DEFAULT_HYPERSIM_ROOT,
    DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS,
    DEFAULT_NORMAL_TRAIN_DATASETS,
    DEFAULT_VKITTI2_ROOT,
)


class NormalShellEntrypointTest(unittest.TestCase):
    def assert_uses_python_defaults(self, path: str) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("from infinity.normal_estimation.defaults import", text)
        self.assertIn('DATA_ROOT_DEFAULT="${NORMAL_DATA_ROOT:-${DEFAULT_HYPERSIM_ROOT}}"', text)
        self.assertIn('TRAIN_DATASETS_DEFAULT="${NORMAL_TRAIN_DATASETS:-${DEFAULT_TRAIN_DATASETS}}"', text)
        self.assertIn('TRAIN_DATASET_WEIGHTS_DEFAULT="${NORMAL_TRAIN_DATASET_WEIGHTS:-${DEFAULT_TRAIN_DATASET_WEIGHTS}}"', text)
        self.assertIn('VKITTI2_ROOT_DEFAULT="${NORMAL_VKITTI2_ROOT:-${DEFAULT_VKITTI2_ROOT}}"', text)
        self.assertNotIn(f'NORMAL_DATA_ROOT:-{DEFAULT_HYPERSIM_ROOT}', text)
        self.assertNotIn(f'NORMAL_TRAIN_DATASETS:-{DEFAULT_NORMAL_TRAIN_DATASETS}', text)
        self.assertNotIn(f'NORMAL_TRAIN_DATASET_WEIGHTS:-{DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS}', text)
        self.assertNotIn(f'NORMAL_VKITTI2_ROOT:-{DEFAULT_VKITTI2_ROOT}', text)
        self.assertNotIn("TOKEN_CACHE_" + "MEMORY", text)
        self.assertNotIn("--token-cache-" + "memory", text)

    def test_train_normal_uses_python_defaults(self) -> None:
        self.assert_uses_python_defaults("scripts/train_normal.sh")

    def test_train_normal_tokenizer_uses_python_defaults(self) -> None:
        self.assert_uses_python_defaults("scripts/train_normal_tokenizer.sh")


if __name__ == "__main__":
    unittest.main()
