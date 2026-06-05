from __future__ import annotations

import unittest

from infinity.normal_estimation.training import require_positive_steps_per_epoch


class NormalTrainingTest(unittest.TestCase):
    def test_require_positive_steps_per_epoch_returns_valid_count(self) -> None:
        self.assertEqual(require_positive_steps_per_epoch(3, context="normal tokenizer training"), 3)

    def test_require_positive_steps_per_epoch_rejects_empty_loader(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "normal estimation training has no training batches"):
            require_positive_steps_per_epoch(0, context="normal estimation training")


if __name__ == "__main__":
    unittest.main()
