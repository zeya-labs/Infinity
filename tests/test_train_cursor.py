from __future__ import annotations

import unittest

from train import next_train_cursor


class TrainCursorTest(unittest.TestCase):
    def test_mid_epoch_checkpoint_resumes_at_next_iteration(self) -> None:
        self.assertEqual(next_train_cursor(ep=3, it=4, iters_train=10), (3, 5))

    def test_end_epoch_checkpoint_resumes_at_next_epoch(self) -> None:
        self.assertEqual(next_train_cursor(ep=3, it=9, iters_train=10), (4, 0))

    def test_rejects_empty_epoch(self) -> None:
        with self.assertRaisesRegex(ValueError, "iters_train must be positive"):
            next_train_cursor(ep=0, it=0, iters_train=0)


if __name__ == "__main__":
    unittest.main()
