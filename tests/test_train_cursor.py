from __future__ import annotations

import unittest

from train import next_train_cursor, train_perf_log_frequency


class TrainCursorTest(unittest.TestCase):
    def test_mid_epoch_checkpoint_resumes_at_next_iteration(self) -> None:
        self.assertEqual(next_train_cursor(ep=3, it=4, iters_train=10), (3, 5))

    def test_end_epoch_checkpoint_resumes_at_next_epoch(self) -> None:
        self.assertEqual(next_train_cursor(ep=3, it=9, iters_train=10), (4, 0))

    def test_rejects_empty_epoch(self) -> None:
        with self.assertRaisesRegex(ValueError, "iters_train must be positive"):
            next_train_cursor(ep=0, it=0, iters_train=0)

    def test_perf_log_frequency_matches_existing_large_epoch_behavior(self) -> None:
        self.assertEqual(train_perf_log_frequency(prof_freq=100, iters_train=1000), 100)
        self.assertEqual(train_perf_log_frequency(prof_freq=100, iters_train=20), 9)

    def test_perf_log_frequency_is_safe_for_tiny_epochs(self) -> None:
        self.assertEqual(train_perf_log_frequency(prof_freq=100, iters_train=1), 1)
        self.assertEqual(train_perf_log_frequency(prof_freq=100, iters_train=2), 1)

    def test_perf_log_frequency_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "prof_freq must be positive"):
            train_perf_log_frequency(prof_freq=0, iters_train=10)
        with self.assertRaisesRegex(ValueError, "iters_train must be positive"):
            train_perf_log_frequency(prof_freq=1, iters_train=0)


if __name__ == "__main__":
    unittest.main()
