from __future__ import annotations

import unittest

import torch

from infinity.normal_estimation.training import (
    convert_ar_batch_target_to_eval_convention,
    require_positive_steps_per_epoch,
)


class NormalTrainingTest(unittest.TestCase):
    def test_require_positive_steps_per_epoch_returns_valid_count(self) -> None:
        self.assertEqual(require_positive_steps_per_epoch(3, context="normal tokenizer training"), 3)

    def test_require_positive_steps_per_epoch_rejects_empty_loader(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "normal estimation training has no training batches"):
            require_positive_steps_per_epoch(0, context="normal estimation training")

    def test_ar_eval_converts_nyuv2_target_to_eval_convention(self) -> None:
        target = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_ar_batch_target_to_eval_convention(target, [{"dataset": "nyuv2"}])

        expected = torch.tensor([[[[-0.2]], [[0.4]], [[0.3]]]], dtype=torch.float32)
        expected = expected / torch.linalg.norm(expected, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_ar_eval_keeps_hypersim_target_convention(self) -> None:
        target = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_ar_batch_target_to_eval_convention(target, [{"dataset": "hypersim"}])

        expected = target / torch.linalg.norm(target, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_ar_eval_rejects_mixed_convention_batch(self) -> None:
        target = torch.zeros(2, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, "one dataset convention"):
            convert_ar_batch_target_to_eval_convention(
                target,
                [{"dataset": "hypersim"}, {"dataset": "nyuv2"}],
            )


if __name__ == "__main__":
    unittest.main()
