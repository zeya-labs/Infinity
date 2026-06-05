import unittest
from types import SimpleNamespace

import torch

from infinity.utils.text_condition import build_text_condition


class FakeTokenizer:
    model_max_length = 4

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, max_length, padding, truncation, return_tensors):
        self.calls.append(
            {
                "text": text,
                "max_length": max_length,
                "padding": padding,
                "truncation": truncation,
                "return_tensors": return_tensors,
            }
        )
        input_ids = torch.tensor(
            [
                [10, 11, 12, 0],
                [20, 21, 0, 0],
            ],
            dtype=torch.long,
        )[:, :max_length]
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.long,
        )[:, :max_length]
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class FakeEncoder:
    def __init__(self) -> None:
        self.input_ids = None
        self.attention_mask = None

    def __call__(self, *, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        batch, length = input_ids.shape
        values = torch.arange(batch * length * 2, dtype=torch.float64).reshape(batch, length, 2)
        return {"last_hidden_state": values}


class TextConditionTest(unittest.TestCase):
    def test_builds_compact_condition_tuple_on_cpu(self) -> None:
        tokenizer = FakeTokenizer()
        encoder = FakeEncoder()

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = build_text_condition(
            tokenizer,
            encoder,
            ["first caption", "second caption"],
            device="cpu",
        )

        self.assertEqual(
            tokenizer.calls,
            [
                {
                    "text": ["first caption", "second caption"],
                    "max_length": tokenizer.model_max_length,
                    "padding": "max_length",
                    "truncation": True,
                    "return_tensors": "pt",
                }
            ],
        )
        self.assertEqual(lens, [3, 2])
        self.assertEqual(max_seqlen_k, 3)
        torch.testing.assert_close(cu_seqlens_k, torch.tensor([0, 3, 5], dtype=torch.int32))
        expected_kv = torch.cat(
            [
                torch.arange(8, dtype=torch.float32).reshape(4, 2)[:3],
                torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)[:2],
            ],
            dim=0,
        )
        torch.testing.assert_close(kv_compact, expected_kv)
        self.assertEqual(kv_compact.dtype, torch.float32)
        torch.testing.assert_close(encoder.attention_mask, torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))

    def test_respects_explicit_max_length_and_no_device_move(self) -> None:
        tokenizer = FakeTokenizer()
        encoder = FakeEncoder()

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = build_text_condition(
            tokenizer,
            encoder,
            ["first caption", "second caption"],
            max_length=3,
            device=None,
        )

        self.assertEqual(tokenizer.calls[0]["max_length"], 3)
        self.assertEqual(lens, [3, 2])
        self.assertEqual(max_seqlen_k, 3)
        self.assertEqual(tuple(kv_compact.shape), (5, 2))
        torch.testing.assert_close(cu_seqlens_k, torch.tensor([0, 3, 5], dtype=torch.int32))
        torch.testing.assert_close(encoder.input_ids, torch.tensor([[10, 11, 12], [20, 21, 0]]))


if __name__ == "__main__":
    unittest.main()
