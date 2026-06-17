from __future__ import annotations

import unittest

import torch

from infinity.normal_estimation.modeling import InfinityNormalPrefixModel


def make_layout_model(layout: str) -> InfinityNormalPrefixModel:
    model = object.__new__(InfinityNormalPrefixModel)
    model.normal_token_layout = layout
    model.normal_prefix_layout_cache = {}
    return model


class NormalTokenLayoutTest(unittest.TestCase):
    def test_prefix_layout_keeps_all_rgb_before_normal_tokens(self) -> None:
        model = make_layout_model("prefix")
        schedule = [(1, 1, 2), (1, 1, 3)]

        layout = model._normal_prefix_layout(schedule, schedule, torch.device("cpu"))

        self.assertEqual(layout["segments"], ((0, 5, 5), (5, 7, 7), (7, 10, 10)))
        self.assertEqual(layout["sequence_schedule"], schedule + schedule)
        self.assertTrue(torch.equal(layout["normal_token_indices"], torch.arange(5, 10)))

    def test_interleaved_layout_alternates_rgb_and_normal_by_scale(self) -> None:
        model = make_layout_model("interleaved")
        schedule = [(1, 1, 2), (1, 1, 3)]

        layout = model._normal_prefix_layout(schedule, schedule, torch.device("cpu"))

        self.assertEqual(layout["segments"], ((0, 2, 2), (2, 4, 4), (4, 7, 7), (7, 10, 10)))
        self.assertEqual(layout["sequence_schedule"], [schedule[0], schedule[0], schedule[1], schedule[1]])
        self.assertTrue(torch.equal(layout["normal_token_indices"], torch.tensor([2, 3, 7, 8, 9])))

    def test_interleaved_uses_contiguous_prefix_segments_for_flash_attention(self) -> None:
        model = make_layout_model("interleaved")
        schedule = [(1, 1, 2), (1, 1, 3)]

        layout = model._normal_prefix_layout(schedule, schedule, torch.device("cpu"))

        self.assertIsNone(layout["segment_key_indices"])
        for query_start, query_end, key_end in layout["segments"]:
            self.assertEqual(query_end, key_end)
            self.assertLessEqual(query_start, query_end)

    def test_interleaved_source_keeps_rgb_source_tokens_from_attending_normals(self) -> None:
        model = make_layout_model("interleaved_source")
        schedule = [(1, 1, 2), (1, 1, 3)]

        layout = model._normal_prefix_layout(schedule, schedule, torch.device("cpu"))
        segment_key_indices = layout["segment_key_indices"]

        self.assertIsNotNone(segment_key_indices)
        self.assertTrue(torch.equal(layout["normal_token_indices"], torch.tensor([2, 3, 7, 8, 9])))
        self.assertTrue(torch.equal(segment_key_indices[2], torch.tensor([0, 1, 4, 5, 6])))
        self.assertFalse(torch.isin(torch.tensor([2, 3]), segment_key_indices[2]).any())
        self.assertTrue(torch.equal(torch.sort(segment_key_indices[3]).values, torch.arange(10)))


if __name__ == "__main__":
    unittest.main()
