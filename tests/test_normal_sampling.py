from __future__ import annotations

import unittest

from torch.utils.data import Dataset

from infinity.normal_estimation.sampling import (
    GroupedTargetSizeBatchSampler,
    RepeatDataset,
    dataset_metadata_at,
    parse_train_dataset_names,
    parse_train_dataset_weights,
)


class ToyNormalDataset(Dataset):
    def __init__(self, dataset_name: str, target_size: tuple[int, int], length: int) -> None:
        self.dataset_name = dataset_name
        self.target_size = target_size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {"metadata": self.get_metadata(index)}

    def get_metadata(self, index: int) -> dict:
        return {
            "dataset": self.dataset_name,
            "index": index,
            "target_size": list(self.target_size),
        }


class NormalSamplingTest(unittest.TestCase):
    def test_parse_train_dataset_weights(self) -> None:
        names = parse_train_dataset_names("hypersim,vkitti2")
        self.assertEqual(names, ["hypersim", "vkitti2"])
        self.assertEqual(parse_train_dataset_weights("hypersim:3,vkitti2:1", names), {"hypersim": 3, "vkitti2": 1})
        with self.assertRaises(ValueError):
            parse_train_dataset_weights("vkitti2:0", names)
        with self.assertRaises(ValueError):
            parse_train_dataset_weights("vkitti2:0.5", names)
        with self.assertRaises(ValueError):
            parse_train_dataset_names("hypersim,unknown")
        with self.assertRaises(ValueError):
            parse_train_dataset_weights("unknown:1", names)
        with self.assertRaises(ValueError):
            parse_train_dataset_names("hypersim,hypersim")
        with self.assertRaises(ValueError):
            parse_train_dataset_weights("hypersim:3,hypersim:2", names)

    def test_parse_train_dataset_weights_ignores_supported_unselected_datasets(self) -> None:
        names = parse_train_dataset_names("hypersim")
        self.assertEqual(parse_train_dataset_weights("hypersim:3,vkitti2:1", names), {"hypersim": 3})

    def test_repeat_dataset_metadata(self) -> None:
        dataset = RepeatDataset(ToyNormalDataset("hypersim", (240, 320), 2), repeat=3)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset_metadata_at(dataset, 3)["index"], 1)

    def test_grouped_sampler_keeps_distributed_step_homogeneous(self) -> None:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(
            [
                ToyNormalDataset("hypersim", (864, 1152), 80),
                ToyNormalDataset("vkitti2", (592, 1776), 80),
            ]
        )
        weights = {"hypersim": 3, "vkitti2": 1}
        rank_batches = []
        for rank in range(8):
            sampler = GroupedTargetSizeBatchSampler(
                dataset,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                distributed=True,
                seed=0,
                rank=rank,
                world_size=8,
                dataset_weights=weights,
            )
            rank_batches.append(list(sampler))

        self.assertEqual([len(batches) for batches in rank_batches], [7] * 8)

        for step in range(6):
            step_datasets = []
            for rank in range(8):
                metadata = dataset_metadata_at(dataset, rank_batches[rank][step][0])
                step_datasets.append(metadata["dataset"])
            self.assertEqual(len(set(step_datasets)), 1)

        rank0_datasets = [dataset_metadata_at(dataset, batch[0])["dataset"] for batch in rank_batches[0][:4]]
        self.assertEqual(rank0_datasets, ["hypersim", "hypersim", "hypersim", "vkitti2"])

    def test_weighted_sampler_uses_first_dataset_as_epoch_anchor(self) -> None:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(
            [
                ToyNormalDataset("hypersim", (864, 1152), 96),
                ToyNormalDataset("vkitti2", (592, 1776), 1000),
            ]
        )
        sampler = GroupedTargetSizeBatchSampler(
            dataset,
            batch_size=2,
            shuffle=False,
            drop_last=True,
            distributed=True,
            seed=0,
            rank=0,
            world_size=8,
            dataset_weights={"hypersim": 3, "vkitti2": 1},
        )

        batches = list(sampler)
        self.assertEqual(len(batches), 8)
        datasets = [dataset_metadata_at(dataset, batch[0])["dataset"] for batch in batches]
        self.assertEqual(datasets, ["hypersim", "hypersim", "hypersim", "vkitti2", "hypersim", "hypersim", "hypersim", "vkitti2"])

    def test_weighted_sampler_shuffles_dataset_sequence(self) -> None:
        from torch.utils.data import ConcatDataset

        dataset = ConcatDataset(
            [
                ToyNormalDataset("hypersim", (864, 1152), 96),
                ToyNormalDataset("vkitti2", (592, 1776), 1000),
            ]
        )
        sampler = GroupedTargetSizeBatchSampler(
            dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
            distributed=True,
            seed=0,
            rank=0,
            world_size=8,
            dataset_weights={"hypersim": 3, "vkitti2": 1},
        )

        batches = list(sampler)
        datasets = [dataset_metadata_at(dataset, batch[0])["dataset"] for batch in batches]
        self.assertEqual(datasets.count("hypersim"), 6)
        self.assertEqual(datasets.count("vkitti2"), 2)
        self.assertNotEqual(
            datasets,
            ["hypersim", "hypersim", "hypersim", "vkitti2", "hypersim", "hypersim", "hypersim", "vkitti2"],
        )


if __name__ == "__main__":
    unittest.main()
