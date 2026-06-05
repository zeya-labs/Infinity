from __future__ import annotations

import bisect
import random
from typing import Any, Callable

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from .data import HypersimNormalDataset, VKITTI2NormalDataset


SUPPORTED_NORMAL_TRAIN_DATASETS = {"hypersim", "vkitti2"}


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeat: int) -> None:
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]

    def get_metadata(self, index: int) -> dict[str, Any]:
        return dataset_metadata_at(self.dataset, index % len(self.dataset))


class GroupedTargetSizeBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps each global DDP step on one dataset/target size."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        distributed: bool,
        seed: int,
        rank: int,
        world_size: int,
        dataset_weights: dict[str, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.distributed = distributed
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.dataset_weights = dataset_weights or {}
        self.epoch = 0
        self.groups = self._build_groups()

    def _build_groups(self) -> dict[tuple[str, int, int], list[int]]:
        groups: dict[tuple[str, int, int], list[int]] = {}
        for index in range(len(self.dataset)):
            metadata = dataset_metadata_at(self.dataset, index)
            target_height, target_width = (int(item) for item in metadata["target_size"])
            key = (str(metadata.get("dataset", "unknown")).lower(), target_height, target_width)
            groups.setdefault(key, []).append(index)
        return groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _group_batches(self) -> dict[tuple[str, int, int], list[list[int]]]:
        rng = random.Random(self.seed + self.epoch)
        group_batches: dict[tuple[str, int, int], list[list[int]]] = {}
        for key in sorted(self.groups):
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            batches = []
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
            if self.shuffle:
                rng.shuffle(batches)
            group_batches[key] = batches
        return group_batches

    def _weighted_group_pattern(self, keys: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        if not self.dataset_weights:
            return keys
        pattern: list[tuple[str, int, int]] = []
        for key in keys:
            dataset_name = key[0]
            repeats = self.dataset_weights.get(dataset_name, 1)
            pattern.extend([key] * repeats)
        return pattern or keys

    def _weighted_dataset_batches(
        self,
        group_batches: dict[tuple[str, int, int], list[list[int]]],
        rng: random.Random,
    ) -> list[list[int]]:
        needed = self.world_size if self.distributed else 1
        original_batches = {key: list(batches) for key, batches in group_batches.items() if len(batches) >= needed}
        if not original_batches:
            return []

        dataset_to_keys: dict[str, list[tuple[str, int, int]]] = {}
        for key in sorted(original_batches):
            dataset_to_keys.setdefault(key[0], []).append(key)
        dataset_names = [name for name in sorted(dataset_to_keys) if self.dataset_weights.get(name, 1) > 0]
        dataset_pattern = [
            name
            for name in dataset_names
            for _ in range(self.dataset_weights.get(name, 1))
        ]
        if not dataset_pattern:
            return []

        total_global_steps = sum(len(batches) // needed for batches in original_batches.values())
        queues = {key: list(batches) for key, batches in original_batches.items()}
        key_cursors = {name: 0 for name in dataset_to_keys}
        batches: list[list[int]] = []

        def refill(key: tuple[str, int, int]) -> None:
            queues[key] = list(original_batches[key])
            if self.shuffle:
                rng.shuffle(queues[key])

        for step in range(total_global_steps):
            dataset_name = dataset_pattern[step % len(dataset_pattern)]
            keys = dataset_to_keys.get(dataset_name, [])
            if not keys:
                continue

            selected_key = None
            for _ in range(len(keys)):
                key = keys[key_cursors[dataset_name] % len(keys)]
                key_cursors[dataset_name] += 1
                if len(queues[key]) < needed:
                    refill(key)
                if len(queues[key]) >= needed:
                    selected_key = key
                    break
            if selected_key is None:
                continue

            for _ in range(needed):
                batches.append(queues[selected_key].pop())
        return batches

    def _all_batches(self) -> list[list[int]]:
        group_batches = self._group_batches()
        if self.dataset_weights:
            return self._weighted_dataset_batches(group_batches, random.Random(self.seed + self.epoch + 1000003))

        keys = [key for key in sorted(group_batches) if group_batches[key]]
        pattern = self._weighted_group_pattern(keys)
        if not pattern:
            return []

        batches: list[list[int]] = []
        cursor = 0
        empty_rounds = 0
        while keys and empty_rounds < len(pattern):
            key = pattern[cursor % len(pattern)]
            cursor += 1
            available = group_batches.get(key, [])
            needed = self.world_size if self.distributed else 1
            if len(available) < needed:
                empty_rounds += 1
                continue
            empty_rounds = 0
            for _ in range(needed):
                batches.append(available.pop())
        return batches

    def __iter__(self):
        batches = self._all_batches()
        if self.distributed:
            batches = batches[self.rank :: self.world_size]
        return iter(batches)

    def __len__(self) -> int:
        batches = self._all_batches()
        if self.distributed:
            return len(batches[self.rank :: self.world_size])
        return len(batches)


def parse_train_dataset_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    if not names:
        raise ValueError("--train-datasets must include at least one dataset")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate --train-datasets entries: {duplicates}")
    unknown = sorted(set(names) - SUPPORTED_NORMAL_TRAIN_DATASETS)
    if unknown:
        raise ValueError(
            f"Unsupported --train-datasets entries: {unknown}. "
            f"Supported: {sorted(SUPPORTED_NORMAL_TRAIN_DATASETS)}"
        )
    return names


def parse_train_dataset_weights(value: str, dataset_names: list[str]) -> dict[str, int]:
    weights = {name: 1 for name in dataset_names}
    seen_weights: set[str] = set()
    if not value.strip():
        return weights
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --train-dataset-weights item {item!r}; expected name:weight")
        name, raw_weight = item.split(":", 1)
        name = name.strip().lower()
        if name not in weights:
            raise ValueError(f"Weight specified for dataset {name!r}, but --train-datasets={dataset_names}")
        if name in seen_weights:
            raise ValueError(f"Duplicate --train-dataset-weights entry for dataset {name!r}")
        seen_weights.add(name)
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Dataset weight must be a positive integer step ratio for {name}, got {raw_weight!r}"
            ) from exc
        if weight <= 0:
            raise ValueError(f"Dataset weight must be > 0 for {name}, got {weight}")
        weights[name] = weight
    return weights


def dataset_metadata_at(dataset: Dataset, index: int) -> dict[str, Any]:
    if isinstance(dataset, Subset):
        return dataset_metadata_at(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
        sample_index = index if dataset_index == 0 else index - dataset.cumulative_sizes[dataset_index - 1]
        return dataset_metadata_at(dataset.datasets[dataset_index], sample_index)
    get_metadata = getattr(dataset, "get_metadata", None)
    if callable(get_metadata):
        return get_metadata(index)
    sample = dataset[index]
    return sample["metadata"]


def build_normal_train_dataset(
    *,
    train_datasets: str,
    hypersim_root: str,
    vkitti2_root: str,
    partition: str,
    pn: str,
    max_samples: int,
    metadata_only: bool = False,
) -> Dataset:
    datasets: list[Dataset] = []
    for name in parse_train_dataset_names(train_datasets):
        if name == "hypersim":
            datasets.append(
                HypersimNormalDataset(
                    root=hypersim_root,
                    partition=partition,
                    pn=pn,
                    max_samples=max_samples,
                    metadata_only=metadata_only,
                )
            )
        elif name == "vkitti2":
            datasets.append(
                VKITTI2NormalDataset(
                    root=vkitti2_root,
                    partition=partition,
                    pn=pn,
                    max_samples=max_samples,
                    metadata_only=metadata_only,
                )
            )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_normal_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    pin_memory: bool,
    group_by_target_size: bool = False,
    dataset_weights: dict[str, int] | None = None,
) -> tuple[DataLoader, Sampler | None]:
    if group_by_target_size:
        batch_sampler = GroupedTargetSizeBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            distributed=distributed,
            seed=0,
            rank=rank,
            world_size=world_size,
            dataset_weights=dataset_weights,
        )
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**kwargs), batch_sampler

    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if distributed else None
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "sampler": sampler,
        "shuffle": False if sampler is not None else shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs), sampler
