from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
try:
    import swanlab
except ImportError:
    swanlab = None
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
try:
    from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
except ImportError:
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    StateDictType = None
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infinity.normal_estimation import (  # noqa: E402
    HypersimNormalDataset,
    VKITTI2NormalDataset,
    build_bsq_vae,
    build_prefix_tokens_from_image,
    build_infinity_normal_model,
    build_multiscale_var_inputs,
    collate_normal_estimation_batch,
    compute_normal_metrics,
    decode_logits_to_normal,
    load_normal_sample_from_metadata,
    load_infinity_state_dict,
    normals_to_vis,
    resolve_scale_schedule_from_hw,
)
from infinity.models.infinity import MultipleLayers  # noqa: E402


LOGGER = logging.getLogger("train_normal_estimation")
TOKEN_CACHE_MEMORY: dict[str, dict[str, torch.Tensor]] = {}
FULLSTATE_SAVE_POLICY = (
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if FullStateDictConfig is not None
    else None
)


class GroupedTargetSizeBatchSampler(Sampler[list[int]]):
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
        dataset_weights: dict[str, float] | None = None,
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
            weight = self.dataset_weights.get(dataset_name, 1.0)
            repeats = max(1, int(round(weight)))
            pattern.extend([key] * repeats)
        return pattern or keys

    def _all_batches(self) -> list[list[int]]:
        group_batches = self._group_batches()
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
FULLOPTSTATE_SAVE_POLICY = (
    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if FullOptimStateDictConfig is not None
    else None
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Infinity for RGB-to-normal estimation on raw Hypersim.")
    parser.add_argument("--data-root", type=str, default="/root/vepfs/Infinity/data/hypersim/processed/hypersim")
    parser.add_argument(
        "--train-datasets",
        type=str,
        default="hypersim",
        help="Comma-separated train datasets. Supported: hypersim,vkitti2.",
    )
    parser.add_argument(
        "--train-dataset-weights",
        type=str,
        default="",
        help="Comma-separated dataset sampling weights, e.g. hypersim:3,vkitti2:1. Empty uses 1 for each train dataset.",
    )
    parser.add_argument("--vkitti2-root", type=str, default="/root/vepfs/Infinity/data/VKITTI2")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Micro-batches to accumulate before each optimizer step.")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--resume-lr-step",
        type=int,
        default=-1,
        help="When checkpoint has no scheduler state, fast-forward the LR scheduler to this step; -1 uses checkpoint step.",
    )
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--optimizer-backend",
        type=str,
        choices=("loop", "foreach", "fused"),
        default="loop",
        help="AdamW implementation. 'loop' preserves the previous foreach=False path; 'fused' is fastest on supported CUDA builds.",
    )
    parser.add_argument("--zero", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=float,
        default=8.0,
        help="DDP gradient bucket size in MiB. Smaller buckets avoid very large first allreduces after resume.",
    )
    parser.add_argument("--disable-ddp-static-graph", dest="ddp_static_graph", action="store_false")
    parser.set_defaults(ddp_static_graph=True)
    parser.add_argument("--enable-hybrid-shard", action="store_true", default=False)
    parser.add_argument("--inner-shard-degree", type=int, default=1)
    parser.add_argument("--fsdp-use-orig-params", action="store_true", default=True)
    parser.add_argument("--disable-fsdp-use-orig-params", dest="fsdp_use_orig_params", action="store_false")
    parser.add_argument(
        "--checkpointing",
        type=str,
        choices=("full-block", "self-attn", "none"),
        default="full-block",
        help="Activation checkpointing mode for Infinity blocks. Use 'none' for speed if memory allows.",
    )
    parser.add_argument(
        "--full-block-checkpoint-skip-interval",
        type=int,
        default=0,
        help="For --checkpointing=full-block, leave every Nth block uncheckpointed. 0 preserves the previous every-block policy.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-log-every", type=int, default=200)
    parser.add_argument(
        "--train-normal-metrics-every",
        type=int,
        default=100,
        help="Compute decoded normal train metrics every N optimizer steps; 0 disables them during train. Validation is unchanged.",
    )
    parser.add_argument("--ar-eval-every", type=int, default=0)
    parser.add_argument("--ar-eval-samples", type=int, default=32)
    parser.add_argument("--ar-eval-top-k", type=int, default=1)
    parser.add_argument("--ar-eval-top-p", type=float, default=0.0)
    parser.add_argument("--ar-eval-tau", type=float, default=1.0)
    parser.add_argument("--save-every-steps", type=int, default=0, help="Overwrite checkpoints/last_step.pth every N optimizer steps; 0 disables step checkpointing.")
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--save-optimizer-state", action="store_true", default=False)
    parser.add_argument("--pn", type=str, choices=("0.06M", "0.25M", "1M"), default="0.06M")
    parser.add_argument("--train-partition", type=str, default="train")
    parser.add_argument("--val-partition", type=str, default="val")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--token-cache-dir",
        type=str,
        default="",
        help="Optional directory for deterministic train token cache. Skips RGB/normal VAE tokenization on cache hits.",
    )
    parser.add_argument("--token-cache-memory", action="store_true", default=False, help="Keep token cache entries in host memory after first load/write.")
    parser.add_argument(
        "--token-cache-metadata-only",
        action="store_true",
        default=False,
        help="For train data, return metadata only and materialize raw samples only on token-cache misses or decoded-metric/image steps.",
    )
    parser.add_argument(
        "--token-cache-require-hit",
        action="store_true",
        default=False,
        help="Fail instead of falling back to raw VAE tokenization if a train token-cache entry is missing.",
    )
    parser.add_argument(
        "--token-cache-filter-missing",
        action="store_true",
        default=False,
        help="Filter the train dataset to samples with existing token-cache entries before training.",
    )
    parser.add_argument("--model-name", type=str, default="infinity_8b")
    parser.add_argument(
        "--fast-model-init",
        action="store_true",
        default=False,
        help="Skip expensive default parameter initialization for the Infinity model when it will be immediately warm-started.",
    )
    parser.add_argument("--use-bit-label", action="store_true", default=True)
    parser.add_argument("--disable-bit-label", dest="use_bit_label", action="store_false")
    parser.add_argument("--add-lvl-embeding-only-first-block", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rope2d-each-sa-layer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rope2d-normalized-by-hw", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--normal-use-flex-attn", action="store_true", default=False)
    parser.add_argument("--normal-use-segmented-flash-attn", action="store_true", default=False)
    parser.add_argument("--normal-bf16-activations", action="store_true", default=False)
    parser.add_argument(
        "--normal-save-activations-on-cpu",
        action="store_true",
        default=False,
        help="Offload tensors saved for autograd during the normal transformer forward to CPU pinned memory. This avoids recompute but trades GPU memory for PCIe traffic.",
    )
    parser.add_argument("--fused-mlp", action="store_true", default=False)
    parser.add_argument("--fused-norm", action="store_true", default=False)
    parser.add_argument("--always-training-scales", type=int, default=100)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--normal-l1-weight", type=float, default=0.25)
    parser.add_argument("--normal-angular-weight", type=float, default=0.5)
    parser.add_argument("--normal-latent-weight", type=float, default=0.1)
    parser.add_argument("--normal-norm-weight", type=float, default=0.05)
    parser.add_argument("--noise-apply-layers", type=int, default=-1)
    parser.add_argument("--noise-apply-strength", type=float, default=0.0)
    parser.add_argument("--noise-apply-requant", action="store_true", default=True)
    parser.add_argument("--disable-noise-requant", dest="noise_apply_requant", action="store_false")
    parser.add_argument("--normal-vae-ckpt", type=str, required=True)
    parser.add_argument("--normal-vae-type", type=int, default=14)
    parser.add_argument("--normal-apply-spatial-patchify", action="store_true", default=False)
    parser.add_argument("--disable-normal-spatial-patchify", dest="normal_apply_spatial_patchify", action="store_false")
    parser.add_argument("--rgb-vae-ckpt", type=str, required=True)
    parser.add_argument("--rgb-vae-type", type=int, default=14)
    parser.add_argument("--rgb-apply-spatial-patchify", action="store_true", default=False)
    parser.add_argument("--rgb-no-spatial-patchify", dest="rgb_apply_spatial_patchify", action="store_false")
    parser.add_argument("--swanlab-project", type=str, default=os.environ.get("SWANLAB_PROJECT", "infinity_normal_estimation_hypersim"))
    parser.add_argument("--swanlab-workspace", type=str, default=os.environ.get("SWANLAB_WORKSPACE", ""))
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-job-type", type=str, default="train_normal_estimation")
    parser.add_argument("--swanlab-tags", nargs="*", default=["normal_estimation", "hypersim"])
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        choices=("cloud", "local", "offline", "disabled"),
        default=os.environ.get("SWANLAB_MODE", "cloud"),
    )
    parser.add_argument("--swanlab-logdir", type=str, default="")
    parser.add_argument("--profile-timings", action="store_true", default=False, help="Log coarse init and per-step timing breakdowns.")
    parser.add_argument("--profile-warmup-steps", type=int, default=1, help="Skip this many optimizer steps before logging profile_step timings.")
    parser.add_argument("--profile-max-steps", type=int, default=0, help="Maximum number of profile_step logs after warmup; 0 logs all.")
    parser.add_argument(
        "--profile-torch-step",
        type=int,
        default=0,
        help="Capture one torch.profiler optimizer step at this 1-based global step; 0 disables.",
    )
    parser.add_argument(
        "--profile-torch-row-limit",
        type=int,
        default=40,
        help="Number of rows to write in the torch.profiler CUDA time table.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path, is_main: bool) -> None:
    handlers: list[logging.Handler] = []
    if is_main:
        handlers.append(logging.StreamHandler())
        handlers.append(logging.FileHandler(output_dir / "train.log", encoding="utf-8"))
    else:
        handlers.append(logging.FileHandler(output_dir / f"train_rank{dist.get_rank():02d}.log", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def init_distributed() -> tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, torch.device("cuda", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return False, 0, 1, device


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def reduce_tensor(value: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def reduce_metrics(metrics: dict[str, torch.Tensor], enabled: bool) -> dict[str, float]:
    return {key: float(reduce_tensor(value, enabled).item()) for key, value in metrics.items()}


def dist_barrier_if_initialized() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@contextmanager
def skip_torch_weight_init(enabled: bool):
    if not enabled:
        yield
        return

    init_names = (
        "uniform_",
        "normal_",
        "trunc_normal_",
        "constant_",
        "ones_",
        "zeros_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
    )
    saved_init_fns = {name: getattr(torch.nn.init, name) for name in init_names if hasattr(torch.nn.init, name)}
    saved_resets = {
        torch.nn.Linear: torch.nn.Linear.reset_parameters,
        torch.nn.Embedding: torch.nn.Embedding.reset_parameters,
        torch.nn.LayerNorm: torch.nn.LayerNorm.reset_parameters,
    }

    def no_init_(tensor, *args, **kwargs):
        return tensor

    try:
        for name in saved_init_fns:
            setattr(torch.nn.init, name, no_init_)
        for module_cls in saved_resets:
            module_cls.reset_parameters = lambda self: None
        yield
    finally:
        for name, fn in saved_init_fns.items():
            setattr(torch.nn.init, name, fn)
        for module_cls, reset in saved_resets.items():
            module_cls.reset_parameters = reset


def cuda_synchronize_if_needed(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_stage(timings: dict[str, float] | None, name: str, device: torch.device):
    if timings is None:
        yield
        return
    cuda_synchronize_if_needed(device, True)
    start = time.perf_counter()
    try:
        yield
    finally:
        cuda_synchronize_if_needed(device, True)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)


def format_timing_dict(timings: dict[str, float]) -> str:
    parts = []
    for key, value in sorted(timings.items()):
        suffix = "GiB" if key.endswith("_gib") else "s"
        parts.append(f"{key}={value:.3f}{suffix}")
    return " ".join(parts)


def reduce_timing_dict(timings: dict[str, float], distributed: bool, device: torch.device) -> dict[str, float]:
    if not distributed:
        return dict(timings)
    keys = sorted(timings)
    values = torch.tensor([timings[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed: bool,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool,
    drop_last: bool,
    group_by_target_size: bool = False,
    dataset_weights: dict[str, float] | None = None,
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
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": collate_normal_estimation_batch,
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
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
        "collate_fn": collate_normal_estimation_batch,
        "sampler": sampler,
        "shuffle": False if sampler is not None else shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs), sampler


def dataset_metadata_at(dataset: Dataset, index: int) -> dict[str, Any]:
    if isinstance(dataset, Subset):
        return dataset_metadata_at(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
        sample_index = index if dataset_index == 0 else index - dataset.cumulative_sizes[dataset_index - 1]
        return dataset_metadata_at(dataset.datasets[dataset_index], sample_index)
    if isinstance(dataset, HypersimNormalDataset):
        return dataset._metadata_only_sample(index)["metadata"]
    if isinstance(dataset, VKITTI2NormalDataset):
        return dataset._metadata_for_record(index, dataset.records[index])
    sample = dataset[index]
    return sample["metadata"]


def parse_train_dataset_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    if not names:
        raise ValueError("--train-datasets must include at least one dataset")
    supported = {"hypersim", "vkitti2"}
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(f"Unsupported --train-datasets entries: {unknown}. Supported: {sorted(supported)}")
    return names


def parse_train_dataset_weights(value: str, dataset_names: list[str]) -> dict[str, float]:
    weights = {name: 1.0 for name in dataset_names}
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
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"Dataset weight must be > 0 for {name}, got {weight}")
        weights[name] = weight
    return weights


def build_train_dataset(args: argparse.Namespace) -> Dataset:
    metadata_only = args.token_cache_metadata_only and token_cache_enabled(args)
    datasets: list[Dataset] = []
    for name in parse_train_dataset_names(args.train_datasets):
        if name == "hypersim":
            datasets.append(
                HypersimNormalDataset(
                    root=args.data_root,
                    partition=args.train_partition,
                    pn=args.pn,
                    max_samples=args.max_train_samples,
                    metadata_only=metadata_only,
                )
            )
        elif name == "vkitti2":
            datasets.append(
                VKITTI2NormalDataset(
                    root=args.vkitti2_root,
                    partition=args.train_partition,
                    pn=args.pn,
                    max_samples=args.max_train_samples,
                    metadata_only=metadata_only,
                )
            )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_optimizer_and_scheduler(model: torch.nn.Module, args: argparse.Namespace, total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs: dict[str, Any] = {}
    if args.optimizer_backend == "fused":
        optimizer_kwargs["fused"] = True
    elif args.optimizer_backend == "foreach":
        optimizer_kwargs["foreach"] = True
    else:
        optimizer_kwargs["foreach"] = False
    try:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
            **optimizer_kwargs,
        )
    except (TypeError, RuntimeError) as exc:
        if args.optimizer_backend != "fused":
            raise
        LOGGER.warning("Fused AdamW is unavailable (%s); falling back to foreach AdamW.", exc)
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
            foreach=True,
        )

    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def maybe_resize_target_for_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape[-2:] == target.shape[-2:]:
        return target, mask
    target = F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    target = target / torch.linalg.norm(target, dim=1, keepdim=True).clamp_min(1e-6)
    mask = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="nearest") > 0.5
    return target, mask


def token_cache_enabled(args: argparse.Namespace) -> bool:
    return bool(args.token_cache_dir) and not (
        args.noise_apply_layers >= 0 and args.noise_apply_strength > 0
    )


def batch_has_full_tensors(batch: dict[str, Any]) -> bool:
    return bool(batch["image"].numel() > 0 and batch["target"].numel() > 0 and batch["mask"].numel() > 0)


def materialize_batch_from_metadata(batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    samples = [load_normal_sample_from_metadata(metadata, args.pn) for metadata in batch["metadata"]]
    return collate_normal_estimation_batch(samples)


def resolve_batch_scale_schedule(batch: dict[str, Any], args: argparse.Namespace) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    target_cpu = batch["target"]
    if target_cpu.numel() > 0:
        target_hw = (int(target_cpu.shape[-2]), int(target_cpu.shape[-1]))
    else:
        target_hw = tuple(int(item) for item in batch["metadata"][0]["target_size"])
    scale_schedule = [tuple(item) for item in resolve_scale_schedule_from_hw(target_hw[0], target_hw[1], args.pn)[1]]
    training_scales = min(args.always_training_scales, len(scale_schedule))
    return target_cpu, scale_schedule[:training_scales]


def token_cache_signature(args: argparse.Namespace, scale_schedule: list[tuple[int, int, int]]) -> str:
    payload = {
        "version": 1,
        "pn": args.pn,
        "scale_schedule": scale_schedule,
        "always_training_scales": args.always_training_scales,
        "normal_vae_ckpt": str(Path(args.normal_vae_ckpt).resolve()),
        "normal_vae_type": args.normal_vae_type,
        "normal_apply_spatial_patchify": args.normal_apply_spatial_patchify,
        "rgb_vae_ckpt": str(Path(args.rgb_vae_ckpt).resolve()),
        "rgb_vae_type": args.rgb_vae_type,
        "rgb_apply_spatial_patchify": args.rgb_apply_spatial_patchify,
        "use_bit_label": args.use_bit_label,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def token_cache_sample_key(metadata: dict[str, Any], signature: str) -> str:
    source = "|".join(
        str(metadata.get(key, ""))
        for key in ("dataset", "partition", "index", "image_path", "normal_path", "target_size")
    )
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:24]
    return f"{signature}_{digest}.pt"


def load_token_cache_batch(
    cache_dir: Path,
    metadata: list[dict[str, Any]],
    signature: str,
    device: torch.device,
    use_memory_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    payloads = []
    for item in metadata:
        path = cache_dir / token_cache_sample_key(item, signature)
        path_key = str(path)
        payload = TOKEN_CACHE_MEMORY.get(path_key) if use_memory_cache else None
        if payload is None:
            if not path.is_file():
                return None
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if use_memory_cache:
                TOKEN_CACHE_MEMORY[path_key] = payload
        payloads.append(payload)
    return (
        torch.stack([payload["rgb_prefix_blc"] for payload in payloads], dim=0).to(device, non_blocking=True),
        torch.stack([payload["x_blc_without_prefix"] for payload in payloads], dim=0).to(device, non_blocking=True),
        torch.stack([payload["gt_bl"] for payload in payloads], dim=0).to(device, non_blocking=True),
        torch.stack([payload["raw_features"] for payload in payloads], dim=0).to(device, non_blocking=True),
    )


def missing_token_cache_paths(cache_dir: Path, metadata: list[dict[str, Any]], signature: str) -> list[Path]:
    return [
        cache_dir / token_cache_sample_key(item, signature)
        for item in metadata
        if not (cache_dir / token_cache_sample_key(item, signature)).is_file()
    ]


def filter_dataset_to_token_cache_hits(dataset: Dataset, args: argparse.Namespace, cache_dir: Path) -> tuple[Dataset, int]:
    kept_indices = []
    for index in range(len(dataset)):
        metadata = dataset_metadata_at(dataset, index)
        target_hw = tuple(int(item) for item in metadata["target_size"])
        scale_schedule = [tuple(item) for item in resolve_scale_schedule_from_hw(target_hw[0], target_hw[1], args.pn)[1]]
        scale_schedule = scale_schedule[: min(args.always_training_scales, len(scale_schedule))]
        signature = token_cache_signature(args, scale_schedule)
        if (cache_dir / token_cache_sample_key(metadata, signature)).is_file():
            kept_indices.append(index)
    original_len = len(dataset)
    return Subset(dataset, kept_indices), original_len


def save_token_cache_batch(
    cache_dir: Path,
    metadata: list[dict[str, Any]],
    signature: str,
    rgb_prefix_blc: torch.Tensor,
    x_blc_without_prefix: torch.Tensor,
    gt_bl: torch.Tensor,
    raw_features: torch.Tensor,
    use_memory_cache: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "rgb_prefix_blc": rgb_prefix_blc.detach().cpu(),
        "x_blc_without_prefix": x_blc_without_prefix.detach().cpu(),
        "gt_bl": gt_bl.detach().cpu(),
        "raw_features": raw_features.detach().cpu(),
    }
    for sample_index, item in enumerate(metadata):
        path = cache_dir / token_cache_sample_key(item, signature)
        if path.is_file():
            continue
        payload = {key: value[sample_index].clone().contiguous() for key, value in tensors.items()}
        if use_memory_cache:
            TOKEN_CACHE_MEMORY[str(path)] = payload
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)


def compute_ce_loss(
    logits_blv: torch.Tensor,
    gt_bl: torch.Tensor,
    *,
    use_bit_label: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits_for_loss = logits_blv.float()
    if use_bit_label:
        batch_size, seq_len, _ = logits_blv.shape
        loss = F.cross_entropy(
            logits_for_loss.reshape(batch_size, seq_len, -1, 2).permute(0, 3, 1, 2),
            gt_bl,
            reduction="none",
        ).mean(dim=-1)
        bitwise_acc = (logits_blv.reshape(batch_size, seq_len, -1, 2).argmax(dim=-1) == gt_bl).float()
        token_acc = (bitwise_acc.sum(dim=-1) == bitwise_acc.shape[-1]).float().mean()
        bit_acc = bitwise_acc.mean()
    else:
        batch_size, seq_len, vocab = logits_blv.shape
        loss = F.cross_entropy(
            logits_for_loss.reshape(batch_size * seq_len, vocab),
            gt_bl.reshape(batch_size * seq_len),
            reduction="none",
        ).reshape(batch_size, seq_len)
        pred_tokens = logits_blv.argmax(dim=-1)
        bit_acc = (pred_tokens == gt_bl).float().mean()
        token_acc = bit_acc

    ce_loss = loss.mean()
    return ce_loss, {
        "loss_ce": ce_loss.detach(),
        "acc_bit": bit_acc.detach(),
        "acc_token": token_acc.detach(),
    }


def precision_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def convert_model_precision(model: torch.nn.Module, precision: str) -> torch.nn.Module:
    if precision not in {"bf16", "fp16"}:
        return model
    cast_fn = torch.nn.Module.bfloat16 if precision == "bf16" else torch.nn.Module.half
    for block in model.unregistered_blocks:
        cast_fn(block)
    return model


def is_fsdp_model(model: torch.nn.Module) -> bool:
    return isinstance(model, FSDP)


def require_fsdp_state_dict_api() -> None:
    if StateDictType is None or FULLSTATE_SAVE_POLICY is None or FULLOPTSTATE_SAVE_POLICY is None:
        raise ImportError(
            "Current PyTorch does not provide FSDP full-state-dict APIs. "
            "Use --zero 0 or run with a newer PyTorch environment."
        )


def build_training_model(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
) -> torch.nn.Module:
    if args.zero == 0:
        if distributed:
            LOGGER.info(
                "DDP init bucket_cap_mb=%.2f static_graph=%s gradient_as_bucket_view=True",
                args.ddp_bucket_cap_mb,
                args.ddp_static_graph,
            )
            return DDP(
                model,
                device_ids=[device.index],
                broadcast_buffers=False,
                find_unused_parameters=False,
                bucket_cap_mb=args.ddp_bucket_cap_mb,
                gradient_as_bucket_view=True,
                static_graph=args.ddp_static_graph,
            )
        return model

    if not distributed:
        raise ValueError("--zero requires torchrun/distributed launch")
    if device.type != "cuda":
        raise ValueError("FSDP training requires CUDA")

    if model.num_block_chunks == 1:
        auto_wrap_policy = ModuleWrapPolicy([type(model.unregistered_blocks[0])])
    else:
        auto_wrap_policy = ModuleWrapPolicy([MultipleLayers])

    if args.enable_hybrid_shard:
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        if world_size % args.inner_shard_degree != 0:
            raise ValueError(f"world_size={world_size} is not divisible by inner_shard_degree={args.inner_shard_degree}")
        if not (1 < args.inner_shard_degree < world_size):
            raise ValueError(f"inner_shard_degree must be in (1, {world_size}), got {args.inner_shard_degree}")
        sharding_strategy = (
            ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
        )
        device_mesh = init_device_mesh("cuda", (world_size // args.inner_shard_degree, args.inner_shard_degree))
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
        device_mesh = None

    LOGGER.info("FSDP init zero=%s sharding=%s auto_wrap=%s", args.zero, sharding_strategy, auto_wrap_policy)
    return FSDP(
        model,
        device_id=device.index,
        sharding_strategy=sharding_strategy,
        mixed_precision=None,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=args.fsdp_use_orig_params,
        sync_module_states=True,
        limit_all_gathers=True,
        device_mesh=device_mesh,
    )


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> None:
    if max_norm <= 0:
        return
    if is_fsdp_model(model):
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads:
            return
        local_sq_norm = torch.zeros(1, device=grads[0].device, dtype=torch.float32)
        for grad in grads:
            local_sq_norm += grad.detach().float().pow(2).sum()
        dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM)
        total_norm = local_sq_norm.sqrt()
        clip_coef = float(max_norm) / float(total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def build_swanlab_experiment_name(args: argparse.Namespace, output_dir: Path) -> str:
    if args.swanlab_experiment_name:
        return args.swanlab_experiment_name
    if output_dir.parent.name and output_dir.parent.name != output_dir.anchor:
        return f"{args.swanlab_job_type}_{output_dir.parent.name}_{output_dir.name}"
    return f"{args.swanlab_job_type}_{output_dir.name}"


def swanlab_state_path(output_dir: Path) -> Path:
    return output_dir / "swanlab_run.json"


def find_swanlab_local_run_dir(logdir: Path, run_id: str, state: dict[str, Any]) -> Path | None:
    saved = str(state.get("local_run_dir", "")).strip()
    if saved and Path(saved).is_dir():
        return Path(saved)
    if not run_id:
        return None
    matches = sorted(logdir.glob(f"run-*-{run_id}"))
    return matches[-1] if matches else None


@contextmanager
def swanlab_local_resume_patch(logdir: Path, run_id: str, run_dir: Path | None):
    if not run_id or run_dir is None:
        yield
        return
    try:
        import swanlab.data.callbacker.local as swanlab_local
        import swanlab.data.sdk as swanlab_sdk
    except Exception:
        yield
        return

    parts = run_dir.name.split("-")
    if len(parts) < 3:
        yield
        return
    try:
        fixed_now = datetime.strptime(parts[1], "%Y%m%d_%H%M%S")
    except ValueError:
        yield
        return

    original_generate_run_id = swanlab_local.N.generate_run_id
    original_datetime = swanlab_sdk.datetime
    original_mkdir = swanlab_sdk.os.mkdir
    target_run_dir = run_dir.resolve()

    class FixedDatetime:
        @classmethod
        def now(cls):
            return fixed_now

    def mkdir_existing_run(path, mode=0o777, *args, **kwargs):
        if Path(path).resolve() == target_run_dir and target_run_dir.is_dir():
            return None
        return original_mkdir(path, mode, *args, **kwargs)

    swanlab_local.N.generate_run_id = lambda: run_id
    swanlab_sdk.datetime = FixedDatetime
    swanlab_sdk.os.mkdir = mkdir_existing_run
    try:
        yield
    finally:
        swanlab_local.N.generate_run_id = original_generate_run_id
        swanlab_sdk.datetime = original_datetime
        swanlab_sdk.os.mkdir = original_mkdir


def init_swanlab(args: argparse.Namespace, output_dir: Path, enabled: bool) -> Any | None:
    if not enabled or args.swanlab_mode == "disabled":
        return None
    if args.swanlab_mode == "local" and importlib.util.find_spec("swanboard") is None:
        LOGGER.warning("Disabled SwanLab local mode because swanboard is not installed.")
        return None
    if swanlab is None:
        raise ImportError("swanlab is not installed, but SwanLab logging is enabled.")

    state_path = swanlab_state_path(output_dir)
    state: dict[str, Any] = {}
    run_id = ""
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        run_id = str(state.get("run_id", "")).strip()

    logdir = Path(args.swanlab_logdir) if args.swanlab_logdir else output_dir / "swanlab"
    logdir.mkdir(parents=True, exist_ok=True)

    init_kwargs: dict[str, Any] = {
        "project": args.swanlab_project,
        "experiment_name": build_swanlab_experiment_name(args, output_dir),
        "job_type": args.swanlab_job_type,
        "tags": list(args.swanlab_tags),
        "config": vars(args),
        "logdir": str(logdir),
        "mode": args.swanlab_mode,
        "reinit": True,
    }
    workspace = args.swanlab_workspace.strip()
    if workspace:
        init_kwargs["workspace"] = workspace
    if run_id and args.swanlab_mode == "cloud":
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    local_run_dir = find_swanlab_local_run_dir(logdir, run_id, state) if args.swanlab_mode == "local" else None
    with swanlab_local_resume_patch(logdir, run_id, local_run_dir):
        try:
            run = swanlab.init(**init_kwargs)
        except Exception:
            LOGGER.exception("SwanLab init failed mode=%s logdir=%s", args.swanlab_mode, logdir)
            raise
    run_public = getattr(run, "public", None)
    current_run_id = str(getattr(run_public, "run_id", "")).strip()
    current_run_dir = str(getattr(run_public, "run_dir", "")).strip()
    state_payload = {
        "project": args.swanlab_project,
        "workspace": workspace,
        "experiment_name": build_swanlab_experiment_name(args, output_dir),
        "mode": args.swanlab_mode,
        "logdir": str(logdir),
    }
    if current_run_id:
        state_payload["run_id"] = current_run_id
    if args.swanlab_mode == "local" and current_run_dir:
        state_payload["local_run_dir"] = current_run_dir
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def save_visuals(
    output_dir: Path,
    swanlab_run: Any | None,
    stage: str,
    step: int,
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    is_main: bool,
) -> None:
    if not is_main:
        return

    if image.shape[-2:] != target.shape[-2:]:
        image = F.interpolate(image.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(prediction.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
        prediction = prediction / torch.linalg.norm(prediction, dim=1, keepdim=True).clamp_min(1e-6)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_vis = image[:4].detach().cpu().float().clamp(0.0, 1.0)
    target_vis = normals_to_vis(target[:4])
    prediction_vis = normals_to_vis(prediction[:4])
    angle_vis = torch.rad2deg(
        torch.acos((F.normalize(prediction[:4].float(), dim=1) * F.normalize(target[:4].float(), dim=1)).sum(dim=1).clamp(-1.0, 1.0))
    ).unsqueeze(1).detach().cpu().float().div(45.0).clamp(0.0, 1.0)
    angle_vis = angle_vis.repeat(1, 3, 1, 1)

    image_grid = make_grid(image_vis, nrow=min(4, image_vis.shape[0]))
    target_grid = make_grid(target_vis, nrow=min(4, target_vis.shape[0]))
    prediction_grid = make_grid(prediction_vis, nrow=min(4, prediction_vis.shape[0]))
    angle_grid = make_grid(angle_vis, nrow=min(4, angle_vis.shape[0]))
    compare = make_grid(torch.cat([image_vis, target_vis, prediction_vis, angle_vis], dim=0), nrow=min(4, image_vis.shape[0]))
    save_image(image_grid, image_dir / f"{stage}_rgb_step_{step:07d}.png")
    save_image(target_grid, image_dir / f"{stage}_normal_gt_step_{step:07d}.png")
    save_image(prediction_grid, image_dir / f"{stage}_normal_pred_step_{step:07d}.png")
    save_image(angle_grid, image_dir / f"{stage}_normal_angle_step_{step:07d}.png")
    save_image(compare, image_dir / f"{stage}_compare_step_{step:07d}.png")
    if swanlab_run is not None:
        swanlab_run.log(
            {
                f"{stage}/rgb": swanlab.Image(image_grid),
                f"{stage}/normal_gt": swanlab.Image(target_grid),
                f"{stage}/normal_pred": swanlab.Image(prediction_grid),
                f"{stage}/normal_angle": swanlab.Image(angle_grid),
                f"{stage}/compare": swanlab.Image(compare),
            },
            step=step,
        )


def save_checkpoint(
    checkpoint_path: Path,
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    step: int,
    best_val_angle: float,
    is_main: bool,
) -> None:
    payload = {
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "best_val_angle": best_val_angle,
        "zero": args.zero,
    }
    if is_fsdp_model(model):
        require_fsdp_state_dict_api()
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FULLSTATE_SAVE_POLICY, FULLOPTSTATE_SAVE_POLICY):
            payload["gpt"] = model.state_dict()
            if args.save_optimizer_state:
                payload["optimizer"] = FSDP.optim_state_dict(
                    model=model,
                    optim=optimizer,
                    optim_state_dict=optimizer.state_dict(),
                )
                payload["scheduler"] = scheduler.state_dict()
                payload["scaler"] = scaler.state_dict() if scaler is not None else None
    else:
        raw_model = model.module if isinstance(model, DDP) else model
        payload["gpt"] = raw_model.state_dict()
        if args.save_optimizer_state:
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
            payload["scaler"] = scaler.state_dict() if scaler is not None else None

    if is_main:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    dist_barrier_if_initialized()


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    if args.resume:
        requested = Path(args.resume)
        if requested.name == "last.pth":
            step_checkpoint = requested.with_name("last_step.pth")
            if step_checkpoint.is_file():
                return step_checkpoint
        if requested.is_file():
            return requested
        return None

    for candidate in (checkpoint_dir / "last_step.pth", checkpoint_dir / "last.pth"):
        if candidate.is_file():
            return candidate
    return None


def maybe_resume(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    is_main: bool,
) -> tuple[int, int, float]:
    resume_path = resolve_resume_path(args)
    if resume_path is None:
        return 0, 0, float("inf")

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_zero = int(checkpoint.get("zero", 0))
    if is_fsdp_model(model):
        require_fsdp_state_dict_api()
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FULLSTATE_SAVE_POLICY, FULLOPTSTATE_SAVE_POLICY):
            model.load_state_dict(checkpoint["gpt"], strict=True)
    else:
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(checkpoint["gpt"], strict=True)

    has_optimizer_state = "optimizer" in checkpoint and "scheduler" in checkpoint
    optimizer_loaded = False
    if has_optimizer_state:
        if is_fsdp_model(model) and checkpoint_zero > 0:
            optim_state_dict = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=checkpoint["optimizer"],
            )
            optimizer.load_state_dict(optim_state_dict)
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            optimizer_loaded = True
        elif not is_fsdp_model(model) and checkpoint_zero == 0:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            optimizer_loaded = True
        elif is_main:
            LOGGER.info(
                "Resumed weights from %s but skipped optimizer state because checkpoint zero=%s and current zero=%s",
                resume_path,
                checkpoint_zero,
                args.zero,
            )
    start_epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    best_val_angle = float(checkpoint.get("best_val_angle", float("inf")))
    if not optimizer_loaded and step > 0:
        lr_step = args.resume_lr_step if args.resume_lr_step >= 0 else step
        for _ in range(lr_step):
            scheduler.step()
        if is_main:
            LOGGER.info(
                "Fast-forwarded LR scheduler to step=%d because checkpoint has no optimizer/scheduler state",
                lr_step,
            )
    if is_main:
        resume_mode = "weights+optimizer" if optimizer_loaded else "weights-only"
        LOGGER.info("Resumed from %s (epoch=%s, step=%s, mode=%s)", resume_path, start_epoch, step, resume_mode)
    return start_epoch, step, best_val_angle


def maybe_load_init_model(model: torch.nn.Module, init_model: str, is_main: bool) -> None:
    if not init_model:
        return
    missing, unexpected = load_infinity_state_dict(model.module if isinstance(model, DDP) else model, init_model)
    if is_main:
        LOGGER.info(
            "Warm-started Infinity weights from %s (missing=%d, unexpected=%d)",
            init_model,
            len(missing),
            len(unexpected),
        )


def forward_batch(
    *,
    model: torch.nn.Module,
    normal_vae: torch.nn.Module,
    rgb_vae: torch.nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    precision: str,
    compute_decoded_metrics: bool = True,
    include_aux_loss: bool = True,
    require_token_cache_hit: bool = True,
    timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    with timed_stage(timings, "schedule", device):
        target_cpu, scale_schedule = resolve_batch_scale_schedule(batch, args)
    image: torch.Tensor | None = None
    target: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    with torch.no_grad():
        cache_dir = Path(args.token_cache_dir) if token_cache_enabled(args) else None
        cache_signature = token_cache_signature(args, scale_schedule) if cache_dir is not None else ""
        with timed_stage(timings, "cache_load", device):
            cached_tokens = (
                load_token_cache_batch(cache_dir, batch["metadata"], cache_signature, device, args.token_cache_memory)
                if cache_dir is not None
                else None
            )
        if cached_tokens is not None:
            rgb_prefix_blc, x_blc_without_prefix, gt_bl, raw_features = cached_tokens
        else:
            if cache_dir is not None and args.token_cache_require_hit and require_token_cache_hit:
                missing_paths = missing_token_cache_paths(cache_dir, batch["metadata"], cache_signature)
                preview = ", ".join(str(path) for path in missing_paths[:3])
                raise FileNotFoundError(f"Token cache miss for {len(missing_paths)} samples. First missing paths: {preview}")
            if not batch_has_full_tensors(batch):
                with timed_stage(timings, "raw_materialize", device):
                    batch = materialize_batch_from_metadata(batch, args)
                with timed_stage(timings, "schedule", device):
                    target_cpu, scale_schedule = resolve_batch_scale_schedule(batch, args)
                    cache_signature = token_cache_signature(args, scale_schedule) if cache_dir is not None else ""
            with timed_stage(timings, "batch_to_gpu", device):
                image = batch["image"].to(device, non_blocking=True)
                target = target_cpu.to(device, non_blocking=True)
            with timed_stage(timings, "rgb_tokenize", device):
                rgb_prefix_blc = build_prefix_tokens_from_image(
                    image_01=image,
                    rgb_vae=rgb_vae,
                    scale_schedule=scale_schedule,
                    apply_spatial_patchify=args.rgb_apply_spatial_patchify,
                )
            normal_vae_scale_schedule = [
                (pt, 2 * ph, 2 * pw) if args.normal_apply_spatial_patchify else (pt, ph, pw)
                for pt, ph, pw in scale_schedule
            ]
            with timed_stage(timings, "normal_tokenize", device):
                raw_features, _, _ = normal_vae.encode_for_raw_features(target, scale_schedule=normal_vae_scale_schedule)
                x_blc_without_prefix, gt_ms_idx_bl = build_multiscale_var_inputs(
                    vae=normal_vae,
                    raw_features=raw_features,
                    vae_scale_schedule=normal_vae_scale_schedule,
                    apply_spatial_patchify=args.normal_apply_spatial_patchify,
                    noise_apply_layers=args.noise_apply_layers,
                    noise_apply_strength=args.noise_apply_strength,
                    noise_apply_requant=args.noise_apply_requant,
                )
                total_seq_len = int(sum(np.array(item).prod() for item in scale_schedule))
                gt_bl = torch.cat(gt_ms_idx_bl, dim=1)[:, :total_seq_len].contiguous().long()
            if cache_dir is not None:
                first_stage_len = int(np.array(scale_schedule[0]).prod())
                cache_x_blc = x_blc_without_prefix[:, : total_seq_len - first_stage_len, :].contiguous()
                with timed_stage(timings, "cache_save", device):
                    save_token_cache_batch(
                        cache_dir,
                        batch["metadata"],
                        cache_signature,
                        rgb_prefix_blc,
                        cache_x_blc,
                        gt_bl,
                        raw_features,
                        args.token_cache_memory,
                    )

    total_seq_len = int(sum(np.array(item).prod() for item in scale_schedule))
    first_stage_len = int(np.array(scale_schedule[0]).prod())
    x_blc_without_prefix = x_blc_without_prefix[:, : total_seq_len - first_stage_len, :]
    gt_bl = gt_bl[:, :total_seq_len].contiguous().long()

    activation_context = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if getattr(args, "normal_save_activations_on_cpu", False)
        else nullcontext()
    )

    with precision_context(precision):
        with activation_context:
            with timed_stage(timings, "model_forward", device):
                logits_blv = model(rgb_prefix_blc, x_blc_without_prefix, scale_schedule=scale_schedule)
        with timed_stage(timings, "ce_loss", device):
            ce_loss, ce_metrics = compute_ce_loss(logits_blv, gt_bl, use_bit_label=args.use_bit_label)
        if compute_decoded_metrics:
            with timed_stage(timings, "decoded_metrics", device):
                if not batch_has_full_tensors(batch):
                    batch = materialize_batch_from_metadata(batch, args)
                    target_cpu = batch["target"]
                if target is None:
                    target = target_cpu.to(device, non_blocking=True)
                if mask is None:
                    mask = batch["mask"].to(device, non_blocking=True)
                with torch.set_grad_enabled(include_aux_loss and torch.is_grad_enabled()):
                    prediction, latent_prediction = decode_logits_to_normal(
                        logits_blv=logits_blv,
                        vae=normal_vae,
                        scale_schedule=scale_schedule,
                        use_bit_label=args.use_bit_label,
                        apply_spatial_patchify=args.normal_apply_spatial_patchify,
                    )
                    target_for_prediction, mask_for_prediction = maybe_resize_target_for_prediction(prediction, target, mask)
                    if latent_prediction.shape == raw_features.unsqueeze(2).shape:
                        latent_target = raw_features.unsqueeze(2)
                    else:
                        latent_target = None
                    aux_loss, aux_metrics = compute_normal_metrics(
                        prediction=prediction,
                        target=target_for_prediction,
                        mask=mask_for_prediction,
                        latent_prediction=latent_prediction,
                        latent_target=latent_target,
                        l1_weight=args.normal_l1_weight,
                        angular_weight=args.normal_angular_weight,
                        latent_weight=args.normal_latent_weight,
                        norm_weight=args.normal_norm_weight,
                    )
        else:
            prediction = target if target is not None else target_cpu
            aux_loss = ce_loss.new_tensor(0.0)
            aux_metrics = {
                "loss_l1": aux_loss.detach(),
                "loss_angular_rad": aux_loss.detach(),
                "loss_latent": aux_loss.detach(),
                "loss_norm": aux_loss.detach(),
                "angle_deg": aux_loss.new_tensor(float("nan")),
                "acc_11_25": aux_loss.new_tensor(float("nan")),
                "acc_22_5": aux_loss.new_tensor(float("nan")),
                "acc_30": aux_loss.new_tensor(float("nan")),
                "loss_total_aux": aux_loss.detach(),
            }
        total_loss = args.ce_weight * ce_loss + (aux_loss if include_aux_loss else ce_loss.new_tensor(0.0))

    metrics = {
        **ce_metrics,
        **aux_metrics,
        "loss_total": total_loss.detach(),
    }
    if image is None:
        image = batch["image"] if batch["image"].numel() > 0 else torch.empty(0)
    return total_loss, metrics, prediction.detach(), image.detach()


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    normal_vae: torch.nn.Module,
    rgb_vae: torch.nn.Module,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    training = model.training
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0
    visual_payload = None
    for batch in dataloader:
        _, metrics, prediction, image = forward_batch(
            model=model,
            normal_vae=normal_vae,
            rgb_vae=rgb_vae,
            batch=batch,
            args=args,
            device=device,
            precision=args.precision,
            require_token_cache_hit=False,
        )
        if visual_payload is None:
            visual_payload = {
                "image": image.detach().cpu(),
                "target": batch["target"].detach().cpu(),
                "prediction": prediction.detach().cpu(),
            }
        batch_size = batch["image"].shape[0]
        count += batch_size
        for key, value in metrics.items():
            sums[key] = sums.get(key, value.new_tensor(0.0)) + value * batch_size

    count_tensor = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(count_tensor.item()))
    result = {key: float(value.item() / total_count) for key, value in sums.items()}
    if training:
        model.train()
    return result, visual_payload


@torch.no_grad()
def evaluate_ar(
    *,
    model: torch.nn.Module,
    normal_vae: torch.nn.Module,
    rgb_vae: torch.nn.Module,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    training = model.training
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    sums: dict[str, torch.Tensor] = {}
    count = 0
    visual_payload = None

    for batch in dataloader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        scale_schedule = [tuple(item) for item in resolve_scale_schedule_from_hw(target.shape[-2], target.shape[-1], args.pn)[1]]
        training_scales = min(args.always_training_scales, len(scale_schedule))
        scale_schedule = scale_schedule[:training_scales]

        rgb_prefix_blc = build_prefix_tokens_from_image(
            image_01=image,
            rgb_vae=rgb_vae,
            scale_schedule=scale_schedule,
            apply_spatial_patchify=args.rgb_apply_spatial_patchify,
        )
        with precision_context(args.precision):
            prediction = raw_model.autoregressive_infer_prefix(
                vae=normal_vae,
                rgb_prefix_blc=rgb_prefix_blc,
                scale_schedule=scale_schedule,
                top_k=args.ar_eval_top_k,
                top_p=args.ar_eval_top_p,
                tau=args.ar_eval_tau,
                vae_type=args.normal_vae_type,
            )
        target_for_prediction, mask_for_prediction = maybe_resize_target_for_prediction(prediction, target, mask)
        _, metrics = compute_normal_metrics(
            prediction=prediction,
            target=target_for_prediction,
            mask=mask_for_prediction,
            l1_weight=0.0,
            angular_weight=0.0,
            latent_weight=0.0,
            norm_weight=0.0,
        )
        if visual_payload is None:
            visual_payload = {
                "image": image.detach().cpu(),
                "target": target.detach().cpu(),
                "prediction": prediction.detach().cpu(),
            }
        batch_size = image.shape[0]
        count += batch_size
        for key, value in metrics.items():
            sums[key] = sums.get(key, value.new_tensor(0.0)) + value * batch_size

    count_tensor = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(count_tensor.item()))
    result = {key: float(value.item() / total_count) for key, value in sums.items()}
    result["samples"] = total_count
    if training:
        model.train()
    return result, visual_payload


def main() -> int:
    script_start_time = time.perf_counter()
    args = parse_args()
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    init_timings: dict[str, float] = {}
    distributed, rank, world_size, device = init_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, is_main=(rank == 0))
    seed_everything(args.seed, rank)

    if rank == 0:
        (output_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("world_size=%s device=%s", world_size, device)

    swanlab_run = None
    try:
        if rank == 0:
            LOGGER.info(
                "batch_size_per_gpu=%d grad_accum_steps=%d global_batch_size=%d effective_global_batch_size=%d val_batch_size_per_gpu=%d zero=%d swanlab_mode=%s",
                args.batch_size,
                args.grad_accum_steps,
                args.batch_size * world_size,
                args.batch_size * world_size * args.grad_accum_steps,
                args.val_batch_size,
                args.zero,
                args.swanlab_mode,
            )
            if args.token_cache_metadata_only and not token_cache_enabled(args):
                LOGGER.warning("--token-cache-metadata-only was requested, but token cache is disabled for this run.")

        with timed_stage(init_timings if args.profile_timings else None, "dataset_loader", device):
            train_dataset_names = parse_train_dataset_names(args.train_datasets)
            train_dataset_weights = parse_train_dataset_weights(args.train_dataset_weights, train_dataset_names)
            train_dataset = build_train_dataset(args)
            if rank == 0:
                LOGGER.info(
                    "train_datasets=%s train_dataset_weights=%s",
                    ",".join(train_dataset_names),
                    ",".join(f"{name}:{train_dataset_weights[name]:g}" for name in train_dataset_names),
                )
            if args.token_cache_filter_missing:
                if not token_cache_enabled(args):
                    raise ValueError("--token-cache-filter-missing requires --token-cache-dir.")
                train_dataset, original_len = filter_dataset_to_token_cache_hits(train_dataset, args, Path(args.token_cache_dir))
                if rank == 0:
                    LOGGER.info(
                        "Filtered train dataset by token cache hits: kept=%d original=%d",
                        len(train_dataset),
                        original_len,
                    )
                if len(train_dataset) == 0:
                    raise RuntimeError("No train samples have token-cache entries after --token-cache-filter-missing.")
            val_dataset = HypersimNormalDataset(
                root=args.data_root,
                partition=args.val_partition,
                pn=args.pn,
                max_samples=args.max_val_samples,
            )
            ar_val_dataset = HypersimNormalDataset(
                root=args.data_root,
                partition=args.val_partition,
                pn=args.pn,
                max_samples=args.ar_eval_samples,
            ) if args.ar_eval_samples > 0 else None
            train_shuffle = not args.token_cache_filter_missing
            if rank == 0 and not train_shuffle:
                LOGGER.info("Disabled train shuffle because --token-cache-filter-missing is enabled.")
            train_loader, train_sampler = make_dataloader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                shuffle=train_shuffle,
                drop_last=True,
                group_by_target_size=True,
                dataset_weights=train_dataset_weights,
            )
            val_loader, _ = make_dataloader(
                val_dataset,
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                drop_last=False,
            )
            ar_val_loader = None
            if ar_val_dataset is not None:
                ar_val_loader, _ = make_dataloader(
                    ar_val_dataset,
                    batch_size=args.val_batch_size,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                    drop_last=False,
                )

        with timed_stage(init_timings if args.profile_timings else None, "vae_load", device):
            normal_vae = build_bsq_vae(
                ckpt_path=args.normal_vae_ckpt,
                codebook_dim=args.normal_vae_type,
                apply_spatial_patchify=args.normal_apply_spatial_patchify,
                device=device,
            )
            rgb_vae = build_bsq_vae(
                ckpt_path=args.rgb_vae_ckpt,
                codebook_dim=args.rgb_vae_type,
                apply_spatial_patchify=args.rgb_apply_spatial_patchify,
                device=device,
            )

        with timed_stage(init_timings if args.profile_timings else None, "model_build", device):
            with skip_torch_weight_init(args.fast_model_init and bool(args.init_model)):
                model = build_infinity_normal_model(
                    model_name=args.model_name,
                    vae_local=normal_vae,
                    pn=args.pn,
                    batch_size=args.batch_size,
                    use_bit_label=args.use_bit_label,
                    add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
                    rope2d_each_sa_layer=args.rope2d_each_sa_layer,
                    rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
                    apply_spatial_patchify=args.normal_apply_spatial_patchify,
                    checkpointing=None if args.checkpointing == "none" else args.checkpointing,
                    normal_use_flex_attn=args.normal_use_flex_attn,
                    normal_use_segmented_flash_attn=args.normal_use_segmented_flash_attn,
                    normal_bf16_activations=args.normal_bf16_activations,
                    fused_mlp=args.fused_mlp,
                    fused_norm=args.fused_norm,
                    device=device,
                )
            if args.checkpointing == "full-block":
                model.checkpointing_full_block_skip_interval = max(0, int(args.full_block_checkpoint_skip_interval))
            model = convert_model_precision(model, args.precision)
        with timed_stage(init_timings if args.profile_timings else None, "init_model_load", device):
            maybe_load_init_model(model, args.init_model, is_main=(rank == 0))
        with timed_stage(init_timings if args.profile_timings else None, "distributed_wrap", device):
            model = build_training_model(model, args=args, device=device, distributed=distributed)

        optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
        total_steps = optimizer_steps_per_epoch * args.epochs
        if args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)
        with timed_stage(init_timings if args.profile_timings else None, "optimizer_build", device):
            optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_steps=max(1, total_steps))
            scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
        with timed_stage(init_timings if args.profile_timings else None, "resume_load", device):
            start_epoch, global_step, best_val_angle = maybe_resume(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if args.precision == "fp16" else None,
                is_main=(rank == 0),
            )
        if global_step > 0:
            step_epoch = min(global_step // optimizer_steps_per_epoch, args.epochs)
            if step_epoch > start_epoch:
                if rank == 0:
                    LOGGER.info(
                        "Adjusted start_epoch from %d to %d based on global_step=%d and optimizer_steps_per_epoch=%d",
                        start_epoch,
                        step_epoch,
                        global_step,
                        optimizer_steps_per_epoch,
                    )
                start_epoch = step_epoch

        with timed_stage(init_timings if args.profile_timings else None, "swanlab_init", device):
            swanlab_run = init_swanlab(args, output_dir, enabled=(rank == 0))
        if args.profile_timings:
            init_timings["startup_total"] = time.perf_counter() - script_start_time
            reduced_init_timings = reduce_timing_dict(init_timings, distributed, device)
            if rank == 0:
                LOGGER.info("profile_init %s", format_timing_dict(reduced_init_timings))

        model.train()
        start_time = time.time()
        profile_steps_logged = 0
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            resume_batch_offset = (
                (global_step % optimizer_steps_per_epoch) * args.grad_accum_steps
                if epoch == start_epoch
                else 0
            )
            if resume_batch_offset and rank == 0:
                LOGGER.info(
                    "Skipping %d already-completed batches in resumed epoch %d",
                    resume_batch_offset,
                    epoch + 1,
                )
            for batch_idx, batch in enumerate(train_loader):
                step_timings = {} if args.profile_timings else None
                if step_timings is not None and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if batch_idx < resume_batch_offset:
                    continue
                is_last_batch = batch_idx + 1 == len(train_loader)
                accum_index = (batch_idx - resume_batch_offset) % args.grad_accum_steps
                accum_count = accum_index + 1
                accum_start_batch = batch_idx - accum_index
                current_accum_steps = min(args.grad_accum_steps, len(train_loader) - accum_start_batch)
                should_step_optimizer = accum_count == args.grad_accum_steps or is_last_batch
                next_step = global_step + 1
                profile_this_step = should_step_optimizer and args.profile_torch_step == next_step and rank == 0
                profiler_context = (
                    profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                    )
                    if profile_this_step
                    else nullcontext()
                )
                should_log_step = should_step_optimizer and (next_step % args.log_every == 0 or next_step == 1)
                should_log_image = should_step_optimizer and args.image_log_every > 0 and (
                    next_step % args.image_log_every == 0 or next_step == 1
                )
                should_compute_train_normals = should_log_image or (
                    should_step_optimizer
                    and args.train_normal_metrics_every > 0
                    and next_step % args.train_normal_metrics_every == 0
                )
                with profiler_context as torch_prof:
                    if accum_index == 0:
                        with timed_stage(step_timings, "zero_grad", device):
                            optimizer.zero_grad(set_to_none=True)
                    sync_context = (
                        model.no_sync()
                        if (
                            distributed
                            and world_size > 1
                            and isinstance(model, DDP)
                            and not args.ddp_static_graph
                            and not should_step_optimizer
                        )
                        else nullcontext()
                    )
                    with sync_context:
                        total_loss, metrics, prediction, image = forward_batch(
                            model=model,
                            normal_vae=normal_vae,
                            rgb_vae=rgb_vae,
                            batch=batch,
                            args=args,
                            device=device,
                            precision=args.precision,
                            compute_decoded_metrics=should_compute_train_normals,
                            include_aux_loss=False,
                            require_token_cache_hit=True,
                            timings=step_timings,
                        )
                        backward_loss = total_loss / current_accum_steps

                        with timed_stage(step_timings, "backward", device):
                            if args.precision == "fp16":
                                scaler.scale(backward_loss).backward()
                            else:
                                backward_loss.backward()
                    if not should_step_optimizer:
                        continue

                    if args.precision == "fp16":
                        if args.grad_clip > 0:
                            with timed_stage(step_timings, "grad_clip", device):
                                scaler.unscale_(optimizer)
                                clip_grad_norm(model, args.grad_clip)
                        with timed_stage(step_timings, "optimizer_step", device):
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        with timed_stage(step_timings, "grad_clip", device):
                            clip_grad_norm(model, args.grad_clip)
                        with timed_stage(step_timings, "optimizer_step", device):
                            optimizer.step()
                    with timed_stage(step_timings, "scheduler_step", device):
                        scheduler.step()
                if profile_this_step:
                    torch_profile_dir = output_dir / "profiles"
                    torch_profile_dir.mkdir(parents=True, exist_ok=True)
                    table = torch_prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=args.profile_torch_row_limit,
                    )
                    table_path = torch_profile_dir / f"torch_step_{next_step:06d}.txt"
                    trace_path = torch_profile_dir / f"torch_step_{next_step:06d}.json"
                    table_path.write_text(table)
                    torch_prof.export_chrome_trace(str(trace_path))
                    LOGGER.info("Wrote torch profiler table to %s and trace to %s", table_path, trace_path)
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                if should_log_step:
                    reduced = reduce_metrics(metrics, distributed)
                if rank == 0 and should_log_step:
                    normal_metric_keys = (
                        "loss_total_aux",
                        "loss_l1",
                        "loss_angular_rad",
                        "loss_latent",
                        "loss_norm",
                        "angle_deg",
                        "acc_11_25",
                        "acc_22_5",
                        "acc_30",
                    )
                    has_normal_metrics = all(
                        math.isfinite(reduced[key])
                        for key in ("angle_deg", "acc_11_25", "acc_22_5", "acc_30")
                    )
                    if has_normal_metrics:
                        LOGGER.info(
                            "epoch=%d batch=%d/%d global_step=%d/%d loss=%.4f ce=%.4f aux=%.4f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f lr=%.2e",
                            epoch + 1,
                            batch_idx + 1,
                            len(train_loader),
                            global_step,
                            total_steps,
                            reduced["loss_total"],
                            reduced["loss_ce"],
                            reduced["loss_total_aux"],
                            reduced["angle_deg"],
                            reduced["acc_11_25"],
                            reduced["acc_22_5"],
                            reduced["acc_30"],
                            current_lr,
                        )
                    else:
                        LOGGER.info(
                            "epoch=%d batch=%d/%d global_step=%d/%d loss=%.4f ce=%.4f lr=%.2e",
                            epoch + 1,
                            batch_idx + 1,
                            len(train_loader),
                            global_step,
                            total_steps,
                            reduced["loss_total"],
                            reduced["loss_ce"],
                            current_lr,
                        )
                    if swanlab_run is not None:
                        payload = {
                            f"train/{key}": value
                            for key, value in reduced.items()
                            if has_normal_metrics or key not in normal_metric_keys
                        }
                        payload["train/lr"] = current_lr
                        payload["train/epoch"] = epoch + 1
                        swanlab_run.log(payload, step=global_step)
                if step_timings is not None:
                    step_timings["step_total"] = sum(step_timings.values())
                    if device.type == "cuda":
                        step_timings["cuda_peak_alloc_gib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        step_timings["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                    if (
                        global_step > args.profile_warmup_steps
                        and (args.profile_max_steps <= 0 or profile_steps_logged < args.profile_max_steps)
                    ):
                        reduced_step_timings = reduce_timing_dict(step_timings, distributed, device)
                        if rank == 0:
                            LOGGER.info("profile_step global_step=%d %s", global_step, format_timing_dict(reduced_step_timings))
                        profile_steps_logged += 1
                if rank == 0 and should_log_image:
                    if batch["target"].numel() == 0:
                        batch = materialize_batch_from_metadata(batch, args)
                    if image.numel() == 0:
                        image = batch["image"]
                    save_visuals(
                        output_dir,
                        swanlab_run,
                        "train",
                        global_step,
                        image,
                        batch["target"].to(device),
                        prediction,
                        is_main=True,
                    )

                if (
                    ar_val_loader is not None
                    and args.ar_eval_every > 0
                    and global_step % args.ar_eval_every == 0
                ):
                    ar_metrics, ar_visuals = evaluate_ar(
                        model=model,
                        normal_vae=normal_vae,
                        rgb_vae=rgb_vae,
                        dataloader=ar_val_loader,
                        args=args,
                        device=device,
                        distributed=distributed,
                    )
                    if rank == 0:
                        LOGGER.info(
                            "val_ar step=%d samples=%.0f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                            global_step,
                            ar_metrics["samples"],
                            ar_metrics["angle_deg"],
                            ar_metrics["acc_11_25"],
                            ar_metrics["acc_22_5"],
                            ar_metrics["acc_30"],
                        )
                        if swanlab_run is not None:
                            payload = {f"val_ar/{key}": value for key, value in ar_metrics.items()}
                            swanlab_run.log(payload, step=global_step)
                        if ar_visuals is not None:
                            save_visuals(
                                output_dir,
                                swanlab_run,
                                "val_ar",
                                global_step,
                                ar_visuals["image"],
                                ar_visuals["target"],
                                ar_visuals["prediction"],
                                is_main=True,
                            )
                    dist_barrier_if_initialized()

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / "last_step.pth",
                        args=args,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if args.precision == "fp16" else None,
                        epoch=epoch,
                        step=global_step,
                        best_val_angle=best_val_angle,
                        is_main=(rank == 0),
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            val_metrics, val_visuals = evaluate(
                model=model,
                normal_vae=normal_vae,
                rgb_vae=rgb_vae,
                dataloader=val_loader,
                args=args,
                device=device,
                distributed=distributed,
            )
            should_save_last = args.save_every_epoch > 0 and (epoch + 1) % args.save_every_epoch == 0
            is_best = val_metrics["angle_deg"] < best_val_angle
            if is_best:
                best_val_angle = val_metrics["angle_deg"]
            if rank == 0:
                LOGGER.info(
                    "val epoch=%d loss=%.4f ce=%.4f aux=%.4f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                    epoch + 1,
                    val_metrics["loss_total"],
                    val_metrics["loss_ce"],
                    val_metrics["loss_total_aux"],
                    val_metrics["angle_deg"],
                    val_metrics["acc_11_25"],
                    val_metrics["acc_22_5"],
                    val_metrics["acc_30"],
                )
                if swanlab_run is not None:
                    payload = {f"val/{key}": value for key, value in val_metrics.items()}
                    payload["val/epoch"] = epoch + 1
                    swanlab_run.log(payload, step=global_step)
                if val_visuals is not None:
                    save_visuals(
                        output_dir,
                        swanlab_run,
                        "val",
                        global_step,
                        val_visuals["image"],
                        val_visuals["target"],
                        val_visuals["prediction"],
                        is_main=True,
                    )
            if ar_val_loader is not None and args.ar_eval_every > 0:
                ar_metrics, ar_visuals = evaluate_ar(
                    model=model,
                    normal_vae=normal_vae,
                    rgb_vae=rgb_vae,
                    dataloader=ar_val_loader,
                    args=args,
                    device=device,
                    distributed=distributed,
                )
                if rank == 0:
                    LOGGER.info(
                        "val_ar epoch=%d samples=%.0f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                        epoch + 1,
                        ar_metrics["samples"],
                        ar_metrics["angle_deg"],
                        ar_metrics["acc_11_25"],
                        ar_metrics["acc_22_5"],
                        ar_metrics["acc_30"],
                    )
                    if swanlab_run is not None:
                        payload = {f"val_ar/{key}": value for key, value in ar_metrics.items()}
                        payload["val_ar/epoch"] = epoch + 1
                        swanlab_run.log(payload, step=global_step)
                    if ar_visuals is not None:
                        save_visuals(
                            output_dir,
                            swanlab_run,
                            "val_ar",
                            global_step,
                            ar_visuals["image"],
                            ar_visuals["target"],
                            ar_visuals["prediction"],
                            is_main=True,
                        )
                dist_barrier_if_initialized()
            if should_save_last:
                save_checkpoint(
                    output_dir / "checkpoints" / "last.pth",
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if args.precision == "fp16" else None,
                    epoch=epoch + 1,
                    step=global_step,
                    best_val_angle=best_val_angle,
                    is_main=(rank == 0),
                )
            if is_best:
                save_checkpoint(
                    output_dir / "checkpoints" / f"best_angle_{best_val_angle:.4f}.pth",
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if args.precision == "fp16" else None,
                    epoch=epoch + 1,
                    step=global_step,
                    best_val_angle=best_val_angle,
                    is_main=(rank == 0),
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if rank == 0:
            elapsed = time.time() - start_time
            LOGGER.info("finished in %.1fs, best_val_angle=%.4f", elapsed, best_val_angle)
        return 0
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    raise SystemExit(main())
