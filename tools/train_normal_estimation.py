from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.utils import make_grid, save_image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infinity.normal_estimation import (  # noqa: E402
    HypersimNormalDataset,
    NYUv2ParquetNormalDataset,
    build_normal_dataloader,
    build_normal_train_dataset,
    build_bsq_vae,
    build_prefix_tokens_from_image,
    build_infinity_normal_model,
    build_multiscale_var_inputs,
    collate_normal_estimation_batch,
    compute_normal_metrics,
    decode_logits_to_normal,
    dataset_metadata_at,
    load_normal_sample_from_metadata,
    load_infinity_state_dict,
    normals_to_vis,
    atomic_torch_save,
    parse_train_dataset_names,
    parse_train_dataset_weights,
    resolve_checkpoint_resume_path,
    require_positive_steps_per_epoch,
    resolve_scale_schedule_from_hw,
)
from infinity.models.infinity import MultipleLayers  # noqa: E402
from infinity.normal_estimation.defaults import (  # noqa: E402
    DEFAULT_HYPERSIM_ROOT,
    DEFAULT_NORMAL_TRAIN_DATASETS,
    DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS,
    DEFAULT_VKITTI2_ROOT,
)
from infinity.normal_estimation.token_cache import token_cache_sample_key  # noqa: E402
from infinity.utils.swanlab_utils import (  # noqa: E402
    import_swanlab,
    init_swanlab_run,
)


LOGGER = logging.getLogger("train_normal_estimation")
DEFAULT_NYUV2_ROOT = str(ROOT_DIR / "data" / "NYUv2" / "hf-parquet" / "tanganke" / "nyuv2" / "data")
FULLSTATE_SAVE_POLICY = (
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if FullStateDictConfig is not None
    else None
)
FULLOPTSTATE_SAVE_POLICY = (
    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if FullOptimStateDictConfig is not None
    else None
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Infinity for RGB-to-normal estimation on raw Hypersim.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_HYPERSIM_ROOT)
    parser.add_argument(
        "--train-datasets",
        type=str,
        default=DEFAULT_NORMAL_TRAIN_DATASETS,
        help="Comma-separated train datasets. Supported: hypersim,vkitti2.",
    )
    parser.add_argument(
        "--train-dataset-weights",
        type=str,
        default=DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS,
        help="Comma-separated dataset sampling weights, e.g. hypersim:9,vkitti2:1. Empty uses 1 for each train dataset.",
    )
    parser.add_argument("--vkitti2-root", type=str, default=DEFAULT_VKITTI2_ROOT)
    parser.add_argument(
        "--hypersim-filter-depth-nan",
        action="store_true",
        default=False,
        help="Drop Hypersim samples whose depth HDF5 contains NaN before training/validation.",
    )
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
    parser.add_argument(
        "--word-head-lr",
        type=float,
        default=0.0,
        help="LR for token embedding/output head parameters; 0 uses --lr.",
    )
    parser.add_argument(
        "--word-head-min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for token embedding/output head parameters; 0 uses --min-lr.",
    )
    parser.add_argument(
        "--image-word-lr",
        type=float,
        default=0.0,
        help="LR for RGB prefix image_word_embed parameters; 0 uses --lr.",
    )
    parser.add_argument(
        "--image-word-min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for RGB prefix image_word_embed parameters; 0 uses --min-lr.",
    )
    parser.add_argument(
        "--normal-task-lr",
        type=float,
        default=0.0,
        help="LR for normal task/modal embedding parameters; 0 uses --lr.",
    )
    parser.add_argument(
        "--normal-task-min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for normal task/modal embedding parameters; 0 uses --min-lr.",
    )
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
    parser.add_argument("--ar-eval-every", type=int, default=9999999)
    parser.add_argument("--ar-eval-samples", type=int, default=32)
    parser.add_argument("--ar-eval-nyuv2-root", type=str, default=DEFAULT_NYUV2_ROOT)
    parser.add_argument("--ar-eval-nyuv2-samples", type=int, default=32)
    parser.add_argument("--ar-eval-top-k", type=int, default=1)
    parser.add_argument("--ar-eval-top-p", type=float, default=0.0)
    parser.add_argument("--ar-eval-tau", type=float, default=1.0)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=500,
        help="Run validation and save checkpoints every N optimizer steps; 0 disables step validation checkpointing.",
    )
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--save-optimizer-state", action="store_true", default=False)
    parser.add_argument("--pn", type=str, choices=("0.06M", "0.25M", "1M"), default="0.06M")
    parser.add_argument("--train-partition", type=str, default="train")
    parser.add_argument("--val-partition", type=str, default="val")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument(
        "--token-cache-dir",
        type=str,
        default="",
        help="Optional directory for deterministic train token cache. Skips RGB/normal VAE tokenization on cache hits.",
    )
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
    return {key: float(value) for key, value in zip(keys, values.tolist(), strict=True)}


def _normal_lr_group_name(parameter_name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
            if parameter_name.startswith(prefix):
                parameter_name = parameter_name[len(prefix) :]
                changed = True
    if parameter_name.startswith("image_word_embed."):
        return "image_word"
    if parameter_name in {"image_modality_embed", "normal_modality_embed", "normal_task_kv"}:
        return "normal_task"
    if (
        parameter_name.startswith("word_embed.")
        or parameter_name.startswith("norm0_ve.")
        or parameter_name.startswith("head.")
        or parameter_name.startswith("head_nm.")
    ):
        return "word_head"
    return "backbone"


def _resolve_group_lr(value: float, fallback: float) -> float:
    return value if value > 0 else fallback


def build_normal_lr_param_groups(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    lr_config = {
        "backbone": (args.lr, args.min_lr),
        "word_head": (
            _resolve_group_lr(args.word_head_lr, args.lr),
            _resolve_group_lr(args.word_head_min_lr, args.min_lr),
        ),
        "image_word": (
            _resolve_group_lr(args.image_word_lr, args.lr),
            _resolve_group_lr(args.image_word_min_lr, args.min_lr),
        ),
        "normal_task": (
            _resolve_group_lr(args.normal_task_lr, args.lr),
            _resolve_group_lr(args.normal_task_min_lr, args.min_lr),
        ),
    }
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = {name: [] for name in lr_config}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped[_normal_lr_group_name(name)].append((name, parameter))

    param_groups: list[dict[str, Any]] = []
    for group_name in ("backbone", "word_head", "image_word", "normal_task"):
        named_parameters = grouped[group_name]
        if not named_parameters:
            continue
        lr, min_lr = lr_config[group_name]
        if min_lr > lr:
            raise ValueError(f"{group_name} min_lr ({min_lr:g}) must be <= lr ({lr:g}).")
        parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
        LOGGER.info(
            "optimizer_group=%s params=%.2fM lr=%.3g min_lr=%.3g examples=%s",
            group_name,
            parameter_count / 1_000_000,
            lr,
            min_lr,
            ",".join(name for name, _ in named_parameters[:3]),
        )
        param_groups.append(
            {
                "params": [parameter for _, parameter in named_parameters],
                "lr": lr,
                "initial_lr": lr,
                "min_lr": min_lr,
                "group_name": group_name,
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for normal estimation optimizer.")
    return param_groups


def build_optimizer_and_scheduler(model: torch.nn.Module, args: argparse.Namespace, total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    param_groups = build_normal_lr_param_groups(model, args)
    optimizer_kwargs: dict[str, Any] = {}
    if args.optimizer_backend == "fused":
        optimizer_kwargs["fused"] = True
    elif args.optimizer_backend == "foreach":
        optimizer_kwargs["foreach"] = True
    else:
        optimizer_kwargs["foreach"] = False
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
            **optimizer_kwargs,
        )
    except (TypeError, RuntimeError) as exc:
        if args.optimizer_backend != "fused":
            raise
        LOGGER.warning("Fused AdamW is unavailable (%s); falling back to foreach AdamW.", exc)
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
            foreach=True,
        )

    warmup_steps = int(total_steps * args.warmup_ratio)

    def group_lr_lambda(group_index: int):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
            if total_steps <= warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            group = optimizer.param_groups[group_index]
            min_ratio = float(group["min_lr"]) / float(group["initial_lr"])
            return min_ratio + (1.0 - min_ratio) * cosine

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [group_lr_lambda(group_index) for group_index in range(len(optimizer.param_groups))],
    )
    return optimizer, scheduler


def optimizer_lr_by_group(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def batch_dataset_name(batch: dict[str, Any]) -> str | None:
    names = {
        str(metadata.get("dataset", "")).strip()
        for metadata in batch.get("metadata", [])
        if isinstance(metadata, dict) and metadata.get("dataset")
    }
    if len(names) == 1:
        return next(iter(names))
    if len(names) > 1:
        return "mixed"
    return None


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


def load_token_cache_batch(
    cache_dir: Path,
    metadata: list[dict[str, Any]],
    signature: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    payloads = []
    for item in metadata:
        path = cache_dir / token_cache_sample_key(item, signature)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
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
        atomic_torch_save(payload, path)


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


def init_swanlab(args: argparse.Namespace, output_dir: Path, enabled: bool) -> Any | None:
    return init_swanlab_run(
        output_dir=output_dir,
        enabled=enabled,
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.swanlab_experiment_name,
        job_type=args.swanlab_job_type,
        tags=list(args.swanlab_tags),
        config=vars(args),
        logdir=args.swanlab_logdir,
        require_swanboard_for_local=True,
        logger=LOGGER,
    )


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
        swanlab_module = import_swanlab()
        swanlab_run.log(
            {
                f"{stage}/rgb": swanlab_module.Image(image_grid),
                f"{stage}/normal_gt": swanlab_module.Image(target_grid),
                f"{stage}/normal_pred": swanlab_module.Image(prediction_grid),
                f"{stage}/normal_angle": swanlab_module.Image(angle_grid),
                f"{stage}/compare": swanlab_module.Image(compare),
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
    metrics: dict[str, float] | None = None,
) -> None:
    payload = {
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "best_val_angle": best_val_angle,
        "zero": args.zero,
    }
    if metrics:
        payload["metrics"] = metrics
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
        atomic_torch_save(payload, checkpoint_path)
    dist_barrier_if_initialized()


def format_checkpoint_metric(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def prefixed_checkpoint_name(prefix: str, val_ar_nyu_deg: float) -> str:
    return f"{prefix}_val_ar_nyu_deg_{format_checkpoint_metric(val_ar_nyu_deg)}.pth"


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    return resolve_checkpoint_resume_path(
        output_dir=Path(args.output_dir),
        resume_arg=args.resume,
        auto_checkpoint_names=("last_step.pth", "last.pth"),
        prefer_step_checkpoint_for_last=True,
    )


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
        can_load_optimizer_state = (is_fsdp_model(model) and checkpoint_zero > 0) or (
            not is_fsdp_model(model) and checkpoint_zero == 0
        )
        if can_load_optimizer_state:
            try:
                if is_fsdp_model(model):
                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        model=model,
                        optim=optimizer,
                        optim_state_dict=checkpoint["optimizer"],
                    )
                    optimizer.load_state_dict(optim_state_dict)
                else:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                if scaler is not None and checkpoint.get("scaler") is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                optimizer_loaded = True
            except ValueError as exc:
                if is_main:
                    LOGGER.info("Resumed weights from %s but skipped optimizer state: %s", resume_path, exc)
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
                load_token_cache_batch(cache_dir, batch["metadata"], cache_signature, device)
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
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, torch.Tensor] | None]:
    training = model.training
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0
    dataset_names = parse_train_dataset_names(args.train_datasets)
    dataset_sums: dict[str, dict[str, torch.Tensor]] = {name: {} for name in dataset_names}
    dataset_counts: dict[str, int] = {name: 0 for name in dataset_names}
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
        dataset_name = batch_dataset_name(batch)
        if dataset_name in dataset_sums:
            dataset_counts[dataset_name] += batch_size
            dataset_sum = dataset_sums[dataset_name]
            for key, value in metrics.items():
                dataset_sum[key] = dataset_sum.get(key, value.new_tensor(0.0)) + value * batch_size

    count_tensor = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(count_tensor.item()))
    result = {key: float(value.item() / total_count) for key, value in sums.items()}
    dataset_results: dict[str, dict[str, float]] = {}
    for dataset_name in dataset_names:
        dataset_sum = dataset_sums[dataset_name]
        dataset_count_tensor = torch.tensor(float(dataset_counts[dataset_name]), device=device)
        if distributed:
            dist.all_reduce(dataset_count_tensor, op=dist.ReduceOp.SUM)
        if dataset_count_tensor.item() <= 0:
            continue
        for key in sums:
            if key not in dataset_sum:
                dataset_sum[key] = next(iter(sums.values())).new_tensor(0.0)
            if distributed:
                dist.all_reduce(dataset_sum[key], op=dist.ReduceOp.SUM)
        dataset_total_count = max(1.0, float(dataset_count_tensor.item()))
        dataset_results[dataset_name] = {
            key: float(value.item() / dataset_total_count)
            for key, value in dataset_sum.items()
        }
    if training:
        model.train()
    return result, dataset_results, visual_payload


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
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, torch.Tensor] | None]:
    training = model.training
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    sums: dict[str, torch.Tensor] = {}
    count = 0
    dataset_sums: dict[str, dict[str, torch.Tensor]] = {}
    dataset_counts: dict[str, int] = {}
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
        dataset_name = batch_dataset_name(batch)
        if dataset_name is not None:
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + batch_size
            dataset_sum = dataset_sums.setdefault(dataset_name, {})
            for key, value in metrics.items():
                dataset_sum[key] = dataset_sum.get(key, value.new_tensor(0.0)) + value * batch_size

    count_tensor = torch.tensor(float(count), device=device)
    if distributed:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(count_tensor.item()))
    result = {key: float(value.item() / total_count) for key, value in sums.items()}
    result["samples"] = total_count
    dataset_results: dict[str, dict[str, float]] = {}
    for dataset_name, dataset_sum in dataset_sums.items():
        dataset_count_tensor = torch.tensor(float(dataset_counts[dataset_name]), device=device)
        if distributed:
            dist.all_reduce(dataset_count_tensor, op=dist.ReduceOp.SUM)
        if dataset_count_tensor.item() <= 0:
            continue
        for key in sums:
            if key not in dataset_sum:
                dataset_sum[key] = next(iter(sums.values())).new_tensor(0.0)
            if distributed:
                dist.all_reduce(dataset_sum[key], op=dist.ReduceOp.SUM)
        dataset_total_count = max(1.0, float(dataset_count_tensor.item()))
        dataset_results[dataset_name] = {
            key: float(value.item() / dataset_total_count)
            for key, value in dataset_sum.items()
        }
        dataset_results[dataset_name]["samples"] = dataset_total_count
    if training:
        model.train()
    return result, dataset_results, visual_payload


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
            train_dataset = build_normal_train_dataset(
                train_datasets=args.train_datasets,
                hypersim_root=args.data_root,
                vkitti2_root=args.vkitti2_root,
                partition=args.train_partition,
                pn=args.pn,
                max_samples=args.max_train_samples,
                metadata_only=args.token_cache_metadata_only and token_cache_enabled(args),
                hypersim_filter_depth_nan=args.hypersim_filter_depth_nan,
            )
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
                filter_depth_nan=args.hypersim_filter_depth_nan,
            )
            ar_val_datasets: list[Dataset] = []
            if args.ar_eval_samples > 0:
                ar_val_datasets.append(
                    HypersimNormalDataset(
                        root=args.data_root,
                        partition=args.val_partition,
                        pn=args.pn,
                        max_samples=args.ar_eval_samples,
                        filter_depth_nan=args.hypersim_filter_depth_nan,
                    )
                )
            if args.ar_eval_nyuv2_samples > 0:
                nyuv2_root = Path(args.ar_eval_nyuv2_root)
                if nyuv2_root.exists():
                    ar_val_datasets.append(
                        NYUv2ParquetNormalDataset(
                            root=str(nyuv2_root),
                            partition="val",
                            pn=args.pn,
                            max_samples=args.ar_eval_nyuv2_samples,
                        )
                    )
                elif rank == 0:
                    LOGGER.warning("Skipping NYUv2 AR eval because root does not exist: %s", nyuv2_root)
            ar_val_dataset = ConcatDataset(ar_val_datasets) if len(ar_val_datasets) > 1 else (ar_val_datasets[0] if ar_val_datasets else None)
            if rank == 0 and ar_val_dataset is not None:
                LOGGER.info(
                    "ar_eval_samples hypersim=%d nyuv2=%d total=%d",
                    max(0, int(args.ar_eval_samples)),
                    max(0, int(args.ar_eval_nyuv2_samples)) if Path(args.ar_eval_nyuv2_root).exists() else 0,
                    len(ar_val_dataset),
                )
            train_shuffle = not args.token_cache_filter_missing
            if rank == 0 and not train_shuffle:
                LOGGER.info("Disabled train shuffle because --token-cache-filter-missing is enabled.")
            train_loader, train_sampler = build_normal_dataloader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                shuffle=train_shuffle,
                drop_last=True,
                collate_fn=collate_normal_estimation_batch,
                pin_memory=torch.cuda.is_available(),
                group_by_target_size=True,
                dataset_weights=train_dataset_weights,
            )
            val_loader, _ = build_normal_dataloader(
                val_dataset,
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_normal_estimation_batch,
                pin_memory=torch.cuda.is_available(),
            )
            ar_val_loader = None
            if ar_val_dataset is not None:
                ar_val_loader, _ = build_normal_dataloader(
                    ar_val_dataset,
                    batch_size=args.val_batch_size,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_normal_estimation_batch,
                    pin_memory=torch.cuda.is_available(),
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

        train_batches_per_epoch = require_positive_steps_per_epoch(
            len(train_loader),
            context="normal estimation training",
        )
        optimizer_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum_steps)
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

                current_lr_by_group = optimizer_lr_by_group(optimizer)
                current_lr = current_lr_by_group.get("backbone", optimizer.param_groups[0]["lr"])
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
                        dataset_name = batch_dataset_name(batch)
                        if dataset_name is not None:
                            for key, value in reduced.items():
                                if has_normal_metrics or key not in normal_metric_keys:
                                    payload[f"train/{dataset_name}/{key}"] = value
                        payload["train/lr"] = current_lr
                        for group_name, group_lr in current_lr_by_group.items():
                            payload[f"train/lr/{group_name}"] = group_lr
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
                    ar_metrics, ar_dataset_metrics, ar_visuals = evaluate_ar(
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
                        for dataset_name, dataset_metrics in ar_dataset_metrics.items():
                            LOGGER.info(
                                "val_ar/%s step=%d samples=%.0f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                                dataset_name,
                                global_step,
                                dataset_metrics["samples"],
                                dataset_metrics["angle_deg"],
                                dataset_metrics["acc_11_25"],
                                dataset_metrics["acc_22_5"],
                                dataset_metrics["acc_30"],
                            )
                        if swanlab_run is not None:
                            payload = {f"val_ar/{key}": value for key, value in ar_metrics.items()}
                            for dataset_name, dataset_metrics in ar_dataset_metrics.items():
                                for key, value in dataset_metrics.items():
                                    payload[f"val_ar/{dataset_name}/{key}"] = value
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
                    val_metrics, val_dataset_metrics, val_visuals = evaluate(
                        model=model,
                        normal_vae=normal_vae,
                        rgb_vae=rgb_vae,
                        dataloader=val_loader,
                        args=args,
                        device=device,
                        distributed=distributed,
                    )
                    is_best = val_metrics["angle_deg"] < best_val_angle
                    if is_best:
                        best_val_angle = val_metrics["angle_deg"]
                    if rank == 0:
                        LOGGER.info(
                            "val step=%d loss=%.4f ce=%.4f aux=%.4f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                            global_step,
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
                            for dataset_name, dataset_metrics in val_dataset_metrics.items():
                                for key, value in dataset_metrics.items():
                                    payload[f"val/{dataset_name}/{key}"] = value
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
                    step_ar_metrics: dict[str, float] | None = None
                    step_ar_dataset_metrics: dict[str, dict[str, float]] = {}
                    if ar_val_loader is not None:
                        step_ar_metrics, step_ar_dataset_metrics, ar_visuals = evaluate_ar(
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
                                step_ar_metrics["samples"],
                                step_ar_metrics["angle_deg"],
                                step_ar_metrics["acc_11_25"],
                                step_ar_metrics["acc_22_5"],
                                step_ar_metrics["acc_30"],
                            )
                            for dataset_name, dataset_metrics in step_ar_dataset_metrics.items():
                                LOGGER.info(
                                    "val_ar/%s step=%d samples=%.0f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                                    dataset_name,
                                    global_step,
                                    dataset_metrics["samples"],
                                    dataset_metrics["angle_deg"],
                                    dataset_metrics["acc_11_25"],
                                    dataset_metrics["acc_22_5"],
                                    dataset_metrics["acc_30"],
                                )
                            if swanlab_run is not None:
                                payload = {f"val_ar/{key}": value for key, value in step_ar_metrics.items()}
                                for dataset_name, dataset_metrics in step_ar_dataset_metrics.items():
                                    for key, value in dataset_metrics.items():
                                        payload[f"val_ar/{dataset_name}/{key}"] = value
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
                    val_ar_nyu_deg = step_ar_dataset_metrics.get("nyuv2", {}).get("angle_deg", float("nan"))
                    checkpoint_metrics = {
                        "val_angle_deg": float(val_metrics["angle_deg"]),
                        "val_ar_nyu_deg": float(val_ar_nyu_deg),
                    }
                    if step_ar_metrics is not None:
                        checkpoint_metrics["val_ar_angle_deg"] = float(step_ar_metrics["angle_deg"])
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
                        metrics=checkpoint_metrics,
                    )
                    save_checkpoint(
                        output_dir
                        / "checkpoints"
                        / prefixed_checkpoint_name(f"step_{global_step:07d}", val_ar_nyu_deg),
                        args=args,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if args.precision == "fp16" else None,
                        epoch=epoch,
                        step=global_step,
                        best_val_angle=best_val_angle,
                        is_main=(rank == 0),
                        metrics=checkpoint_metrics,
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            val_metrics, val_dataset_metrics, val_visuals = evaluate(
                model=model,
                normal_vae=normal_vae,
                rgb_vae=rgb_vae,
                dataloader=val_loader,
                args=args,
                device=device,
                distributed=distributed,
            )
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
                    for dataset_name, dataset_metrics in val_dataset_metrics.items():
                        for key, value in dataset_metrics.items():
                            payload[f"val/{dataset_name}/{key}"] = value
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
            epoch_ar_metrics: dict[str, float] | None = None
            epoch_ar_dataset_metrics: dict[str, dict[str, float]] = {}
            if ar_val_loader is not None and args.ar_eval_every > 0:
                if rank == 0:
                    LOGGER.info("val_ar epoch=%d start", epoch + 1)
                epoch_ar_metrics, epoch_ar_dataset_metrics, ar_visuals = evaluate_ar(
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
                        epoch_ar_metrics["samples"],
                        epoch_ar_metrics["angle_deg"],
                        epoch_ar_metrics["acc_11_25"],
                        epoch_ar_metrics["acc_22_5"],
                        epoch_ar_metrics["acc_30"],
                    )
                    for dataset_name, dataset_metrics in epoch_ar_dataset_metrics.items():
                        LOGGER.info(
                            "val_ar/%s epoch=%d samples=%.0f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f",
                            dataset_name,
                            epoch + 1,
                            dataset_metrics["samples"],
                            dataset_metrics["angle_deg"],
                            dataset_metrics["acc_11_25"],
                            dataset_metrics["acc_22_5"],
                            dataset_metrics["acc_30"],
                        )
                    if swanlab_run is not None:
                        payload = {f"val_ar/{key}": value for key, value in epoch_ar_metrics.items()}
                        for dataset_name, dataset_metrics in epoch_ar_dataset_metrics.items():
                            for key, value in dataset_metrics.items():
                                payload[f"val_ar/{dataset_name}/{key}"] = value
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
                if rank == 0:
                    LOGGER.info("val_ar epoch=%d done", epoch + 1)
            val_ar_nyu_deg = epoch_ar_dataset_metrics.get("nyuv2", {}).get("angle_deg", float("nan"))
            checkpoint_metrics = {
                "val_angle_deg": float(val_metrics["angle_deg"]),
                "val_ar_nyu_deg": float(val_ar_nyu_deg),
            }
            if epoch_ar_metrics is not None:
                checkpoint_metrics["val_ar_angle_deg"] = float(epoch_ar_metrics["angle_deg"])
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
                metrics=checkpoint_metrics,
            )
            save_checkpoint(
                output_dir
                / "checkpoints"
                / prefixed_checkpoint_name(f"epoch_{epoch + 1:04d}", val_ar_nyu_deg),
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if args.precision == "fp16" else None,
                epoch=epoch + 1,
                step=global_step,
                best_val_angle=best_val_angle,
                is_main=(rank == 0),
                metrics=checkpoint_metrics,
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
