from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infinity.normal_estimation import (  # noqa: E402
    HypersimNormalDataset,
    build_bsq_vae,
    build_prefix_tokens_from_image,
    build_infinity_normal_model,
    build_multiscale_var_inputs,
    collate_normal_estimation_batch,
    compute_normal_metrics,
    decode_logits_to_normal,
    load_infinity_state_dict,
    normals_to_vis,
    resolve_scale_schedule_from_hw,
)
from infinity.models.infinity import MultipleLayers  # noqa: E402


LOGGER = logging.getLogger("train_normal_estimation")
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
    parser.add_argument("--data-root", type=str, default="/root/vepfs/NormalART/datasets/processed/hypersim")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--zero", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument("--enable-hybrid-shard", action="store_true", default=False)
    parser.add_argument("--inner-shard-degree", type=int, default=1)
    parser.add_argument("--fsdp-use-orig-params", action="store_true", default=True)
    parser.add_argument("--disable-fsdp-use-orig-params", dest="fsdp_use_orig_params", action="store_false")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-log-every", type=int, default=200)
    parser.add_argument("--ar-eval-every", type=int, default=0)
    parser.add_argument("--ar-eval-samples", type=int, default=32)
    parser.add_argument("--ar-eval-top-k", type=int, default=1)
    parser.add_argument("--ar-eval-top-p", type=float, default=0.0)
    parser.add_argument("--ar-eval-tau", type=float, default=1.0)
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--save-optimizer-state", action="store_true", default=False)
    parser.add_argument("--pn", type=str, choices=("0.06M", "0.25M", "1M"), default="0.06M")
    parser.add_argument("--train-partition", type=str, default="train")
    parser.add_argument("--val-partition", type=str, default="val")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="infinity_8b")
    parser.add_argument("--use-bit-label", action="store_true", default=True)
    parser.add_argument("--disable-bit-label", dest="use_bit_label", action="store_false")
    parser.add_argument("--add-lvl-embeding-only-first-block", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rope2d-each-sa-layer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rope2d-normalized-by-hw", type=int, choices=(0, 1, 2), default=2)
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
        if dist.get_world_size() > 1:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
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


def make_dataloader(
    dataset: HypersimNormalDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
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


def build_optimizer_and_scheduler(model: torch.nn.Module, args: argparse.Namespace, total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        foreach=False,
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


def compute_ce_loss(
    logits_blv: torch.Tensor,
    gt_bl: torch.Tensor,
    *,
    use_bit_label: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if use_bit_label:
        batch_size, seq_len, _ = logits_blv.shape
        loss = F.cross_entropy(
            logits_blv.reshape(batch_size, seq_len, -1, 2).permute(0, 3, 1, 2),
            gt_bl,
            reduction="none",
        ).mean(dim=-1)
        bitwise_acc = (logits_blv.reshape(batch_size, seq_len, -1, 2).argmax(dim=-1) == gt_bl).float()
        token_acc = (bitwise_acc.sum(dim=-1) == bitwise_acc.shape[-1]).float().mean()
        bit_acc = bitwise_acc.mean()
    else:
        batch_size, seq_len, vocab = logits_blv.shape
        loss = F.cross_entropy(
            logits_blv.reshape(batch_size * seq_len, vocab),
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
            return DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
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


def init_swanlab(args: argparse.Namespace, output_dir: Path, enabled: bool) -> Any | None:
    if not enabled or args.swanlab_mode == "disabled":
        return None
    if swanlab is None:
        raise ImportError("swanlab is not installed, but SwanLab logging is enabled.")

    state_path = swanlab_state_path(output_dir)
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
    if run_id:
        init_kwargs["id"] = run_id
        if args.swanlab_mode == "cloud":
            init_kwargs["resume"] = "allow"

    run = swanlab.init(**init_kwargs)
    run_public = getattr(run, "public", None)
    current_run_id = str(getattr(run_public, "run_id", "")).strip()
    state_payload = {
        "project": args.swanlab_project,
        "workspace": workspace,
        "experiment_name": build_swanlab_experiment_name(args, output_dir),
        "mode": args.swanlab_mode,
        "logdir": str(logdir),
    }
    if current_run_id:
        state_payload["run_id"] = current_run_id
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
        torch.save(payload, checkpoint_path)
    dist_barrier_if_initialized()


def maybe_resume(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    is_main: bool,
) -> tuple[int, int, float]:
    resume_path = Path(args.resume) if args.resume else Path(args.output_dir) / "checkpoints" / "last.pth"
    if not resume_path.is_file():
        return 0, 0, float("inf")

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    scale_schedule = [tuple(item) for item in resolve_scale_schedule_from_hw(target.shape[-2], target.shape[-1], args.pn)[1]]
    training_scales = min(args.always_training_scales, len(scale_schedule))
    scale_schedule = scale_schedule[:training_scales]

    with torch.no_grad():
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
    first_stage_len = int(np.array(scale_schedule[0]).prod())
    x_blc_without_prefix = x_blc_without_prefix[:, : total_seq_len - first_stage_len, :]
    gt_bl = torch.cat(gt_ms_idx_bl, dim=1)[:, :total_seq_len].contiguous().long()

    with precision_context(precision):
        logits_blv = model(rgb_prefix_blc, x_blc_without_prefix, scale_schedule=scale_schedule)
        ce_loss, ce_metrics = compute_ce_loss(logits_blv, gt_bl, use_bit_label=args.use_bit_label)
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
        total_loss = args.ce_weight * ce_loss + aux_loss

    metrics = {
        **ce_metrics,
        **aux_metrics,
        "loss_total": total_loss.detach(),
    }
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
    args = parse_args()
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
                "batch_size_per_gpu=%d global_batch_size=%d val_batch_size_per_gpu=%d zero=%d swanlab_mode=%s",
                args.batch_size,
                args.batch_size * world_size,
                args.val_batch_size,
                args.zero,
                args.swanlab_mode,
            )

        train_dataset = HypersimNormalDataset(
            root=args.data_root,
            partition=args.train_partition,
            pn=args.pn,
            max_samples=args.max_train_samples,
        )
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
        train_loader, train_sampler = make_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            distributed=distributed,
            shuffle=True,
            drop_last=True,
        )
        val_loader, _ = make_dataloader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            distributed=distributed,
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
                shuffle=False,
                drop_last=False,
            )

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
            device=device,
        )
        model = convert_model_precision(model, args.precision)
        maybe_load_init_model(model, args.init_model, is_main=(rank == 0))
        model = build_training_model(model, args=args, device=device, distributed=distributed)

        total_steps = len(train_loader) * args.epochs
        if args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)
        optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_steps=max(1, total_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
        start_epoch, global_step, best_val_angle = maybe_resume(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if args.precision == "fp16" else None,
            is_main=(rank == 0),
        )

        swanlab_run = init_swanlab(args, output_dir, enabled=(rank == 0))

        model.train()
        start_time = time.time()
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                total_loss, metrics, prediction, image = forward_batch(
                    model=model,
                    normal_vae=normal_vae,
                    rgb_vae=rgb_vae,
                    batch=batch,
                    args=args,
                    device=device,
                    precision=args.precision,
                )

                if args.precision == "fp16":
                    scaler.scale(total_loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm(model, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    clip_grad_norm(model, args.grad_clip)
                    optimizer.step()
                scheduler.step()
                global_step += 1

                reduced = reduce_metrics(metrics, distributed)
                current_lr = optimizer.param_groups[0]["lr"]
                if rank == 0 and (global_step % args.log_every == 0 or global_step == 1):
                    LOGGER.info(
                        "epoch=%d step=%d/%d loss=%.4f ce=%.4f aux=%.4f angle=%.2f acc11=%.3f acc22=%.3f acc30=%.3f lr=%.2e",
                        epoch + 1,
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
                    if swanlab_run is not None:
                        payload = {f"train/{key}": value for key, value in reduced.items()}
                        payload["train/lr"] = current_lr
                        payload["train/epoch"] = epoch + 1
                        swanlab_run.log(payload, step=global_step)
                if rank == 0 and (global_step % args.image_log_every == 0 or global_step == 1):
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
            should_save_last = (epoch + 1) % args.save_every_epoch == 0
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
            if ar_val_loader is not None:
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
