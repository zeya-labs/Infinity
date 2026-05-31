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
import swanlab
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infinity.models.bsq_vae.vae import vae_model
from infinity.tokenizer_finetune.data import HypersimNormalCacheDataset, collate_normal_batch


LOGGER = logging.getLogger("train_normal_tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Infinity tokenizer on normal maps.")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-process batch size.")
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--repeat-train", type=int, default=1)
    parser.add_argument("--repeat-val", type=int, default=1)
    parser.add_argument("--mmap-cache", action="store_true", default=True)
    parser.add_argument("--disable-mmap-cache", dest="mmap_cache", action="store_false")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--image-log-every", type=int, default=200)
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--recon-l1-weight", type=float, default=1.0)
    parser.add_argument("--recon-cosine-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--norm-weight", type=float, default=0.1)
    parser.add_argument("--normal-gradient-weight", type=float, default=0.2)
    parser.add_argument("--vae-ckpt", type=str, default=str(ROOT_DIR / "weights" / "infinity_vae_d56_f8_14_patchify.pth"))
    parser.add_argument("--codebook-dim", type=int, default=14)
    parser.add_argument("--apply-spatial-patchify", action="store_true", default=True)
    parser.add_argument("--disable-spatial-patchify", dest="apply_spatial_patchify", action="store_false")
    parser.add_argument("--encoder-dtype", type=str, choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--trainable-scope",
        type=str,
        choices=("all", "decoder_quantizer", "decoder_only"),
        default="all",
        help="Train all modules, freeze encoder only, or freeze encoder+quantizer so latent stays fixed.",
    )
    parser.add_argument("--swanlab-project", type=str, default=os.environ.get("SWANLAB_PROJECT", "infinity_normal_tokenizer_hypersim"))
    parser.add_argument("--swanlab-workspace", type=str, default=os.environ.get("SWANLAB_WORKSPACE", ""))
    parser.add_argument("--swanlab-experiment-name", type=str, default="")
    parser.add_argument("--swanlab-job-type", type=str, default="train_normal_tokenizer")
    parser.add_argument("--swanlab-tags", nargs="*", default=["normal_tokenizer", "hypersim"])
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
        torch.cuda.set_device(device)
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
    result: dict[str, float] = {}
    for key, value in metrics.items():
        result[key] = float(reduce_tensor(value, enabled).item())
    return result


def ensure_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}")
    return mask.bool()


def normalize_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals.float() / torch.linalg.norm(normals.float(), dim=1, keepdim=True).clamp_min(1e-6)


def normals_to_vis(normals: torch.Tensor, invert_x_for_vis: bool = True) -> torch.Tensor:
    normals = normalize_normals(normals).detach().cpu().float().clamp(-1.0, 1.0)
    vis = normals.clone()
    if invert_x_for_vis:
        vis[:, 0] = -vis[:, 0]
    return ((vis + 1.0) / 2.0).clamp(0.0, 1.0)


def masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.float()
    if value.ndim == 4:
        mask_float = mask_float.expand(-1, value.shape[1], -1, -1)
    numerator = (value * mask_float).sum()
    denominator = mask_float.sum().clamp_min(1.0)
    return numerator / denominator


def masked_scalar_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.squeeze(1).bool()
    if not torch.any(valid):
        return value.new_tensor(0.0)
    return value[valid].mean()


def masked_gradient_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]

    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

    loss_dx = masked_channel_mean((pred_dx - target_dx).abs(), mask_dx)
    loss_dy = masked_channel_mean((pred_dy - target_dy).abs(), mask_dy)
    return 0.5 * (loss_dx + loss_dy)


def compute_metrics(
    target: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    vq_loss: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_normalized = normalize_normals(recon)
    target_normalized = normalize_normals(target)
    dot = (recon_normalized * target_normalized).sum(dim=1).clamp(-1.0, 1.0)
    cosine_distance = 1.0 - dot
    angular_deg = torch.rad2deg(torch.acos(dot))
    pred_norm = torch.linalg.norm(recon, dim=1, keepdim=True)

    recon_l1 = masked_channel_mean((recon - target).abs(), mask)
    recon_cosine = masked_scalar_mean(cosine_distance, mask)
    recon_angle_deg = masked_scalar_mean(angular_deg, mask)
    norm_loss = masked_channel_mean((pred_norm - 1.0).abs(), mask)
    gradient_loss = masked_gradient_l1(recon_normalized, target_normalized, mask)

    total_loss = (
        args.recon_l1_weight * recon_l1
        + args.recon_cosine_weight * recon_cosine
        + args.vq_weight * vq_loss
        + args.norm_weight * norm_loss
        + args.normal_gradient_weight * gradient_loss
    )
    metrics = {
        "loss_total": total_loss.detach(),
        "loss_recon_l1": recon_l1.detach(),
        "loss_recon_cosine": recon_cosine.detach(),
        "loss_vq": vq_loss.detach(),
        "loss_norm": norm_loss.detach(),
        "loss_normal_gradient": gradient_loss.detach(),
        "metric_angle_deg": recon_angle_deg.detach(),
    }
    return total_loss, metrics


def save_visuals(
    output_dir: Path,
    swanlab_run: Any | None,
    stage: str,
    step: int,
    target: torch.Tensor,
    recon: torch.Tensor,
    is_main: bool,
) -> None:
    if not is_main:
        return

    target_vis = normals_to_vis(target[:4])
    recon_vis = normals_to_vis(recon[:4])
    angle_vis = torch.rad2deg(
        torch.acos((normalize_normals(recon[:4]) * normalize_normals(target[:4])).sum(dim=1).clamp(-1.0, 1.0))
    ).unsqueeze(1).detach().cpu().float().div(45.0).clamp(0.0, 1.0)
    angle_vis = angle_vis.repeat(1, 3, 1, 1)

    gt_grid = make_grid(target_vis, nrow=min(4, target_vis.shape[0]))
    recon_grid = make_grid(recon_vis, nrow=min(4, recon_vis.shape[0]))
    angle_grid = make_grid(angle_vis, nrow=min(4, angle_vis.shape[0]))
    grid = make_grid(torch.cat([target_vis, recon_vis, angle_vis], dim=0), nrow=4)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_image(gt_grid, image_dir / f"{stage}_normal_gt_step_{step:07d}.png")
    save_image(recon_grid, image_dir / f"{stage}_normal_recon_step_{step:07d}.png")
    save_image(angle_grid, image_dir / f"{stage}_normal_angle_step_{step:07d}.png")
    save_image(grid, image_dir / f"{stage}_normal_compare_step_{step:07d}.png")
    if swanlab_run is not None:
        swanlab_run.log(
            {
                f"{stage}/normal_gt": swanlab.Image(gt_grid),
                f"{stage}/normal_recon": swanlab.Image(recon_grid),
                f"{stage}/normal_angle": swanlab.Image(angle_grid),
                f"{stage}/normal_compare": swanlab.Image(grid),
            },
            step=step,
        )


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((ROOT_DIR / path).resolve())


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def configure_trainable_scope(raw_model: torch.nn.Module, scope: str) -> dict[str, bool]:
    if scope == "all":
        plan = {"encoder": True, "quantizer": True, "decoder": True}
    elif scope == "decoder_quantizer":
        plan = {"encoder": False, "quantizer": True, "decoder": True}
    elif scope == "decoder_only":
        plan = {"encoder": False, "quantizer": False, "decoder": True}
    else:
        raise ValueError(f"Unsupported trainable scope: {scope}")

    set_requires_grad(raw_model.encoder, plan["encoder"])
    set_requires_grad(raw_model.quantizer, plan["quantizer"])
    set_requires_grad(raw_model.decoder, plan["decoder"])
    return plan


def apply_scope_train_modes(raw_model: torch.nn.Module, scope: str) -> None:
    raw_model.train()
    if scope in {"decoder_quantizer", "decoder_only"}:
        raw_model.encoder.eval()
    if scope == "decoder_only":
        raw_model.quantizer.eval()
    else:
        raw_model.quantizer.train()
    raw_model.decoder.train()


def get_trainable_parameters(raw_model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in raw_model.parameters() if parameter.requires_grad]


def count_parameters(parameters: list[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    patch_size = 8 if args.apply_spatial_patchify else 16
    encoder_ch_mult = [1, 2, 4, 4] if args.apply_spatial_patchify else [1, 2, 4, 4, 4]
    decoder_ch_mult = [1, 2, 4, 4] if args.apply_spatial_patchify else [1, 2, 4, 4, 4]
    model = vae_model(
        resolve_repo_path(args.vae_ckpt),
        schedule_mode="dynamic",
        codebook_dim=args.codebook_dim,
        codebook_size=2 ** args.codebook_dim,
        test_mode=False,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
    )
    model.args.encoder_dtype = args.encoder_dtype
    return model


def build_loader(
    dataset: HypersimNormalCacheDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
        "collate_fn": collate_normal_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs), sampler


def auto_resume_path(output_dir: Path, resume_arg: str) -> Path | None:
    if resume_arg:
        return Path(resume_arg)
    candidate = output_dir / "checkpoints" / "last.pth"
    return candidate if candidate.exists() else None


def save_checkpoint(
    output_dir: Path,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    best_val_angle: float,
    args: argparse.Namespace,
    tag: str,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pth"
    payload = {
        "step": step,
        "epoch": epoch,
        "best_val_angle": best_val_angle,
        "args": vars(args),
        "vae": raw_model.state_dict(),
        "image_disc": {},
        "video_disc": {},
        "opt_vae": optimizer.state_dict(),
        "opt_image_disc": None,
        "opt_video_disc": None,
        "sch_vae": scheduler.state_dict() if scheduler is not None else None,
        "sch_image_disc": None,
        "sch_video_disc": None,
        "ema": None,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_resume(
    resume_path: Path | None,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    if resume_path is None:
        return 0, 0, float("inf")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["vae"], strict=True)
    if "opt_vae" in checkpoint and checkpoint["opt_vae"] is not None:
        try:
            optimizer.load_state_dict(checkpoint["opt_vae"])
        except ValueError as exc:
            LOGGER.warning("skip optimizer state restore due to parameter mismatch: %s", exc)
    if scheduler is not None and checkpoint.get("sch_vae") is not None:
        try:
            scheduler.load_state_dict(checkpoint["sch_vae"])
        except ValueError as exc:
            LOGGER.warning("skip scheduler state restore due to parameter mismatch: %s", exc)
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("step", 0))
    best_val_angle = float(checkpoint.get("best_val_angle", float("inf")))
    return start_epoch, global_step, best_val_angle


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr_ratio = args.min_lr / args.lr

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-8, float(current_step + 1) / float(warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_grad_scaler(enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
    state_payload = {
        "run_id": run.public.run_id,
        "project": args.swanlab_project,
        "workspace": workspace,
        "experiment_name": build_swanlab_experiment_name(args, output_dir),
        "mode": args.swanlab_mode,
        "logdir": str(logdir),
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.eval()
    meter_sums: dict[str, torch.Tensor] = {}
    meter_count = torch.tensor(0.0, device=device)
    visual_payload = None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and args.precision == "fp16"
        else nullcontext()
    )

    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            mask = ensure_mask(batch["mask"].to(device, non_blocking=True), target)
            with autocast_ctx:
                recon, vq_output = model(target)
                _, metrics = compute_metrics(target, recon, mask, vq_output["commitment_loss"], args)
            if visual_payload is None:
                visual_payload = {
                    "target": target.detach().cpu(),
                    "recon": recon.detach().cpu(),
                }

            batch_size = target.shape[0]
            meter_count += batch_size
            for key, value in metrics.items():
                meter_sums[key] = meter_sums.get(key, torch.zeros_like(value)) + value.detach() * batch_size

    if distributed:
        dist.all_reduce(meter_count, op=dist.ReduceOp.SUM)
        for value in meter_sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    if float(meter_count.item()) == 0.0:
        return {}, visual_payload
    return {key: float((value / meter_count).item()) for key, value in meter_sums.items()}, visual_payload


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, device = init_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, is_main=(rank == 0))
    seed_everything(args.seed, rank)
    if rank == 0:
        with open(output_dir / "args.json", "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    swanlab_run = None
    try:
        LOGGER.info("distributed=%s rank=%d world_size=%d device=%s", distributed, rank, world_size, device)
        LOGGER.info("train_cache=%s", args.train_cache)
        LOGGER.info("val_cache=%s", args.val_cache)
        LOGGER.info(
            "batch_size_per_gpu=%d global_batch_size=%d val_batch_size_per_gpu=%d",
            args.batch_size,
            args.batch_size * world_size,
            args.val_batch_size,
        )
        LOGGER.info("trainable_scope=%s", args.trainable_scope)
        train_dataset = HypersimNormalCacheDataset(args.train_cache, repeat=args.repeat_train, mmap=args.mmap_cache)
        val_dataset = HypersimNormalCacheDataset(args.val_cache, repeat=args.repeat_val, mmap=args.mmap_cache)
        train_loader, train_sampler = build_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            distributed=distributed,
            shuffle=True,
            drop_last=True,
        )
        val_loader, val_sampler = build_loader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=max(1, args.num_workers // 2),
            prefetch_factor=max(2, args.prefetch_factor // 2),
            distributed=distributed,
            shuffle=False,
            drop_last=False,
        )
        if val_sampler is not None:
            val_sampler.drop_last = False

        raw_model = build_model(args).to(device)
        train_plan = configure_trainable_scope(raw_model, args.trainable_scope)
        trainable_parameters = get_trainable_parameters(raw_model)
        if not trainable_parameters:
            raise RuntimeError(f"No trainable parameters found for scope={args.trainable_scope}")
        LOGGER.info(
            "trainable_modules=%s trainable_params=%.2fM total_params=%.2fM",
            ", ".join(name for name, enabled in train_plan.items() if enabled),
            count_parameters(trainable_parameters) / 1_000_000,
            sum(parameter.numel() for parameter in raw_model.parameters()) / 1_000_000,
        )
        model = DDP(raw_model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if distributed else raw_model

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        steps_per_epoch = len(train_loader)
        total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
        scheduler = build_scheduler(optimizer, args, total_steps=total_steps)

        resume_path = auto_resume_path(output_dir, args.resume)
        start_epoch, global_step, best_val_angle = load_resume(resume_path, raw_model, optimizer, scheduler, device)
        if resume_path is not None:
            LOGGER.info("resume from %s (epoch=%d step=%d best_val_angle=%.4f)", resume_path, start_epoch, global_step, best_val_angle)

        use_fp16_scaler = device.type == "cuda" and args.precision == "fp16"
        scaler = build_grad_scaler(enabled=use_fp16_scaler)
        swanlab_run = init_swanlab(args, output_dir, enabled=(rank == 0))

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and args.precision == "bf16"
            else torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda" and args.precision == "fp16"
            else nullcontext()
        )

        train_start = time.time()
        stop_training = False
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            apply_scope_train_modes(raw_model, args.trainable_scope)

            for batch_idx, batch in enumerate(train_loader):
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

                target = batch["target"].to(device, non_blocking=True)
                mask = ensure_mask(batch["mask"].to(device, non_blocking=True), target)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    recon, vq_output = model(target)
                    total_loss, metrics = compute_metrics(
                        target=target,
                        recon=recon,
                        mask=mask,
                        vq_loss=vq_output["commitment_loss"],
                        args=args,
                    )

                if use_fp16_scaler:
                    scaler.scale(total_loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                    optimizer.step()
                scheduler.step()
                global_step += 1

                reduced_metrics = reduce_metrics(metrics, distributed)
                reduced_lr = optimizer.param_groups[0]["lr"]
                if rank == 0 and (global_step == 1 or global_step % args.log_every == 0):
                    elapsed = time.time() - train_start
                    LOGGER.info(
                        "step=%d epoch=%d batch=%d/%d loss=%.4f l1=%.4f cos=%.4f grad=%.4f vq=%.4f norm=%.4f angle=%.3f lr=%.6g elapsed=%.1fs",
                        global_step,
                        epoch,
                        batch_idx + 1,
                        steps_per_epoch,
                        reduced_metrics["loss_total"],
                        reduced_metrics["loss_recon_l1"],
                        reduced_metrics["loss_recon_cosine"],
                        reduced_metrics["loss_normal_gradient"],
                        reduced_metrics["loss_vq"],
                        reduced_metrics["loss_norm"],
                        reduced_metrics["metric_angle_deg"],
                        reduced_lr,
                        elapsed,
                    )
                    if swanlab_run is not None:
                        payload = {f"train/{key}": value for key, value in reduced_metrics.items()}
                        payload["train/lr"] = reduced_lr
                        swanlab_run.log(payload, step=global_step)

                if global_step == 1 or global_step % args.image_log_every == 0:
                    save_visuals(output_dir, swanlab_run, "train", global_step, target, recon, is_main=(rank == 0))

            val_metrics, val_visuals = run_validation(model, val_loader, device, distributed, args)
            if rank == 0 and val_metrics:
                LOGGER.info(
                    "val epoch=%d loss=%.4f l1=%.4f cos=%.4f grad=%.4f vq=%.4f norm=%.4f angle=%.3f",
                    epoch,
                    val_metrics["loss_total"],
                    val_metrics["loss_recon_l1"],
                    val_metrics["loss_recon_cosine"],
                    val_metrics["loss_normal_gradient"],
                    val_metrics["loss_vq"],
                    val_metrics["loss_norm"],
                    val_metrics["metric_angle_deg"],
                )
                if swanlab_run is not None:
                    swanlab_run.log({f"val/{key}": value for key, value in val_metrics.items()}, step=global_step)
                if val_visuals is not None:
                    save_visuals(
                        output_dir,
                        swanlab_run,
                        "val",
                        global_step,
                        val_visuals["target"],
                        val_visuals["recon"],
                        is_main=True,
                    )

            if rank == 0 and val_metrics and val_metrics["metric_angle_deg"] < best_val_angle:
                best_val_angle = val_metrics["metric_angle_deg"]
                save_checkpoint(
                    output_dir,
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_val_angle,
                    args,
                    f"best_angle_{best_val_angle:.4f}",
                )

            if rank == 0 and ((epoch + 1) % args.save_every_epoch == 0 or stop_training or epoch + 1 == args.epochs):
                save_checkpoint(output_dir, raw_model, optimizer, scheduler, epoch, global_step, best_val_angle, args, "last")

            if stop_training:
                break
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
