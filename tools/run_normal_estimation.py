from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infinity.normal_estimation import (  # noqa: E402
    build_bsq_vae,
    build_prefix_tokens_from_image,
    build_infinity_normal_model,
    load_infinity_state_dict,
    normalize_normals,
    normals_to_vis,
    resolve_scale_schedule_from_hw,
)
from infinity.models.basic import precompute_rope2d_freqs_grid  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Infinity RGB-to-normal inference.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pn", type=str, choices=("0.06M", "0.25M", "1M"), default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--normal-vae-ckpt", type=str, default=None)
    parser.add_argument("--normal-vae-type", type=int, default=None)
    parser.add_argument("--normal-apply-spatial-patchify", type=int, choices=(0, 1), default=None)
    parser.add_argument("--rgb-vae-ckpt", type=str, default=None)
    parser.add_argument("--rgb-vae-type", type=int, default=None)
    parser.add_argument("--rgb-apply-spatial-patchify", type=int, choices=(0, 1), default=None)
    parser.add_argument("--use-bit-label", type=int, choices=(0, 1), default=None)
    parser.add_argument("--add-lvl-embeding-only-first-block", type=int, choices=(0, 1), default=None)
    parser.add_argument("--rope2d-each-sa-layer", type=int, choices=(0, 1), default=None)
    parser.add_argument("--rope2d-normalized-by-hw", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--normal-use-flex-attn", type=int, choices=(0, 1), default=None)
    parser.add_argument("--normal-use-segmented-flash-attn", type=int, choices=(0, 1), default=None)
    parser.add_argument("--normal-bf16-activations", type=int, choices=(0, 1), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--save-npy", action="store_true", default=False)
    parser.add_argument("--save-uncertainty", action="store_true", default=False)
    parser.add_argument("--save-visualization", action="store_true", default=True)
    parser.add_argument("--disable-save-visualization", dest="save_visualization", action="store_false")
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--normal-kv-cache-fast", action="store_true", help="Deprecated; KV-cache AR is enabled by default.")
    parser.add_argument("--normal-disable-kv-cache-fast", action="store_true", help="Disable default KV-cache AR path for debugging.")
    parser.add_argument(
        "--force-original-resolution",
        action="store_true",
        help="Experimental: skip input resizing and build an oversized schedule/cache for the original image size.",
    )
    return parser.parse_args()


def _load_checkpoint_args(model_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def _resolve_option(value: Any, checkpoint_args: dict[str, Any], key: str, default: Any) -> Any:
    if value is not None:
        return value
    if key in checkpoint_args:
        return checkpoint_args[key]
    return default


def _resolve_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _save_png(tensor_chw: torch.Tensor, path: Path) -> None:
    array = tensor_chw.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()
    Image.fromarray(array).save(path)


def _save_visualization(image_chw: torch.Tensor, normal_vis_chw: torch.Tensor, path: Path) -> None:
    image = image_chw.detach().cpu().clamp(0.0, 1.0)
    normal_vis = normal_vis_chw.detach().cpu().clamp(0.0, 1.0)
    cat = torch.cat([image, normal_vis], dim=2)
    _save_png(cat, path)


def _save_grayscale(tensor_chw: torch.Tensor, path: Path) -> None:
    tensor = tensor_chw.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor[0]
    array = tensor.clamp(0.0, 1.0).mul(255).byte().numpy()
    Image.fromarray(array, mode="L").save(path)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_precision_context(model: torch.nn.Module, device: torch.device):
    if device.type == "cuda" and bool(getattr(model, "normal_bf16_activations", False)):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _scale_schedule_to_original_resolution(
    *,
    base_schedule: list[tuple[int, int, int]],
    original_height: int,
    original_width: int,
) -> list[tuple[int, int, int]]:
    if original_height % 16 != 0 or original_width % 16 != 0:
        raise ValueError(
            "--force-original-resolution requires image height and width divisible by 16; "
            f"got {original_height}x{original_width}."
        )
    final_h = original_height // 16
    final_w = original_width // 16
    _, base_final_h, base_final_w = base_schedule[-1]
    scaled: list[tuple[int, int, int]] = []
    for index, (pt, ph, pw) in enumerate(base_schedule):
        if index == len(base_schedule) - 1:
            scaled.append((int(pt), final_h, final_w))
            continue
        scaled_h = max(1, int(round(float(ph) / float(base_final_h) * float(final_h))))
        scaled_w = max(1, int(round(float(pw) / float(base_final_w) * float(final_w))))
        scaled.append((int(pt), scaled_h, scaled_w))
    return scaled


def _scale_schedule_tokens(scale_schedule: list[tuple[int, int, int]]) -> int:
    return int(sum(int(pt) * int(ph) * int(pw) for pt, ph, pw in scale_schedule))


def _ensure_custom_rope_cache(model: torch.nn.Module, scale_schedule: list[tuple[int, int, int]]) -> None:
    if not getattr(model, "rope2d_each_sa_layer", 0):
        return
    rope_key = str(tuple(scale_schedule))
    if rope_key in model.rope2d_freqs_grid:
        return
    max_h = max(2048 // 16, *(int(item[1]) for item in scale_schedule))
    max_w = max(2048 // 16, *(int(item[2]) for item in scale_schedule))
    custom_resolution = {
        "original": {
            "1M": {
                "scales": [(int(pt), int(ph), int(pw)) for pt, ph, pw in scale_schedule],
            }
        }
    }
    cache = precompute_rope2d_freqs_grid(
        dim=model.C // model.num_heads,
        dynamic_resolution_h_w=custom_resolution,
        rope2d_normalized_by_hw=model.rope2d_normalized_by_hw,
        pad_to_multiplier=model.pad_to_multiplier,
        max_height=max_h,
        max_width=max_w,
    )
    model.rope2d_freqs_grid.update(cache)


def _predict_normal(
    *,
    image_tensor: torch.Tensor,
    rgb_vae: torch.nn.Module,
    normal_vae: torch.nn.Module,
    model: torch.nn.Module,
    scale_schedule: list[tuple[int, int, int]],
    normal_vae_type: int,
    rgb_apply_spatial_patchify: bool,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    rgb_prefix_blc = build_prefix_tokens_from_image(
        image_01=image_tensor,
        rgb_vae=rgb_vae,
        scale_schedule=scale_schedule,
        apply_spatial_patchify=rgb_apply_spatial_patchify,
    )
    with _model_precision_context(model, device):
        return model.autoregressive_infer_prefix(
            vae=normal_vae,
            rgb_prefix_blc=rgb_prefix_blc,
            scale_schedule=scale_schedule,
            tau=args.tau,
            top_k=args.top_k,
            top_p=args.top_p,
            vae_type=normal_vae_type,
        )


def _predict_normal_with_uncertainty(
    *,
    image_tensor: torch.Tensor,
    rgb_vae: torch.nn.Module,
    normal_vae: torch.nn.Module,
    model: torch.nn.Module,
    scale_schedule: list[tuple[int, int, int]],
    normal_vae_type: int,
    rgb_apply_spatial_patchify: bool,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rgb_prefix_blc = build_prefix_tokens_from_image(
        image_01=image_tensor,
        rgb_vae=rgb_vae,
        scale_schedule=scale_schedule,
        apply_spatial_patchify=rgb_apply_spatial_patchify,
    )
    with _model_precision_context(model, device):
        return model.autoregressive_infer_prefix_with_uncertainty(
            vae=normal_vae,
            rgb_prefix_blc=rgb_prefix_blc,
            scale_schedule=scale_schedule,
            tau=args.tau,
            top_k=args.top_k,
            top_p=args.top_p,
            vae_type=normal_vae_type,
        )


def main() -> int:
    args = parse_args()
    os.environ.pop("INFINITY_NORMAL_ENABLE_KV_FAST", None)
    if args.normal_disable_kv_cache_fast:
        os.environ["INFINITY_NORMAL_DISABLE_KV_FAST"] = "1"
    else:
        os.environ.pop("INFINITY_NORMAL_DISABLE_KV_FAST", None)
    model_path = Path(args.model_path)
    checkpoint_args = _load_checkpoint_args(model_path) if model_path.is_file() else {}

    pn = _resolve_option(args.pn, checkpoint_args, "pn", "0.06M")
    model_name = _resolve_option(args.model_name, checkpoint_args, "model_name", "infinity_8b")
    normal_vae_ckpt = _resolve_option(args.normal_vae_ckpt, checkpoint_args, "normal_vae_ckpt", None)
    rgb_vae_ckpt = _resolve_option(args.rgb_vae_ckpt, checkpoint_args, "rgb_vae_ckpt", None)
    normal_vae_type = int(_resolve_option(args.normal_vae_type, checkpoint_args, "normal_vae_type", 14))
    rgb_vae_type = int(_resolve_option(args.rgb_vae_type, checkpoint_args, "rgb_vae_type", 14))
    normal_apply_spatial_patchify = bool(
        int(_resolve_option(args.normal_apply_spatial_patchify, checkpoint_args, "normal_apply_spatial_patchify", 0))
    )
    rgb_apply_spatial_patchify = bool(
        int(_resolve_option(args.rgb_apply_spatial_patchify, checkpoint_args, "rgb_apply_spatial_patchify", 0))
    )
    use_bit_label = bool(int(_resolve_option(args.use_bit_label, checkpoint_args, "use_bit_label", 1)))
    add_lvl_embeding_only_first_block = int(
        _resolve_option(
            args.add_lvl_embeding_only_first_block,
            checkpoint_args,
            "add_lvl_embeding_only_first_block",
            1,
        )
    )
    rope2d_each_sa_layer = int(_resolve_option(args.rope2d_each_sa_layer, checkpoint_args, "rope2d_each_sa_layer", 1))
    rope2d_normalized_by_hw = int(
        _resolve_option(args.rope2d_normalized_by_hw, checkpoint_args, "rope2d_normalized_by_hw", 2)
    )
    normal_use_flex_attn = bool(int(_resolve_option(args.normal_use_flex_attn, checkpoint_args, "normal_use_flex_attn", 0)))
    normal_use_segmented_flash_attn = bool(
        int(_resolve_option(args.normal_use_segmented_flash_attn, checkpoint_args, "normal_use_segmented_flash_attn", 0))
    )
    normal_bf16_activations = bool(
        int(_resolve_option(args.normal_bf16_activations, checkpoint_args, "normal_bf16_activations", 0))
    )

    if normal_vae_ckpt is None or rgb_vae_ckpt is None:
        raise ValueError("Both --normal-vae-ckpt and --rgb-vae-ckpt must be provided, either explicitly or via the training checkpoint.")

    input_paths = _resolve_image_paths(Path(args.input_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    normal_vae = build_bsq_vae(
        ckpt_path=str(normal_vae_ckpt),
        codebook_dim=normal_vae_type,
        apply_spatial_patchify=normal_apply_spatial_patchify,
        device=device,
    )
    rgb_vae = build_bsq_vae(
        ckpt_path=str(rgb_vae_ckpt),
        codebook_dim=rgb_vae_type,
        apply_spatial_patchify=rgb_apply_spatial_patchify,
        device=device,
    )

    model = build_infinity_normal_model(
        model_name=model_name,
        vae_local=normal_vae,
        pn=pn,
        batch_size=1,
        use_bit_label=use_bit_label,
        add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
        rope2d_each_sa_layer=rope2d_each_sa_layer,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        apply_spatial_patchify=normal_apply_spatial_patchify,
        normal_use_flex_attn=normal_use_flex_attn,
        normal_use_segmented_flash_attn=normal_use_segmented_flash_attn,
        normal_bf16_activations=normal_bf16_activations,
        device=device,
    )
    missing, unexpected = load_infinity_state_dict(model, str(model_path))
    print(json.dumps({"missing": len(missing), "unexpected": len(unexpected)}, indent=2))
    model.eval()

    timing_rows: list[dict[str, object]] = []
    for image_path in input_paths:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_width, original_height = image.size
            _, base_scale_schedule, target_hw = resolve_scale_schedule_from_hw(original_height, original_width, pn)
            if args.force_original_resolution:
                scale_schedule = _scale_schedule_to_original_resolution(
                    base_schedule=base_scale_schedule,
                    original_height=original_height,
                    original_width=original_width,
                )
                target_hw = (original_height, original_width)
                _ensure_custom_rope_cache(model, scale_schedule)
                print(
                    json.dumps(
                        {
                            "image": str(image_path),
                            "force_original_resolution": True,
                            "target_height": target_hw[0],
                            "target_width": target_hw[1],
                            "scale_schedule": scale_schedule,
                            "tokens": _scale_schedule_tokens(scale_schedule),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                resized = image
            else:
                scale_schedule = base_scale_schedule
                resized = image.resize((target_hw[1], target_hw[0]), resample=Image.LANCZOS)
            image_np = np.asarray(resized, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

        repeat_seconds: list[float] = []
        raw_normal = None
        raw_uncertainties = None
        with torch.no_grad():
            for _ in range(max(0, args.timing_warmup)):
                if args.save_uncertainty:
                    _ = _predict_normal_with_uncertainty(
                        image_tensor=image_tensor,
                        rgb_vae=rgb_vae,
                        normal_vae=normal_vae,
                        model=model,
                        scale_schedule=scale_schedule,
                        normal_vae_type=normal_vae_type,
                        rgb_apply_spatial_patchify=rgb_apply_spatial_patchify,
                        device=device,
                        args=args,
                    )
                else:
                    _ = _predict_normal(
                        image_tensor=image_tensor,
                        rgb_vae=rgb_vae,
                        normal_vae=normal_vae,
                        model=model,
                        scale_schedule=scale_schedule,
                        normal_vae_type=normal_vae_type,
                        rgb_apply_spatial_patchify=rgb_apply_spatial_patchify,
                        device=device,
                        args=args,
                    )
            for _ in range(max(1, args.timing_repeats)):
                _sync_device(device)
                inference_start = time.perf_counter()
                if args.save_uncertainty:
                    raw_normal, raw_uncertainties = _predict_normal_with_uncertainty(
                        image_tensor=image_tensor,
                        rgb_vae=rgb_vae,
                        normal_vae=normal_vae,
                        model=model,
                        scale_schedule=scale_schedule,
                        normal_vae_type=normal_vae_type,
                        rgb_apply_spatial_patchify=rgb_apply_spatial_patchify,
                        device=device,
                        args=args,
                    )
                else:
                    raw_normal = _predict_normal(
                        image_tensor=image_tensor,
                        rgb_vae=rgb_vae,
                        normal_vae=normal_vae,
                        model=model,
                        scale_schedule=scale_schedule,
                        normal_vae_type=normal_vae_type,
                        rgb_apply_spatial_patchify=rgb_apply_spatial_patchify,
                        device=device,
                        args=args,
                    )
                _sync_device(device)
                repeat_seconds.append(time.perf_counter() - inference_start)
            assert raw_normal is not None
            raw_normal = F.interpolate(raw_normal, size=(original_height, original_width), mode="bilinear", align_corners=False)
            raw_normal = normalize_normals(raw_normal)
            if raw_uncertainties is not None:
                raw_uncertainties = {
                    name: F.interpolate(
                        uncertainty,
                        size=(original_height, original_width),
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                    for name, uncertainty in raw_uncertainties.items()
                }
            normal_vis = normals_to_vis(raw_normal)[0]
            image_vis = F.interpolate(image_tensor, size=(original_height, original_width), mode="bilinear", align_corners=False)[0].cpu()

        stem = image_path.stem
        timing_rows.append(
            {
                "id": stem,
                "image": str(image_path),
                "height": int(original_height),
                "width": int(original_width),
                "model_input_height": int(target_hw[0]),
                "model_input_width": int(target_hw[1]),
                "force_original_resolution": bool(args.force_original_resolution),
                "tokens": _scale_schedule_tokens(scale_schedule),
                "warmup": int(max(0, args.timing_warmup)),
                "repeats": int(max(1, args.timing_repeats)),
                "repeat_inference_seconds": repeat_seconds,
                "inference_seconds": float(np.median(np.asarray(repeat_seconds, dtype=np.float64))),
                "timed_section": "rgb VAE prefix encoding + autoregressive normal generation only; excludes model loading, file I/O, resize, normalization, and saving.",
            }
        )
        _save_png(normal_vis, output_dir / f"{stem}_normal.png")
        if raw_uncertainties is not None:
            for uncertainty_name, uncertainty in raw_uncertainties.items():
                suffix = "uncertainty" if uncertainty_name == "mean" else f"uncertainty_{uncertainty_name}"
                _save_grayscale(uncertainty[0], output_dir / f"{stem}_{suffix}.png")
        if args.save_visualization:
            _save_visualization(image_vis, normal_vis, output_dir / f"{stem}_visualization.png")
        if args.save_npy:
            np.save(output_dir / f"{stem}_normal.npy", raw_normal[0].permute(1, 2, 0).cpu().numpy())
            if raw_uncertainties is not None:
                for uncertainty_name, uncertainty in raw_uncertainties.items():
                    suffix = "uncertainty" if uncertainty_name == "mean" else f"uncertainty_{uncertainty_name}"
                    np.save(output_dir / f"{stem}_{suffix}.npy", uncertainty[0, 0].cpu().numpy())

    (output_dir / "inference_times.json").write_text(
        json.dumps({"images": timing_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
