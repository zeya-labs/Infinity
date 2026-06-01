from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--save-npy", action="store_true", default=False)
    parser.add_argument("--save-visualization", action="store_true", default=True)
    parser.add_argument("--disable-save-visualization", dest="save_visualization", action="store_false")
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


def main() -> int:
    args = parse_args()
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
        device=device,
    )
    missing, unexpected = load_infinity_state_dict(model, str(model_path))
    print(json.dumps({"missing": len(missing), "unexpected": len(unexpected)}, indent=2))
    model.eval()

    for image_path in input_paths:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_width, original_height = image.size
            _, scale_schedule, target_hw = resolve_scale_schedule_from_hw(original_height, original_width, pn)
            resized = image.resize((target_hw[1], target_hw[0]), resample=Image.LANCZOS)
            image_np = np.asarray(resized, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            rgb_prefix_blc = build_prefix_tokens_from_image(
                image_01=image_tensor,
                rgb_vae=rgb_vae,
                scale_schedule=scale_schedule,
                apply_spatial_patchify=rgb_apply_spatial_patchify,
            )
            raw_normal = model.autoregressive_infer_prefix(
                vae=normal_vae,
                rgb_prefix_blc=rgb_prefix_blc,
                scale_schedule=scale_schedule,
                tau=args.tau,
                top_k=args.top_k,
                top_p=args.top_p,
                vae_type=normal_vae_type,
            )
            raw_normal = F.interpolate(raw_normal, size=(original_height, original_width), mode="bilinear", align_corners=False)
            raw_normal = normalize_normals(raw_normal)
            normal_vis = normals_to_vis(raw_normal)[0]
            image_vis = F.interpolate(image_tensor, size=(original_height, original_width), mode="bilinear", align_corners=False)[0].cpu()

        stem = image_path.stem
        _save_png(normal_vis, output_dir / f"{stem}_normal.png")
        if args.save_visualization:
            _save_visualization(image_vis, normal_vis, output_dir / f"{stem}_visualization.png")
        if args.save_npy:
            np.save(output_dir / f"{stem}_normal.npy", raw_normal[0].permute(1, 2, 0).cpu().numpy())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
