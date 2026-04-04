from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from infinity.models.infinity import Infinity
from infinity.models import alias_dict
from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def normalize_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals.float() / torch.linalg.norm(normals.float(), dim=1, keepdim=True).clamp_min(1e-6)


def normals_to_vis(normals: torch.Tensor, invert_x_for_vis: bool = True) -> torch.Tensor:
    normals = normalize_normals(normals).detach().cpu().float().clamp(-1.0, 1.0)
    vis = normals.clone()
    if invert_x_for_vis:
        vis[:, 0] = -vis[:, 0]
    return ((vis + 1.0) / 2.0).clamp(0.0, 1.0)


def resolve_scale_schedule_from_hw(height: int, width: int, pn: str) -> tuple[float, list[tuple[int, int, int]], tuple[int, int]]:
    ratio = float(height) / float(width)
    template = float(h_div_w_templates[np.argmin(np.abs(h_div_w_templates - ratio))])
    schedule = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[template][pn]["scales"]]
    pixel_hw = tuple(int(item) for item in dynamic_resolution_h_w[template][pn]["pixel"])
    return template, schedule, pixel_hw


def max_condition_length_for_pn(pn: str) -> int:
    max_len = 1
    for template in h_div_w_templates.tolist():
        schedule = dynamic_resolution_h_w[template][pn]["scales"]
        seq_len = sum(int(t) * int(h) * int(w) for (t, h, w) in schedule)
        max_len = max(max_len, seq_len - int(schedule[0][0]) * int(schedule[0][1]) * int(schedule[0][2]))
    return max_len


def build_vae_scale_schedule(scale_schedule: list[tuple[int, int, int]], apply_spatial_patchify: bool) -> list[tuple[int, int, int]]:
    if apply_spatial_patchify:
        return [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    return list(scale_schedule)


def build_bsq_vae(
    *,
    ckpt_path: str,
    codebook_dim: int,
    apply_spatial_patchify: bool,
    device: torch.device,
) -> torch.nn.Module:
    patch_size = 8 if apply_spatial_patchify else 16
    encoder_ch_mult = [1, 2, 4, 4] if apply_spatial_patchify else [1, 2, 4, 4, 4]
    decoder_ch_mult = [1, 2, 4, 4] if apply_spatial_patchify else [1, 2, 4, 4, 4]
    vae = vae_model(
        ckpt_path,
        schedule_mode="dynamic",
        codebook_dim=codebook_dim,
        codebook_size=2 ** codebook_dim,
        test_mode=True,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
    )
    return vae.to(device)


def _parse_model_name(model_name: str) -> tuple[str, int | None]:
    normalized = alias_dict.get(model_name, model_name)
    if not normalized.startswith("infinity_"):
        normalized = f"infinity_{normalized}"
    if normalized.rsplit("c", maxsplit=1)[-1].isdigit():
        normalized, block_chunks = normalized.rsplit("c", maxsplit=1)
        return normalized, int(block_chunks)
    return normalized, None


def _model_config(model_name: str) -> tuple[str, dict[str, Any]]:
    normalized_name, block_chunks_override = _parse_model_name(model_name)
    configs: dict[str, dict[str, Any]] = {
        "infinity_2b": dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        "infinity_8b": dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        "infinity_layer12": dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
        "infinity_layer16": dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
        "infinity_layer24": dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
        "infinity_layer32": dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
        "infinity_layer40": dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
        "infinity_layer48": dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=1),
    }
    if normalized_name not in configs:
        raise ValueError(f"Unsupported Infinity normal-estimation model: {model_name}")
    config = dict(configs[normalized_name])
    if block_chunks_override is not None:
        config["block_chunks"] = block_chunks_override
    return normalized_name, config


def build_infinity_normal_model(
    *,
    model_name: str,
    vae_local: torch.nn.Module,
    cond_dim: int,
    text_maxlen: int,
    pn: str,
    batch_size: int,
    use_bit_label: bool,
    add_lvl_embeding_only_first_block: int,
    rope2d_each_sa_layer: int,
    rope2d_normalized_by_hw: int,
    apply_spatial_patchify: bool,
    device: torch.device,
) -> torch.nn.Module:
    _, model_config = _model_config(model_name)
    model = Infinity(
        vae_local=vae_local,
        text_channels=cond_dim,
        text_maxlen=text_maxlen,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        shared_aln=True,
        cond_drop_rate=0.1,
        nm0=False,
        tau=1,
        cos_attn=True,
        head_depth=1,
        checkpointing="full-block",
        pad_to_multiplier=1,
        use_flex_attn=False,
        batch_size=batch_size,
        add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
        use_bit_label=use_bit_label,
        rope2d_each_sa_layer=rope2d_each_sa_layer,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        pn=pn,
        train_h_div_w_list=h_div_w_templates.tolist(),
        always_training_scales=100,
        apply_spatial_patchify=apply_spatial_patchify,
        **model_config,
    )
    return model.to(device)


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "gpt" in checkpoint:
        return checkpoint["gpt"]
    if "gpt_wo_ddp" in checkpoint:
        return checkpoint["gpt_wo_ddp"]
    if "trainer" in checkpoint and isinstance(checkpoint["trainer"], dict):
        trainer_state = checkpoint["trainer"]
        if "gpt_fsdp" in trainer_state:
            return trainer_state["gpt_fsdp"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _normalize_state_key(key: str) -> str:
    normalized_key = key
    for prefix in ("module.", "_orig_mod."):
        if normalized_key.startswith(prefix):
            normalized_key = normalized_key[len(prefix) :]
    return normalized_key


def _copy_matching_state_dict_tensors(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    loaded_keys: set[str],
    skipped_keys: list[str],
) -> None:
    ignored_runtime_buffers = {
        "lvl_1L",
        "attn_bias_for_masking",
        "Infinity_visible_kvlen",
        "Infinity_invisible_qlen",
    }
    model_state = model.state_dict(keep_vars=True)
    with torch.no_grad():
        for raw_key, value in state_dict.items():
            key = _normalize_state_key(raw_key)
            if key in ignored_runtime_buffers:
                continue
            if key not in model_state:
                skipped_keys.append(key)
                continue

            target = model_state[key]
            if key == "cfg_uncond":
                if target.ndim != value.ndim or target.shape[1:] != value.shape[1:]:
                    skipped_keys.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
                    continue
                min_tlen = min(int(value.shape[0]), int(target.shape[0]))
                target[:min_tlen].copy_(value[:min_tlen].to(device=target.device, dtype=target.dtype))
                loaded_keys.add(key)
                continue

            if target.shape != value.shape:
                skipped_keys.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
                continue

            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded_keys.add(key)


def _load_directory_state_dict(model: torch.nn.Module, checkpoint_dir: Path) -> tuple[list[str], list[str]]:
    index_paths = sorted(checkpoint_dir.glob("*.index.json"))
    if index_paths:
        index = json.loads(index_paths[0].read_text(encoding="utf-8"))
        shard_names = list(dict.fromkeys(index["weight_map"].values()))
        shard_paths = [checkpoint_dir / shard_name for shard_name in shard_names]
    else:
        shard_paths = sorted(checkpoint_dir.glob("*.safetensors")) + sorted(checkpoint_dir.glob("*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No checkpoint shards found in {checkpoint_dir}")

    loaded_keys: set[str] = set()
    skipped_keys: list[str] = []
    for shard_path in shard_paths:
        if shard_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            shard_state = load_file(str(shard_path))
        else:
            shard_state = _extract_state_dict(torch.load(shard_path, map_location="cpu", weights_only=False))
        _copy_matching_state_dict_tensors(model, shard_state, loaded_keys=loaded_keys, skipped_keys=skipped_keys)

    missing = [key for key in model.state_dict().keys() if key not in loaded_keys]
    return missing, skipped_keys


def load_infinity_state_dict(model: torch.nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    checkpoint_path_str = str(checkpoint_path)
    if Path(checkpoint_path_str).is_dir():
        return _load_directory_state_dict(model, Path(checkpoint_path_str))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    loaded_keys: set[str] = set()
    skipped_keys: list[str] = []
    _copy_matching_state_dict_tensors(model, state_dict, loaded_keys=loaded_keys, skipped_keys=skipped_keys)
    missing = [key for key in model.state_dict().keys() if key not in loaded_keys]
    return missing, skipped_keys


def build_multiscale_var_inputs(
    *,
    vae: torch.nn.Module,
    raw_features: torch.Tensor,
    vae_scale_schedule: list[tuple[int, int, int]],
    apply_spatial_patchify: bool,
    noise_apply_layers: int = -1,
    noise_apply_strength: float = 0.0,
    noise_apply_requant: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.amp.autocast("cuda", enabled=False):
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2)
        else:
            codes_out = raw_features

        batch_size = raw_features.shape[0]
        cum_var_input = 0
        gt_all_bit_indices: list[torch.Tensor] = []
        x_blc_without_prefix: list[torch.Tensor] = []

        for scale_index, scale_item in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if scale_index != len(vae_scale_schedule) - 1:
                residual = F.interpolate(
                    residual,
                    size=scale_item,
                    mode=vae.quantizer.z_interplote_down,
                ).contiguous()
            quantized, _, bit_indices, _ = vae.quantizer.lfq(residual)
            gt_all_bit_indices.append(bit_indices)

            if 0 <= noise_apply_layers and scale_index < noise_apply_layers and noise_apply_strength > 0:
                mask = torch.rand_like(bit_indices.float()) < noise_apply_strength
                noisy_indices = bit_indices.clone()
                noisy_indices[mask] = 1 - noisy_indices[mask]
                if noise_apply_requant:
                    quantized = vae.quantizer.lfq.indices_to_codes(noisy_indices, label_type="bit_label")

            cum_var_input = cum_var_input + F.interpolate(
                quantized,
                size=vae_scale_schedule[-1],
                mode=vae.quantizer.z_interplote_up,
            ).contiguous()

            if scale_index < len(vae_scale_schedule) - 1:
                next_input = F.interpolate(
                    cum_var_input,
                    size=vae_scale_schedule[scale_index + 1],
                    mode=vae.quantizer.z_interplote_up,
                ).contiguous()
                next_input = next_input.squeeze(-3)
                if apply_spatial_patchify:
                    next_input = torch.nn.functional.pixel_unshuffle(next_input, 2)
                x_blc_without_prefix.append(next_input.reshape(*next_input.shape[:2], -1).permute(0, 2, 1))

        if apply_spatial_patchify:
            gt_ms_idx_bl: list[torch.Tensor] = []
            for item in gt_all_bit_indices:
                item = item.squeeze(1).permute(0, 3, 1, 2)
                item = torch.nn.functional.pixel_unshuffle(item, 2)
                item = item.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * vae.codebook_dim)
                gt_ms_idx_bl.append(item)
        else:
            gt_ms_idx_bl = [item.reshape(batch_size, -1, vae.codebook_dim) for item in gt_all_bit_indices]

        return torch.cat(x_blc_without_prefix, dim=1), gt_ms_idx_bl


def build_condition_tuple_from_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor, int]:
    batch_size, seq_len, channels = tokens.shape
    del channels
    lens = [seq_len] * batch_size
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=tokens.device)
    # Materialize a fresh tensor so later in-place condition dropout in Infinity.forward
    # does not mutate a view that was created inside a no_grad region.
    kv_compact = tokens.reshape(batch_size * seq_len, -1).clone()
    return kv_compact, lens, cu_seqlens, seq_len


def build_condition_tuple_from_image(
    *,
    image_01: torch.Tensor,
    rgb_vae: torch.nn.Module,
    scale_schedule: list[tuple[int, int, int]],
    apply_spatial_patchify: bool,
) -> tuple[tuple[torch.Tensor, list[int], torch.Tensor, int], torch.Tensor]:
    image_pm1 = image_01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    vae_scale_schedule = build_vae_scale_schedule(scale_schedule, apply_spatial_patchify)
    raw_features, _, _ = rgb_vae.encode_for_raw_features(image_pm1, scale_schedule=vae_scale_schedule)
    cond_tokens, _ = build_multiscale_var_inputs(
        vae=rgb_vae,
        raw_features=raw_features,
        vae_scale_schedule=vae_scale_schedule,
        apply_spatial_patchify=apply_spatial_patchify,
        noise_apply_layers=-1,
    )
    return build_condition_tuple_from_tokens(cond_tokens), cond_tokens


def decode_logits_to_normal(
    *,
    logits_blv: torch.Tensor,
    vae: torch.nn.Module,
    scale_schedule: list[tuple[int, int, int]],
    use_bit_label: bool,
    apply_spatial_patchify: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits_blv.shape[0]
    vae_scale_schedule = build_vae_scale_schedule(scale_schedule, apply_spatial_patchify)
    summed_codes = 0
    pointer = 0

    for scale_item in scale_schedule:
        stage_len = int(np.array(scale_item).prod())
        stage_logits = logits_blv[:, pointer : pointer + stage_len]
        pointer += stage_len

        if use_bit_label:
            stage_bits = stage_logits.reshape(batch_size, stage_len, -1, 2).argmax(dim=-1).float()
            stage_bits = stage_bits.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2], -1)
            if apply_spatial_patchify:
                assert scale_item[0] == 1
                stage_bits = stage_bits.squeeze(1).permute(0, 3, 1, 2)
                stage_bits = torch.nn.functional.pixel_shuffle(stage_bits, 2)
                stage_bits = stage_bits.permute(0, 2, 3, 1).unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(stage_bits, label_type="bit_label")
        else:
            stage_indices = stage_logits.argmax(dim=-1).reshape(batch_size, scale_item[0], scale_item[1], scale_item[2])
            codes = vae.quantizer.lfq.indices_to_codes(stage_indices, label_type="int_label")

        if scale_item != scale_schedule[-1]:
            summed_codes = summed_codes + F.interpolate(
                codes,
                size=vae_scale_schedule[-1],
                mode=vae.quantizer.z_interplote_up,
            )
        else:
            summed_codes = summed_codes + codes

    prediction = vae.decode(summed_codes.squeeze(-3))
    prediction = prediction.clamp(-1.0, 1.0)
    return prediction, summed_codes


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if value.ndim == 4:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    numerator = (value * mask).sum()
    denominator = mask.sum().clamp_min(1.0)
    return numerator / denominator


def _masked_scalar_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.squeeze(1).bool()
    if not torch.any(valid):
        return value.new_tensor(0.0)
    return value[valid].mean()


def compute_normal_metrics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latent_prediction: torch.Tensor | None = None,
    latent_target: torch.Tensor | None = None,
    l1_weight: float = 0.0,
    angular_weight: float = 0.0,
    latent_weight: float = 0.0,
    norm_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction_normalized = normalize_normals(prediction)
    target_normalized = normalize_normals(target)
    dot = (prediction_normalized * target_normalized).sum(dim=1).clamp(-1.0, 1.0)
    angular_rad = torch.acos(dot)
    angular_deg = torch.rad2deg(angular_rad)
    pred_norm = torch.linalg.norm(prediction, dim=1, keepdim=True)

    loss_l1 = _masked_channel_mean((prediction - target).abs(), mask)
    loss_angular = _masked_scalar_mean(angular_rad, mask)
    loss_norm = _masked_channel_mean((pred_norm - 1.0).abs(), mask)
    if latent_prediction is not None and latent_target is not None:
        loss_latent = F.mse_loss(latent_prediction.float(), latent_target.float())
    else:
        loss_latent = prediction.new_tensor(0.0)

    total_loss = (
        l1_weight * loss_l1
        + angular_weight * loss_angular
        + latent_weight * loss_latent
        + norm_weight * loss_norm
    )

    metrics = {
        "loss_l1": loss_l1.detach(),
        "loss_angular_rad": loss_angular.detach(),
        "loss_latent": loss_latent.detach(),
        "loss_norm": loss_norm.detach(),
        "angle_deg": _masked_scalar_mean(angular_deg, mask).detach(),
        "acc_11_25": _masked_scalar_mean((angular_deg < 11.25).float(), mask).detach(),
        "acc_22_5": _masked_scalar_mean((angular_deg < 22.5).float(), mask).detach(),
        "acc_30": _masked_scalar_mean((angular_deg < 30.0).float(), mask).detach(),
        "loss_total_aux": total_loss.detach(),
    }
    return total_loss, metrics
