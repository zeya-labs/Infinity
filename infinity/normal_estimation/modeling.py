from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import Infinity, sample_with_top_k_top_p_also_inplace_modifying_logits_
from infinity.models import alias_dict
from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    NORMAL_FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    create_block_mask = flex_attention = None
    NORMAL_FLEX_ATTENTION_AVAILABLE = False

try:
    from flash_attn import flash_attn_func as normal_flash_attn_func
    NORMAL_FLASH_ATTENTION_AVAILABLE = True
except (ImportError, OSError):
    normal_flash_attn_func = None
    NORMAL_FLASH_ATTENTION_AVAILABLE = False


def normalize_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals.float() / torch.linalg.norm(normals.float(), dim=1, keepdim=True).clamp_min(1e-6)


def normals_to_vis(normals: torch.Tensor, invert_x_for_vis: bool = False) -> torch.Tensor:
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
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def _scale_seq_len(scale_schedule: list[tuple[int, int, int]]) -> int:
    return int(sum(int(pt) * int(ph) * int(pw) for pt, ph, pw in scale_schedule))


def _stage_ids_for_schedule(scale_schedule: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            torch.full((int(pt) * int(ph) * int(pw),), stage_index, dtype=torch.long, device=device)
            for stage_index, (pt, ph, pw) in enumerate(scale_schedule)
        ],
        dim=0,
    )


def _stage_lengths(scale_schedule: list[tuple[int, int, int]]) -> list[int]:
    return [int(pt) * int(ph) * int(pw) for pt, ph, pw in scale_schedule]


def _stage_offsets(scale_schedule: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for length in _stage_lengths(scale_schedule):
        end = start + length
        offsets.append((start, end))
        start = end
    return offsets


def _flatten_feature_tokens(feature: torch.Tensor, *, apply_spatial_patchify: bool) -> torch.Tensor:
    feature = feature.squeeze(-3)
    if apply_spatial_patchify:
        feature = torch.nn.functional.pixel_unshuffle(feature, 2)
    return feature.reshape(*feature.shape[:2], -1).permute(0, 2, 1).contiguous()


def build_multiscale_prefix_tokens(
    *,
    vae: torch.nn.Module,
    raw_features: torch.Tensor,
    vae_scale_schedule: list[tuple[int, int, int]],
    apply_spatial_patchify: bool,
) -> torch.Tensor:
    """Build same-length RGB prefix tokens for all autoregressive stages."""
    with torch.amp.autocast("cuda", enabled=False):
        codes_out = raw_features.unsqueeze(2) if raw_features.dim() == 4 else raw_features
        cumulative = torch.zeros_like(codes_out)
        prefix_tokens: list[torch.Tensor] = []

        for scale_item in vae_scale_schedule:
            residual = codes_out - cumulative
            if scale_item != vae_scale_schedule[-1]:
                residual = F.interpolate(
                    residual,
                    size=scale_item,
                    mode=vae.quantizer.z_interplote_down,
                ).contiguous()
            quantized, _, _, _ = vae.quantizer.lfq(residual)
            cumulative = cumulative + F.interpolate(
                quantized,
                size=vae_scale_schedule[-1],
                mode=vae.quantizer.z_interplote_up,
            ).contiguous()

            if scale_item == vae_scale_schedule[-1]:
                stage_feature = cumulative
            else:
                stage_feature = F.interpolate(
                    cumulative,
                    size=scale_item,
                    mode=vae.quantizer.z_interplote_down,
                ).contiguous()
            prefix_tokens.append(_flatten_feature_tokens(stage_feature, apply_spatial_patchify=apply_spatial_patchify))

        return torch.cat(prefix_tokens, dim=1)


def build_prefix_tokens_from_image(
    *,
    image_01: torch.Tensor,
    rgb_vae: torch.nn.Module,
    scale_schedule: list[tuple[int, int, int]],
    apply_spatial_patchify: bool,
) -> torch.Tensor:
    image_pm1 = image_01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    vae_scale_schedule = build_vae_scale_schedule(scale_schedule, apply_spatial_patchify)
    raw_features, _, _ = rgb_vae.encode_for_raw_features(image_pm1, scale_schedule=vae_scale_schedule)
    return build_multiscale_prefix_tokens(
        vae=rgb_vae,
        raw_features=raw_features,
        vae_scale_schedule=vae_scale_schedule,
        apply_spatial_patchify=apply_spatial_patchify,
    )


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


class InfinityNormalPrefixModel(Infinity):
    """Infinity normal estimator with RGB VAE tokens as self-attention prefix."""

    def __init__(
        self,
        *args: Any,
        task_condition_len: int = 16,
        normal_token_layout: str = "prefix",
        normal_use_flex_attn: bool = False,
        normal_use_segmented_flash_attn: bool = False,
        normal_bf16_activations: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.t2i:
            raise ValueError("InfinityNormalPrefixModel keeps the pretrained text/cross-attn stack for task conditioning.")
        if task_condition_len <= 0:
            raise ValueError("task_condition_len must be positive.")
        if normal_use_flex_attn and not NORMAL_FLEX_ATTENTION_AVAILABLE:
            raise NotImplementedError(f"FlexAttention requires PyTorch 2.5+, got {torch.__version__}.")
        if normal_use_segmented_flash_attn and not NORMAL_FLASH_ATTENTION_AVAILABLE:
            raise NotImplementedError("flash-attn is required for normal segmented flash attention.")
        if normal_use_flex_attn and normal_use_segmented_flash_attn:
            raise ValueError("Use only one normal attention backend: flex or segmented flash.")
        if normal_token_layout not in {"prefix", "interleaved", "interleaved_source"}:
            raise ValueError(f"Unsupported normal_token_layout: {normal_token_layout}")
        if normal_use_flex_attn:
            torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"

        self.task_condition_len = task_condition_len
        self.normal_token_layout = normal_token_layout
        self.normal_use_flex_attn = normal_use_flex_attn
        self.normal_use_segmented_flash_attn = normal_use_segmented_flash_attn
        self.normal_bf16_activations = normal_bf16_activations
        self.normal_flex_attention = torch.compile(flex_attention) if normal_use_flex_attn else None
        self.normal_flex_block_masks: dict[str, Any] = {}
        self.normal_prefix_layout_cache: dict[str, dict[str, Any]] = {}
        self.image_word_embed = nn.Linear(self.d_vae, self.C)
        self.image_word_embed.load_state_dict(self.word_embed.state_dict())
        self.image_modality_embed = nn.Parameter(torch.zeros(1, 1, self.C))
        self.normal_modality_embed = nn.Parameter(torch.zeros(1, 1, self.C))

        task_kv = torch.empty(task_condition_len, self.Ct5)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(1)
        nn.init.trunc_normal_(task_kv, std=1.2, generator=rng)
        task_kv /= self.Ct5 ** 0.5
        self.normal_task_kv = nn.Parameter(task_kv)

    def initialize_missing_prefix_parameters(self, loaded_keys: set[str]) -> None:
        with torch.no_grad():
            if "image_word_embed.weight" not in loaded_keys:
                self.image_word_embed.weight.copy_(self.word_embed.weight)
            if "image_word_embed.bias" not in loaded_keys and self.word_embed.bias is not None:
                self.image_word_embed.bias.copy_(self.word_embed.bias)
            if "image_modality_embed" not in loaded_keys:
                self.image_modality_embed.zero_()
            if "normal_modality_embed" not in loaded_keys:
                self.normal_modality_embed.zero_()
            if "normal_task_kv" not in loaded_keys and hasattr(self, "cfg_uncond"):
                copy_len = min(self.normal_task_kv.shape[0], self.cfg_uncond.shape[0])
                copy_dim = min(self.normal_task_kv.shape[1], self.cfg_uncond.shape[1])
                self.normal_task_kv[:copy_len, :copy_dim].copy_(self.cfg_uncond[:copy_len, :copy_dim])

    def _task_condition(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        kv = self.normal_task_kv.unsqueeze(0).expand(batch_size, -1, -1)
        kv_compact = kv.reshape(batch_size * self.task_condition_len, self.Ct5).contiguous()
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * self.task_condition_len,
            self.task_condition_len,
            dtype=torch.int32,
            device=kv_compact.device,
        )
        max_seqlen_k = self.task_condition_len
        must_on_graph = self.cfg_uncond[0, 0] * 0 if hasattr(self, "cfg_uncond") else kv_compact.new_tensor(0.0)
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
        kv_compact[0, 0] = kv_compact[0, 0] + must_on_graph
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        return sos, cond_BD, cond_BD_or_gss, ca_kv

    def _embed_rgb_prefix(self, rgb_prefix_blc: torch.Tensor) -> torch.Tensor:
        if rgb_prefix_blc.shape[-1] != self.d_vae:
            raise ValueError(f"RGB prefix channel dim {rgb_prefix_blc.shape[-1]} does not match model VAE dim {self.d_vae}.")
        return self.image_word_embed(self.norm0_ve(rgb_prefix_blc.float())) + self.image_modality_embed

    def _embed_normal_inputs(self, normal_x_blc_wo_prefix: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
        batch_size = normal_x_blc_wo_prefix.shape[0]
        sos = sos.unsqueeze(1).expand(batch_size, self.first_l, -1) + self.pos_start.expand(batch_size, self.first_l, -1)
        if normal_x_blc_wo_prefix.numel() == 0:
            return sos + self.normal_modality_embed
        normal_x = self.word_embed(self.norm0_ve(normal_x_blc_wo_prefix.float()))
        return torch.cat((sos, normal_x), dim=1) + self.normal_modality_embed

    def _ensure_sequence_rope_cache(
        self,
        sequence_schedule: list[tuple[int, int, int]],
        sequence_source_indices: list[int],
        source_schedule: list[tuple[int, int, int]],
        padded_len: int,
    ) -> list[tuple[int, int, int]]:
        if not self.rope2d_each_sa_layer:
            return sequence_schedule
        rope_key = str(tuple(sequence_schedule))
        if rope_key in self.rope2d_freqs_grid and self.rope2d_freqs_grid[rope_key].shape[4] >= padded_len:
            return sequence_schedule
        base_cache = self._find_rope_cache_for_schedule(source_schedule)
        if base_cache is None:
            return sequence_schedule

        source_offsets = _stage_offsets(source_schedule)
        pieces = []
        if len(sequence_source_indices) != len(sequence_schedule):
            return sequence_schedule
        for source_index in sequence_source_indices:
            if source_index < 0 or source_index >= len(source_offsets):
                return sequence_schedule
            start, end = source_offsets[source_index]
            pieces.append(base_cache[..., start:end, :])
        rope_cache = torch.cat(pieces, dim=4)
        if rope_cache.shape[4] < padded_len:
            pad = torch.zeros(
                *rope_cache.shape[:4],
                padded_len - rope_cache.shape[4],
                rope_cache.shape[-1],
                dtype=rope_cache.dtype,
                device=rope_cache.device,
            )
            rope_cache = torch.cat([rope_cache, pad], dim=4)
        self.rope2d_freqs_grid[rope_key] = rope_cache
        return sequence_schedule

    def _find_rope_cache_for_schedule(self, scale_schedule: list[tuple[int, int, int]]) -> torch.Tensor | None:
        exact_key = str(tuple(scale_schedule))
        if exact_key in self.rope2d_freqs_grid:
            return self.rope2d_freqs_grid[exact_key]

        schedule_tuple = tuple(scale_schedule)
        schedule_len = len(schedule_tuple)
        token_len = _scale_seq_len(scale_schedule)
        for cache_key, cache in self.rope2d_freqs_grid.items():
            try:
                cached_schedule = tuple(tuple(int(value) for value in item) for item in ast.literal_eval(cache_key))
            except (SyntaxError, ValueError, TypeError):
                continue
            if (
                len(cached_schedule) >= schedule_len
                and cached_schedule[:schedule_len] == schedule_tuple
                and cache.shape[4] >= token_len
            ):
                return cache
        return None

    def _normal_prefix_layout(
        self,
        prefix_schedule: list[tuple[int, int, int]],
        normal_schedule: list[tuple[int, int, int]],
        device: torch.device,
    ) -> dict[str, Any]:
        key = f"{self.normal_token_layout}:{device}:{tuple(prefix_schedule)}:{tuple(normal_schedule)}"
        layout = self.normal_prefix_layout_cache.get(key)
        if layout is not None:
            return layout

        prefix_len = _scale_seq_len(prefix_schedule)
        normal_len = _scale_seq_len(normal_schedule)
        if self.normal_token_layout == "prefix":
            prefix_stage_ids = _stage_ids_for_schedule(prefix_schedule, device)
            normal_stage_ids = _stage_ids_for_schedule(normal_schedule, device)
            stage_ids = torch.cat((prefix_stage_ids, normal_stage_ids), dim=0)
            normal_token_indices = torch.arange(prefix_len, prefix_len + normal_len, dtype=torch.long, device=device)
            sequence_schedule = list(prefix_schedule) + list(normal_schedule)
            sequence_source_indices = list(range(len(prefix_schedule))) + list(range(len(normal_schedule)))

            segments: list[tuple[int, int, int]] = [(0, prefix_len, prefix_len)]
            start = prefix_len
            for pt, ph, pw in normal_schedule:
                end = start + int(pt) * int(ph) * int(pw)
                segments.append((start, end, end))
                start = end
            segment_key_indices = None
        elif self.normal_token_layout == "interleaved":
            if len(prefix_schedule) != len(normal_schedule):
                raise ValueError(
                    "interleaved normal token layout requires matching RGB and normal schedule lengths; "
                    f"got RGB={len(prefix_schedule)} normal={len(normal_schedule)}."
                )
            stage_ids_list: list[torch.Tensor] = []
            normal_token_indices_list: list[torch.Tensor] = []
            sequence_schedule = []
            sequence_source_indices = []
            segments = []
            start = 0
            for stage_index, ((r_pt, r_ph, r_pw), (n_pt, n_ph, n_pw)) in enumerate(zip(prefix_schedule, normal_schedule, strict=True)):
                if (r_pt, r_ph, r_pw) != (n_pt, n_ph, n_pw):
                    raise ValueError("interleaved normal token layout requires aligned RGB and normal schedules.")
                rgb_len = int(r_pt) * int(r_ph) * int(r_pw)
                normal_stage_len = int(n_pt) * int(n_ph) * int(n_pw)
                stage_ids_list.append(torch.full((rgb_len,), stage_index, dtype=torch.long, device=device))
                stage_ids_list.append(torch.full((normal_stage_len,), stage_index, dtype=torch.long, device=device))
                rgb_end = start + rgb_len
                normal_end = rgb_end + normal_stage_len
                normal_indices = torch.arange(rgb_end, normal_end, dtype=torch.long, device=device)
                normal_token_indices_list.append(normal_indices)
                sequence_schedule.extend([(r_pt, r_ph, r_pw), (n_pt, n_ph, n_pw)])
                sequence_source_indices.extend([stage_index, stage_index])
                segments.append((start, rgb_end, rgb_end))
                segments.append((rgb_end, normal_end, normal_end))
                start = normal_end
            stage_ids = torch.cat(stage_ids_list, dim=0)
            normal_stage_ids = _stage_ids_for_schedule(normal_schedule, device)
            normal_token_indices = torch.cat(normal_token_indices_list, dim=0)
            segment_key_indices = None
        else:
            if len(prefix_schedule) != len(normal_schedule):
                raise ValueError(
                    "interleaved_source normal token layout requires matching RGB and normal schedule lengths; "
                    f"got RGB={len(prefix_schedule)} normal={len(normal_schedule)}."
                )
            stage_ids_list = []
            normal_token_indices_list = []
            sequence_schedule = []
            sequence_source_indices = []
            segments = []
            segment_key_indices_list: list[torch.Tensor] = []
            start = 0
            rgb_seen_indices: list[torch.Tensor] = []
            visible_seen_indices: list[torch.Tensor] = []
            for stage_index, ((r_pt, r_ph, r_pw), (n_pt, n_ph, n_pw)) in enumerate(zip(prefix_schedule, normal_schedule, strict=True)):
                if (r_pt, r_ph, r_pw) != (n_pt, n_ph, n_pw):
                    raise ValueError("interleaved_source normal token layout requires aligned RGB and normal schedules.")
                rgb_len = int(r_pt) * int(r_ph) * int(r_pw)
                normal_stage_len = int(n_pt) * int(n_ph) * int(n_pw)
                stage_ids_list.append(torch.full((rgb_len,), stage_index, dtype=torch.long, device=device))
                stage_ids_list.append(torch.full((normal_stage_len,), stage_index, dtype=torch.long, device=device))
                rgb_end = start + rgb_len
                normal_end = rgb_end + normal_stage_len
                rgb_indices = torch.arange(start, rgb_end, dtype=torch.long, device=device)
                normal_indices = torch.arange(rgb_end, normal_end, dtype=torch.long, device=device)
                rgb_seen_indices.append(rgb_indices)
                visible_seen_indices.append(rgb_indices)
                normal_token_indices_list.append(normal_indices)
                sequence_schedule.extend([(r_pt, r_ph, r_pw), (n_pt, n_ph, n_pw)])
                sequence_source_indices.extend([stage_index, stage_index])
                segments.append((start, rgb_end, rgb_end))
                segment_key_indices_list.append(torch.cat(rgb_seen_indices, dim=0))
                segments.append((rgb_end, normal_end, normal_end))
                segment_key_indices_list.append(torch.cat(visible_seen_indices + [normal_indices], dim=0))
                visible_seen_indices.append(normal_indices)
                start = normal_end
            stage_ids = torch.cat(stage_ids_list, dim=0)
            normal_stage_ids = _stage_ids_for_schedule(normal_schedule, device)
            normal_token_indices = torch.cat(normal_token_indices_list, dim=0)
            segment_key_indices = tuple(segment_key_indices_list)

        layout = {
            "prefix_len": prefix_len,
            "normal_len": normal_len,
            "stage_ids": stage_ids,
            "normal_stage_ids": normal_stage_ids,
            "normal_token_indices": normal_token_indices,
            "sequence_schedule": sequence_schedule,
            "sequence_source_indices": sequence_source_indices,
            "segments": tuple(segments),
            "segment_key_indices": segment_key_indices,
        }
        self.normal_prefix_layout_cache[key] = layout
        return layout

    def _add_prefix_level_embedding(self, x: torch.Tensor, stage_ids: torch.Tensor) -> torch.Tensor:
        if stage_ids.numel() == 0:
            return x
        level_emb = self.lvl_embed(stage_ids).to(dtype=x.dtype).unsqueeze(0)
        if level_emb.shape[1] < x.shape[1]:
            level_emb = F.pad(level_emb, (0, 0, 0, x.shape[1] - level_emb.shape[1]))
        return x + level_emb

    def _normal_prefix_flex_attn(
        self,
        *,
        stage_ids: torch.Tensor,
        batch_size: int,
        segments: tuple[tuple[int, int, int], ...],
        segment_key_indices: tuple[torch.Tensor, ...] | None,
    ):
        if not self.normal_use_flex_attn:
            return None
        if self.normal_flex_attention is None:
            raise RuntimeError("Normal FlexAttention was not initialized.")

        if segment_key_indices is None:
            key_end_by_query = torch.empty(stage_ids.shape[0], dtype=torch.int32, device=stage_ids.device)
            for query_start, query_end, key_end in segments:
                key_end_by_query[query_start:query_end] = int(key_end)
            key = f"{batch_size}:{self.num_heads}:{stage_ids.device}:{tuple(segments)}"
        else:
            visible = torch.zeros((stage_ids.shape[0], stage_ids.shape[0]), dtype=torch.bool, device=stage_ids.device)
            for (query_start, query_end, _), key_indices in zip(segments, segment_key_indices, strict=True):
                visible[query_start:query_end, key_indices] = True
            key = f"{batch_size}:{self.num_heads}:{stage_ids.device}:{tuple(segments)}:{tuple(tuple(int(v) for v in item.detach().cpu().tolist()) for item in segment_key_indices)}"
        block_mask = self.normal_flex_block_masks.get(key)
        if block_mask is None:
            if segment_key_indices is None:
                def normal_prefix_mask(_b, _h, q_idx, kv_idx):
                    return kv_idx < key_end_by_query[q_idx]
            else:
                def normal_prefix_mask(_b, _h, q_idx, kv_idx):
                    return visible[q_idx, kv_idx]

            block_mask = create_block_mask(
                normal_prefix_mask,
                B=batch_size,
                H=self.num_heads,
                Q_LEN=int(stage_ids.shape[0]),
                KV_LEN=int(stage_ids.shape[0]),
                device=stage_ids.device,
                _compile=True,
            )
            self.normal_flex_block_masks[key] = block_mask

        def attn_fn(q, k, v, scale=None):
            return self.normal_flex_attention(q.to(v.dtype), k.to(v.dtype), v, block_mask=block_mask, scale=scale)

        return attn_fn

    def _normal_prefix_segmented_flash_attn(
        self,
        *,
        segments: tuple[tuple[int, int, int], ...],
        segment_key_indices: tuple[torch.Tensor, ...] | None,
    ):
        if not self.normal_use_segmented_flash_attn:
            return None
        if normal_flash_attn_func is None:
            raise RuntimeError("flash-attn was not initialized.")
        if segment_key_indices is not None:
            raise NotImplementedError("Segmented flash attention only supports contiguous-prefix normal layouts.")
        def attn_fn(q, k, v, scale=None):
            pieces = []
            for query_start, query_end, key_end in segments:
                q_part = q[:, query_start:query_end, :, :].contiguous()
                k_part = k[:, :key_end, :, :].contiguous()
                v_part = v[:, :key_end, :, :].contiguous()
                out = normal_flash_attn_func(
                    q_part.to(v_part.dtype),
                    k_part.to(v_part.dtype),
                    v_part,
                    dropout_p=0,
                    softmax_scale=scale,
                )
                pieces.append(out)
            return torch.cat(pieces, dim=1)

        attn_fn.expects_blhc = True
        return attn_fn

    def _prepare_prefix_sequence(
        self,
        *,
        rgb_prefix_emb: torch.Tensor,
        normal_input_emb: torch.Tensor,
        prefix_schedule: list[tuple[int, int, int]],
        normal_schedule: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]], torch.Tensor, int, int, tuple[tuple[int, int, int], ...]]:
        prefix_len = _scale_seq_len(prefix_schedule)
        normal_len = _scale_seq_len(normal_schedule)
        layout = self._normal_prefix_layout(prefix_schedule, normal_schedule, rgb_prefix_emb.device)
        if rgb_prefix_emb.shape[1] != prefix_len:
            raise ValueError(f"RGB prefix length {rgb_prefix_emb.shape[1]} does not match schedule length {prefix_len}.")
        if normal_input_emb.shape[1] != normal_len:
            raise ValueError(f"Normal input length {normal_input_emb.shape[1]} does not match schedule length {normal_len}.")

        if self.normal_token_layout == "prefix":
            x = torch.cat((rgb_prefix_emb, normal_input_emb), dim=1)
        else:
            pieces: list[torch.Tensor] = []
            for (rgb_start, rgb_end), (normal_start, normal_end) in zip(
                _stage_offsets(prefix_schedule),
                _stage_offsets(normal_schedule),
                strict=True,
            ):
                pieces.append(rgb_prefix_emb[:, rgb_start:rgb_end])
                pieces.append(normal_input_emb[:, normal_start:normal_end])
            x = torch.cat(pieces, dim=1)
        stage_ids = layout["stage_ids"]
        normal_token_indices = layout["normal_token_indices"]
        segment_key_indices = layout["segment_key_indices"]

        l_end = x.shape[1]
        need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end
        if self.normal_use_flex_attn or self.normal_use_segmented_flash_attn:
            if need_to_pad:
                raise NotImplementedError("Normal memory-efficient attention path does not support padded sequence lengths yet.")
            attn_bias = x.new_empty(0)
        else:
            attn_bias = x.new_full((1, 1, l_end, l_end), -torch.inf)
            if segment_key_indices is None:
                for query_start, query_end, key_end in layout["segments"]:
                    attn_bias[:, :, query_start:query_end, :key_end] = 0
            else:
                for (query_start, query_end, _), key_indices in zip(layout["segments"], segment_key_indices, strict=True):
                    attn_bias[:, :, query_start:query_end, key_indices] = 0

        if need_to_pad:
            attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
            attn_bias[0, 0, l_end:, 0] = 0
            x = F.pad(x, (0, 0, 0, need_to_pad))

        rope_schedule = self._ensure_sequence_rope_cache(
            layout["sequence_schedule"],
            layout["sequence_source_indices"],
            prefix_schedule,
            x.shape[1],
        )
        return x, attn_bias, rope_schedule, stage_ids, normal_token_indices, normal_len, layout["segments"], segment_key_indices

    def _run_prefix_blocks(
        self,
        *,
        x: torch.Tensor,
        cond_BD_or_gss: torch.Tensor,
        ca_kv: tuple[torch.Tensor, torch.Tensor, int],
        attn_bias: torch.Tensor,
        rope_schedule: list[tuple[int, int, int]],
        stage_ids: torch.Tensor,
        attn_fn: Any | None = None,
    ) -> torch.Tensor:
        checkpointing_full_block = self.checkpointing == "full-block" and self.training
        checkpointing_full_block_skip_interval = int(getattr(self, "checkpointing_full_block_skip_interval", 0))
        attn_bias_or_mask = None if attn_fn is not None else attn_bias
        if self.num_block_chunks == 1:
            for block_index, block in enumerate(self.blocks):
                if (self.add_lvl_embeding_only_first_block and block_index == 0) or not self.add_lvl_embeding_only_first_block:
                    x = self._add_prefix_level_embedding(x, stage_ids)
                skip_checkpoint = (
                    checkpointing_full_block_skip_interval > 0
                    and (block_index + 1) % checkpointing_full_block_skip_interval == 0
                )
                should_checkpoint = (
                    checkpointing_full_block
                    and not skip_checkpoint
                )
                if should_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(
                        block,
                        x,
                        cond_BD_or_gss,
                        ca_kv,
                        attn_bias_or_mask,
                        attn_fn,
                        rope_schedule,
                        self.rope2d_freqs_grid,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x=x,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv,
                        attn_bias_or_two_vector=attn_bias_or_mask,
                        attn_fn=attn_fn,
                        scale_schedule=rope_schedule,
                        rope2d_freqs_grid=self.rope2d_freqs_grid,
                    )
            return x

        for chunk_index, chunk in enumerate(self.block_chunks):
            if (self.add_lvl_embeding_only_first_block and chunk_index == 0) or not self.add_lvl_embeding_only_first_block:
                x = self._add_prefix_level_embedding(x, stage_ids)
            x = chunk(
                x=x,
                cond_BD=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=attn_bias_or_mask,
                attn_fn=attn_fn,
                scale_schedule=rope_schedule,
                checkpointing_full_block=checkpointing_full_block,
                checkpointing_full_block_skip_interval=checkpointing_full_block_skip_interval,
                rope2d_freqs_grid=self.rope2d_freqs_grid,
            )
        return x

    def _prefix_attention_for_sequence(
        self,
        *,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        segments: tuple[tuple[int, int, int], ...],
        segment_key_indices: tuple[torch.Tensor, ...] | None,
        stage_ids: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, Any | None]:
        attn_fn = self._normal_prefix_segmented_flash_attn(segments=segments, segment_key_indices=segment_key_indices)
        if attn_fn is None:
            attn_fn = self._normal_prefix_flex_attn(
                stage_ids=stage_ids,
                batch_size=batch_size,
                segments=segments,
                segment_key_indices=segment_key_indices,
            )
        if attn_fn is not None:
            return attn_bias, attn_fn
        return attn_bias.type_as(x).to(x.device), None

    def forward(
        self,
        rgb_prefix_blc: torch.Tensor,
        normal_x_blc_wo_prefix: torch.Tensor,
        scale_schedule: list[tuple[int, int, int]],
        **_: Any,
    ) -> torch.Tensor:
        rgb_prefix_blc = rgb_prefix_blc.float()
        normal_x_blc_wo_prefix = normal_x_blc_wo_prefix.float()
        batch_size = rgb_prefix_blc.shape[0]

        with torch.amp.autocast("cuda", enabled=False):
            sos, cond_BD, cond_BD_or_gss, ca_kv = self._task_condition(batch_size)
            rgb_prefix_emb = self._embed_rgb_prefix(rgb_prefix_blc)
            normal_input_emb = self._embed_normal_inputs(normal_x_blc_wo_prefix, sos)
            x, attn_bias, rope_schedule, stage_ids, normal_token_indices, normal_len, segments, segment_key_indices = self._prepare_prefix_sequence(
                rgb_prefix_emb=rgb_prefix_emb,
                normal_input_emb=normal_input_emb,
                prefix_schedule=scale_schedule,
                normal_schedule=scale_schedule,
            )
            attn_bias, attn_fn = self._prefix_attention_for_sequence(
                x=x,
                attn_bias=attn_bias,
                segments=segments,
                segment_key_indices=segment_key_indices,
                stage_ids=stage_ids,
                batch_size=batch_size,
            )
            if self.normal_bf16_activations:
                x = x.to(torch.bfloat16)

        x = self._run_prefix_blocks(
            x=x,
            cond_BD_or_gss=cond_BD_or_gss,
            ca_kv=ca_kv,
            attn_bias=attn_bias,
            rope_schedule=rope_schedule,
            stage_ids=stage_ids,
            attn_fn=attn_fn,
        )
        hidden = x.index_select(1, normal_token_indices)
        return self.get_logits(hidden, cond_BD)

    def _iter_inference_blocks(self):
        if self.num_block_chunks == 1:
            for block_index, block in enumerate(self.blocks):
                yield block_index, block
            return
        block_index = 0
        for chunk in self.block_chunks:
            for block in chunk.module:
                yield block_index, block
                block_index += 1

    def _set_kv_caching(self, enabled: bool) -> None:
        for _, block in self._iter_inference_blocks():
            if isinstance(block, CrossAttnBlock):
                block.sa.kv_caching(enabled)
            else:
                block.attn.kv_caching(enabled)

    def _run_incremental_blocks(
        self,
        *,
        x: torch.Tensor,
        stage_ids: torch.Tensor,
        cond_BD_or_gss: torch.Tensor,
        ca_kv: tuple[torch.Tensor, torch.Tensor, int],
        rope_schedule: list[tuple[int, int, int]],
        scale_ind: int,
    ) -> torch.Tensor:
        for block_index, block in self._iter_inference_blocks():
            if (self.add_lvl_embeding_only_first_block and block_index == 0) or not self.add_lvl_embeding_only_first_block:
                x = self._add_prefix_level_embedding(x, stage_ids)
            if isinstance(block, CrossAttnBlock):
                x = block(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=rope_schedule,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                    scale_ind=scale_ind,
                )
            else:
                x = block(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=rope_schedule,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                )
        return x

    def autoregressive_infer_prefix_fast(
        self,
        *,
        vae: torch.nn.Module,
        rgb_prefix_blc: torch.Tensor,
        scale_schedule: list[tuple[int, int, int]],
        top_k: int = 1,
        top_p: float = 0.0,
        tau: float = 1.0,
        vae_type: int = 0,
    ) -> torch.Tensor:
        del vae_type
        batch_size = rgb_prefix_blc.shape[0]
        vae_scale_schedule = build_vae_scale_schedule(scale_schedule, self.apply_spatial_patchify)
        sos, cond_BD, cond_BD_or_gss, ca_kv = self._task_condition(batch_size)
        rgb_prefix_emb = self._embed_rgb_prefix(rgb_prefix_blc.float())
        prefix_stage_ids = _stage_ids_for_schedule(scale_schedule, rgb_prefix_emb.device)
        prefix_len = _scale_seq_len(scale_schedule)
        full_rope_schedule = self._ensure_sequence_rope_cache(
            list(scale_schedule) + list(scale_schedule),
            list(range(len(scale_schedule))) + list(range(len(scale_schedule))),
            scale_schedule,
            2 * prefix_len,
        )

        self._set_kv_caching(True)
        try:
            _ = self._run_incremental_blocks(
                x=rgb_prefix_emb,
                stage_ids=prefix_stage_ids,
                cond_BD_or_gss=cond_BD_or_gss,
                ca_kv=ca_kv,
                rope_schedule=full_rope_schedule,
                scale_ind=0,
            )

            summed_codes: torch.Tensor | int = 0
            for scale_index, scale_item in enumerate(scale_schedule):
                stage_len = int(np.array(scale_item).prod())
                if scale_index == 0:
                    current_emb = sos.unsqueeze(1).expand(batch_size, self.first_l, -1) + self.pos_start.expand(
                        batch_size, self.first_l, -1
                    )
                    current_emb = current_emb + self.normal_modality_embed
                else:
                    next_input = F.interpolate(
                        summed_codes,
                        size=vae_scale_schedule[scale_index],
                        mode=vae.quantizer.z_interplote_down,
                    ).contiguous()
                    next_input = _flatten_feature_tokens(next_input, apply_spatial_patchify=self.apply_spatial_patchify)
                    current_emb = self.word_embed(self.norm0_ve(next_input.float())) + self.normal_modality_embed

                normal_stage_ids = torch.full(
                    (stage_len,),
                    fill_value=scale_index,
                    dtype=torch.long,
                    device=current_emb.device,
                )
                x = self._run_incremental_blocks(
                    x=current_emb,
                    stage_ids=normal_stage_ids,
                    cond_BD_or_gss=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    rope_schedule=full_rope_schedule,
                    scale_ind=len(scale_schedule) + scale_index,
                )
                logits = self.get_logits(x, cond_BD).mul(1.0 / max(float(tau), 1e-6))

                if self.use_bit_label:
                    stage_bits = logits.reshape(batch_size, stage_len, -1, 2)
                    sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        stage_bits.reshape(batch_size, -1, 2),
                        rng=None,
                        top_k=top_k,
                        top_p=top_p,
                        num_samples=1,
                    )[:, :, 0]
                    stage_bits = sampled.reshape(batch_size, stage_len, -1).float()
                    stage_bits = stage_bits.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2], -1)
                    if self.apply_spatial_patchify:
                        assert scale_item[0] == 1
                        stage_bits = stage_bits.squeeze(1).permute(0, 3, 1, 2)
                        stage_bits = torch.nn.functional.pixel_shuffle(stage_bits, 2)
                        stage_bits = stage_bits.permute(0, 2, 3, 1).unsqueeze(1)
                    codes = vae.quantizer.lfq.indices_to_codes(stage_bits, label_type="bit_label")
                else:
                    stage_indices = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        logits,
                        rng=None,
                        top_k=top_k,
                        top_p=top_p,
                        num_samples=1,
                    )[:, :, 0]
                    stage_indices = stage_indices.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2])
                    codes = vae.quantizer.lfq.indices_to_codes(stage_indices, label_type="int_label")

                if scale_index == len(scale_schedule) - 1:
                    summed_codes = summed_codes + codes
                else:
                    summed_codes = summed_codes + F.interpolate(
                        codes,
                        size=vae_scale_schedule[-1],
                        mode=vae.quantizer.z_interplote_up,
                    ).contiguous()
        finally:
            self._set_kv_caching(False)

        return vae.decode(summed_codes.squeeze(-3)).clamp(-1.0, 1.0)

    @torch.no_grad()
    def autoregressive_infer_prefix(
        self,
        *,
        vae: torch.nn.Module,
        rgb_prefix_blc: torch.Tensor,
        scale_schedule: list[tuple[int, int, int]],
        top_k: int = 1,
        top_p: float = 0.0,
        tau: float = 1.0,
        vae_type: int = 0,
    ) -> torch.Tensor:
        disable_fast_ar = os.environ.get("INFINITY_NORMAL_DISABLE_KV_FAST", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        if not disable_fast_ar and self.normal_token_layout == "prefix":
            return self.autoregressive_infer_prefix_fast(
                vae=vae,
                rgb_prefix_blc=rgb_prefix_blc,
                scale_schedule=scale_schedule,
                top_k=top_k,
                top_p=top_p,
                tau=tau,
                vae_type=vae_type,
            )
        del vae_type
        batch_size = rgb_prefix_blc.shape[0]
        vae_scale_schedule = build_vae_scale_schedule(scale_schedule, self.apply_spatial_patchify)
        sos, cond_BD, cond_BD_or_gss, ca_kv = self._task_condition(batch_size)
        rgb_prefix_emb = self._embed_rgb_prefix(rgb_prefix_blc.float())

        normal_embeds: list[torch.Tensor] = []
        summed_codes: torch.Tensor | int = 0
        for scale_index, scale_item in enumerate(scale_schedule):
            stage_len = int(np.array(scale_item).prod())
            if scale_index == 0:
                current_emb = sos.unsqueeze(1).expand(batch_size, self.first_l, -1) + self.pos_start.expand(
                    batch_size, self.first_l, -1
                )
                current_emb = current_emb + self.normal_modality_embed
            else:
                next_input = F.interpolate(
                    summed_codes,
                    size=vae_scale_schedule[scale_index],
                    mode=vae.quantizer.z_interplote_down,
                ).contiguous()
                next_input = _flatten_feature_tokens(next_input, apply_spatial_patchify=self.apply_spatial_patchify)
                current_emb = self.word_embed(self.norm0_ve(next_input.float())) + self.normal_modality_embed
            normal_embeds.append(current_emb)

            normal_input_emb = torch.cat(normal_embeds, dim=1)
            normal_schedule = scale_schedule[: scale_index + 1]
            current_prefix_schedule = scale_schedule if self.normal_token_layout == "prefix" else normal_schedule
            current_prefix_len = _scale_seq_len(current_prefix_schedule)
            x, attn_bias, rope_schedule, stage_ids, normal_token_indices, normal_len, segments, segment_key_indices = self._prepare_prefix_sequence(
                rgb_prefix_emb=rgb_prefix_emb[:, :current_prefix_len],
                normal_input_emb=normal_input_emb,
                prefix_schedule=current_prefix_schedule,
                normal_schedule=normal_schedule,
            )
            attn_bias, attn_fn = self._prefix_attention_for_sequence(
                x=x,
                attn_bias=attn_bias,
                segments=segments,
                segment_key_indices=segment_key_indices,
                stage_ids=stage_ids,
                batch_size=batch_size,
            )
            x = self._run_prefix_blocks(
                x=x,
                cond_BD_or_gss=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias=attn_bias,
                rope_schedule=rope_schedule,
                stage_ids=stage_ids,
                attn_fn=attn_fn,
            )
            logits = self.get_logits(x.index_select(1, normal_token_indices[-stage_len:]), cond_BD)
            logits = logits.mul(1.0 / max(float(tau), 1e-6))

            if self.use_bit_label:
                stage_bits = logits.reshape(batch_size, stage_len, -1, 2)
                sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    stage_bits.reshape(batch_size, -1, 2),
                    rng=None,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                stage_bits = sampled.reshape(batch_size, stage_len, -1).float()
                stage_bits = stage_bits.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2], -1)
                if self.apply_spatial_patchify:
                    assert scale_item[0] == 1
                    stage_bits = stage_bits.squeeze(1).permute(0, 3, 1, 2)
                    stage_bits = torch.nn.functional.pixel_shuffle(stage_bits, 2)
                    stage_bits = stage_bits.permute(0, 2, 3, 1).unsqueeze(1)
                codes = vae.quantizer.lfq.indices_to_codes(stage_bits, label_type="bit_label")
            else:
                stage_indices = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits,
                    rng=None,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                stage_indices = stage_indices.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2])
                codes = vae.quantizer.lfq.indices_to_codes(stage_indices, label_type="int_label")

            if scale_index == len(scale_schedule) - 1:
                summed_codes = summed_codes + codes
            else:
                summed_codes = summed_codes + F.interpolate(
                    codes,
                    size=vae_scale_schedule[-1],
                    mode=vae.quantizer.z_interplote_up,
                ).contiguous()

        return vae.decode(summed_codes.squeeze(-3)).clamp(-1.0, 1.0)

    @torch.no_grad()
    def autoregressive_infer_prefix_with_uncertainty(
        self,
        *,
        vae: torch.nn.Module,
        rgb_prefix_blc: torch.Tensor,
        scale_schedule: list[tuple[int, int, int]],
        top_k: int = 1,
        top_p: float = 0.0,
        tau: float = 1.0,
        vae_type: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del vae_type
        batch_size = rgb_prefix_blc.shape[0]
        vae_scale_schedule = build_vae_scale_schedule(scale_schedule, self.apply_spatial_patchify)
        sos, cond_BD, cond_BD_or_gss, ca_kv = self._task_condition(batch_size)
        rgb_prefix_emb = self._embed_rgb_prefix(rgb_prefix_blc.float())

        normal_embeds: list[torch.Tensor] = []
        uncertainty_maps: list[torch.Tensor] = []
        contribution_weights: list[torch.Tensor] = []
        final_hw = tuple(int(value) for value in vae_scale_schedule[-1][-2:])
        summed_codes: torch.Tensor | int = 0
        for scale_index, scale_item in enumerate(scale_schedule):
            stage_len = int(np.array(scale_item).prod())
            if scale_index == 0:
                current_emb = sos.unsqueeze(1).expand(batch_size, self.first_l, -1) + self.pos_start.expand(
                    batch_size, self.first_l, -1
                )
                current_emb = current_emb + self.normal_modality_embed
            else:
                next_input = F.interpolate(
                    summed_codes,
                    size=vae_scale_schedule[scale_index],
                    mode=vae.quantizer.z_interplote_down,
                ).contiguous()
                next_input = _flatten_feature_tokens(next_input, apply_spatial_patchify=self.apply_spatial_patchify)
                current_emb = self.word_embed(self.norm0_ve(next_input.float())) + self.normal_modality_embed
            normal_embeds.append(current_emb)

            normal_input_emb = torch.cat(normal_embeds, dim=1)
            normal_schedule = scale_schedule[: scale_index + 1]
            current_prefix_schedule = scale_schedule if self.normal_token_layout == "prefix" else normal_schedule
            current_prefix_len = _scale_seq_len(current_prefix_schedule)
            x, attn_bias, rope_schedule, stage_ids, normal_token_indices, normal_len, segments, segment_key_indices = self._prepare_prefix_sequence(
                rgb_prefix_emb=rgb_prefix_emb[:, :current_prefix_len],
                normal_input_emb=normal_input_emb,
                prefix_schedule=current_prefix_schedule,
                normal_schedule=normal_schedule,
            )
            attn_bias, attn_fn = self._prefix_attention_for_sequence(
                x=x,
                attn_bias=attn_bias,
                segments=segments,
                segment_key_indices=segment_key_indices,
                stage_ids=stage_ids,
                batch_size=batch_size,
            )
            x = self._run_prefix_blocks(
                x=x,
                cond_BD_or_gss=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias=attn_bias,
                rope_schedule=rope_schedule,
                stage_ids=stage_ids,
                attn_fn=attn_fn,
            )
            logits = self.get_logits(x.index_select(1, normal_token_indices[-stage_len:]), cond_BD)
            logits = logits.mul(1.0 / max(float(tau), 1e-6))

            if self.use_bit_label:
                stage_logits = logits.reshape(batch_size, stage_len, -1, 2)
                probs = stage_logits.float().softmax(dim=-1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / math.log(2.0)
                stage_uncertainty = entropy.mean(dim=-1)
                sampled = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    stage_logits.reshape(batch_size, -1, 2),
                    rng=None,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                stage_bits = sampled.reshape(batch_size, stage_len, -1).float()
                stage_bits = stage_bits.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2], -1)
                if self.apply_spatial_patchify:
                    assert scale_item[0] == 1
                    stage_bits = stage_bits.squeeze(1).permute(0, 3, 1, 2)
                    stage_bits = torch.nn.functional.pixel_shuffle(stage_bits, 2)
                    stage_bits = stage_bits.permute(0, 2, 3, 1).unsqueeze(1)
                codes = vae.quantizer.lfq.indices_to_codes(stage_bits, label_type="bit_label")
            else:
                probs = logits.float().softmax(dim=-1)
                stage_uncertainty = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / math.log(float(logits.shape[-1]))
                stage_indices = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits,
                    rng=None,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                )[:, :, 0]
                stage_indices = stage_indices.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2])
                codes = vae.quantizer.lfq.indices_to_codes(stage_indices, label_type="int_label")

            stage_uncertainty = stage_uncertainty.reshape(batch_size, scale_item[0], scale_item[1], scale_item[2])
            stage_uncertainty = stage_uncertainty.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            uncertainty_maps.append(
                F.interpolate(stage_uncertainty, size=final_hw, mode="bilinear", align_corners=False)
            )

            if scale_index == len(scale_schedule) - 1:
                stage_contribution = codes
            else:
                stage_contribution = F.interpolate(
                    codes,
                    size=vae_scale_schedule[-1],
                    mode=vae.quantizer.z_interplote_up,
                ).contiguous()
            if stage_contribution.dim() == 5:
                contribution_weight = stage_contribution.float().pow(2).mean(dim=1).mean(dim=1, keepdim=True).sqrt()
            elif stage_contribution.dim() == 4:
                contribution_weight = stage_contribution.float().pow(2).mean(dim=1, keepdim=True).sqrt()
            else:
                raise ValueError(f"Unexpected code contribution shape: {tuple(stage_contribution.shape)}")
            contribution_weights.append(
                F.interpolate(contribution_weight, size=final_hw, mode="bilinear", align_corners=False)
            )

            if scale_index == len(scale_schedule) - 1:
                summed_codes = summed_codes + codes
            else:
                summed_codes = summed_codes + stage_contribution

        normal = vae.decode(summed_codes.squeeze(-3)).clamp(-1.0, 1.0)
        stacked_uncertainty = torch.stack(uncertainty_maps, dim=0)
        mean_uncertainty = stacked_uncertainty.mean(dim=0).clamp(0.0, 1.0)
        fine_count = min(3, len(uncertainty_maps))
        fine_mean_uncertainty = torch.stack(uncertainty_maps[-fine_count:], dim=0).mean(dim=0).clamp(0.0, 1.0)
        last_uncertainty = uncertainty_maps[-1].clamp(0.0, 1.0)
        stacked_weights = torch.stack(contribution_weights, dim=0).clamp_min(0.0)
        weighted_uncertainty = (stacked_uncertainty * stacked_weights).sum(dim=0)
        weighted_uncertainty = weighted_uncertainty / stacked_weights.sum(dim=0).clamp_min(1e-8)
        uncertainty_by_name = {
            "mean": mean_uncertainty,
            "fine_mean": fine_mean_uncertainty,
            "last": last_uncertainty,
            "latent_weighted": weighted_uncertainty.clamp(0.0, 1.0),
        }
        uncertainty_by_name = {
            name: F.interpolate(value, size=normal.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            for name, value in uncertainty_by_name.items()
        }
        return normal, uncertainty_by_name


def build_infinity_normal_model(
    *,
    model_name: str,
    vae_local: torch.nn.Module,
    pn: str,
    batch_size: int,
    use_bit_label: bool,
    add_lvl_embeding_only_first_block: int,
    rope2d_each_sa_layer: int,
    rope2d_normalized_by_hw: int,
    apply_spatial_patchify: bool,
    device: torch.device,
    checkpointing: str | None = "full-block",
    normal_token_layout: str = "prefix",
    normal_use_flex_attn: bool = False,
    normal_use_segmented_flash_attn: bool = False,
    normal_bf16_activations: bool = False,
    fused_mlp: bool = False,
    fused_norm: bool = False,
    text_channels: int = 2048,
    text_maxlen: int = 512,
    task_condition_len: int = 16,
) -> torch.nn.Module:
    _, model_config = _model_config(model_name)
    model = InfinityNormalPrefixModel(
        vae_local=vae_local,
        text_channels=text_channels,
        text_maxlen=text_maxlen,
        task_condition_len=task_condition_len,
        normal_token_layout=normal_token_layout,
        normal_use_flex_attn=normal_use_flex_attn,
        normal_use_segmented_flash_attn=normal_use_segmented_flash_attn,
        normal_bf16_activations=normal_bf16_activations,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        shared_aln=True,
        cond_drop_rate=0.1,
        nm0=False,
        tau=1,
        cos_attn=True,
        head_depth=1,
        fused_mlp=fused_mlp,
        fused_norm=fused_norm,
        checkpointing=checkpointing,
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
        bf16_activations=normal_bf16_activations,
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
            shard_state = _extract_state_dict(torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True))
        _copy_matching_state_dict_tensors(model, shard_state, loaded_keys=loaded_keys, skipped_keys=skipped_keys)

    if hasattr(model, "initialize_missing_prefix_parameters"):
        model.initialize_missing_prefix_parameters(loaded_keys)
    missing = [key for key in model.state_dict().keys() if key not in loaded_keys]
    return missing, skipped_keys


def load_infinity_state_dict(model: torch.nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    checkpoint_path_str = str(checkpoint_path)
    if Path(checkpoint_path_str).is_dir():
        return _load_directory_state_dict(model, Path(checkpoint_path_str))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = _extract_state_dict(checkpoint)
    loaded_keys: set[str] = set()
    skipped_keys: list[str] = []
    _copy_matching_state_dict_tensors(model, state_dict, loaded_keys=loaded_keys, skipped_keys=skipped_keys)
    if hasattr(model, "initialize_missing_prefix_parameters"):
        model.initialize_missing_prefix_parameters(loaded_keys)
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
                stage_noise_strength = np.random.randint(0, int(100 * noise_apply_strength) + 1) * 0.01
                mask = torch.rand_like(bit_indices.float()) < stage_noise_strength
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
                    mode=vae.quantizer.z_interplote_down,
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
