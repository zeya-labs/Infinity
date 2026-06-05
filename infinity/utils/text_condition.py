from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F


TextCondition = tuple[torch.Tensor, list[int], torch.Tensor, int]


def build_text_condition(
    text_tokenizer,
    text_encoder,
    captions: Sequence[str],
    *,
    max_length: int | None = None,
    device: torch.device | str | None = "cuda",
    non_blocking: bool = True,
) -> TextCondition:
    """Encode captions into Infinity's compact text-conditioning tuple."""

    if max_length is None:
        max_length = text_tokenizer.model_max_length
    tokens = text_tokenizer(
        text=list(captions),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids
    mask = tokens.attention_mask
    if device is not None:
        input_ids = input_ids.to(device, non_blocking=non_blocking)
        mask = mask.to(device, non_blocking=non_blocking)

    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen_k = max(lens)
    kv_compact = torch.cat(
        [feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0), strict=True)],
        dim=0,
    )
    return kv_compact, lens, cu_seqlens_k, max_seqlen_k
