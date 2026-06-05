from .modeling import (
    build_bsq_vae,
    build_infinity_normal_model,
    build_multiscale_var_inputs,
    build_prefix_tokens_from_image,
    compute_normal_metrics,
    decode_logits_to_normal,
    load_infinity_state_dict,
    normalize_normals,
    normals_to_vis,
    resolve_scale_schedule_from_hw,
)

_DATA_EXPORTS = {
    "HypersimNormalDataset",
    "NYUv2ParquetNormalDataset",
    "collate_normal_estimation_batch",
    "load_hypersim_normal_sample_from_metadata",
}


def __getattr__(name: str):
    if name in _DATA_EXPORTS:
        from .data import (
            HypersimNormalDataset,
            NYUv2ParquetNormalDataset,
            collate_normal_estimation_batch,
            load_hypersim_normal_sample_from_metadata,
        )

        exports = {
            "HypersimNormalDataset": HypersimNormalDataset,
            "NYUv2ParquetNormalDataset": NYUv2ParquetNormalDataset,
            "collate_normal_estimation_batch": collate_normal_estimation_batch,
            "load_hypersim_normal_sample_from_metadata": load_hypersim_normal_sample_from_metadata,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HypersimNormalDataset",
    "NYUv2ParquetNormalDataset",
    "build_bsq_vae",
    "build_infinity_normal_model",
    "build_multiscale_var_inputs",
    "build_prefix_tokens_from_image",
    "collate_normal_estimation_batch",
    "load_hypersim_normal_sample_from_metadata",
    "compute_normal_metrics",
    "decode_logits_to_normal",
    "load_infinity_state_dict",
    "normalize_normals",
    "normals_to_vis",
    "resolve_scale_schedule_from_hw",
]
