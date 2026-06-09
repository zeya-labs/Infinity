from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "atomic_torch_save": ".checkpoints",
    "resolve_checkpoint_resume_path": ".checkpoints",
    "HypersimNormalDataset": ".data",
    "DSINEEvalNormalDataset": ".data",
    "NYUv2ParquetNormalDataset": ".data",
    "VKITTI2NormalDataset": ".data",
    "collate_normal_estimation_batch": ".data",
    "load_dsine_eval_normal_sample_from_metadata": ".data",
    "load_hypersim_normal_sample_from_metadata": ".data",
    "load_normal_sample_from_metadata": ".data",
    "load_vkitti2_normal_sample_from_metadata": ".data",
    "GroupedTargetSizeBatchSampler": ".sampling",
    "RepeatDataset": ".sampling",
    "build_normal_dataloader": ".sampling",
    "build_normal_train_dataset": ".sampling",
    "dataset_metadata_at": ".sampling",
    "parse_train_dataset_names": ".sampling",
    "parse_train_dataset_weights": ".sampling",
    "build_bsq_vae": ".modeling",
    "build_infinity_normal_model": ".modeling",
    "build_multiscale_var_inputs": ".modeling",
    "build_prefix_tokens_from_image": ".modeling",
    "compute_normal_metrics": ".modeling",
    "decode_logits_to_normal": ".modeling",
    "load_infinity_state_dict": ".modeling",
    "normalize_normals": ".modeling",
    "normals_to_vis": ".modeling",
    "resolve_scale_schedule_from_hw": ".modeling",
    "token_cache_sample_key": ".token_cache",
    "require_positive_steps_per_epoch": ".training",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
