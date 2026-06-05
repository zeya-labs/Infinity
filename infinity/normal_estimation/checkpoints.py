from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def resolve_checkpoint_resume_path(
    *,
    output_dir: Path,
    resume_arg: str,
    auto_checkpoint_names: Sequence[str],
    prefer_step_checkpoint_for_last: bool = False,
) -> Path | None:
    checkpoint_dir = output_dir / "checkpoints"
    if resume_arg:
        requested = Path(resume_arg)
        if prefer_step_checkpoint_for_last and requested.name == "last.pth":
            step_checkpoint = requested.with_name("last_step.pth")
            if step_checkpoint.is_file():
                return step_checkpoint
        if requested.is_file():
            return requested
        raise FileNotFoundError(f"--resume checkpoint not found: {requested}")

    for checkpoint_name in auto_checkpoint_names:
        candidate = checkpoint_dir / checkpoint_name
        if candidate.is_file():
            return candidate
    return None
