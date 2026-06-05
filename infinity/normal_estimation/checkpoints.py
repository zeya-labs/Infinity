from __future__ import annotations

from pathlib import Path
from typing import Sequence

from infinity.utils.torch_io import atomic_torch_save


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
