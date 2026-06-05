from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
