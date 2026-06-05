from __future__ import annotations


def require_positive_steps_per_epoch(steps_per_epoch: int, *, context: str) -> int:
    if steps_per_epoch <= 0:
        raise RuntimeError(
            f"{context} has no training batches. "
            "Check dataset size, batch size, world size, drop_last, and target-size grouping."
        )
    return steps_per_epoch
