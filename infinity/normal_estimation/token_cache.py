from __future__ import annotations

import hashlib
from typing import Any


def token_cache_sample_key(metadata: dict[str, Any], signature: str) -> str:
    identity_fields = ["dataset", "partition", "index", "image_path", "normal_path", "target_size"]
    if metadata.get("manifest_dir"):
        identity_fields.append("manifest_dir")
    source = "|".join(str(metadata.get(key, "")) for key in identity_fields)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:24]
    return f"{signature}_{digest}.pt"
