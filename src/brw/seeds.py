from __future__ import annotations

import hashlib


def stable_seed(base: int, *parts: object) -> int:
    """Derive a deterministic uint32 seed from a base seed and labels."""
    payload = "|".join([str(base), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little", signed=False)
