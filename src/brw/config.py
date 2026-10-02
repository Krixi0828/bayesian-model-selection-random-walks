from __future__ import annotations

import json
from pathlib import Path


def load_profile(profile: str, root: str | Path) -> dict:
    root = Path(root)
    obj = json.loads((root / "configs" / "profiles.json").read_text(encoding="utf-8"))
    if profile not in obj:
        raise KeyError(f"Unknown profile {profile!r}; choose from {sorted(obj)}")
    cfg = obj[profile]
    cfg["profile_name"] = profile
    return cfg
