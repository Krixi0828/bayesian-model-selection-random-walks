from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
import matplotlib


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(rows: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _git_commit(root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def write_environment(path: str | Path, profile: str, base_seed: int, extra: dict | None = None) -> None:
    path = Path(path)
    root = path.resolve().parents[2] if len(path.resolve().parents) >= 3 else Path.cwd()
    obj = {
        "profile": profile,
        "base_seed": base_seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "timestamp_unix": time.time(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(root),
    }
    if extra:
        obj.update(extra)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
