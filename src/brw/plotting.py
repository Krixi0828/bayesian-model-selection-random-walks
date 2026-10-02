from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finish(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def add_identity(ax, xlim=None, ylim=None):
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    lo = max(xlim[0], ylim[0])
    hi = min(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
