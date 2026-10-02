from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_balanced_accuracy(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    n_boot: int = 2000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    m1 = np.where(yt == "M1")[0]
    m2 = np.where(yt == "M2")[0]
    if len(m1) == 0 or len(m2) == 0:
        return float("nan"), float("nan"), float("nan")
    tnr = np.mean(yp[m1] == "M1")
    tpr = np.mean(yp[m2] == "M2")
    point = 0.5 * (tnr + tpr)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b1 = rng.choice(m1, size=len(m1), replace=True)
        b2 = rng.choice(m2, size=len(m2), replace=True)
        vals[i] = 0.5 * (np.mean(yp[b1] == "M1") + np.mean(yp[b2] == "M2"))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def iqr(x: np.ndarray) -> tuple[float, float, float]:
    q1, med, q3 = np.quantile(np.asarray(x, dtype=float), [0.25, 0.5, 0.75])
    return float(q1), float(med), float(q3)
