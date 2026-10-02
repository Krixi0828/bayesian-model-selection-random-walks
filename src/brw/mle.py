from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .likelihoods import log_binom_pmf


def mle_m1(k: np.ndarray, n_steps: int) -> tuple[float, float]:
    k = np.asarray(k, dtype=np.int64)
    p = float(np.clip(k.sum() / (len(k) * n_steps), 1e-8, 1.0 - 1e-8))
    ll = float(np.sum(log_binom_pmf(k, n_steps, p)))
    return p, ll


def _mix_loglik(k: np.ndarray, n_steps: int, p1: float, p2: float, pi: float) -> float:
    l1 = np.log(pi) + log_binom_pmf(k, n_steps, p1)
    l2 = np.log1p(-pi) + log_binom_pmf(k, n_steps, p2)
    return float(np.sum(np.logaddexp(l1, l2)))


def mle_m2_em(
    k: np.ndarray,
    n_steps: int,
    n_starts: int = 8,
    max_iter: int = 500,
    tol: float = 1e-9,
    seed: int = 0,
) -> tuple[dict[str, float], float]:
    k = np.asarray(k, dtype=np.int64)
    x = k / n_steps
    rng = np.random.default_rng(seed)
    quantiles = np.quantile(x, [0.2, 0.35, 0.65, 0.8])
    starts: list[tuple[float, float, float]] = [
        (float(quantiles[0]), float(quantiles[3]), 0.5),
        (float(quantiles[1]), float(quantiles[2]), 0.5),
        (max(0.01, float(x.mean() - 0.10)), min(0.99, float(x.mean() + 0.10)), 0.5),
        (max(0.01, float(x.mean() - 0.03)), min(0.99, float(x.mean() + 0.03)), 0.5),
    ]
    while len(starts) < n_starts:
        center = float(np.clip(x.mean() + rng.normal(0.0, 0.03), 0.05, 0.95))
        gap = float(rng.uniform(0.02, 0.25))
        starts.append((max(0.001, center - gap / 2), min(0.999, center + gap / 2), float(rng.uniform(0.2, 0.8))))

    best_ll = -np.inf
    best = None
    eps = 1e-8
    for p1, p2, pi in starts[:n_starts]:
        p1 = float(np.clip(p1, eps, 1 - eps))
        p2 = float(np.clip(p2, eps, 1 - eps))
        pi = float(np.clip(pi, eps, 1 - eps))
        prev = -np.inf
        for _ in range(max_iter):
            l1 = np.log(pi) + log_binom_pmf(k, n_steps, p1)
            l2 = np.log1p(-pi) + log_binom_pmf(k, n_steps, p2)
            denom = np.logaddexp(l1, l2)
            r = np.exp(l1 - denom)
            sr = float(r.sum())
            s1 = float(np.dot(r, k))
            s0 = float(np.dot(1.0 - r, k))
            pi = float(np.clip(sr / len(k), eps, 1 - eps))
            p1 = float(np.clip(s1 / (n_steps * max(sr, eps)), eps, 1 - eps))
            p2 = float(np.clip(s0 / (n_steps * max(len(k) - sr, eps)), eps, 1 - eps))
            ll = float(denom.sum())
            if abs(ll - prev) <= tol * (1.0 + abs(prev)):
                break
            prev = ll
        ll = _mix_loglik(k, n_steps, p1, p2, pi)
        if ll > best_ll:
            if p1 > p2:
                p1, p2, pi = p2, p1, 1.0 - pi
            best_ll = ll
            best = {"p1": p1, "p2": p2, "pi": pi}
    assert best is not None
    return best, float(best_ll)


def model_selection_scores(k: np.ndarray, n_steps: int, seed: int = 0) -> dict[str, float | str]:
    _, ll1 = mle_m1(k, n_steps)
    _, ll2 = mle_m2_em(k, n_steps, seed=seed)
    n = len(k)
    k1, k2 = 1, 3
    aic1 = 2 * k1 - 2 * ll1
    aic2 = 2 * k2 - 2 * ll2
    bic1 = k1 * np.log(n) - 2 * ll1
    bic2 = k2 * np.log(n) - 2 * ll2
    return {
        "loglik_M1": ll1,
        "loglik_M2": ll2,
        "ml_selected": "M2" if ll2 > ll1 + 1e-10 else "M1",
        "aic_M1": float(aic1),
        "aic_M2": float(aic2),
        "aic_selected": "M1" if aic1 <= aic2 else "M2",
        "bic_M1": float(bic1),
        "bic_M2": float(bic2),
        "bic_selected": "M1" if bic1 <= bic2 else "M2",
    }
