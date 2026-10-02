from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.special import betaln, gammaln, log_ndtr, logsumexp, ndtr
from scipy.stats import beta as beta_dist
from scipy.stats import qmc

_EPS = 1e-10


def _clip_prob(p: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def log_binom_coeff(k: np.ndarray | int, n: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def log_binom_pmf(k: np.ndarray | int, n: int, p: np.ndarray | float) -> np.ndarray:
    p = _clip_prob(p)
    k_arr = np.asarray(k, dtype=np.float64)
    return log_binom_coeff(k_arr, n) + k_arr * np.log(p) + (n - k_arr) * np.log1p(-p)


def compress_counts(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(np.asarray(k, dtype=np.int64), return_counts=True)
    return values, counts.astype(np.int64)


def log_evidence_m1_binomial(k: np.ndarray, n_steps: int, a: float = 1.0, b: float = 1.0) -> float:
    """Closed-form Beta-Binomial evidence for independent trajectory counts."""
    k = np.asarray(k, dtype=np.int64)
    s = int(k.sum())
    n_obs = len(k)
    combin = float(np.sum(log_binom_coeff(k, n_steps)))
    return combin + float(betaln(a + s, b + n_obs * n_steps - s) - betaln(a, b))


def _logdiffexp(logx: np.ndarray, logy: np.ndarray) -> np.ndarray:
    """Compute log(exp(logx)-exp(logy)) for logx >= logy."""
    logx = np.asarray(logx, dtype=np.float64)
    logy = np.asarray(logy, dtype=np.float64)
    delta = logy - logx
    # log1p(-1) is correctly -inf when the interval has zero mass.  Avoid
    # positive roundoff in delta, which would make the expression invalid.
    delta = np.minimum(delta, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = logx + np.log1p(-np.exp(delta))
    return out


def _log_normal_interval_mass(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Stable log[Phi(hi)-Phi(lo)] across central and tail regions.

    Direct subtraction of two CDF values near one is catastrophically unstable
    in the right tail.  There we use Phi(hi)-Phi(lo)=Phi(-lo)-Phi(-hi), so the
    subtraction occurs in the left tail in log space.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lo, hi = np.broadcast_arrays(lo, hi)
    out = np.empty_like(lo, dtype=np.float64)

    left = hi <= 0.0
    right = lo >= 0.0
    center = ~(left | right)

    if np.any(left):
        out[left] = _logdiffexp(log_ndtr(hi[left]), log_ndtr(lo[left]))
    if np.any(right):
        out[right] = _logdiffexp(log_ndtr(-lo[right]), log_ndtr(-hi[right]))
    if np.any(center):
        mass = ndtr(hi[center]) - ndtr(lo[center])
        out[center] = np.log(np.maximum(mass, np.finfo(float).tiny))

    return np.maximum(out, math.log(np.finfo(float).tiny))


def log_gaussian_cc_mass(k: int, n_steps: int, p: np.ndarray) -> np.ndarray:
    """Continuity-corrected Gaussian approximation to a Binomial point mass."""
    p = _clip_prob(p)
    mu = n_steps * p
    var = np.maximum(n_steps * p * (1.0 - p), 1e-12)
    sd = np.sqrt(var)
    lo = (k - 0.5 - mu) / sd
    hi = (k + 0.5 - mu) / sd
    return _log_normal_interval_mass(lo, hi)


def log_gaussian_pdf(k: int, n_steps: int, p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    mu = n_steps * p
    var = np.maximum(n_steps * p * (1.0 - p), 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (k - mu) ** 2 / var)


@lru_cache(maxsize=128)
def prior_draws_3d(
    budget: int,
    a: float,
    b: float,
    pi_a: float,
    pi_b: float,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "sobol":
        if budget <= 0 or budget & (budget - 1):
            raise ValueError("Sobol budget must be a positive power of two")
        m = int(round(math.log2(budget)))
        u = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(m)
    elif method == "mc":
        u = np.random.default_rng(seed).random((budget, 3))
    else:
        raise ValueError(f"unknown method: {method}")
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    p1 = beta_dist.ppf(u[:, 0], a, b)
    p2 = beta_dist.ppf(u[:, 1], a, b)
    pi = beta_dist.ppf(u[:, 2], pi_a, pi_b)
    return _clip_prob(p1), _clip_prob(p2), _clip_prob(pi)


@lru_cache(maxsize=128)
def prior_draws_1d(budget: int, a: float, b: float, method: str, seed: int) -> np.ndarray:
    if method == "sobol":
        if budget <= 0 or budget & (budget - 1):
            raise ValueError("Sobol budget must be a positive power of two")
        m = int(round(math.log2(budget)))
        u = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(m).reshape(-1)
    elif method == "mc":
        u = np.random.default_rng(seed).random(budget)
    else:
        raise ValueError(f"unknown method: {method}")
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return _clip_prob(beta_dist.ppf(u, a, b))


def mixture_loglik_samples(
    k: np.ndarray,
    n_steps: int,
    p1: np.ndarray,
    p2: np.ndarray,
    pi: np.ndarray,
    likelihood: str = "binomial",
) -> np.ndarray:
    values, counts = compress_counts(k)
    log_pi = np.log(_clip_prob(pi))
    log_1m_pi = np.log1p(-_clip_prob(pi))
    out = np.zeros_like(p1, dtype=np.float64)
    for kval, count in zip(values, counts):
        if likelihood == "binomial":
            l1 = log_binom_pmf(int(kval), n_steps, p1)
            l2 = log_binom_pmf(int(kval), n_steps, p2)
        elif likelihood == "gaussian_cc":
            l1 = log_gaussian_cc_mass(int(kval), n_steps, p1)
            l2 = log_gaussian_cc_mass(int(kval), n_steps, p2)
        elif likelihood == "gaussian_pdf":
            l1 = log_gaussian_pdf(int(kval), n_steps, p1)
            l2 = log_gaussian_pdf(int(kval), n_steps, p2)
        else:
            raise ValueError(f"unknown likelihood: {likelihood}")
        out += count * np.logaddexp(log_pi + l1, log_1m_pi + l2)
    return out


def single_loglik_samples(
    k: np.ndarray,
    n_steps: int,
    p: np.ndarray,
    likelihood: str = "gaussian_cc",
) -> np.ndarray:
    values, counts = compress_counts(k)
    out = np.zeros_like(p, dtype=np.float64)
    for kval, count in zip(values, counts):
        if likelihood == "binomial":
            lp = log_binom_pmf(int(kval), n_steps, p)
        elif likelihood == "gaussian_cc":
            lp = log_gaussian_cc_mass(int(kval), n_steps, p)
        elif likelihood == "gaussian_pdf":
            lp = log_gaussian_pdf(int(kval), n_steps, p)
        else:
            raise ValueError(f"unknown likelihood: {likelihood}")
        out += count * lp
    return out


def logmeanexp(x: np.ndarray) -> float:
    return float(logsumexp(x) - math.log(len(x)))


def log_evidence_m2_numeric(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    pi_a: float = 1.0,
    pi_b: float = 1.0,
    method: str = "sobol",
    seed: int = 1234,
    likelihood: str = "binomial",
) -> float:
    p1, p2, pi = prior_draws_3d(budget, a, b, pi_a, pi_b, method, seed)
    ll = mixture_loglik_samples(k, n_steps, p1, p2, pi, likelihood=likelihood)
    return logmeanexp(ll)


def log_evidence_m1_numeric(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    method: str = "sobol",
    seed: int = 4321,
    likelihood: str = "gaussian_cc",
) -> float:
    p = prior_draws_1d(budget, a, b, method, seed)
    ll = single_loglik_samples(k, n_steps, p, likelihood=likelihood)
    return logmeanexp(ll)


def running_log_evidence_from_samples(loglik: np.ndarray, budgets: list[int]) -> dict[int, float]:
    loglik = np.asarray(loglik, dtype=np.float64)
    cum = np.logaddexp.accumulate(loglik)
    result: dict[int, float] = {}
    for b in budgets:
        if b > len(loglik):
            raise ValueError("budget checkpoint exceeds number of samples")
        result[b] = float(cum[b - 1] - math.log(b))
    return result
