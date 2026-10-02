from __future__ import annotations

import numpy as np


def generate_iid_counts(rng: np.random.Generator, n_trajectories: int, steps: int, p: float) -> np.ndarray:
    """Number of +1 steps for independent biased 1-D random walks."""
    return rng.binomial(steps, p, size=n_trajectories).astype(np.int64)


def generate_mixture_counts(
    rng: np.random.Generator,
    n_trajectories: int,
    steps: int,
    p1: float,
    p2: float,
    mix_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate latent-regime walks; mix_weight is P(z=1), using p1 for z=1."""
    z = rng.random(n_trajectories) < mix_weight
    probs = np.where(z, p1, p2)
    k = rng.binomial(steps, probs).astype(np.int64)
    return k, z.astype(np.int8)


def generate_persistent_counts(
    rng: np.random.Generator,
    n_trajectories: int,
    steps: int,
    persistence_q: float,
) -> np.ndarray:
    """
    Symmetric two-state persistent random walk.

    X_1 is +/-1 equiprobably and P(X_t = X_{t-1}) = q.  The marginal
    direction is 0.5/0.5, but steps are temporally correlated for q != 0.5.
    Returns the number of +1 steps in each trajectory.
    """
    if not (0.0 <= persistence_q <= 1.0):
        raise ValueError("persistence_q must be in [0,1]")
    state = np.where(rng.random(n_trajectories) < 0.5, 1, -1).astype(np.int8)
    positive = (state == 1).astype(np.int64)
    for _ in range(1, steps):
        switch = rng.random(n_trajectories) >= persistence_q
        state[switch] *= -1
        positive += (state == 1)
    return positive


def persistent_variance_inflation(steps: int, persistence_q: float) -> float:
    """Finite-T variance inflation for the count of +1 states in a symmetric persistent walk.

    With rho=2q-1, Corr(I_t, I_{t+h})=rho^h.  Relative to the iid
    Binomial(T, 1/2) variance T/4, the exact finite-horizon inflation is
    1 + 2 * sum_{h=1}^{T-1} (1-h/T) rho^h.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not (0.0 <= persistence_q <= 1.0):
        raise ValueError("persistence_q must be in [0,1]")
    rho = 2.0 * persistence_q - 1.0
    if steps == 1:
        return 1.0
    h = np.arange(1, steps, dtype=np.float64)
    return float(1.0 + 2.0 * np.sum((1.0 - h / steps) * rho ** h))


def endpoint_from_counts(k: np.ndarray, steps: int) -> np.ndarray:
    return 2 * np.asarray(k, dtype=np.int64) - int(steps)


def simulate_occupation_proportion(
    rng: np.random.Generator,
    n_trajectories: int,
    steps: int,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Simulate A=(n_+ + 0.5 n_0)/T for symmetric 1-D random walks.

    Chunking bounds memory while preserving independent trajectories.
    """
    out = np.empty(n_trajectories, dtype=np.float64)
    start = 0
    while start < n_trajectories:
        m = min(chunk_size, n_trajectories - start)
        increments = np.where(rng.random((m, steps)) < 0.5, 1, -1).astype(np.int8)
        pos = np.cumsum(increments, axis=1, dtype=np.int32)
        n_pos = np.sum(pos > 0, axis=1)
        n_zero = np.sum(pos == 0, axis=1)
        out[start : start + m] = (n_pos + 0.5 * n_zero) / steps
        start += m
    return out
