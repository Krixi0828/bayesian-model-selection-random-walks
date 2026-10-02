from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, expit, logsumexp

from .likelihoods import mixture_loglik_samples, single_loglik_samples
from .mle import mle_m2_em

_EPS = 1e-12


def _logit(p: float | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _log_beta_pdf(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return (a - 1.0) * np.log(p) + (b - 1.0) * np.log1p(-p) - betaln(a, b)


def _log_beta_x(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Stable log-density in logit space induced by p~Beta(a,b).

    If p = sigmoid(x), then

        Beta(p; a,b) * dp/dx
        = p**a * (1-p)**b / B(a,b).

    Compute log(p) and log(1-p) directly from x to avoid expit saturation
    at extreme logits.
    """
    x = np.asarray(x, dtype=float)
    log_p = -np.logaddexp(0.0, -x)
    log_1mp = -np.logaddexp(0.0, x)
    return a * log_p + b * log_1mp - betaln(a, b)


def _log_product_beta_x(x: np.ndarray, params: tuple[tuple[float, float], ...]) -> np.ndarray:
    x = np.atleast_2d(x)
    out = np.zeros(x.shape[0], dtype=float)
    for j, (a, b) in enumerate(params):
        out += _log_beta_x(x[:, j], a, b)
    return out


def _sample_product_beta_x(
    rng: np.random.Generator,
    n: int,
    params: tuple[tuple[float, float], ...],
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, len(params)), dtype=float)
    cols = []
    for a, b in params:
        cols.append(_logit(rng.beta(a, b, size=n)))
    return np.column_stack(cols)


def _log_prior_x(x: np.ndarray, a: float, b: float, pi_a: float, pi_b: float) -> np.ndarray:
    return _log_product_beta_x(np.atleast_2d(x), ((a, b), (a, b), (pi_a, pi_b)))


def _log_target_x(
    x: np.ndarray,
    k: np.ndarray,
    n_steps: int,
    a: float,
    b: float,
    pi_a: float,
    pi_b: float,
    likelihood: str,
) -> np.ndarray:
    x2 = np.atleast_2d(x)
    p = expit(x2)
    ll = mixture_loglik_samples(k, n_steps, p[:, 0], p[:, 1], p[:, 2], likelihood=likelihood)
    return ll + _log_prior_x(x2, a, b, pi_a, pi_b)


def _finite_hessian(f, x: np.ndarray, h: float = 2e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = len(x)
    H = np.zeros((d, d), dtype=float)
    f0 = float(f(x))
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = h
        H[i, i] = (float(f(x + ei)) - 2.0 * f0 + float(f(x - ei))) / (h * h)
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = h
            v = (
                float(f(x + ei + ej))
                - float(f(x + ei - ej))
                - float(f(x - ei + ej))
                + float(f(x - ei - ej))
            ) / (4.0 * h * h)
            H[i, j] = H[j, i] = v
    return H


def _regularized_cov_from_hessian(H: np.ndarray) -> np.ndarray:
    H = 0.5 * (H + H.T)
    vals, vecs = np.linalg.eigh(H)
    vals = np.clip(vals, 1e-3, 1e5)
    cov = vecs @ np.diag(1.0 / vals) @ vecs.T
    return 0.5 * (cov + cov.T)


def _mvn_logpdf_many(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    cov = 0.5 * (cov + cov.T)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(d) * 1e-8
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    delta = x - mean
    quad = np.einsum("ni,ij,nj->n", delta, inv, delta)
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)


def _find_mode(
    k: np.ndarray,
    n_steps: int,
    a: float,
    b: float,
    pi_a: float,
    pi_b: float,
    likelihood: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    mle, _ = mle_m2_em(k, n_steps, seed=seed)
    x0 = np.array([_logit(mle["p1"]), _logit(mle["p2"]), _logit(mle["pi"])], dtype=float).reshape(3)

    def objective(x):
        val = _log_target_x(np.asarray(x), k, n_steps, a, b, pi_a, pi_b, likelihood)[0]
        return -float(val)

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(-15.0, 15.0)] * 3,
        options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-6},
    )
    mode = np.asarray(res.x if np.isfinite(res.fun) else x0, dtype=float)
    try:
        H = _finite_hessian(objective, mode)
        cov = _regularized_cov_from_hessian(H)
    except Exception:
        cov = np.eye(3) * 2.0
    return mode, cov, bool(np.isfinite(res.fun))


def _local_gaussian_components(mode: np.ndarray, cov: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    swap = np.array([mode[1], mode[0], -mode[2]])
    P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    cov_swap = P @ cov @ P.T
    scales = [0.5, 1.0, 2.0, 4.0]
    centers: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    for center, c0 in [(mode, cov), (swap, cov_swap)]:
        for scale in scales:
            centers.append(center)
            covs.append(c0 * scale)
    return centers, covs


def summarize_logweights(logw: np.ndarray) -> dict:
    logw = np.asarray(logw, dtype=float)
    logsumw = float(logsumexp(logw))
    ess = float(np.exp(2.0 * logsumw - logsumexp(2.0 * logw)))
    norm_w = np.exp(logw - logsumw)
    return {
        "ess": ess,
        "ess_fraction": ess / len(logw),
        "max_normalized_weight": float(norm_w.max()),
    }


def local_importance_logweight_samples(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    pi_a: float = 1.0,
    pi_b: float = 1.0,
    seed: int = 1234,
    likelihood: str = "binomial",
    defensive_weight: float = 0.20,
) -> tuple[np.ndarray, dict]:
    """V1 local-mode defensive importance sampler retained as a baseline."""
    if budget < 64:
        raise ValueError("importance budget should be at least 64")
    rng = np.random.default_rng(seed)
    mode, cov, opt_success = _find_mode(k, n_steps, a, b, pi_a, pi_b, likelihood, seed)
    centers, covs = _local_gaussian_components(mode, cov)
    n_gauss = len(centers)
    gauss_weight_total = 1.0 - defensive_weight
    component_probs = np.array([defensive_weight] + [gauss_weight_total / n_gauss] * n_gauss)
    component = rng.choice(1 + n_gauss, size=budget, p=component_probs)
    x = np.empty((budget, 3), dtype=float)

    idx = np.where(component == 0)[0]
    if len(idx):
        x[idx] = _sample_product_beta_x(rng, len(idx), ((a, b), (a, b), (pi_a, pi_b)))
    for j, (mean, c0) in enumerate(zip(centers, covs), start=1):
        idx = np.where(component == j)[0]
        if len(idx):
            x[idx] = rng.multivariate_normal(mean, c0, size=len(idx), check_valid="ignore")

    log_target = _log_target_x(x, k, n_steps, a, b, pi_a, pi_b, likelihood)
    log_prior_prop = _log_prior_x(x, a, b, pi_a, pi_b)
    gaussian_terms = [
        _mvn_logpdf_many(x, mean, c0) + math.log(gauss_weight_total / n_gauss)
        for mean, c0 in zip(centers, covs)
    ]
    log_q_gauss = logsumexp(np.vstack(gaussian_terms), axis=0)
    log_q = np.logaddexp(math.log(defensive_weight) + log_prior_prop, log_q_gauss)
    logw = log_target - log_q
    diagnostics = summarize_logweights(logw)
    diagnostics.update(
        {
            "optimizer_success": opt_success,
            "mode_p1": float(expit(mode[0])),
            "mode_p2": float(expit(mode[1])),
            "mode_pi": float(expit(mode[2])),
            "proposal": "local_importance",
        }
    )
    return logw, diagnostics


def log_evidence_m2_local_importance(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    pi_a: float = 1.0,
    pi_b: float = 1.0,
    seed: int = 1234,
    likelihood: str = "binomial",
    defensive_weight: float = 0.20,
) -> tuple[float, dict]:
    logw, diagnostics = local_importance_logweight_samples(
        k, n_steps, budget, a, b, pi_a, pi_b, seed, likelihood, defensive_weight
    )
    return float(logsumexp(logw) - math.log(budget)), diagnostics


def ridge_importance_logweight_samples(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    pi_a: float = 1.0,
    pi_b: float = 1.0,
    seed: int = 1234,
    likelihood: str = "binomial",
    prior_weight: float = 0.10,
    equal_ridge_weight: float = 0.20,
    boundary1_weight: float = 0.15,
    boundary2_weight: float = 0.15,
    local_weight: float = 0.40,
    boundary_strength: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Ridge-aware defensive IS for the two-component mixture evidence.

    The proposal explicitly covers the nested/singular geometry where M2 can
    mimic M1: p1≈p2, pi≈1, and pi≈0, in addition to both label-switched local
    posterior modes and a full-support prior component.
    """
    if budget < 64:
        raise ValueError("importance budget should be at least 64")
    weights = np.array(
        [prior_weight, equal_ridge_weight, boundary1_weight, boundary2_weight, local_weight],
        dtype=float,
    )
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("ridge proposal top-level weights must be positive and sum to one")

    k = np.asarray(k, dtype=np.int64)
    rng = np.random.default_rng(seed)
    mode, cov, opt_success = _find_mode(k, n_steps, a, b, pi_a, pi_b, likelihood, seed)
    centers, covs = _local_gaussian_components(mode, cov)
    n_local = len(centers)

    s = int(k.sum())
    post_a = float(a + s)
    post_b = float(b + len(k) * n_steps - s)
    prior_params = ((a, b), (a, b), (pi_a, pi_b))
    equal_params = ((post_a, post_b), (post_a, post_b), (pi_a, pi_b))
    boundary1_params = ((post_a, post_b), (a, b), (boundary_strength, 1.0))
    boundary2_params = ((a, b), (post_a, post_b), (1.0, boundary_strength))

    component_probs = np.array(
        [prior_weight, equal_ridge_weight, boundary1_weight, boundary2_weight]
        + [local_weight / n_local] * n_local,
        dtype=float,
    )
    component = rng.choice(len(component_probs), size=budget, p=component_probs)
    x = np.empty((budget, 3), dtype=float)

    beta_params = [prior_params, equal_params, boundary1_params, boundary2_params]
    for j, params in enumerate(beta_params):
        idx = np.where(component == j)[0]
        if len(idx):
            x[idx] = _sample_product_beta_x(rng, len(idx), params)
    for local_j, (mean, c0) in enumerate(zip(centers, covs)):
        j = 4 + local_j
        idx = np.where(component == j)[0]
        if len(idx):
            x[idx] = rng.multivariate_normal(mean, c0, size=len(idx), check_valid="ignore")

    log_target = _log_target_x(x, k, n_steps, a, b, pi_a, pi_b, likelihood)
    proposal_terms = [
        math.log(prior_weight) + _log_product_beta_x(x, prior_params),
        math.log(equal_ridge_weight) + _log_product_beta_x(x, equal_params),
        math.log(boundary1_weight) + _log_product_beta_x(x, boundary1_params),
        math.log(boundary2_weight) + _log_product_beta_x(x, boundary2_params),
    ]
    proposal_terms.extend(
        math.log(local_weight / n_local) + _mvn_logpdf_many(x, mean, c0)
        for mean, c0 in zip(centers, covs)
    )
    log_q = logsumexp(np.vstack(proposal_terms), axis=0)
    logw = log_target - log_q
    diagnostics = summarize_logweights(logw)
    diagnostics.update(
        {
            "optimizer_success": opt_success,
            "mode_p1": float(expit(mode[0])),
            "mode_p2": float(expit(mode[1])),
            "mode_pi": float(expit(mode[2])),
            "proposal": "ridge_importance",
            "m1_posterior_a": post_a,
            "m1_posterior_b": post_b,
        }
    )
    return logw, diagnostics


def log_evidence_m2_ridge_importance(
    k: np.ndarray,
    n_steps: int,
    budget: int,
    a: float = 1.0,
    b: float = 1.0,
    pi_a: float = 1.0,
    pi_b: float = 1.0,
    seed: int = 1234,
    likelihood: str = "binomial",
) -> tuple[float, dict]:
    logw, diagnostics = ridge_importance_logweight_samples(
        k, n_steps, budget, a, b, pi_a, pi_b, seed, likelihood
    )
    return float(logsumexp(logw) - math.log(budget)), diagnostics


# Backward-compatible aliases: v1 code and old tests still resolve, while E4
# can explicitly compare the old local estimator against the v2 ridge proposal.
importance_logweight_samples = local_importance_logweight_samples
log_evidence_m2_importance = log_evidence_m2_local_importance


def log_evidence_m1_gaussian_laplace(
    k: np.ndarray,
    n_steps: int,
    a: float = 1.0,
    b: float = 1.0,
    likelihood: str = "gaussian_cc",
) -> float:
    """Fast 1-D Laplace approximation retained only for diagnostics."""

    def logtarget_scalar(x: float) -> float:
        p = float(expit(x))
        ll = float(single_loglik_samples(k, n_steps, np.array([p]), likelihood=likelihood)[0])
        lp = float(_log_beta_pdf(np.array([p]), a, b)[0] + math.log(p) + math.log1p(-p))
        return ll + lp

    res = minimize_scalar(
        lambda x: -logtarget_scalar(x),
        bounds=(-15.0, 15.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    mode = float(res.x)
    f0 = logtarget_scalar(mode)
    h = 1e-3
    second = (logtarget_scalar(mode + h) - 2.0 * f0 + logtarget_scalar(mode - h)) / (h * h)
    curvature = max(-float(second), 1e-8)
    return float(f0 + 0.5 * math.log(2.0 * math.pi) - 0.5 * math.log(curvature))


def log_evidence_m1_gaussian_quad(
    k: np.ndarray,
    n_steps: int,
    a: float = 1.0,
    b: float = 1.0,
    likelihood: str = "gaussian_cc",
) -> float:
    """Accurate 1-D evidence integral in logit-p space for Gaussian M1."""
    from scipy.integrate import quad

    def logtarget_scalar(x: float) -> float:
        p = float(expit(x))
        ll = float(single_loglik_samples(k, n_steps, np.array([p]), likelihood=likelihood)[0])
        lp = float(_log_beta_pdf(np.array([p]), a, b)[0] + math.log(p) + math.log1p(-p))
        return ll + lp

    res = minimize_scalar(
        lambda x: -logtarget_scalar(x),
        bounds=(-15.0, 15.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    mode = float(res.x)
    shift = logtarget_scalar(mode)
    h = 1e-3
    curvature = (
        -logtarget_scalar(mode + h) + 2.0 * logtarget_scalar(mode) - logtarget_scalar(mode - h)
    ) / (h * h)
    curvature = max(float(curvature), 1e-6)
    scale = 1.0 / math.sqrt(curvature)

    def integrand_u(u: float) -> float:
        return math.exp(logtarget_scalar(mode + scale * u) - shift)

    val, _ = quad(integrand_u, -14.0, 14.0, epsabs=1e-11, epsrel=1e-9, limit=300)
    return float(math.log(val * scale) + shift)
