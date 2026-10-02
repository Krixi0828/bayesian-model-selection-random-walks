from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.stats import norm

from brw.evaluation import evaluate_binomial_dataset
from brw.importance import (
    log_evidence_m2_local_importance,
    log_evidence_m2_ridge_importance,
)
from brw.likelihoods import (
    log_binom_pmf,
    log_evidence_m1_binomial,
    log_evidence_m2_numeric,
    log_gaussian_cc_mass,
    mixture_loglik_samples,
)
from brw.random_walk import (
    generate_iid_counts,
    persistent_variance_inflation,
)


def test_random_walk_count_mean():
    rng = np.random.default_rng(123)
    T = 100
    k = generate_iid_counts(rng, 20000, T, 0.4)
    assert abs(k.mean() - 40.0) < 0.2


def test_m1_evidence_matches_numeric_integral():
    k = np.array([1, 2, 3], dtype=int)
    T = 4
    a = b = 1.0
    analytic = log_evidence_m1_binomial(k, T, a, b)

    def integrand(p):
        ll = float(np.sum(log_binom_pmf(k, T, p)))
        logprior = (a - 1) * math.log(p) + (b - 1) * math.log(1 - p) - betaln(a, b)
        return math.exp(ll + logprior)

    numeric, _ = quad(integrand, 1e-10, 1 - 1e-10, epsabs=1e-12, epsrel=1e-12)
    assert abs(analytic - math.log(numeric)) < 1e-8


def test_mixture_likelihood_label_symmetry():
    k = np.array([10, 12, 18, 20], dtype=int)
    T = 30
    p1 = np.array([0.3, 0.4])
    p2 = np.array([0.7, 0.6])
    pi = np.array([0.2, 0.8])
    a = mixture_loglik_samples(k, T, p1, p2, pi)
    b = mixture_loglik_samples(k, T, p2, p1, 1 - pi)
    assert np.allclose(a, b, atol=1e-12)


def test_local_importance_estimator_is_finite():
    rng = np.random.default_rng(9)
    k = generate_iid_counts(rng, 30, 50, 0.5)
    logz, diag = log_evidence_m2_local_importance(k, 50, 256, seed=99)
    assert np.isfinite(logz)
    assert 0 < diag["ess"] <= 256
    assert 0 < diag["max_normalized_weight"] <= 1


def test_gaussian_cc_right_tail_matches_direct_normal_integral():
    T = 1000
    p = 0.4
    k = 600
    got = float(log_gaussian_cc_mass(k, T, np.array([p]))[0])
    mu = T * p
    sd = math.sqrt(T * p * (1 - p))
    lo = (k - 0.5 - mu) / sd
    hi = (k + 0.5 - mu) / sd
    # Integrate the standard-normal density directly; this is independent of
    # the CDF-difference implementation under test.
    mass, _ = quad(norm.pdf, lo, hi, epsabs=1e-300, epsrel=1e-10, limit=200)
    expected = math.log(mass)
    assert abs(got - expected) < 1e-8
    assert got < -80.0  # catches the v1 catastrophic-cancellation failure


def test_gaussian_cc_symmetry():
    T = 200
    k = 71
    p = 0.37
    a = log_gaussian_cc_mass(k, T, np.array([p]))[0]
    b = log_gaussian_cc_mass(T - k, T, np.array([1.0 - p]))[0]
    assert abs(float(a - b)) < 1e-12


def test_gaussian_cc_central_mass_is_finite():
    got = float(log_gaussian_cc_mass(50, 100, np.array([0.5]))[0])
    assert np.isfinite(got)
    assert got < 0.0


def test_ridge_importance_agrees_with_high_budget_sobol_on_small_case():
    rng = np.random.default_rng(2026)
    k = generate_iid_counts(rng, 24, 30, 0.5)
    ref = log_evidence_m2_numeric(k, 30, 65536, method="sobol", seed=77)
    est, diag = log_evidence_m2_ridge_importance(k, 30, 16384, seed=88)
    assert np.isfinite(est)
    assert abs(est - ref) < 0.20
    assert diag["ess"] > 0


def test_persistent_vif_iid_case_is_one():
    assert abs(persistent_variance_inflation(200, 0.5) - 1.0) < 1e-15


def test_persistent_vif_matches_explicit_covariance_sum():
    T = 37
    q = 0.72
    rho = 2 * q - 1
    expected = 1.0 + 2.0 * sum((1.0 - h / T) * rho**h for h in range(1, T))
    assert abs(persistent_variance_inflation(T, q) - expected) < 1e-12


def test_adaptive_evaluator_returns_numerical_status():
    rng = np.random.default_rng(314)
    k = generate_iid_counts(rng, 30, 40, 0.5)
    cfg = {
        "base_budget": 128,
        "escalation_budget": 512,
        "max_budget": 2048,
        "ess_fraction_min": 0.01,
        "max_weight_max": 0.20,
        "near_boundary_abs_logbf": 0.50,
        "repeat_sd_max": 0.10,
    }
    row = evaluate_binomial_dataset(k, 40, 1.0, cfg, 12345)
    assert row["evidence_numerical_status"] in {"pass", "reestimated_pass", "unresolved"}
    assert row["evidence_method"] == "ridge_importance_adaptive"
    assert np.isfinite(row["logBF"])


def test_log_beta_x_is_stable_for_extreme_logits():
    from brw.importance import _log_beta_x

    x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    got = _log_beta_x(x, 1.0, 1.0)

    assert np.all(np.isfinite(got))
    assert np.allclose(got, got[::-1], atol=1e-12)
