from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp

from .importance import (
    log_evidence_m1_gaussian_quad,
    ridge_importance_logweight_samples,
    summarize_logweights,
)
from .likelihoods import log_evidence_m1_binomial
from .mle import model_selection_scores
from .seeds import stable_seed


def _logz_from_logweights(logw: np.ndarray) -> float:
    return float(logsumexp(logw) - math.log(len(logw)))


def _adaptive_m2_evidence(
    k: np.ndarray,
    n_steps: int,
    logz1: float,
    prior_a: float,
    evidence_cfg: dict,
    evidence_seed: int,
    likelihood: str,
) -> tuple[float, dict]:
    """Adaptive ridge-aware marginal-likelihood estimation for M2.

    Stage 1 uses the preregistered base budget.  Numerically difficult or
    decision-boundary cases are re-estimated with two independent runs at a
    larger budget.  A final maximum-budget stage is used only when the two
    independent estimates still disagree materially.
    """
    base_budget = int(evidence_cfg["base_budget"])
    escalation_budget = int(evidence_cfg["escalation_budget"])
    max_budget = int(evidence_cfg["max_budget"])
    ess_min = float(evidence_cfg["ess_fraction_min"])
    max_weight_max = float(evidence_cfg["max_weight_max"])
    boundary = float(evidence_cfg["near_boundary_abs_logbf"])
    repeat_sd_max = float(evidence_cfg["repeat_sd_max"])

    def one_run(budget: int, stage: str, repeat: int) -> tuple[float, np.ndarray, dict]:
        seed = stable_seed(evidence_seed, "ridge", stage, repeat)
        logw, diag = ridge_importance_logweight_samples(
            k,
            n_steps,
            budget,
            a=prior_a,
            b=prior_a,
            pi_a=1.0,
            pi_b=1.0,
            seed=seed,
            likelihood=likelihood,
        )
        return _logz_from_logweights(logw), logw, diag

    logz_base, logw_base, diag_base = one_run(base_budget, "base", 0)
    logbf_base = logz_base - logz1
    base_ok = (
        diag_base["ess_fraction"] >= ess_min
        and diag_base["max_normalized_weight"] <= max_weight_max
        and abs(logbf_base) >= boundary
    )
    if base_ok:
        out = {
            **diag_base,
            "method": "ridge_importance_adaptive",
            "initial_budget": base_budget,
            "final_budget": base_budget,
            "total_samples": base_budget,
            "num_repeats": 1,
            "reestimated": 0,
            "repeat_sd_logZ": 0.0,
            "repeat_sign_agree": 1,
            "numerical_status": "pass",
            "base_logZ_M2": logz_base,
            "base_logBF": logbf_base,
        }
        return logz_base, out

    def repeated_stage(budget: int, stage: str) -> tuple[float, dict]:
        runs = [one_run(budget, stage, r) for r in range(2)]
        logs = np.array([r[0] for r in runs], dtype=float)
        pooled_logw = np.concatenate([r[1] for r in runs])
        combined = _logz_from_logweights(pooled_logw)
        pooled_diag = summarize_logweights(pooled_logw)
        pooled_diag["optimizer_success"] = bool(all(bool(r[2]["optimizer_success"]) for r in runs))
        repeat_sd = float(np.std(logs, ddof=1))
        signs = [(x - logz1) > 0.0 for x in logs]
        return combined, {
            **pooled_diag,
            "repeat_sd_logZ": repeat_sd,
            "repeat_sign_agree": int(signs[0] == signs[1]),
            "repeat_logZ_1": float(logs[0]),
            "repeat_logZ_2": float(logs[1]),
            "final_budget": budget,
            "total_samples": 2 * budget,
            "num_repeats": 2,
        }

    logz2, stage2 = repeated_stage(escalation_budget, "escalation")
    needs_max = (
        stage2["repeat_sd_logZ"] > repeat_sd_max
        or stage2["repeat_sign_agree"] == 0
    )
    if not needs_max:
        stage2.update(
            {
                "method": "ridge_importance_adaptive",
                "initial_budget": base_budget,
                "reestimated": 1,
                "numerical_status": "reestimated_pass",
                "base_logZ_M2": logz_base,
                "base_logBF": logbf_base,
            }
        )
        return logz2, stage2

    logz3, stage3 = repeated_stage(max_budget, "max")
    final_ok = (
        stage3["repeat_sd_logZ"] <= repeat_sd_max
        and stage3["repeat_sign_agree"] == 1
    )
    stage3.update(
        {
            "method": "ridge_importance_adaptive",
            "initial_budget": base_budget,
            "reestimated": 1,
            "numerical_status": "reestimated_pass" if final_ok else "unresolved",
            "base_logZ_M2": logz_base,
            "base_logBF": logbf_base,
        }
    )
    return logz3, stage3


def _row_from_evidence(logz1: float, logz2: float, diag: dict) -> dict:
    logbf = float(logz2 - logz1)
    return {
        "logZ_M1": float(logz1),
        "logZ_M2": float(logz2),
        "logBF": logbf,
        "bayes_selected": "M2" if logbf > 0.0 else "M1",
        "evidence_method": diag["method"],
        "evidence_initial_budget": int(diag["initial_budget"]),
        "evidence_final_budget": int(diag["final_budget"]),
        "evidence_total_samples": int(diag["total_samples"]),
        "evidence_num_repeats": int(diag["num_repeats"]),
        "evidence_reestimated": int(diag["reestimated"]),
        "evidence_ess": float(diag["ess"]),
        "evidence_ess_fraction": float(diag["ess_fraction"]),
        "evidence_max_weight_fraction": float(diag["max_normalized_weight"]),
        "evidence_optimizer_success": int(bool(diag["optimizer_success"])),
        "evidence_repeat_sd_logZ": float(diag["repeat_sd_logZ"]),
        "evidence_repeat_sign_agree": int(diag["repeat_sign_agree"]),
        "evidence_numerical_status": str(diag["numerical_status"]),
        "evidence_base_logZ_M2": float(diag["base_logZ_M2"]),
        "evidence_base_logBF": float(diag["base_logBF"]),
    }


def evaluate_binomial_dataset(
    k: np.ndarray,
    n_steps: int,
    prior_a: float,
    evidence_cfg: dict,
    evidence_seed: int,
    include_baselines: bool = False,
) -> dict:
    logz1 = log_evidence_m1_binomial(k, n_steps, prior_a, prior_a)
    logz2, diag = _adaptive_m2_evidence(
        k,
        n_steps,
        logz1,
        prior_a,
        evidence_cfg,
        evidence_seed,
        likelihood="binomial",
    )
    row = _row_from_evidence(logz1, logz2, diag)
    if include_baselines:
        row.update(model_selection_scores(k, n_steps, seed=evidence_seed))
    return row


def evaluate_gaussian_approx_dataset(
    k: np.ndarray,
    n_steps: int,
    prior_a: float,
    evidence_cfg: dict,
    evidence_seed: int,
    likelihood: str = "gaussian_cc",
) -> dict:
    # E2 is intended to isolate likelihood approximation, so M1 evidence is
    # integrated by accurate 1-D quadrature rather than an additional Laplace
    # approximation.
    logz1 = log_evidence_m1_gaussian_quad(k, n_steps, a=prior_a, b=prior_a, likelihood=likelihood)
    logz2, diag = _adaptive_m2_evidence(
        k,
        n_steps,
        logz1,
        prior_a,
        evidence_cfg,
        evidence_seed,
        likelihood=likelihood,
    )
    return _row_from_evidence(logz1, logz2, diag)
