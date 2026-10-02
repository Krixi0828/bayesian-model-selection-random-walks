from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from brw.common import ensure_dir, save_csv
from brw.importance import (
    local_importance_logweight_samples,
    ridge_importance_logweight_samples,
    summarize_logweights,
)
from brw.likelihoods import (
    log_evidence_m1_binomial,
    log_evidence_m2_numeric,
    mixture_loglik_samples,
    prior_draws_3d,
    running_log_evidence_from_samples,
)
from brw.parallel import map_tasks
from brw.plotting import finish
from brw.random_walk import generate_iid_counts, generate_mixture_counts
from brw.seeds import stable_seed


def _case_spec(case: str) -> tuple[str, tuple[float, float, float] | float]:
    if case == "M1_center":
        return "M1", 0.50
    if case == "M1_offcenter":
        return "M1", 0.35
    if case == "M2_hard":
        return "M2", (0.47, 0.53, 0.50)
    if case == "M2_medium":
        return "M2", (0.40, 0.60, 0.50)
    if case == "M2_imbalanced":
        return "M2", (0.40, 0.60, 0.10)
    raise KeyError(case)


def _make_dataset(task: dict) -> tuple[np.ndarray, str]:
    N, T = int(task["N"]), int(task["T"])
    case = str(task["case"])
    rng = np.random.default_rng(int(task["data_seed"]))
    true_model, spec = _case_spec(case)
    if true_model == "M1":
        return generate_iid_counts(rng, N, T, float(spec)), true_model
    p1, p2, w = spec  # type: ignore[misc]
    k, _ = generate_mixture_counts(rng, N, T, float(p1), float(p2), float(w))
    return k, true_model


def _reference_sobol(task: dict, k: np.ndarray, logz1: float) -> tuple[float, dict]:
    c = task
    scrambles = int(c["reference_scrambles"])
    initial = int(c["reference_initial_budget"])
    maximum = int(c["reference_max_budget"])
    escalate_sd = float(c["reference_escalate_sd"])
    sd_max = float(c["reference_sd_max"])
    boundary = float(c["reference_boundary_abs_logbf"])

    def runs_at(budget: int) -> np.ndarray:
        vals = []
        for r in range(scrambles):
            # Same low-discrepancy designs are reused across datasets, which is
            # valid and greatly reduces repeated inverse-CDF construction.
            seed = stable_seed(int(c["base_seed"]), "e4", "reference_sobol", budget, r)
            vals.append(
                log_evidence_m2_numeric(
                    k,
                    int(c["T"]),
                    budget,
                    a=1.0,
                    b=1.0,
                    pi_a=1.0,
                    pi_b=1.0,
                    method="sobol",
                    seed=seed,
                    likelihood="binomial",
                )
            )
        return np.asarray(vals, dtype=float)

    logs = runs_at(initial)
    combined = float(logsumexp(logs) - math.log(len(logs)))
    sd = float(np.std(logs, ddof=1)) if len(logs) > 1 else 0.0
    used = initial
    if (sd > escalate_sd or abs(combined - logz1) < boundary) and maximum > initial:
        logs = runs_at(maximum)
        combined = float(logsumexp(logs) - math.log(len(logs)))
        sd = float(np.std(logs, ddof=1)) if len(logs) > 1 else 0.0
        used = maximum
    return combined, {
        "reference_budget": used,
        "reference_scrambles": scrambles,
        "reference_between_run_sd_logZ": sd,
        "reference_status": "pass" if sd <= sd_max else "unresolved",
        "reference_logZ_min": float(logs.min()),
        "reference_logZ_max": float(logs.max()),
    }


def _worker(task: dict) -> tuple[list[dict], dict]:
    k, true_model = _make_dataset(task)
    T = int(task["T"])
    logz1 = log_evidence_m1_binomial(k, T, 1.0, 1.0)
    logz2_ref, ref_diag = _reference_sobol(task, k, logz1)
    logbf_ref = logz2_ref - logz1

    rows: list[dict] = []
    budgets = list(map(int, task["budgets"]))
    max_budget = max(budgets)
    for method in ["mc", "sobol", "local_is", "ridge_is"]:
        for s in range(int(task["estimator_seeds"])):
            est_seed = stable_seed(
                int(task["base_seed"]), "e4", "estimate", method, task["case"], task["dataset_id"], s
            )
            if method in ["mc", "sobol"]:
                p1, p2, pi = prior_draws_3d(max_budget, 1.0, 1.0, 1.0, 1.0, method, est_seed)
                samples = mixture_loglik_samples(k, T, p1, p2, pi, likelihood="binomial")
            elif method == "local_is":
                samples, _ = local_importance_logweight_samples(
                    k, T, max_budget, a=1.0, b=1.0, seed=est_seed, likelihood="binomial"
                )
            else:
                samples, _ = ridge_importance_logweight_samples(
                    k, T, max_budget, a=1.0, b=1.0, seed=est_seed, likelihood="binomial"
                )
            path = running_log_evidence_from_samples(samples, budgets)
            for budget, logz2 in path.items():
                if method in ["local_is", "ridge_is"]:
                    d = summarize_logweights(samples[:budget])
                    ess_fraction = d["ess_fraction"]
                    max_weight = d["max_normalized_weight"]
                else:
                    ess_fraction = np.nan
                    max_weight = np.nan
                logbf = logz2 - logz1
                rows.append(
                    {
                        "experiment": "E4",
                        "case": task["case"],
                        "dataset_id": int(task["dataset_id"]),
                        "data_seed": int(task["data_seed"]),
                        "true_model": true_model,
                        "method": method,
                        "estimator_seed_index": s,
                        "budget": budget,
                        "logZ_M1_analytic": logz1,
                        "logZ_M2_estimate": logz2,
                        "logZ_M2_reference": logz2_ref,
                        "logBF_estimate": logbf,
                        "logBF_reference": logbf_ref,
                        "abs_logZ_error": abs(logz2 - logz2_ref),
                        "squared_logZ_error": (logz2 - logz2_ref) ** 2,
                        "bf_sign_correct": int((logbf > 0) == (logbf_ref > 0)),
                        "reference_decisive": int(abs(logbf_ref) >= float(task["reference_boundary_abs_logbf"])),
                        "estimator_ess_fraction": ess_fraction,
                        "estimator_max_weight_fraction": max_weight,
                        **ref_diag,
                    }
                )
    reference = {
        "case": task["case"],
        "dataset_id": int(task["dataset_id"]),
        "data_seed": int(task["data_seed"]),
        "true_model": true_model,
        "logZ_M1_analytic": logz1,
        "logZ_M2_reference": logz2_ref,
        "logBF_reference": logbf_ref,
        **ref_diag,
    }
    return rows, reference


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e4"]
    out = ensure_dir(out_root / "exp4_convergence")
    tasks = []
    for case, count in c["cases"].items():
        for dataset_id in range(int(count)):
            data_seed = stable_seed(cfg["base_seed"], "e4", "dataset", case, dataset_id)
            tasks.append(
                {
                    **c,
                    "case": case,
                    "dataset_id": dataset_id,
                    "data_seed": data_seed,
                    "base_seed": cfg["base_seed"],
                }
            )
    outputs = map_tasks(_worker, tasks, jobs)
    df = save_csv([r for block, _ in outputs for r in block], out / "raw.csv")
    refs = pd.DataFrame([ref for _, ref in outputs])
    refs.to_csv(out / "reference_datasets.csv", index=False)

    summary_rows = []
    for (method, budget), g in df.groupby(["method", "budget"]):
        decisive = g[g["reference_decisive"] == 1]
        summary_rows.append(
            {
                "method": method,
                "budget": int(budget),
                "RMSE_logZ": float(np.sqrt(np.mean(g["squared_logZ_error"]))),
                "median_abs_logZ_error": float(g["abs_logZ_error"].median()),
                "p95_abs_logZ_error": float(g["abs_logZ_error"].quantile(0.95)),
                "sign_error_rate_all": float(1.0 - g["bf_sign_correct"].mean()),
                "sign_error_rate_decisive": float(1.0 - decisive["bf_sign_correct"].mean()) if len(decisive) else np.nan,
                "rows": len(g),
                "decisive_rows": len(decisive),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)

    production_budget = int(cfg["evidence"]["base_budget"])
    prod = summary[(summary["method"] == "ridge_is") & (summary["budget"] == production_budget)]
    if len(prod) != 1:
        raise RuntimeError(f"E4 budgets must contain production base budget {production_budget}")
    prod_row = prod.iloc[0]
    ref_pass = bool((refs["reference_status"] == "pass").all())
    sign_pass = bool(float(prod_row["sign_error_rate_decisive"]) <= float(c["gate_sign_error_max"]))
    p95_target = bool(float(prod_row["p95_abs_logZ_error"]) <= float(c["target_p95_abs_logz_error"]))
    gate = pd.DataFrame(
        [
            {
                "production_method": "ridge_is",
                "production_budget": production_budget,
                "decisive_sign_error_rate": float(prod_row["sign_error_rate_decisive"]),
                "sign_error_threshold": float(c["gate_sign_error_max"]),
                "reference_all_pass": int(ref_pass),
                "reference_sd_threshold": float(c["reference_sd_max"]),
                "p95_abs_logZ_error": float(prod_row["p95_abs_logZ_error"]),
                "p95_target": float(c["target_p95_abs_logz_error"]),
                "p95_target_met": int(p95_target),
                "numerical_gate_pass": int(ref_pass and sign_pass),
            }
        ]
    )
    gate.to_csv(out / "numerical_gate.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for method, g in summary.groupby("method"):
        g = g.sort_values("budget")
        ax.loglog(g["budget"], g["RMSE_logZ"], marker="o", label=method.upper())
    ax.set_xlabel("evidence sample budget B")
    ax.set_ylabel("RMSE of log evidence")
    ax.set_title("E4: evidence-estimator convergence")
    ax.legend()
    finish(out / "figure_convergence_rmse.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for method, g in summary.groupby("method"):
        g = g.sort_values("budget")
        ax.plot(g["budget"], g["sign_error_rate_decisive"], marker="o", label=method.upper())
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("evidence sample budget B")
    ax.set_ylabel("BF-sign error rate (decisive references)")
    ax.set_title("E4: numerical model-decision stability")
    ax.legend()
    finish(out / "figure_convergence_sign_error.png")

    return {
        "rows": len(df),
        "reference_datasets": len(refs),
        "reference_unresolved": int((refs["reference_status"] != "pass").sum()),
        "numerical_gate_pass": int(gate.iloc[0]["numerical_gate_pass"]),
    }
