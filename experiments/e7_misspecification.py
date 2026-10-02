from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brw.common import ensure_dir, save_csv
from brw.evaluation import evaluate_binomial_dataset
from brw.parallel import map_tasks
from brw.plotting import finish
from brw.random_walk import generate_persistent_counts, persistent_variance_inflation
from brw.seeds import stable_seed
from brw.stats import wilson_interval


def _worker(task: dict) -> list[dict]:
    rows = []
    N, T = int(task["N"]), int(task["T"])
    q = float(task["q_value"])
    theory_vif = persistent_variance_inflation(T, q)
    for rep in range(int(task["reps"])):
        seed = stable_seed(task["base_seed"], "e7", q, rep)
        rng = np.random.default_rng(seed)
        k = generate_persistent_counts(rng, N, T, q)
        evidence_seed = stable_seed(task["evidence_seed"], "e7", seed)
        ev = evaluate_binomial_dataset(k, T, float(task["prior_a"]), task["evidence"], evidence_seed)
        empirical_vif = float(np.var(k, ddof=1) / (T * 0.25))
        rows.append(
            {
                "experiment": "E7",
                "replicate": rep,
                "data_seed": seed,
                "true_model": "M1_correlated",
                "N": N,
                "T": T,
                "q": q,
                "rho": 2 * q - 1,
                "sample_var_K": float(np.var(k, ddof=1)),
                "binomial_var_reference": T * 0.25,
                "variance_inflation": empirical_vif,
                "theoretical_variance_inflation": theory_vif,
                **ev,
            }
        )
    return rows


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e7"]
    out = ensure_dir(out_root / "exp7_misspecification")
    tasks = [
        {
            **c,
            "q_value": q,
            "base_seed": cfg["base_seed"],
            "evidence_seed": cfg["evidence_seed"],
            "evidence": cfg["evidence"],
        }
        for q in c["q"]
    ]
    chunks = map_tasks(_worker, tasks, jobs)
    df = save_csv([r for ch in chunks for r in ch], out / "raw.csv")
    rows = []
    for q, g in df.groupby("q"):
        s = int((g.bayes_selected == "M2").sum())
        lo, hi = wilson_interval(s, len(g))
        rows.append(
            {
                "q": q,
                "rho": 2 * q - 1,
                "false_positive_rate": s / len(g),
                "ci_low": lo,
                "ci_high": hi,
                "median_logBF": float(g.logBF.median()),
                "mean_variance_inflation": float(g.variance_inflation.mean()),
                "theoretical_variance_inflation": float(g.theoretical_variance_inflation.iloc[0]),
                "rows": len(g),
            }
        )
    summary = pd.DataFrame(rows).sort_values("q")
    summary.to_csv(out / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(summary.q, summary.false_positive_rate, marker="o")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"persistence $q=P(X_t=X_{t-1})$")
    ax.set_ylabel("false-positive rate for M2")
    ax.set_title("E7: temporal dependence can mimic latent heterogeneity")
    finish(out / "figure_misspecification_fpr.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(summary.q, summary.mean_variance_inflation, marker="o", label="simulation")
    ax.plot(summary.q, summary.theoretical_variance_inflation, linestyle="--", label="finite-T theory")
    ax.axhline(1.0, linestyle=":", linewidth=1)
    ax.set_xlabel("persistence q")
    ax.set_ylabel("variance inflation of K")
    ax.set_title("E7: endpoint dispersion under temporal dependence")
    ax.legend()
    finish(out / "figure_variance_inflation.png")
    return {
        "rows": len(df),
        "unresolved_evidence_rows": int((df["evidence_numerical_status"] == "unresolved").sum()),
    }
