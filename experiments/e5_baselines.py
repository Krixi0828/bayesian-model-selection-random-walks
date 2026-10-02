from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brw.common import ensure_dir, save_csv
from brw.evaluation import evaluate_binomial_dataset
from brw.parallel import map_tasks
from brw.plotting import finish
from brw.random_walk import generate_iid_counts, generate_mixture_counts
from brw.seeds import stable_seed
from brw.stats import bootstrap_balanced_accuracy


def _worker(task: dict) -> list[dict]:
    rows = []
    N, T, delta = int(task["N"]), int(task["T"]), float(task["delta_p"])
    p1, p2 = 0.5 - delta / 2.0, 0.5 + delta / 2.0
    true_model = task["true_model"]
    for rep in range(int(task["reps"])):
        seed = stable_seed(task["base_seed"], "e5", N, delta, true_model, rep)
        rng = np.random.default_rng(seed)
        if true_model == "M1":
            k = generate_iid_counts(rng, N, T, 0.5)
        else:
            k, _ = generate_mixture_counts(rng, N, T, p1, p2, 0.5)
        evidence_seed = stable_seed(task["evidence_seed"], "e5", seed)
        ev = evaluate_binomial_dataset(
            k,
            T,
            float(task["prior_a"]),
            task["evidence"],
            evidence_seed,
            include_baselines=True,
        )
        ev.update(
            {
                "experiment": "E5",
                "replicate": rep,
                "true_model": true_model,
                "N": N,
                "T": T,
                "delta_p": delta,
                "kappa": 2 * delta * math.sqrt(T),
                "data_seed": seed,
            }
        )
        rows.append(ev)
    return rows


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e5"]
    out = ensure_dir(out_root / "exp5_baselines")
    tasks = []
    for N in c["N"]:
        for d in c["delta_p"]:
            for tm in ["M1", "M2"]:
                tasks.append(
                    {
                        **c,
                        "N": N,
                        "delta_p": d,
                        "true_model": tm,
                        "base_seed": cfg["base_seed"],
                        "evidence_seed": cfg["evidence_seed"],
                        "evidence": cfg["evidence"],
                    }
                )
    chunks = map_tasks(_worker, tasks, jobs)
    df = save_csv([r for ch in chunks for r in ch], out / "raw.csv")
    methods = {"ML": "ml_selected", "AIC": "aic_selected", "BIC": "bic_selected", "Bayesian": "bayes_selected"}
    rows = []
    for (N, d), g in df.groupby(["N", "delta_p"]):
        for name, col in methods.items():
            point, lo, hi = bootstrap_balanced_accuracy(
                g["true_model"],
                g[col],
                seed=stable_seed(cfg["base_seed"], "e5", "boot", N, d, name),
            )
            m1 = g[g.true_model == "M1"]
            m2 = g[g.true_model == "M2"]
            rows.append(
                {
                    "N": N,
                    "delta_p": d,
                    "method": name,
                    "TPR": float(np.mean(m2[col] == "M2")),
                    "FPR": float(np.mean(m1[col] == "M2")),
                    "balanced_accuracy": point,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    for d, g0 in summary.groupby("delta_p"):
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        for method, g in g0.groupby("method"):
            g = g.sort_values("N")
            ax.plot(g["N"], g["balanced_accuracy"], marker="o", label=method)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("N")
        ax.set_ylabel("balanced accuracy")
        ax.set_title(f"E5: model-selection criteria, Δp={d:.2f}")
        ax.legend()
        finish(out / f"figure_baselines_delta_{d:.2f}.png")
    return {
        "rows": len(df),
        "unresolved_evidence_rows": int((df["evidence_numerical_status"] == "unresolved").sum()),
    }
