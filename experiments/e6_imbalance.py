from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brw.common import ensure_dir, save_csv
from brw.evaluation import evaluate_binomial_dataset
from brw.parallel import map_tasks
from brw.plotting import finish
from brw.random_walk import generate_mixture_counts
from brw.seeds import stable_seed
from brw.stats import wilson_interval


def _worker(task: dict) -> list[dict]:
    rows = []
    N, T = int(task["N"]), int(task["T"])
    w = float(task["minority_weight_value"])
    delta = float(task["delta_p_value"])
    p1, p2 = 0.5 - delta / 2.0, 0.5 + delta / 2.0
    for rep in range(int(task["reps"])):
        seed = stable_seed(task["base_seed"], "e6", N, delta, w, rep)
        rng = np.random.default_rng(seed)
        k, z = generate_mixture_counts(rng, N, T, p1, p2, w)
        evidence_seed = stable_seed(task["evidence_seed"], "e6", seed)
        ev = evaluate_binomial_dataset(k, T, float(task["prior_a"]), task["evidence"], evidence_seed)
        rows.append(
            {
                "experiment": "E6",
                "replicate": rep,
                "data_seed": seed,
                "true_model": "M2",
                "N": N,
                "T": T,
                "delta_p": delta,
                "p1": p1,
                "p2": p2,
                "minority_weight": w,
                "realized_minor_fraction": float(z.mean()),
                **ev,
            }
        )
    return rows


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e6"]
    out = ensure_dir(out_root / "exp6_imbalance")
    tasks = []
    for N in c["N"]:
        for delta in c["delta_p"]:
            for w in c["minority_weight"]:
                tasks.append(
                    {
                        **c,
                        "N": N,
                        "delta_p_value": delta,
                        "minority_weight_value": w,
                        "base_seed": cfg["base_seed"],
                        "evidence_seed": cfg["evidence_seed"],
                        "evidence": cfg["evidence"],
                    }
                )
    chunks = map_tasks(_worker, tasks, jobs)
    df = save_csv([r for ch in chunks for r in ch], out / "raw.csv")
    rows = []
    for (N, delta, w), g in df.groupby(["N", "delta_p", "minority_weight"]):
        s = int((g.bayes_selected == "M2").sum())
        lo, hi = wilson_interval(s, len(g))
        rows.append(
            {
                "N": N,
                "delta_p": delta,
                "minority_weight": w,
                "TPR": s / len(g),
                "ci_low": lo,
                "ci_high": hi,
                "median_logBF": float(g.logBF.median()),
                "rows": len(g),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    for delta, g0 in summary.groupby("delta_p"):
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        for N, g in g0.groupby("N"):
            g = g.sort_values("minority_weight")
            ax.plot(g["minority_weight"], g["TPR"], marker="o", label=f"N={int(N)}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("minority regime weight")
        ax.set_ylabel("P(select M2 | true M2)")
        ax.set_title(f"E6: minority-regime detection, Δp={delta:.2f}")
        ax.legend()
        finish(out / f"figure_imbalance_delta_{delta:.2f}.png")
    return {
        "rows": len(df),
        "unresolved_evidence_rows": int((df["evidence_numerical_status"] == "unresolved").sum()),
    }
