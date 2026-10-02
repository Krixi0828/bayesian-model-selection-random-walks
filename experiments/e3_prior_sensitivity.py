from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brw.common import ensure_dir, save_csv
from brw.evaluation import evaluate_binomial_dataset
from brw.parallel import map_tasks
from brw.plotting import finish
from brw.random_walk import generate_iid_counts, generate_mixture_counts
from brw.seeds import stable_seed
from brw.stats import wilson_interval


def _worker(task: dict) -> list[dict]:
    rows = []
    N, T = int(task["N"]), int(task["T"])
    a = float(task["prior_a_value"])
    true_model = str(task["true_model"])
    delta = float(task["delta_p"])
    p1, p2 = 0.5 - delta / 2.0, 0.5 + delta / 2.0

    for rep in range(int(task["reps"])):
        # prior_a is deliberately excluded so the same dataset is reused across
        # priors (paired sensitivity comparison).
        data_seed = stable_seed(int(task["base_seed"]), "e3", true_model, delta, rep)
        rng = np.random.default_rng(data_seed)
        if true_model == "M1":
            k = generate_iid_counts(rng, N, T, 0.5)
        else:
            k, _ = generate_mixture_counts(rng, N, T, p1, p2, 0.5)
        evidence_seed = stable_seed(int(task["evidence_seed"]), "e3", true_model, delta, rep)
        ev = evaluate_binomial_dataset(k, T, a, task["evidence"], evidence_seed)
        rows.append(
            {
                "experiment": "E3",
                "replicate": rep,
                "data_seed": data_seed,
                "true_model": true_model,
                "N": N,
                "T": T,
                "delta_p": delta,
                "p1": p1,
                "p2": p2,
                "prior_a": a,
                **ev,
            }
        )
    return rows


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e3"]
    out = ensure_dir(out_root / "exp3_prior_sensitivity")
    tasks = []
    for a in c["prior_a"]:
        # One null scenario; delta_p is undefined under M1 and is encoded as 0.
        tasks.append(
            {
                **c,
                "prior_a_value": a,
                "true_model": "M1",
                "delta_p": 0.0,
                "base_seed": cfg["base_seed"],
                "evidence_seed": cfg["evidence_seed"],
                "evidence": cfg["evidence"],
            }
        )
        for delta in c["delta_p"]:
            tasks.append(
                {
                    **c,
                    "prior_a_value": a,
                    "true_model": "M2",
                    "delta_p": delta,
                    "base_seed": cfg["base_seed"],
                    "evidence_seed": cfg["evidence_seed"],
                    "evidence": cfg["evidence"],
                }
            )
    chunks = map_tasks(_worker, tasks, jobs)
    df = save_csv([r for ch in chunks for r in ch], out / "raw.csv")

    rows = []
    for (a, true_model, delta), g in df.groupby(["prior_a", "true_model", "delta_p"]):
        select_m2 = int((g["bayes_selected"] == "M2").sum())
        lo, hi = wilson_interval(select_m2, len(g))
        q1, med, q3 = np.quantile(g["logBF"], [0.25, 0.5, 0.75])
        rows.append(
            {
                "prior_a": a,
                "true_model": true_model,
                "delta_p": delta,
                "rows": len(g),
                "mean_logBF": float(g["logBF"].mean()),
                "median_logBF": float(med),
                "q1_logBF": float(q1),
                "q3_logBF": float(q3),
                "p_select_M2": select_m2 / len(g),
                "p_select_M2_ci_low": lo,
                "p_select_M2_ci_high": hi,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    m1 = summary[summary["true_model"] == "M1"].sort_values("prior_a")
    ax.plot(m1["prior_a"], m1["p_select_M2"], marker="o", label="M1 false-positive rate")
    for delta, g in summary[summary["true_model"] == "M2"].groupby("delta_p"):
        g = g.sort_values("prior_a")
        ax.plot(g["prior_a"], g["p_select_M2"], marker="o", label=f"M2 Δp={delta:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Beta prior concentration $a$")
    ax.set_ylabel("P(select M2)")
    ax.set_title("E3: prior sensitivity across regime separations")
    ax.legend()
    finish(out / "figure_prior_selection.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for (true_model, delta), g in summary.groupby(["true_model", "delta_p"]):
        g = g.sort_values("prior_a")
        label = "M1" if true_model == "M1" else f"M2 Δp={delta:.2f}"
        ax.plot(g["prior_a"], g["median_logBF"], marker="o", label=label)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Beta prior concentration $a$")
    ax.set_ylabel("median log Bayes factor")
    ax.set_title("E3: prior sensitivity of evidence")
    ax.legend()
    finish(out / "figure_prior_logbf.png")

    return {
        "rows": len(df),
        "unresolved_evidence_rows": int((df["evidence_numerical_status"] == "unresolved").sum()),
    }
