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
from brw.summarize import balanced_accuracy_summary, selection_rate_summary


def _worker(task: dict) -> list[dict]:
    rows = []
    N, T, delta = int(task["N"]), int(task["T"]), float(task["delta_p"])
    true_model = task["true_model"]
    p1, p2 = 0.5 - delta / 2.0, 0.5 + delta / 2.0
    for rep in range(int(task["reps"])):
        data_seed = stable_seed(int(task["base_seed"]), "e1", N, T, delta, true_model, rep)
        rng = np.random.default_rng(data_seed)
        if true_model == "M1":
            k = generate_iid_counts(rng, N, T, 0.5)
        else:
            k, _ = generate_mixture_counts(rng, N, T, p1, p2, 0.5)
        evidence_seed = stable_seed(int(task["evidence_seed"]), "e1", data_seed)
        ev = evaluate_binomial_dataset(
            k,
            T,
            prior_a=float(task["prior_a"]),
            evidence_cfg=task["evidence"],
            evidence_seed=evidence_seed,
            include_baselines=False,
        )
        ev.update(
            {
                "experiment": "E1",
                "replicate": rep,
                "data_seed": data_seed,
                "true_model": true_model,
                "N": N,
                "T": T,
                "delta_p": delta,
                "p1": p1,
                "p2": p2,
                "mix_weight": 0.5,
                "kappa": 2.0 * delta * math.sqrt(T),
                "prior_a": float(task["prior_a"]),
            }
        )
        ev["bayes_correct"] = int(ev["bayes_selected"] == true_model)
        rows.append(ev)
    return rows


def _plot_heatmaps(summary: pd.DataFrame, out: Path) -> None:
    m2 = summary[summary["true_model"] == "M2"]
    for T in sorted(m2["T"].unique()):
        g = m2[m2["T"] == T]
        pivot = g.pivot(index="delta_p", columns="N", values="p_select_M2").sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("number of trajectories N")
        ax.set_ylabel(r"regime separation $\Delta p$")
        ax.set_title(f"E1: true-positive rate, T={int(T)}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, label="P(select M2 | true M2)")
        finish(out / f"figure_identifiability_heatmap_T{int(T)}.png")


def run(cfg: dict, out_root: Path, jobs: int = 1) -> dict:
    c = cfg["e1"]
    out = ensure_dir(out_root / "exp1_identifiability")
    tasks = []
    for N in c["N"]:
        for T in c["T"]:
            for delta in c["delta_p"]:
                for true_model in ["M1", "M2"]:
                    tasks.append(
                        {
                            "N": N,
                            "T": T,
                            "delta_p": delta,
                            "true_model": true_model,
                            "reps": c["reps"],
                            "prior_a": c["prior_a"],
                            "evidence": cfg["evidence"],
                            "base_seed": cfg["base_seed"],
                            "evidence_seed": cfg["evidence_seed"],
                        }
                    )
    chunks = map_tasks(_worker, tasks, jobs)
    df = save_csv([row for chunk in chunks for row in chunk], out / "raw.csv")

    group_cols = ["N", "T", "delta_p", "kappa"]
    sel = selection_rate_summary(df, group_cols)
    sel.to_csv(out / "selection_summary.csv", index=False)
    bal = balanced_accuracy_summary(
        df,
        group_cols,
        seed=stable_seed(cfg["base_seed"], "e1", "bootstrap"),
    )
    bal.to_csv(out / "balanced_accuracy_summary.csv", index=False)
    _plot_heatmaps(sel, out)

    m2 = sel[sel["true_model"] == "M2"].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for N, g in m2.groupby("N"):
        gg = g.groupby("kappa", as_index=False)["p_select_M2"].mean().sort_values("kappa")
        ax.plot(gg["kappa"], gg["p_select_M2"], marker="o", label=f"N={int(N)}")
    ax.set_xlabel(r"standardized separation $\kappa=2\Delta p\sqrt{T}$")
    ax.set_ylabel("P(select M2 | true M2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("E1: identifiability versus standardized separation")
    ax.legend()
    finish(out / "figure_kappa_detection.png")

    return {
        "rows": len(df),
        "min_balanced_accuracy": float(bal["balanced_accuracy"].min()),
        "max_balanced_accuracy": float(bal["balanced_accuracy"].max()),
        "unresolved_evidence_rows": int((df["evidence_numerical_status"] == "unresolved").sum()),
    }
