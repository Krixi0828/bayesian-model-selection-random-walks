from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from brw.common import ensure_dir, save_csv
from brw.plotting import finish
from brw.random_walk import endpoint_from_counts, generate_iid_counts, simulate_occupation_proportion
from brw.seeds import stable_seed


def arcsine_cdf(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(x))


def empirical_ks_to_arcsine(x: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=float))
    n = len(xs)
    F = arcsine_cdf(xs)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - F)), np.max(np.abs(lower - F))))


def _bootstrap_slope(logT: np.ndarray, logmeans: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(logT)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(logT[idx])) < 2:
            continue
        vals.append(linregress(logT[idx], logmeans[idx]).slope)
    if not vals:
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(vals, [0.025, 0.975])))


def run(cfg: dict, out_root: Path) -> dict:
    c = cfg["e0"]
    base_seed = int(cfg["base_seed"])
    out = ensure_dir(out_root / "exp0_validation")

    diffusion_rows = []
    for T in c["diffusion_T"]:
        seed = stable_seed(base_seed, "e0", "diffusion", T)
        rng = np.random.default_rng(seed)
        k = generate_iid_counts(rng, int(c["diffusion_reps"]), int(T), 0.5)
        endpoints = endpoint_from_counts(k, int(T))
        abs_s = np.abs(endpoints)
        diffusion_rows.append({
            "T": int(T),
            "reps": int(c["diffusion_reps"]),
            "mean_abs_endpoint": float(abs_s.mean()),
            "sd_abs_endpoint": float(abs_s.std(ddof=1)),
            "se_abs_endpoint": float(abs_s.std(ddof=1) / math.sqrt(len(abs_s))),
            "asymptotic_mean_abs": float(math.sqrt(2.0 * T / math.pi)),
            "seed": seed,
        })
    dfd = save_csv(diffusion_rows, out / "diffusion_summary.csv")
    lr = linregress(np.log(dfd["T"].to_numpy(float)), np.log(dfd["mean_abs_endpoint"].to_numpy(float)))
    lo, hi = _bootstrap_slope(
        np.log(dfd["T"].to_numpy(float)),
        np.log(dfd["mean_abs_endpoint"].to_numpy(float)),
        5000,
        stable_seed(base_seed, "e0", "slope_bootstrap"),
    )
    slope_df = pd.DataFrame([{
        "estimated_alpha": float(lr.slope),
        "intercept": float(lr.intercept),
        "r_squared": float(lr.rvalue ** 2),
        "alpha_ci_low_bootstrap": lo,
        "alpha_ci_high_bootstrap": hi,
        "theory_alpha": 0.5,
    }])
    slope_df.to_csv(out / "diffusion_scaling_fit.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.loglog(dfd["T"], dfd["mean_abs_endpoint"], marker="o", label="simulation")
    ax.loglog(dfd["T"], dfd["asymptotic_mean_abs"], linestyle="--", label=r"$\sqrt{2T/\pi}$")
    fitted = np.exp(lr.intercept) * dfd["T"].to_numpy(float) ** lr.slope
    ax.loglog(dfd["T"], fitted, linestyle=":", label=fr"fit $\alpha={lr.slope:.3f}$")
    ax.set_xlabel("steps T")
    ax.set_ylabel(r"mean $|S_T|$")
    ax.set_title("Diffusion scaling validation")
    ax.legend()
    finish(out / "figure_diffusion_scaling.png")

    arc_rows = []
    arc_samples: dict[int, np.ndarray] = {}
    for T in c["arcsine_T"]:
        seed = stable_seed(base_seed, "e0", "arcsine", T)
        rng = np.random.default_rng(seed)
        A = simulate_occupation_proportion(rng, int(c["arcsine_reps"]), int(T))
        ks = empirical_ks_to_arcsine(A)
        arc_samples[int(T)] = A
        arc_rows.append({
            "T": int(T),
            "reps": int(c["arcsine_reps"]),
            "mean_A": float(A.mean()),
            "median_A": float(np.median(A)),
            "ks_distance_to_arcsine": ks,
            "seed": seed,
        })
    dfa = save_csv(arc_rows, out / "arcsine_summary.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    grid = np.linspace(0.0, 1.0, 1001)
    ax.plot(grid, arcsine_cdf(grid), linestyle="--", linewidth=2, label="arcsine CDF")
    for T, A in arc_samples.items():
        xs = np.sort(A)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", label=f"T={T}", alpha=0.85)
    ax.set_xlabel("occupation proportion A")
    ax.set_ylabel("CDF")
    ax.set_title("Occupation-time validation")
    ax.legend()
    finish(out / "figure_arcsine_ecdf.png")

    return {
        "diffusion_alpha": float(lr.slope),
        "diffusion_alpha_ci": [lo, hi],
        "max_arcsine_ks": float(dfa["ks_distance_to_arcsine"].max()),
    }
