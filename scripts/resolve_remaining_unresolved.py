from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from brw.likelihoods import (
    log_evidence_m1_binomial,
    log_evidence_m2_numeric,
    prior_draws_3d,
)
from brw.plotting import finish
from brw.random_walk import (
    generate_iid_counts,
    generate_mixture_counts,
)
from brw.seeds import stable_seed
from brw.stats import (
    bootstrap_balanced_accuracy,
    wilson_interval,
)


PROFILE = "paper_v2"

INITIAL_BUDGET = 2**18
MAX_BUDGET = 2**20
N_SCRAMBLES = 4

ESCALATE_SD = 0.02
FINAL_SD_MAX = 0.05
NEAR_BOUNDARY_ABS_LOGBF = 0.50


EXPERIMENTS = {
    "E3": "exp3_prior_sensitivity",
    "E5": "exp5_baselines",
    "E6": "exp6_imbalance",
}


def combine_log_evidence(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    return float(
        logsumexp(values)
        - math.log(len(values))
    )


def regenerate_dataset(
    experiment: str,
    row: pd.Series,
) -> np.ndarray:

    rng = np.random.default_rng(
        int(row["data_seed"])
    )

    N = int(row["N"])
    T = int(row["T"])

    if experiment == "E3":
        if row["true_model"] == "M1":
            return generate_iid_counts(
                rng,
                N,
                T,
                0.5,
            )

        k, _ = generate_mixture_counts(
            rng,
            N,
            T,
            float(row["p1"]),
            float(row["p2"]),
            0.5,
        )

        return k

    if experiment == "E5":
        delta = float(row["delta_p"])

        p1 = 0.5 - delta / 2.0
        p2 = 0.5 + delta / 2.0

        if row["true_model"] == "M1":
            return generate_iid_counts(
                rng,
                N,
                T,
                0.5,
            )

        k, _ = generate_mixture_counts(
            rng,
            N,
            T,
            p1,
            p2,
            0.5,
        )

        return k

    if experiment == "E6":
        k, _ = generate_mixture_counts(
            rng,
            N,
            T,
            float(row["p1"]),
            float(row["p2"]),
            float(row["minority_weight"]),
        )

        return k

    raise ValueError(
        f"Unknown experiment: {experiment}"
    )


def get_prior_a(
    experiment: str,
    row: pd.Series,
    cfg: dict,
) -> float:

    if experiment == "E3":
        return float(row["prior_a"])

    if experiment == "E5":
        return float(
            cfg["e5"]["prior_a"]
        )

    if experiment == "E6":
        return float(
            cfg["e6"]["prior_a"]
        )

    raise ValueError(experiment)


def sobol_runs(
    k: np.ndarray,
    T: int,
    prior_a: float,
    evidence_seed: int,
    experiment: str,
    data_seed: int,
    budget: int,
) -> np.ndarray:

    values = []

    for scramble in range(N_SCRAMBLES):

        seed = stable_seed(
            evidence_seed,
            "remaining_resolution",
            experiment,
            data_seed,
            prior_a,
            budget,
            scramble,
        )

        logz = log_evidence_m2_numeric(
            k,
            T,
            budget,
            a=prior_a,
            b=prior_a,
            pi_a=1.0,
            pi_b=1.0,
            method="sobol",
            seed=seed,
            likelihood="binomial",
        )

        values.append(
            float(logz)
        )

        # Avoid retaining large 2^20 Sobol arrays.
        prior_draws_3d.cache_clear()

    return np.asarray(
        values,
        dtype=float,
    )


def resolve_one(
    experiment: str,
    row_index: int,
    row: pd.Series,
    cfg: dict,
) -> dict:

    k = regenerate_dataset(
        experiment,
        row,
    )

    T = int(row["T"])

    prior_a = get_prior_a(
        experiment,
        row,
        cfg,
    )

    logz1 = log_evidence_m1_binomial(
        k,
        T,
        prior_a,
        prior_a,
    )

    logs = sobol_runs(
        k,
        T,
        prior_a,
        int(cfg["evidence_seed"]),
        experiment,
        int(row["data_seed"]),
        INITIAL_BUDGET,
    )

    combined = combine_log_evidence(
        logs
    )

    sd = float(
        np.std(
            logs,
            ddof=1,
        )
    )

    logbf = float(
        combined - logz1
    )

    signs = np.sign(
        logs - logz1
    )

    sign_stable = bool(
        np.all(signs == signs[0])
        and signs[0] != 0
    )

    budget = INITIAL_BUDGET

    if (
        sd > ESCALATE_SD
        or abs(logbf)
        < NEAR_BOUNDARY_ABS_LOGBF
        or not sign_stable
    ):
        logs = sobol_runs(
            k,
            T,
            prior_a,
            int(cfg["evidence_seed"]),
            experiment,
            int(row["data_seed"]),
            MAX_BUDGET,
        )

        combined = combine_log_evidence(
            logs
        )

        sd = float(
            np.std(
                logs,
                ddof=1,
            )
        )

        logbf = float(
            combined - logz1
        )

        signs = np.sign(
            logs - logz1
        )

        sign_stable = bool(
            np.all(signs == signs[0])
            and signs[0] != 0
        )

        budget = MAX_BUDGET

    # Numerical convergence and decision stability are different.
    #
    # The marginal-likelihood estimate is numerically resolved once
    # independent Sobol estimates agree sufficiently in log-evidence.
    # If log BF is extremely close to zero, different scrambles may
    # straddle zero even though the evidence estimate itself has
    # converged. That is a genuine model near-tie, not numerical failure.
    reference_pass = bool(
        sd <= FINAL_SD_MAX
    )

    return {
        "experiment":
            experiment,

        "row_index":
            int(row_index),

        "data_seed":
            int(row["data_seed"]),

        "true_model":
            str(row["true_model"]),

        "N":
            int(row["N"]),

        "T":
            int(row["T"]),

        "original_logZ_M2":
            float(row["logZ_M2"]),

        "original_logBF":
            float(row["logBF"]),

        "reference_logZ_M1":
            float(logz1),

        "reference_logZ_M2":
            float(combined),

        "reference_logBF":
            float(logbf),

        "budget_per_scramble":
            int(budget),

        "scrambles":
            int(N_SCRAMBLES),

        "total_sobol_points":
            int(budget * N_SCRAMBLES),

        "reference_sd_logZ":
            float(sd),

        "BF_min":
            float(
                (logs - logz1).min()
            ),

        "BF_max":
            float(
                (logs - logz1).max()
            ),

        "sign_stable":
            int(sign_stable),

        "reference_status":
            (
                "pass"
                if reference_pass
                else "unresolved"
            ),

        "decision_status":
            (
                "stable"
                if sign_stable
                else "boundary_indeterminate"
            ),

        "original_reference_sign_agree":
            int(
                (
                    float(row["logBF"]) > 0
                )
                ==
                (
                    logbf > 0
                )
            ),
    }


def rebuild_e3(
    df: pd.DataFrame,
    out: Path,
) -> None:

    rows = []

    for (
        a,
        true_model,
        delta,
    ), g in df.groupby(
        [
            "prior_a",
            "true_model",
            "delta_p",
        ]
    ):

        select_m2 = int(
            (
                g["bayes_selected"]
                == "M2"
            ).sum()
        )

        lo, hi = wilson_interval(
            select_m2,
            len(g),
        )

        q1, med, q3 = np.quantile(
            g["logBF"],
            [0.25, 0.5, 0.75],
        )

        rows.append(
            {
                "prior_a": a,
                "true_model": true_model,
                "delta_p": delta,
                "rows": len(g),
                "mean_logBF":
                    float(g["logBF"].mean()),
                "median_logBF":
                    float(med),
                "q1_logBF":
                    float(q1),
                "q3_logBF":
                    float(q3),
                "p_select_M2":
                    select_m2 / len(g),
                "p_select_M2_ci_low":
                    lo,
                "p_select_M2_ci_high":
                    hi,
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out / "summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(6.6, 4.7)
    )

    m1 = summary[
        summary["true_model"] == "M1"
    ].sort_values("prior_a")

    ax.plot(
        m1["prior_a"],
        m1["p_select_M2"],
        marker="o",
        label="M1 false-positive rate",
    )

    for delta, g in (
        summary[
            summary["true_model"] == "M2"
        ]
        .groupby("delta_p")
    ):
        g = g.sort_values(
            "prior_a"
        )

        ax.plot(
            g["prior_a"],
            g["p_select_M2"],
            marker="o",
            label=f"M2 Δp={delta:.2f}",
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        r"Beta prior concentration $a$"
    )
    ax.set_ylabel("P(select M2)")
    ax.set_title(
        "E3: prior sensitivity across "
        "regime separations"
    )
    ax.legend()

    finish(
        out / "figure_prior_selection.png"
    )

    fig, ax = plt.subplots(
        figsize=(6.6, 4.7)
    )

    for (
        true_model,
        delta,
    ), g in summary.groupby(
        [
            "true_model",
            "delta_p",
        ]
    ):
        g = g.sort_values(
            "prior_a"
        )

        label = (
            "M1"
            if true_model == "M1"
            else f"M2 Δp={delta:.2f}"
        )

        ax.plot(
            g["prior_a"],
            g["median_logBF"],
            marker="o",
            label=label,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Beta prior concentration $a$"
    )
    ax.set_ylabel(
        "median log Bayes factor"
    )
    ax.set_title(
        "E3: prior sensitivity of evidence"
    )
    ax.legend()

    finish(
        out / "figure_prior_logbf.png"
    )


def rebuild_e5(
    df: pd.DataFrame,
    out: Path,
    cfg: dict,
) -> None:

    methods = {
        "ML": "ml_selected",
        "AIC": "aic_selected",
        "BIC": "bic_selected",
        "Bayesian": "bayes_selected",
    }

    rows = []

    for (
        N,
        delta,
    ), g in df.groupby(
        [
            "N",
            "delta_p",
        ]
    ):

        for name, col in methods.items():

            point, lo, hi = (
                bootstrap_balanced_accuracy(
                    g["true_model"],
                    g[col],
                    seed=stable_seed(
                        cfg["base_seed"],
                        "e5",
                        "boot",
                        N,
                        delta,
                        name,
                    ),
                )
            )

            m1 = g[
                g["true_model"] == "M1"
            ]

            m2 = g[
                g["true_model"] == "M2"
            ]

            rows.append(
                {
                    "N": N,
                    "delta_p": delta,
                    "method": name,
                    "TPR":
                        float(
                            np.mean(
                                m2[col]
                                == "M2"
                            )
                        ),
                    "FPR":
                        float(
                            np.mean(
                                m1[col]
                                == "M2"
                            )
                        ),
                    "balanced_accuracy":
                        point,
                    "ci_low":
                        lo,
                    "ci_high":
                        hi,
                }
            )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out / "summary.csv",
        index=False,
    )

    for delta, g0 in (
        summary.groupby("delta_p")
    ):

        fig, ax = plt.subplots(
            figsize=(6.4, 4.5)
        )

        for method, g in (
            g0.groupby("method")
        ):
            g = g.sort_values("N")

            ax.plot(
                g["N"],
                g["balanced_accuracy"],
                marker="o",
                label=method,
            )

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("N")
        ax.set_ylabel(
            "balanced accuracy"
        )
        ax.set_title(
            "E5: model-selection criteria, "
            f"Δp={delta:.2f}"
        )
        ax.legend()

        finish(
            out
            / (
                "figure_baselines_delta_"
                f"{delta:.2f}.png"
            )
        )


def rebuild_e6(
    df: pd.DataFrame,
    out: Path,
) -> None:

    rows = []

    for (
        N,
        delta,
        weight,
    ), g in df.groupby(
        [
            "N",
            "delta_p",
            "minority_weight",
        ]
    ):

        selected = int(
            (
                g["bayes_selected"]
                == "M2"
            ).sum()
        )

        lo, hi = wilson_interval(
            selected,
            len(g),
        )

        rows.append(
            {
                "N": N,
                "delta_p": delta,
                "minority_weight": weight,
                "TPR":
                    selected / len(g),
                "ci_low":
                    lo,
                "ci_high":
                    hi,
                "median_logBF":
                    float(
                        g["logBF"].median()
                    ),
                "rows":
                    len(g),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out / "summary.csv",
        index=False,
    )

    for delta, g0 in (
        summary.groupby("delta_p")
    ):

        fig, ax = plt.subplots(
            figsize=(6.4, 4.5)
        )

        for N, g in g0.groupby("N"):

            g = g.sort_values(
                "minority_weight"
            )

            ax.plot(
                g["minority_weight"],
                g["TPR"],
                marker="o",
                label=f"N={int(N)}",
            )

        ax.set_ylim(-0.02, 1.02)

        ax.set_xlabel(
            "minority regime weight"
        )

        ax.set_ylabel(
            "P(select M2 | true M2)"
        )

        ax.set_title(
            "E6: minority-regime detection, "
            f"Δp={delta:.2f}"
        )

        ax.legend()

        finish(
            out
            / (
                "figure_imbalance_delta_"
                f"{delta:.2f}.png"
            )
        )


def main() -> None:

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    profile_root = (
        root
        / "results"
        / PROFILE
    )

    config_path = (
        profile_root
        / "config_snapshot.json"
    )

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    frames = {}
    resolutions = []

    print()
    print(
        "======================================="
    )
    print(
        "Remaining targeted numerical resolution"
    )
    print(
        "======================================="
    )

    for experiment, folder in (
        EXPERIMENTS.items()
    ):

        raw_path = (
            profile_root
            / folder
            / "raw.csv"
        )

        df = pd.read_csv(
            raw_path
        )

        frames[experiment] = df

        unresolved = df[
            df[
                "evidence_numerical_status"
            ]
            == "unresolved"
        ]

        print(
            f"{experiment}: "
            f"{len(unresolved)} unresolved rows"
        )

        for row_index, row in (
            unresolved.iterrows()
        ):

            print()
            print(
                "---------------------------------------"
            )

            print(
                f"{experiment} "
                f"row={row_index}"
            )

            print(
                f"data_seed={int(row['data_seed'])}"
            )

            print(
                "original logBF="
                f"{float(row['logBF']):.8f}"
            )

            resolution = resolve_one(
                experiment,
                int(row_index),
                row,
                cfg,
            )

            resolutions.append(
                resolution
            )

            print(
                "reference logBF="
                f"{resolution['reference_logBF']:.8f}"
            )

            print(
                "reference SD="
                f"{resolution['reference_sd_logZ']:.8f}"
            )

            print(
                "BF range="
                f"[{resolution['BF_min']:.8f}, "
                f"{resolution['BF_max']:.8f}]"
            )

            print(
                "budget="
                f"{resolution['budget_per_scramble']}"
                f" x {resolution['scrambles']}"
            )

            print(
                "status="
                f"{resolution['reference_status']}"
            )

    resolution_df = pd.DataFrame(
        resolutions
    )

    resolution_path = (
        profile_root
        / "remaining_unresolved_resolution.csv"
    )

    resolution_df.to_csv(
        resolution_path,
        index=False,
    )

    print()
    print(
        "Resolution diagnostics written to:"
    )
    print(
        resolution_path
    )

    unresolved_ref = resolution_df[
        resolution_df[
            "reference_status"
        ]
        != "pass"
    ]

    if len(unresolved_ref):
        print()
        print(
            "STOP: at least one high-budget "
            "reference remains unresolved."
        )
        print(
            "No production raw.csv files "
            "were modified."
        )
        raise SystemExit(2)

    # --------------------------------------------------
    # Only after ALL references pass do we patch results.
    # --------------------------------------------------

    for experiment, folder in (
        EXPERIMENTS.items()
    ):

        out = (
            profile_root
            / folder
        )

        raw_path = (
            out
            / "raw.csv"
        )

        df = frames[
            experiment
        ].copy()

        backup = (
            out
            / "raw_before_targeted_resolution.csv"
        )

        if not backup.exists():
            shutil.copy2(
                raw_path,
                backup,
            )

        exp_resolutions = resolution_df[
            resolution_df[
                "experiment"
            ]
            == experiment
        ]

        for _, r in (
            exp_resolutions.iterrows()
        ):

            idx = int(
                r["row_index"]
            )

            logz2 = float(
                r["reference_logZ_M2"]
            )

            logz1 = float(
                df.loc[
                    idx,
                    "logZ_M1",
                ]
            )

            logbf = (
                logz2
                - logz1
            )

            df.loc[
                idx,
                "logZ_M2",
            ] = logz2

            df.loc[
                idx,
                "logBF",
            ] = logbf

            df.loc[
                idx,
                "bayes_selected",
            ] = (
                "M2"
                if logbf > 0
                else "M1"
            )

            decision_stable = bool(
                int(r["sign_stable"])
            )

            df.loc[
                idx,
                "evidence_numerical_status",
            ] = (
                "resolved_sobol_reference"
                if decision_stable
                else "resolved_sobol_boundary"
            )

            # Preserve targeted-resolution diagnostics explicitly.
            df.loc[
                idx,
                "evidence_resolution_method",
            ] = "independent_sobol_qmc"

            df.loc[
                idx,
                "evidence_decision_stable",
            ] = int(decision_stable)

            df.loc[
                idx,
                "evidence_reference_sd_logZ",
            ] = float(
                r["reference_sd_logZ"]
            )

            df.loc[
                idx,
                "evidence_resolution_budget",
            ] = int(
                r["budget_per_scramble"]
            )

            df.loc[
                idx,
                "evidence_resolution_scrambles",
            ] = int(
                r["scrambles"]
            )

        df.to_csv(
            raw_path,
            index=False,
        )

        if experiment == "E3":
            rebuild_e3(
                df,
                out,
            )

        elif experiment == "E5":
            rebuild_e5(
                df,
                out,
                cfg,
            )

        elif experiment == "E6":
            rebuild_e6(
                df,
                out,
            )

    print()
    print(
        "======================================="
    )
    print(
        "All remaining references passed."
    )
    print(
        "E3/E5/E6 raw results, summaries, "
        "and figures were updated."
    )
    print(
        "======================================="
    )


if __name__ == "__main__":
    main()

