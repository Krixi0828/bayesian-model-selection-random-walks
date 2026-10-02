from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from brw.likelihoods import (
    log_evidence_m1_binomial,
    log_evidence_m2_numeric,
    prior_draws_3d,
)
from brw.random_walk import (
    generate_iid_counts,
    generate_mixture_counts,
)
from brw.seeds import stable_seed


PROFILE = "paper_v2"

# Independent Sobol-QMC numerical reference.
INITIAL_BUDGET = 2**18       # 262,144
MAX_BUDGET = 2**20           # 1,048,576
N_SCRAMBLES = 4

# Escalate early if repeatability is not already very tight.
ESCALATE_SD = 0.02

# Final numerical-reference acceptance criterion.
FINAL_SD_MAX = 0.05

# Any Bayes factor close to zero receives the maximum-budget check.
NEAR_BOUNDARY_ABS_LOGBF = 0.50


def combine_log_evidence(log_values: np.ndarray) -> float:
    """
    Average independent evidence estimates on the evidence scale,
    then return the result in log space.
    """
    log_values = np.asarray(log_values, dtype=float)

    return float(
        logsumexp(log_values)
        - math.log(len(log_values))
    )


def regenerate_dataset(row: pd.Series) -> np.ndarray:
    """
    Reconstruct exactly the same dataset using the original data_seed.

    No new dataset is generated.  This is a numerical re-evaluation
    of the original observation only.
    """
    rng = np.random.default_rng(
        int(row["data_seed"])
    )

    N = int(row["N"])
    T = int(row["T"])

    if str(row["true_model"]) == "M1":
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


def run_sobol_scrambles(
    k: np.ndarray,
    T: int,
    prior_a: float,
    evidence_seed: int,
    data_seed: int,
    budget: int,
) -> np.ndarray:
    """
    Compute four independently scrambled Sobol-QMC estimates of Z_M2.
    """
    values = []

    for scramble in range(N_SCRAMBLES):

        seed = stable_seed(
            evidence_seed,
            "e2",
            "targeted_sobol_reference",
            data_seed,
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

        # prior_draws_3d is cached in the normal implementation.
        # These reference arrays are large and are not reused here,
        # so explicitly release them to avoid memory accumulation.
        prior_draws_3d.cache_clear()

    return np.asarray(
        values,
        dtype=float,
    )


def resolve_one(
    row: pd.Series,
    evidence_seed: int,
) -> dict:
    """
    Construct an independent numerical reference for one unresolved row.
    """
    k = regenerate_dataset(
        row
    )

    T = int(row["T"])
    prior_a = float(row["prior_a"])

    # M1 Binomial evidence is analytical.
    logz1 = log_evidence_m1_binomial(
        k,
        T,
        prior_a,
        prior_a,
    )

    # ------------------------------------------------------------
    # Stage 1: 2^18 Sobol points x 4 independent scrambles
    # ------------------------------------------------------------

    logs = run_sobol_scrambles(
        k,
        T,
        prior_a,
        evidence_seed,
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
        np.all(
            signs == signs[0]
        )
        and signs[0] != 0
    )

    budget = INITIAL_BUDGET

    # ------------------------------------------------------------
    # Stage 2:
    # Every near-boundary case is deliberately checked with 2^20.
    # Also escalate if the first-stage reference is not sufficiently
    # repeatable or different scrambles disagree in sign.
    # ------------------------------------------------------------

    if (
        sd > ESCALATE_SD
        or abs(logbf) < NEAR_BOUNDARY_ABS_LOGBF
        or not sign_stable
    ):
        logs = run_sobol_scrambles(
            k,
            T,
            prior_a,
            evidence_seed,
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
            np.all(
                signs == signs[0]
            )
            and signs[0] != 0
        )

        budget = MAX_BUDGET

    evidence_resolved = bool(
        sd <= FINAL_SD_MAX
    )

    if evidence_resolved:
        reference_status = "pass"
    else:
        reference_status = "unresolved"

    if sign_stable:
        decision_status = "stable"
    else:
        decision_status = "boundary_indeterminate"

    return {
        "data_seed":
            int(row["data_seed"]),

        "design":
            str(row["design"]),

        "replicate":
            int(row["replicate"]),

        "true_model":
            str(row["true_model"]),

        "N":
            int(row["N"]),

        "T":
            T,

        "delta_p":
            float(row["delta_p"]),

        "kappa":
            float(row["kappa"]),

        "original_logZ1_binomial":
            float(row["logZ1_binomial"]),

        "original_logZ2_binomial":
            float(row["logZ2_binomial"]),

        "original_logBF_binomial":
            float(row["logBF_binomial"]),

        "reference_logZ1_binomial":
            float(logz1),

        "reference_logZ2_binomial":
            float(combined),

        "reference_logBF_binomial":
            float(logbf),

        "reference_budget_per_scramble":
            int(budget),

        "reference_scrambles":
            int(N_SCRAMBLES),

        "reference_total_points":
            int(budget * N_SCRAMBLES),

        "reference_sd_logZ":
            float(sd),

        "reference_logZ_min":
            float(logs.min()),

        "reference_logZ_max":
            float(logs.max()),

        "reference_BF_min":
            float(
                (logs - logz1).min()
            ),

        "reference_BF_max":
            float(
                (logs - logz1).max()
            ),

        "reference_status":
            reference_status,

        "decision_status":
            decision_status,

        "scramble_sign_agreement":
            int(sign_stable),

        "original_reference_sign_agree":
            int(
                (
                    float(
                        row["logBF_binomial"]
                    ) > 0
                )
                ==
                (
                    logbf > 0
                )
            ),
    }


def find_checkpoint(
    checkpoint_dir: Path,
    data_seed: int,
) -> Path:
    """
    Identify the unique checkpoint chunk containing a data_seed.
    """
    matches = []

    for path in checkpoint_dir.glob(
        "chunk_*.csv"
    ):
        small = pd.read_csv(
            path,
            usecols=[
                "data_seed",
            ],
        )

        found = (
            small["data_seed"]
            .astype(np.int64)
            .eq(
                int(data_seed)
            )
            .any()
        )

        if found:
            matches.append(
                path
            )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one checkpoint "
            f"for data_seed={data_seed}; "
            f"found {len(matches)}."
        )

    return matches[0]


def patch_checkpoint(
    checkpoint_path: Path,
    resolution: pd.Series,
) -> None:
    """
    Replace only the numerical M2 Binomial evidence of one previously
    unresolved row.  The dataset itself is never changed.
    """
    df = pd.read_csv(
        checkpoint_path
    )

    seed = int(
        resolution["data_seed"]
    )

    mask = (
        df["data_seed"]
        .astype(np.int64)
        .eq(seed)
    )

    if int(mask.sum()) != 1:
        raise RuntimeError(
            f"Expected one matching row for seed={seed}"
        )

    idx = df.index[
        mask
    ][0]

    new_logz2 = float(
        resolution[
            "reference_logZ2_binomial"
        ]
    )

    logz1 = float(
        df.loc[
            idx,
            "logZ1_binomial",
        ]
    )

    new_logbf = float(
        new_logz2 - logz1
    )

    # Replace numerical evidence only.
    df.loc[
        idx,
        "logZ2_binomial",
    ] = new_logz2

    df.loc[
        idx,
        "logBF_binomial",
    ] = new_logbf

    # The combined reference has a sign, so retain it for the existing
    # pipeline.  decision_status separately records whether all independent
    # scrambles agreed on that sign.
    selected = (
        "M2"
        if new_logbf > 0
        else "M1"
    )

    df.loc[
        idx,
        "binomial_selected",
    ] = selected

    # Recompute all dependent E2 quantities.
    gaussian_logz2 = float(
        df.loc[
            idx,
            "logZ2_gaussian",
        ]
    )

    gaussian_logbf = float(
        df.loc[
            idx,
            "logBF_gaussian",
        ]
    )

    gaussian_selected = str(
        df.loc[
            idx,
            "gaussian_selected",
        ]
    )

    df.loc[
        idx,
        "abs_logZ2_error",
    ] = abs(
        gaussian_logz2
        - new_logz2
    )

    df.loc[
        idx,
        "abs_logBF_error",
    ] = abs(
        gaussian_logbf
        - new_logbf
    )

    df.loc[
        idx,
        "decision_agree",
    ] = int(
        selected
        == gaussian_selected
    )

    # Do not pretend that this was an importance-sampling estimate.
    df.loc[
        idx,
        "binomial_numerical_status",
    ] = "resolved_sobol_reference"

    # Preserve original IS diagnostics, but append explicit
    # reference-resolution diagnostics in new columns.
    df.loc[
        idx,
        "binomial_resolution_method",
    ] = "independent_sobol_qmc"

    df.loc[
        idx,
        "binomial_resolution_budget",
    ] = int(
        resolution[
            "reference_budget_per_scramble"
        ]
    )

    df.loc[
        idx,
        "binomial_resolution_scrambles",
    ] = int(
        resolution[
            "reference_scrambles"
        ]
    )

    df.loc[
        idx,
        "binomial_reference_sd_logZ",
    ] = float(
        resolution[
            "reference_sd_logZ"
        ]
    )

    df.loc[
        idx,
        "binomial_decision_stable",
    ] = int(
        resolution[
            "scramble_sign_agreement"
        ]
    )

    # Atomic replacement.
    tmp = checkpoint_path.with_suffix(
        checkpoint_path.suffix
        + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        checkpoint_path
    )


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    result_dir = (
        project_root
        / "results"
        / PROFILE
        / "exp2_exact_vs_gaussian"
    )

    raw_path = (
        result_dir
        / "raw.csv"
    )

    checkpoint_dir = (
        result_dir
        / "checkpoints"
    )

    if not raw_path.exists():
        raise SystemExit(
            f"Missing E2 raw result: {raw_path}"
        )

    if not checkpoint_dir.exists():
        raise SystemExit(
            f"Missing E2 checkpoint directory: {checkpoint_dir}"
        )

    raw = pd.read_csv(
        raw_path
    )

    unresolved = raw[
        raw[
            "binomial_numerical_status"
        ]
        == "unresolved"
    ].copy()

    print(
        "\n======================================="
    )
    print(
        "E2 targeted numerical resolution"
    )
    print(
        "======================================="
    )
    print(
        f"Unresolved Binomial rows: {len(unresolved)}"
    )

    if len(unresolved) == 0:
        print(
            "Nothing to resolve."
        )
        return

    config_path = (
        project_root
        / "results"
        / PROFILE
        / "config_snapshot.json"
    )

    if not config_path.exists():
        raise SystemExit(
            f"Missing config snapshot: {config_path}"
        )

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    evidence_seed = int(
        cfg["evidence_seed"]
    )

    results = []

    for counter, (_, row) in enumerate(
        unresolved.iterrows(),
        start=1,
    ):
        print()
        print(
            "---------------------------------------"
        )
        print(
            f"Resolving {counter}/{len(unresolved)}"
        )
        print(
            f"data_seed   = {int(row['data_seed'])}"
        )
        print(
            f"design      = {row['design']}"
        )
        print(
            f"true_model  = {row['true_model']}"
        )
        print(
            f"N            = {int(row['N'])}"
        )
        print(
            f"T            = {int(row['T'])}"
        )
        print(
            f"delta_p      = {float(row['delta_p']):.8f}"
        )
        print(
            f"kappa        = {float(row['kappa']):.8f}"
        )
        print(
            f"original BF  = {float(row['logBF_binomial']):.8f}"
        )

        result = resolve_one(
            row,
            evidence_seed,
        )

        results.append(
            result
        )

        print(
            f"reference BF = "
            f"{result['reference_logBF_binomial']:.8f}"
        )

        print(
            f"reference SD = "
            f"{result['reference_sd_logZ']:.8f}"
        )

        print(
            f"BF range     = "
            f"[{result['reference_BF_min']:.8f}, "
            f"{result['reference_BF_max']:.8f}]"
        )

        print(
            f"budget       = "
            f"{result['reference_budget_per_scramble']} "
            f"x {result['reference_scrambles']}"
        )

        print(
            f"evidence     = "
            f"{result['reference_status']}"
        )

        print(
            f"decision     = "
            f"{result['decision_status']}"
        )

    resolution_df = pd.DataFrame(
        results
    )

    resolution_path = (
        result_dir
        / "unresolved_resolution.csv"
    )

    resolution_df.to_csv(
        resolution_path,
        index=False,
    )

    print()
    print(
        "======================================="
    )
    print(
        f"Diagnostics written to:\n{resolution_path}"
    )

    # ------------------------------------------------------------
    # Hard gate:
    # do not modify production checkpoints unless every evidence
    # reference itself satisfies the preregistered repeatability bound.
    # ------------------------------------------------------------

    bad = resolution_df[
        resolution_df[
            "reference_status"
        ]
        != "pass"
    ]

    if len(bad) > 0:
        print()
        print(
            "STOP: at least one evidence reference is "
            "still numerically unresolved."
        )
        print(
            "No E2 checkpoint has been modified."
        )
        print(
            "Send unresolved_resolution.csv for review."
        )
        raise SystemExit(2)

    # ------------------------------------------------------------
    # Preserve the original raw result as an audit artifact.
    # ------------------------------------------------------------

    raw_backup = (
        result_dir
        / "raw_before_targeted_resolution.csv"
    )

    if not raw_backup.exists():
        shutil.copy2(
            raw_path,
            raw_backup,
        )

    # ------------------------------------------------------------
    # Patch only the checkpoint chunks containing the three rows.
    # ------------------------------------------------------------

    checkpoint_backup_dir = (
        result_dir
        / "checkpoints_before_targeted_resolution"
    )

    checkpoint_backup_dir.mkdir(
        exist_ok=True
    )

    for _, resolution in resolution_df.iterrows():

        checkpoint = find_checkpoint(
            checkpoint_dir,
            int(
                resolution[
                    "data_seed"
                ]
            ),
        )

        backup = (
            checkpoint_backup_dir
            / checkpoint.name
        )

        if not backup.exists():
            shutil.copy2(
                checkpoint,
                backup,
            )

        patch_checkpoint(
            checkpoint,
            resolution,
        )

        print(
            "Patched checkpoint: "
            f"{checkpoint.name}"
        )

    print()
    print(
        "All evidence references passed."
    )
    print(
        "Only the previously unresolved checkpoint rows "
        "were updated."
    )
    print()
    print(
        "Next step:"
    )
    print(
        "rerun E2 normally. Existing checkpoints will be "
        "reused, so the 8400 datasets will NOT be recomputed."
    )


if __name__ == "__main__":
    main()
