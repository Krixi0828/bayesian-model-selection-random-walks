from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brw.common import ensure_dir
from brw.evaluation import (
    evaluate_binomial_dataset,
    evaluate_gaussian_approx_dataset,
)
from brw.plotting import finish
from brw.random_walk import (
    generate_iid_counts,
    generate_mixture_counts,
)
from brw.seeds import stable_seed


# Execution-only parameter.
# This does NOT change the statistical design.
CHECKPOINT_REPS = 10

# Increment this only if checkpoint format changes.
CHECKPOINT_SCHEMA_VERSION = 1


def _worker(task: dict) -> list[dict]:
    rows = []

    N = int(task["N"])
    T = int(task["T"])
    design = str(task["design"])
    true_model = str(task["true_model"])
    delta = float(task["delta_p"])

    if task.get("kappa") is not None:
        kappa = float(task["kappa"])
    else:
        kappa = float(2.0 * delta * math.sqrt(T))

    p1 = 0.5 - delta / 2.0
    p2 = 0.5 + delta / 2.0

    rep_start = int(task["rep_start"])
    rep_end = int(task["rep_end"])

    for rep in range(rep_start, rep_end):
        data_seed = stable_seed(
            int(task["base_seed"]),
            "e2",
            design,
            T,
            delta,
            true_model,
            rep,
        )

        rng = np.random.default_rng(data_seed)

        if true_model == "M1":
            k = generate_iid_counts(
                rng,
                N,
                T,
                0.5,
            )
        else:
            k, _ = generate_mixture_counts(
                rng,
                N,
                T,
                p1,
                p2,
                0.5,
            )

        bin_seed = stable_seed(
            int(task["evidence_seed"]),
            "e2",
            "binomial",
            data_seed,
        )

        gau_seed = stable_seed(
            int(task["evidence_seed"]),
            "e2",
            "gaussian",
            data_seed,
        )

        binom = evaluate_binomial_dataset(
            k,
            T,
            float(task["prior_a"]),
            task["evidence"],
            bin_seed,
        )

        gauss = evaluate_gaussian_approx_dataset(
            k,
            T,
            float(task["prior_a"]),
            task["evidence"],
            gau_seed,
            likelihood="gaussian_cc",
        )

        rows.append(
            {
                "experiment": "E2",
                "design": design,
                "replicate": rep,
                "data_seed": data_seed,
                "true_model": true_model,
                "N": N,
                "T": T,
                "delta_p": delta,
                "kappa": kappa,
                "p1": p1,
                "p2": p2,

                "logZ1_binomial":
                    binom["logZ_M1"],

                "logZ2_binomial":
                    binom["logZ_M2"],

                "logZ1_gaussian":
                    gauss["logZ_M1"],

                "logZ2_gaussian":
                    gauss["logZ_M2"],

                "logBF_binomial":
                    binom["logBF"],

                "logBF_gaussian":
                    gauss["logBF"],

                "abs_logZ1_error":
                    abs(
                        gauss["logZ_M1"]
                        - binom["logZ_M1"]
                    ),

                "abs_logZ2_error":
                    abs(
                        gauss["logZ_M2"]
                        - binom["logZ_M2"]
                    ),

                "abs_logBF_error":
                    abs(
                        gauss["logBF"]
                        - binom["logBF"]
                    ),

                "binomial_selected":
                    binom["bayes_selected"],

                "gaussian_selected":
                    gauss["bayes_selected"],

                "decision_agree":
                    int(
                        binom["bayes_selected"]
                        == gauss["bayes_selected"]
                    ),

                "prior_a":
                    float(task["prior_a"]),

                "binomial_numerical_status":
                    binom[
                        "evidence_numerical_status"
                    ],

                "gaussian_numerical_status":
                    gauss[
                        "evidence_numerical_status"
                    ],

                "binomial_ess_fraction":
                    binom[
                        "evidence_ess_fraction"
                    ],

                "gaussian_ess_fraction":
                    gauss[
                        "evidence_ess_fraction"
                    ],

                "binomial_max_weight_fraction":
                    binom[
                        "evidence_max_weight_fraction"
                    ],

                "gaussian_max_weight_fraction":
                    gauss[
                        "evidence_max_weight_fraction"
                    ],

                "binomial_final_budget":
                    binom[
                        "evidence_final_budget"
                    ],

                "gaussian_final_budget":
                    gauss[
                        "evidence_final_budget"
                    ],
            }
        )

    return rows


def _source_hashes(
    project_root: Path,
) -> dict[str, str]:
    """
    Record hashes of all files that can alter E2
    numerical results.

    This prevents accidentally resuming checkpoints
    generated by a different code version.
    """

    rel_paths = [
        "experiments/e2_exact_vs_gaussian.py",
        "src/brw/evaluation.py",
        "src/brw/importance.py",
        "src/brw/likelihoods.py",
    ]

    hashes = {}

    for rel in rel_paths:
        path = project_root / rel
        hashes[rel] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    return hashes


def _run_fingerprint(
    cfg: dict,
    project_root: Path,
) -> tuple[str, dict]:
    """
    Fingerprint the complete configuration and
    numerical implementation used by E2.
    """

    payload = {
        "schema_version":
            CHECKPOINT_SCHEMA_VERSION,

        "checkpoint_reps":
            CHECKPOINT_REPS,

        "base_seed":
            cfg["base_seed"],

        "evidence_seed":
            cfg["evidence_seed"],

        "evidence":
            cfg["evidence"],

        "e2":
            cfg["e2"],

        "source_sha256":
            _source_hashes(project_root),
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    fingerprint = hashlib.sha256(
        raw
    ).hexdigest()

    return fingerprint, payload


def _task_id(task: dict) -> str:
    """
    Deterministic ID for one checkpoint chunk.
    """

    key = {
        "design":
            task["design"],

        "T":
            int(task["T"]),

        "delta_p":
            float(task["delta_p"]),

        "kappa":
            float(task["kappa"]),

        "true_model":
            task["true_model"],

        "rep_start":
            int(task["rep_start"]),

        "rep_end":
            int(task["rep_end"]),
    }

    raw = json.dumps(
        key,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha1(
        raw
    ).hexdigest()[:16]


def _checkpoint_path(
    checkpoint_dir: Path,
    task: dict,
) -> Path:
    return (
        checkpoint_dir
        / f"chunk_{_task_id(task)}.csv"
    )


def _checkpoint_is_valid(
    path: Path,
    task: dict,
) -> bool:
    """
    A checkpoint is accepted only if:
    1. it can be parsed;
    2. row count matches the requested chunk;
    3. replicate IDs exactly match.
    """

    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    expected = (
        int(task["rep_end"])
        - int(task["rep_start"])
    )

    if len(df) != expected:
        return False

    if "replicate" not in df.columns:
        return False

    expected_reps = list(
        range(
            int(task["rep_start"]),
            int(task["rep_end"]),
        )
    )

    actual_reps = (
        df["replicate"]
        .astype(int)
        .tolist()
    )

    return actual_reps == expected_reps


def _write_checkpoint_atomic(
    path: Path,
    rows: list[dict],
) -> None:
    """
    Atomic checkpoint write.

    If the process is interrupted during writing,
    the existing completed checkpoint will not
    become partially corrupted.
    """

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def _build_tasks(
    cfg: dict,
) -> list[dict]:
    c = cfg["e2"]

    conditions = []

    # -------------------------------------------------
    # E2-A:
    # Fixed Δp while trajectory length T changes.
    # -------------------------------------------------

    for T in c["T"]:
        delta = float(
            c["fixed_delta_p"]
        )

        for true_model in [
            "M1",
            "M2",
        ]:
            conditions.append(
                {
                    **c,

                    "design":
                        "fixed_delta",

                    "T":
                        T,

                    "delta_p":
                        delta,

                    "kappa":
                        2.0
                        * delta
                        * math.sqrt(T),

                    "true_model":
                        true_model,

                    "base_seed":
                        cfg["base_seed"],

                    "evidence_seed":
                        cfg["evidence_seed"],

                    "evidence":
                        cfg["evidence"],
                }
            )

    # -------------------------------------------------
    # E2-B:
    # Fixed standardized separation κ.
    #
    # M1 is independent of κ and is already
    # represented by E2-A.
    # -------------------------------------------------

    for kappa in c["fixed_kappa"]:
        for T in c["T"]:
            delta = (
                float(kappa)
                / (
                    2.0
                    * math.sqrt(T)
                )
            )

            conditions.append(
                {
                    **c,

                    "design":
                        "fixed_kappa",

                    "T":
                        T,

                    "delta_p":
                        delta,

                    "kappa":
                        float(kappa),

                    "true_model":
                        "M2",

                    "base_seed":
                        cfg["base_seed"],

                    "evidence_seed":
                        cfg["evidence_seed"],

                    "evidence":
                        cfg["evidence"],
                }
            )

    # -------------------------------------------------
    # Divide each condition into small independent
    # checkpoint chunks.
    # -------------------------------------------------

    tasks = []

    reps = int(
        c["reps"]
    )

    for condition in conditions:
        for rep_start in range(
            0,
            reps,
            CHECKPOINT_REPS,
        ):
            task = dict(
                condition
            )

            task["rep_start"] = (
                rep_start
            )

            task["rep_end"] = min(
                rep_start
                + CHECKPOINT_REPS,
                reps,
            )

            tasks.append(
                task
            )

    return tasks


def _run_pending_tasks(
    tasks: list[dict],
    checkpoint_dir: Path,
    jobs: int,
) -> None:

    complete = []
    pending = []

    for task in tasks:
        path = _checkpoint_path(
            checkpoint_dir,
            task,
        )

        if _checkpoint_is_valid(
            path,
            task,
        ):
            complete.append(
                task
            )

        else:
            # Remove corrupted/incomplete checkpoint.
            if path.exists():
                path.unlink()

            pending.append(
                task
            )

    total = len(tasks)
    completed_count = len(
        complete
    )

    print(
        "[E2] checkpoint status: "
        f"{completed_count}/{total} "
        "chunks already complete; "
        f"{len(pending)} pending.",
        flush=True,
    )

    if not pending:
        return

    def report_progress():
        pct = (
            100.0
            * completed_count
            / total
        )

        print(
            "[E2] progress: "
            f"{completed_count}/{total} "
            f"chunks ({pct:.1f}%)",
            flush=True,
        )

    # -------------------------------------------------
    # Sequential execution
    # -------------------------------------------------

    if jobs is None or jobs <= 1:
        try:
            for task in pending:
                rows = _worker(
                    task
                )

                _write_checkpoint_atomic(
                    _checkpoint_path(
                        checkpoint_dir,
                        task,
                    ),
                    rows,
                )

                completed_count += 1
                report_progress()

        except KeyboardInterrupt:
            print(
                "\n[E2] interrupted. "
                "Completed checkpoints are preserved. "
                "Rerun the same command to resume.",
                flush=True,
            )

            raise

        return

    # -------------------------------------------------
    # Parallel execution
    # -------------------------------------------------

    executor = ProcessPoolExecutor(
        max_workers=jobs
    )

    futures = {
        executor.submit(
            _worker,
            task,
        ): task
        for task in pending
    }

    try:
        for future in as_completed(
            futures
        ):
            task = futures[
                future
            ]

            rows = future.result()

            _write_checkpoint_atomic(
                _checkpoint_path(
                    checkpoint_dir,
                    task,
                ),
                rows,
            )

            completed_count += 1
            report_progress()

    except KeyboardInterrupt:
        print(
            "\n[E2] interrupt received. "
            "Completed checkpoints are preserved. "
            "Terminating worker processes...",
            flush=True,
        )

        for future in futures:
            future.cancel()

        # Python 3.14 provides terminate_workers(),
        # which avoids the long shutdown(wait=True)
        # behavior observed in the original E2 runner.
        if hasattr(
            executor,
            "terminate_workers",
        ):
            executor.terminate_workers()

        else:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

        raise

    else:
        executor.shutdown(
            wait=True
        )


def _collect_checkpoints(
    tasks: list[dict],
    checkpoint_dir: Path,
) -> pd.DataFrame:

    frames = []
    missing = []

    for task in tasks:
        path = _checkpoint_path(
            checkpoint_dir,
            task,
        )

        if not _checkpoint_is_valid(
            path,
            task,
        ):
            missing.append(
                path.name
            )
            continue

        frames.append(
            pd.read_csv(path)
        )

    if missing:
        raise RuntimeError(
            "E2 cannot finalize: "
            f"{len(missing)} checkpoint chunks "
            "are missing or invalid."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df.sort_values(
            [
                "design",
                "T",
                "true_model",
                "kappa",
                "delta_p",
                "replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


def run(
    cfg: dict,
    out_root: Path,
    jobs: int = 1,
) -> dict:

    out = ensure_dir(
        out_root
        / "exp2_exact_vs_gaussian"
    )

    checkpoint_dir = ensure_dir(
        out
        / "checkpoints"
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # -------------------------------------------------
    # Checkpoint provenance protection
    # -------------------------------------------------

    fingerprint, state_payload = (
        _run_fingerprint(
            cfg,
            project_root,
        )
    )

    state_path = (
        checkpoint_dir
        / "checkpoint_state.json"
    )

    if state_path.exists():
        state = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            state.get(
                "fingerprint"
            )
            != fingerprint
        ):
            raise RuntimeError(
                "Existing E2 checkpoints were created "
                "with different code/configuration. "
                "They must not be mixed with the "
                "current run. If you intentionally "
                "want to restart E2, remove only "
                "results/<profile>/"
                "exp2_exact_vs_gaussian/checkpoints."
            )

    else:
        state_path.write_text(
            json.dumps(
                {
                    "fingerprint":
                        fingerprint,

                    "payload":
                        state_payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    tasks = _build_tasks(
        cfg
    )

    expected_rows = sum(
        int(t["rep_end"])
        - int(t["rep_start"])
        for t in tasks
    )

    print(
        f"[E2] {len(tasks)} checkpoint chunks, "
        f"{expected_rows} total datasets, "
        f"chunk_size={CHECKPOINT_REPS}, "
        f"jobs={jobs}",
        flush=True,
    )

    _run_pending_tasks(
        tasks,
        checkpoint_dir,
        jobs,
    )

    # -------------------------------------------------
    # Finalize raw data only after every checkpoint
    # has been verified.
    # -------------------------------------------------

    df = _collect_checkpoints(
        tasks,
        checkpoint_dir,
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            "E2 row-count mismatch: "
            f"expected {expected_rows}, "
            f"got {len(df)}"
        )

    df.to_csv(
        out / "raw.csv",
        index=False,
    )

    # -------------------------------------------------
    # Summary statistics
    # -------------------------------------------------

    rows = []

    group_cols = [
        "design",
        "T",
        "true_model",
        "delta_p",
        "kappa",
    ]

    for keys, g in df.groupby(
        group_cols,
        dropna=False,
    ):
        (
            design,
            T,
            true_model,
            delta,
            kappa,
        ) = keys

        q1, med, q3 = np.quantile(
            g["abs_logBF_error"],
            [
                0.25,
                0.5,
                0.75,
            ],
        )

        rows.append(
            {
                "design":
                    design,

                "T":
                    int(T),

                "true_model":
                    true_model,

                "delta_p":
                    float(delta),

                "kappa":
                    float(kappa),

                "rows":
                    len(g),

                "median_abs_logZ1_error":
                    float(
                        g[
                            "abs_logZ1_error"
                        ].median()
                    ),

                "median_abs_logZ2_error":
                    float(
                        g[
                            "abs_logZ2_error"
                        ].median()
                    ),

                "median_abs_logBF_error":
                    float(med),

                "q1_abs_logBF_error":
                    float(q1),

                "q3_abs_logBF_error":
                    float(q3),

                "decision_agreement":
                    float(
                        g[
                            "decision_agree"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        out / "summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Figure: fixed Δp
    # -------------------------------------------------

    fixed = summary[
        summary["design"]
        == "fixed_delta"
    ]

    fig, ax = plt.subplots(
        figsize=(
            6.4,
            4.5,
        )
    )

    for true_model, g in (
        fixed.groupby(
            "true_model"
        )
    ):
        g = g.sort_values(
            "T"
        )

        ax.plot(
            g["T"],
            g[
                "median_abs_logBF_error"
            ],
            marker="o",
            label=true_model,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "trajectory length T"
    )

    ax.set_ylabel(
        r"median $|\Delta\log BF|$"
    )

    ax.set_title(
        "E2-A: fixed regime separation"
    )

    ax.legend()

    finish(
        out
        / "figure_fixed_delta_logbf_error.png"
    )

    # -------------------------------------------------
    # Figure: fixed κ
    # -------------------------------------------------

    fk = summary[
        summary["design"]
        == "fixed_kappa"
    ]

    fig, ax = plt.subplots(
        figsize=(
            6.4,
            4.5,
        )
    )

    for kappa, g in (
        fk.groupby(
            "kappa"
        )
    ):
        g = g.sort_values(
            "T"
        )

        ax.plot(
            g["T"],
            g[
                "median_abs_logBF_error"
            ],
            marker="o",
            label=fr"$\kappa={kappa:g}$",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "trajectory length T"
    )

    ax.set_ylabel(
        r"median $|\Delta\log BF|$"
    )

    ax.set_title(
        "E2-B: Gaussian error at "
        "fixed standardized separation"
    )

    ax.legend()

    finish(
        out
        / "figure_fixed_kappa_logbf_error.png"
    )

    # -------------------------------------------------
    # Figure: model-selection agreement
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            6.4,
            4.5,
        )
    )

    for design, g0 in (
        summary.groupby(
            "design"
        )
    ):
        gg = (
            g0.groupby(
                "T",
                as_index=False,
            )[
                "decision_agreement"
            ]
            .mean()
            .sort_values(
                "T"
            )
        )

        ax.plot(
            gg["T"],
            gg[
                "decision_agreement"
            ],
            marker="o",
            label=design,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_xlabel(
        "trajectory length T"
    )

    ax.set_ylabel(
        "model-selection agreement"
    )

    ax.set_title(
        "E2: Binomial versus Gaussian "
        "decision agreement"
    )

    ax.legend()

    finish(
        out
        / "figure_decision_agreement.png"
    )

    unresolved = int(
        (
            (
                df[
                    "binomial_numerical_status"
                ]
                == "unresolved"
            )
            |
            (
                df[
                    "gaussian_numerical_status"
                ]
                == "unresolved"
            )
        ).sum()
    )

    return {
        "rows":
            len(df),

        "unresolved_evidence_rows":
            unresolved,

        "checkpoint_chunks":
            len(tasks),

        "checkpoint_reps":
            CHECKPOINT_REPS,
    }
