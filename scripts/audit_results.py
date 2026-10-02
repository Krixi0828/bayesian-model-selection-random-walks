from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _status_summary(df: pd.DataFrame, lines: list[str]) -> None:
    for col in [c for c in df.columns if c.endswith("numerical_status")]:
        counts = df[col].fillna("<NA>").value_counts().to_dict()
        lines.append(f"{col}: {counts}")
    if "evidence_reestimated" in df.columns:
        x = pd.to_numeric(df["evidence_reestimated"], errors="coerce").dropna()
        if len(x):
            lines.append(f"evidence_reestimated_rows={int((x==1).sum())}/{len(x)}")
    if "evidence_repeat_sd_logZ" in df.columns:
        x = pd.to_numeric(df["evidence_repeat_sd_logZ"], errors="coerce").dropna()
        if len(x):
            lines.append(f"evidence_repeat_sd_logZ_max={x.max():.6g} p95={x.quantile(.95):.6g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="paper_v2")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "results" / args.profile
    lines = [f"RESULT AUDIT: {args.profile}", f"base={base}"]
    if not base.exists():
        raise SystemExit(f"Missing {base}")

    manifest_path = base / "manifest.json"
    manifest_experiments = set()
    if manifest_path.exists():
        try:
            import json
            manifest_experiments = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("experiments", {}))
        except Exception:
            manifest_experiments = set()

    expected = [
        "exp0_validation",
        "exp1_identifiability",
        "exp2_exact_vs_gaussian",
        "exp3_prior_sensitivity",
        "exp4_convergence",
        "exp5_baselines",
        "exp6_imbalance",
        "exp7_misspecification",
    ]
    total_unresolved = 0
    for name in expected:
        d = base / name
        lines.append(f"\n[{name}]")
        if not d.exists():
            exp_id = name.split('_')[0].replace('exp','e')
            if exp_id not in manifest_experiments:
                lines.append("NOT RUN (not present in manifest)")
            else:
                lines.append("MISSING DIRECTORY")
            continue
        raw = d / "raw.csv"
        if not raw.exists():
            if name == "exp0_validation":
                lines.append(
                    "validation summaries present="
                    + str((d / "diffusion_summary.csv").exists() and (d / "arcsine_summary.csv").exists())
                )
                continue
            lines.append("MISSING raw.csv")
            continue

        df = pd.read_csv(raw)
        lines.append(f"rows={len(df)} cols={len(df.columns)}")
        numeric = df.select_dtypes(include=[np.number]).copy()
        # Some diagnostics are conditionally defined.
        #
        # E4:
        # MC/Sobol rows intentionally do not have importance-sampling
        # ESS / maximum-weight diagnostics.
        #
        # E2:
        # targeted Sobol-resolution diagnostics exist only for rows whose
        # production importance estimate was originally unresolved.
        optional = [
            c
            for c in numeric.columns
            if (
                c.startswith("estimator_ess")
                or c.startswith("estimator_max_weight")
                or c.startswith("binomial_resolution_")
                or c.startswith("binomial_reference_")
                or c == "binomial_decision_stable"
                or c.startswith("evidence_resolution_")
                or c.startswith("evidence_reference_")
                or c == "evidence_decision_stable"
            )
        ]

        numeric_required = numeric.drop(
            columns=optional,
            errors="ignore",
        )

        bad = (
            int(
                (
                    ~np.isfinite(
                        numeric_required.to_numpy()
                    )
                ).sum()
            )
            if numeric_required.size
            else 0
        )

        lines.append(
            f"unexpected_nonfinite_numeric_cells={bad}"
        )

        # Validate E2 targeted numerical resolutions separately instead of
        # treating their intentionally missing diagnostics as failures.
        if (
            name == "exp2_exact_vs_gaussian"
            and "binomial_numerical_status" in df.columns
        ):
            resolved = df[
                df["binomial_numerical_status"]
                == "resolved_sobol_reference"
            ]

            resolution_cols = [
                c
                for c in [
                    "binomial_resolution_budget",
                    "binomial_resolution_scrambles",
                    "binomial_reference_sd_logZ",
                    "binomial_decision_stable",
                ]
                if c in df.columns
            ]

            if len(resolved):
                resolution_numeric = (
                    resolved[resolution_cols]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                )

                resolution_bad = int(
                    (
                        ~np.isfinite(
                            resolution_numeric.to_numpy()
                        )
                    ).sum()
                )

                lines.append(
                    "targeted_resolution_rows="
                    f"{len(resolved)} "
                    "targeted_resolution_nonfinite_cells="
                    f"{resolution_bad}"
                )

        _status_summary(df, lines)
        for col in [c for c in df.columns if c.endswith("numerical_status")]:
            total_unresolved += int((df[col] == "unresolved").sum())

        ess_cols = [c for c in df.columns if c.endswith("ess_fraction")]
        for col in ess_cols:
            ess = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(ess):
                lines.append(
                    f"{col}: min={ess.min():.6g} median={ess.median():.6g} "
                    f"lt_0.01={(ess < 0.01).sum()} lt_0.05={(ess < 0.05).sum()}"
                )
        weight_cols = [c for c in df.columns if c.endswith("max_weight_fraction")]
        for col in weight_cols:
            w = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(w):
                lines.append(f"{col}: max={w.max():.6g} rows_gt_0.20={(w > 0.20).sum()}")

        if name == "exp4_convergence":
            refs = pd.read_csv(d / "reference_datasets.csv")
            lines.append(f"reference_between_run_sd_max={refs.reference_between_run_sd_logZ.max():.6g}")
            lines.append(f"reference_unresolved={(refs.reference_status != 'pass').sum()}/{len(refs)}")
            gate_path = d / "numerical_gate.csv"
            if gate_path.exists():
                gate = pd.read_csv(gate_path).iloc[0]
                lines.append(
                    "numerical_gate: "
                    f"pass={int(gate.numerical_gate_pass)} "
                    f"decisive_sign_error_rate={gate.decisive_sign_error_rate:.6g} "
                    f"p95_abs_logZ_error={gate.p95_abs_logZ_error:.6g} "
                    f"p95_target_met={int(gate.p95_target_met)}"
                )

    gate_path = base / "exp4_convergence" / "numerical_gate.csv"
    if gate_path.exists():
        gate_pass = int(pd.read_csv(gate_path).iloc[0]["numerical_gate_pass"])
        lines.append(f"\nE4_NUMERICAL_GATE_PASS={gate_pass}")
    else:
        lines.append("\nE4_NUMERICAL_GATE_PASS=NOT_AVAILABLE")
    lines.append(f"TOTAL_UNRESOLVED_EVIDENCE_ROWS={total_unresolved}")
    lines.append("manifest_present=" + ("yes" if (base / "manifest.json").exists() else "NO"))
    lines.append("config_snapshot_present=" + ("yes" if (base / "config_snapshot.json").exists() else "NO"))

    report = "\n".join(lines) + "\n"
    (base / "audit_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
