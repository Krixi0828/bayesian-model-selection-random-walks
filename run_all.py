from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from brw.common import ensure_dir, write_environment
from brw.config import load_profile
from experiments import (
    e0_validation,
    e1_identifiability,
    e2_exact_vs_gaussian,
    e3_prior_sensitivity,
    e4_convergence,
    e5_baselines,
    e6_imbalance,
    e7_misspecification,
)

EXPERIMENTS = {
    "e0": e0_validation.run,
    "e1": e1_identifiability.run,
    "e2": e2_exact_vs_gaussian.run,
    "e3": e3_prior_sensitivity.run,
    "e4": e4_convergence.run,
    "e5": e5_baselines.run,
    "e6": e6_imbalance.run,
    "e7": e7_misspecification.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian random-walk research experiments")
    parser.add_argument("--profile", default="smoke_v2")
    parser.add_argument("--experiments", default="e0,e1,e2,e3,e4,e5,e6,e7", help="comma-separated experiment IDs")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    try:
        cfg = load_profile(args.profile, root)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    out = ensure_dir(root / args.out / args.profile)
    (out / "config_snapshot.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_environment(out / "environment.json", args.profile, int(cfg["base_seed"]), {"jobs": args.jobs})

    selected = [x.strip() for x in args.experiments.split(",") if x.strip()]
    unknown = [x for x in selected if x not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}")

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("experiments", {})
            manifest["profile"] = args.profile
            manifest.setdefault("started_unix", time.time())
        except Exception:
            manifest = {"profile": args.profile, "experiments": {}, "started_unix": time.time()}
    else:
        manifest = {"profile": args.profile, "experiments": {}, "started_unix": time.time()}

    for exp_id in selected:
        print(f"\n=== Running {exp_id.upper()} ({args.profile}) ===", flush=True)
        t0 = time.time()
        if exp_id == "e0":
            summary = EXPERIMENTS[exp_id](cfg, out)
        else:
            summary = EXPERIMENTS[exp_id](cfg, out, jobs=args.jobs)
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = elapsed
        manifest["experiments"][exp_id] = summary
        print(f"{exp_id.upper()} finished in {elapsed:.2f}s: {summary}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest["finished_unix"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll requested experiments completed. Results: {out}")


if __name__ == "__main__":
    main()
