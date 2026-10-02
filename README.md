# Bayesian Random-Walk Model Selection Study — V2

This repository studies whether Bayesian model selection can distinguish a single random-walk regime from a latent two-regime mixture under finite data, prior sensitivity, approximation error, numerical marginal-likelihood estimation, minority regimes, and temporal-dependence misspecification.

V2 was created after a numerical audit of the pilot (`paper`) run. It fixes a Gaussian tail-cancellation bug and replaces the production M2 evidence estimator with a ridge-aware adaptive importance sampler. The original `smoke` and `paper` profiles are preserved for provenance; final research results must use `smoke_v2` / `paper_v2`.

## Read first

- `PROTOCOL_V2.md` — frozen research/numerical protocol.
- `CHANGELOG_V2.md` — why V2 exists and exactly what changed.
- `MIGRATION_V2.md` — how to replace the old project and run V2.
- `RUNBOOK_zh-TW.md` — Traditional Chinese execution checklist.

## Experiments

- **E0** simulator validation: diffusion exponent + arcsine occupation law.
- **E1** identifiability: `N`, `T`, `delta_p`, and standardized separation `kappa`.
- **E2** Binomial likelihood versus continuity-corrected Gaussian approximation; fixed-`delta_p` and fixed-`kappa` designs.
- **E3** prior sensitivity across hard/medium/easy latent-regime separations.
- **E4** numerical certification: prior MC, Sobol QMC, V1 local IS, V2 ridge-aware IS.
- **E5** ML / AIC / BIC / Bayesian model-selection behavior.
- **E6** minority-regime detectability across prevalence and separation.
- **E7** misspecification: persistent/anti-persistent walks and false latent heterogeneity.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Do not continue if tests fail.

## V2 execution order

### Smoke the repaired numerical components

```bash
python run_all.py --profile smoke_v2 --experiments e2,e4 --jobs 2
bash scripts/audit_results.sh smoke_v2
```

### Formal E4 numerical gate

```bash
JOBS=4 bash scripts/run_v2_gate.sh
```

Open `results/paper_v2/audit_report.txt`. Continue only if:

```text
E4_NUMERICAL_GATE_PASS=1
```

### Then run E2 and the scientific experiments

```bash
python run_all.py --profile paper_v2 --experiments e2 --jobs 4 | tee e2_v2.log
bash scripts/audit_results.sh paper_v2

JOBS=4 bash scripts/run_paper_v2.sh
```

The helper runs E1, E3, E5, E6, E7, and E0 after the separate E2 validation step.

### Final audit and bundle

```bash
bash scripts/audit_results.sh paper_v2
bash scripts/make_results_bundle.sh paper_v2
```

Send back `results_bundle_paper_v2.zip`. If any warning/error appears, also send the relevant log.

## Reproducibility

- Never overwrite `results/paper/` from V1.
- Do not change `configs/profiles.json` after beginning `paper_v2`.
- Do not rerun selected “bad-looking” data seeds.
- Numerical re-estimation uses the same dataset and only changes the integration budget/seed according to the frozen protocol.
- `unresolved` evidence rows are retained and reported, never silently discarded.
- Every V2 result directory contains `environment.json`, `config_snapshot.json`, and `manifest.json`.

## Statistical note on BIC

BIC is included only as a conventional penalized-likelihood baseline. Finite mixtures are singular models, so the usual regular-model asymptotic justification of standard BIC does not strictly apply.
