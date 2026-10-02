#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
JOBS="${JOBS:-4}"
python run_all.py --profile paper_v2 --experiments e1 --jobs "$JOBS"
python run_all.py --profile paper_v2 --experiments e3 --jobs "$JOBS"
python run_all.py --profile paper_v2 --experiments e5 --jobs "$JOBS"
python run_all.py --profile paper_v2 --experiments e6 --jobs "$JOBS"
python run_all.py --profile paper_v2 --experiments e7 --jobs "$JOBS"
python run_all.py --profile paper_v2 --experiments e0 --jobs "$JOBS"
bash scripts/audit_results.sh paper_v2
