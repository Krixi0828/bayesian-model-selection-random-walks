#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/audit_results.py --profile "${1:-paper_v2}"
