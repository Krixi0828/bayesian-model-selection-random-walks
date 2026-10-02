#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python - <<'PY'
import brw, numpy, pandas, scipy, matplotlib
print("Environment OK")
print("brw", brw.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
PY
