#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-paper_v2}"
BASE="results/${PROFILE}"
if [[ ! -d "$BASE" ]]; then
  echo "Missing $BASE. Run experiments first." >&2
  exit 1
fi
OUT="results_bundle_${PROFILE}.zip"
rm -f "$OUT"
python3 - "$BASE" "$OUT" <<'PY'
import sys, zipfile
from pathlib import Path
base=Path(sys.argv[1]); out=Path(sys.argv[2]); root=Path.cwd()
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv",".json",".png",".txt"}:
            z.write(p,p.relative_to(base.parent))
    for name in ["PROTOCOL_V2.md","CHANGELOG_V2.md","MIGRATION_V2.md","README.md","RUNBOOK_zh-TW.md"]:
        p=root/name
        if p.exists():
            z.write(p,Path("protocol")/p.name)
    cfg=root/"configs"/"profiles.json"
    if cfg.exists():
        z.write(cfg,Path("protocol")/"profiles.json")
print(out)
PY
