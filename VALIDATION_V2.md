# V2 implementation validation

The delivered V2 code was checked before packaging.

- `pytest -q`: **11 passed**.
- Python byte-code compilation: passed for `src/`, `experiments/`, `run_all.py`, and the audit script.
- `smoke_v2` E2 + E4: completed without unresolved evidence rows.
- Smoke E4 numerical gate: passed in the delivery environment.

Smoke results are intentionally not bundled into the source ZIP because they are not paper results. The user should recreate them in their own environment before the formal `paper_v2` run.
