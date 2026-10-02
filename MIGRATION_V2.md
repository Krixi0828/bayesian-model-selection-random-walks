# MIGRATION_V2 — 直接取代舊專案的方式

`PROTOCOL_V2.md`、`CHANGELOG_V2.md` 與本檔都放在**專案根目錄**，也就是與 `README.md`、`run_all.py`、`configs/` 同一層。

## 最簡單、最安全的方法：整個資料夾換掉

不要手動複製十幾個小片段。建議：

1. 把舊專案改名保留，例如 `bayesian_random_walk_study_v1`。
2. 解壓 `bayesian_random_walk_study_v2.zip`。
3. 新資料夾直接當正式 V2 專案使用。
4. 舊 `results/paper/` 留在 V1 資料夾，不搬進 V2，也不覆寫。

這是最推薦方式。

## 如果一定要覆蓋原本專案

### 新增檔案

- `PROTOCOL_V2.md`
- `CHANGELOG_V2.md`
- `MIGRATION_V2.md`
- `scripts/run_v2_gate.sh`
- `scripts/run_paper_v2.sh`

### 直接覆蓋

- `src/brw/likelihoods.py`
- `src/brw/importance.py`
- `src/brw/evaluation.py`
- `src/brw/random_walk.py`
- `src/brw/common.py`
- `experiments/e1_identifiability.py`
- `experiments/e2_exact_vs_gaussian.py`
- `experiments/e3_prior_sensitivity.py`
- `experiments/e4_convergence.py`
- `experiments/e5_baselines.py`
- `experiments/e6_imbalance.py`
- `experiments/e7_misspecification.py`
- `tests/test_core.py`
- `configs/profiles.json`
- `run_all.py`
- `scripts/audit_results.py`
- `scripts/make_results_bundle.sh`
- `README.md`
- `RUNBOOK_zh-TW.md`

### 不要覆蓋／不要刪

- 舊 `results/paper/`
- 舊 `results_bundle_paper.zip`
- 任何 V1 audit log

## V2 執行流程

### 1. 建立環境

```bash
bash scripts/setup.sh
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Unit tests

```bash
pytest -q
```

應全部通過。任何 test fail 都不要繼續跑正式實驗。

### 3. Smoke：只先測最危險的 E2 + E4

```bash
python run_all.py --profile smoke_v2 --experiments e2,e4 --jobs 2
bash scripts/audit_results.sh smoke_v2
```

### 4. 正式 numerical gate：只跑 E4

```bash
JOBS=4 bash scripts/run_v2_gate.sh
```

查看：

`results/paper_v2/audit_report.txt`

必須看到：

`E4_NUMERICAL_GATE_PASS=1`

若不是 1，停止，不要跑 E1/E3/E5/E6/E7，將 E4 results + log 交回分析。

### 5. Gate 通過後，跑 E2

```bash
python run_all.py --profile paper_v2 --experiments e2 --jobs 4 | tee e2_v2.log
bash scripts/audit_results.sh paper_v2
```

確認無 NaN/Inf、無大量 unresolved，並確認 T=1000 不再出現 V1 的 Gaussian-tail 爆炸。

### 6. 主實驗

可以用：

```bash
JOBS=4 bash scripts/run_paper_v2.sh
```

或逐一跑：

```bash
python run_all.py --profile paper_v2 --experiments e1 --jobs 4 | tee e1_v2.log
python run_all.py --profile paper_v2 --experiments e3 --jobs 4 | tee e3_v2.log
python run_all.py --profile paper_v2 --experiments e5 --jobs 4 | tee e5_v2.log
python run_all.py --profile paper_v2 --experiments e6 --jobs 4 | tee e6_v2.log
python run_all.py --profile paper_v2 --experiments e7 --jobs 4 | tee e7_v2.log
python run_all.py --profile paper_v2 --experiments e0 --jobs 4 | tee e0_v2.log
```

### 7. Final audit

```bash
bash scripts/audit_results.sh paper_v2
```

### 8. 打包回傳

```bash
bash scripts/make_results_bundle.sh paper_v2
```

回傳：

`results_bundle_paper_v2.zip`

如果 audit 有 warning，再一起附對應 `*_v2.log`。
