# V2 正式實驗執行手冊

## 先確認你在哪裡

`PROTOCOL_V2.md` 與 `CHANGELOG_V2.md` 都放在專案根目錄，與 `README.md`、`run_all.py` 同一層。

V2 最安全的使用方式不是手動複製小段程式，而是直接使用完整 `bayesian_random_walk_study_v2` 資料夾。舊 V1 專案與 `results/paper/` 留著當 pilot/audit 紀錄。

## 1. 環境

```bash
cd bayesian_random_walk_study_v2
bash scripts/setup.sh
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 2. Unit tests

```bash
pytest -q
```

應全部 PASS。任何 fail 都停止，不進正式實驗。

## 3. Smoke V2：只先測 E2 / E4

```bash
python run_all.py --profile smoke_v2 --experiments e2,e4 --jobs 2
bash scripts/audit_results.sh smoke_v2
```

這一步只確認 repaired Gaussian tail、ridge-aware estimator、reference logic、CSV/figure pipeline 都能運作。Smoke 結果不寫進正式論文。

## 4. 正式 E4 numerical certification

```bash
JOBS=4 bash scripts/run_v2_gate.sh
```

MacBook Air 若過熱或記憶體壓力大：

```bash
JOBS=2 bash scripts/run_v2_gate.sh
```

跑完看：

```text
results/paper_v2/audit_report.txt
```

必須看到：

```text
E4_NUMERICAL_GATE_PASS=1
```

如果是 0：停止。不要繼續 E1/E3/E5/E6/E7，也不要改 seed。把 `results/paper_v2/exp4_convergence/` 與終端 log 交回分析。

## 5. E4 通過後先跑 E2

```bash
python run_all.py --profile paper_v2 --experiments e2 --jobs 4 | tee e2_v2.log
bash scripts/audit_results.sh paper_v2
```

這一步確認 Gaussian approximation 的正式結果，尤其檢查 V1 在 far-tail 出現的 catastrophic cancellation 已消失。

## 6. 正式 scientific experiments

逐一跑最安全：

```bash
python run_all.py --profile paper_v2 --experiments e1 --jobs 4 | tee e1_v2.log
python run_all.py --profile paper_v2 --experiments e3 --jobs 4 | tee e3_v2.log
python run_all.py --profile paper_v2 --experiments e5 --jobs 4 | tee e5_v2.log
python run_all.py --profile paper_v2 --experiments e6 --jobs 4 | tee e6_v2.log
python run_all.py --profile paper_v2 --experiments e7 --jobs 4 | tee e7_v2.log
python run_all.py --profile paper_v2 --experiments e0 --jobs 4 | tee e0_v2.log
```

也可以在 E4 gate 已通過後使用：

```bash
JOBS=4 bash scripts/run_paper_v2.sh
```

## 7. Final audit

```bash
bash scripts/audit_results.sh paper_v2
```

至少確認：

- `unexpected_nonfinite_numeric_cells=0`
- E4 numerical gate = 1
- `TOTAL_UNRESOLVED_EVIDENCE_ROWS` 已被明確列出
- `manifest_present=yes`
- `config_snapshot_present=yes`

如果 unresolved 不為 0，不要自己刪除。把結果交回分析。

## 8. 打包

```bash
bash scripts/make_results_bundle.sh paper_v2
```

得到：

```text
results_bundle_paper_v2.zip
```

請回傳這一個 ZIP。它會一起包含 raw CSV、summaries、figures、audit、environment、manifest、config snapshot、V2 protocol 與 changelog。

## 研究紀律

1. 不因圖不好看而改 seed。
2. 不手動刪 low-ESS rows。
3. 不把 numerical instability 當 statistical uncertainty。
4. 任何 bug 修復都保留同一 data-generating seed。
5. E4 gate 未通過以前，不解讀主要 Bayesian model-selection 結果。
6. V1 `results/paper/` 永遠保留，不覆蓋；V2 寫入 `results/paper_v2/`。
