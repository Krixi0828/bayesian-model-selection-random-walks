# Bayesian Random-Walk Study — Protocol V2

## 1. 研究目的

本研究探討有限樣本 random-walk trajectories 中，Bayesian model selection 是否能可靠辨識單一 stochastic regime 與兩個 latent regimes 的 mixture。V2 在 V1 numerical audit 後修正數值方法，但保留原始研究問題與主要資料生成設定；所有 V2 修改均在重新執行主要科學實驗前固定。

核心 competing models：

- `M1`: 所有 trajectory 共享同一個 `p`，`K_i ~ Binomial(T,p)`。
- `M2`: `K_i ~ pi Binomial(T,p1) + (1-pi) Binomial(T,p2)`。

主要判定量：

`log BF_21 = log p(K|M2) - log p(K|M1)`。

`log BF_21 > 0` 選 M2，否則選 M1。除了最終 decision，保留完整 log evidence、numerical diagnostics、TPR、FPR、balanced accuracy 與 uncertainty summaries。

## 2. V2 numerical protocol

### 2.1 M1 evidence

Binomial M1 使用 Beta-Binomial closed form，不含 Monte Carlo integration noise。

### 2.2 M2 production evidence

V2 使用 ridge-aware defensive importance sampling。Proposal 明確涵蓋：

1. 原 prior defensive component，提供 full support。
2. `p1 ≈ p2` 的 equal-regime ridge。
3. `pi ≈ 1` 的 nested boundary。
4. `pi ≈ 0` 的 nested boundary。
5. 兩個 label-switched posterior modes 的 multi-scale Gaussian components。

Top-level proposal weights 固定為：

- prior: 0.10
- equal-regime ridge: 0.20
- `pi≈1`: 0.15
- `pi≈0`: 0.15
- local Gaussian modes total: 0.40

這些權重在 V2 正式實驗前固定，不根據後續結果調整。

### 2.3 Adaptive evidence budget

`paper_v2` 固定：

- base budget = 4096
- escalation budget = 32768
- maximum budget = 131072
- base ESS-fraction trigger = 0.01
- base max-weight trigger = 0.20
- decision-boundary trigger = `|log BF| < 0.5`
- repeated-estimate SD threshold = 0.05 log-evidence units

Base estimate 若同時符合 ESS、max weight 與 decision-boundary 條件則接受。否則以兩個獨立 numerical seeds 進行 escalation。若兩個 estimates 的 log-evidence SD > 0.05 或 Bayes-factor signs 不一致，進入 maximum-budget stage。Maximum stage 若仍不符合 repeatability criterion，標記 `unresolved`；不得刪除該 dataset。

ESS 是 diagnostic，不是排除資料的規則。正式 numerical reliability 由 E4 calibration 決定。

## 3. Numerical certification gate — E4

E4 必須在主要 Bayesian experiments 前完成。

Reference 使用 independent scrambled Sobol QMC，而不是 production importance sampler：

- initial reference budget = 2^18 = 262144
- maximum reference budget = 2^20 = 1048576
- scrambles = 4
- initial reference 若 between-scramble SD > 0.02 或 `|log BF_ref| < 0.5`，升級至 maximum budget
- final reference SD threshold = 0.05

Stress datasets：

- 8 × M1 center (`p=.50`)
- 4 × M1 off-center (`p=.35`)
- 4 × M2 hard (`p1=.47, p2=.53, pi=.5`)
- 2 × M2 medium (`p1=.40, p2=.60, pi=.5`)
- 2 × M2 imbalanced (`p1=.40, p2=.60, pi=.10`)

比較四種 estimators：

- prior Monte Carlo
- scrambled Sobol QMC
- V1 local-mode defensive IS
- V2 ridge-aware IS

正式 gate：對 `|log BF_ref| >= 0.5` 的 reference datasets，V2 production budget 4096 的 BF-sign error rate 必須 <= 1%，且所有 final reference estimates 必須通過 SD <= 0.05。95th-percentile absolute log-evidence error <= 0.25 是 performance target，不是硬性排除標準。

E4 gate 未通過時，不執行 E1/E3/E5/E6/E7 正式分析。

## 4. Experiments

### E0 — simulator validation

驗證 diffusion scaling 與 one-dimensional occupation-time arcsine law。V2 不改動 V1 設定。

### E1 — identifiability

- `N = {20,50,100,200}`
- `T = {50,200,1000}`
- `delta_p = {.02,.05,.10,.20,.30}`
- 200 datasets / condition / true model
- `kappa = 2 delta_p sqrt(T)`

主要指標：TPR、FPR、balanced accuracy、median/IQR log BF、`kappa` collapse。

### E2 — Binomial likelihood vs Gaussian approximation

V2 分為兩個 design：

**E2-A fixed delta-p**

- `delta_p=.20`
- `T={10,20,50,100,200,500,1000}`
- M1 與 M2 均執行

**E2-B fixed kappa**

- `kappa={1,2}`
- 同一組 T
- `delta_p = kappa/(2 sqrt(T))`
- 只需 M2，因為 M1 不依賴 kappa

Gaussian continuity-corrected likelihood 採數值穩定的 tail interval calculation。Gaussian M1 evidence 使用 1-D adaptive quadrature，不使用 Laplace approximation。

分別保存 `logZ1_binomial`, `logZ2_binomial`, `logZ1_gaussian`, `logZ2_gaussian`，避免把兩個模型的 approximation error 混成單一 BF error。

### E3 — prior sensitivity

- `a={.5,1,2,5,10,20}` for `Beta(a,a)`
- M1 null scenario
- M2 `delta_p={.05,.10,.20}`
- N=100, T=200
- 300 datasets / scenario

同一 dataset 必須跨 prior 重複使用；prior_a 不可進入 data seed。

### E5 — model-selection criteria

比較 raw ML、AIC、BIC、Bayesian evidence。BIC 僅作 conventional penalized-likelihood baseline；finite mixture 是 singular model，因此不宣稱其 regular-model asymptotic justification 嚴格成立。

### E6 — minority regime

- N={50,100,200}
- `delta_p={.05,.10,.20}`
- minority weight={.50,.30,.20,.10,.05}
- 300 datasets / condition

研究 minority prevalence 與 regime separation 的交互影響。

### E7 — model misspecification

Persistent walk：`P(X_t=X_{t-1})=q`。

- q={.40,.45,.50,.52,.54,.56,.58,.60,.65,.70,.80,.90}
- N=100, T=200
- 500 datasets / q

除 M2 false-positive rate 外，同時比較 empirical variance inflation 與 finite-T theory：

`VIF_T = 1 + 2 sum_{h=1}^{T-1} (1-h/T) (2q-1)^h`。

q<.5 作為 negative-correlation / underdispersion control。

## 5. Reproducibility rules

1. `paper_v2` 開始後不得改 `configs/profiles.json`。
2. 不因 accuracy、圖形或 Bayes factor 不漂亮而更換 data seed。
3. Bug 修正或 numerical re-estimation 必須保留相同 dataset seed。
4. Raw CSV 不手動編輯。
5. 所有 output 保存 `environment.json`, `config_snapshot.json`, `manifest.json`。
6. `audit_report.txt` 與 E4 `numerical_gate.csv` 是正式結果的一部分。
7. `unresolved` rows 不刪除，必須在 Results/Limitations 中揭露。

## 6. Formal execution order

1. `pytest -q`
2. `smoke_v2` E2 + E4
3. `paper_v2` E4 only
4. audit E4 numerical gate
5. gate pass 後執行 E2
6. 再執行 E1, E3, E5, E6, E7
7. 最後執行 E0，建立完整 V2 bundle
8. full audit
9. `make_results_bundle.sh paper_v2`

任何 E4 gate failure 都視為 numerical-method problem，先處理 estimator，不解讀主實驗。
