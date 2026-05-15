# V13 — Forecast Disagreement Anomaly Score (FDAS)
*(formerly RWMFD; recency-weighted multi-horizon forecast disagreement.)*
## 종합 평가 리포트

**작성일**: 2026-05-07
**데이터셋**: 199개 (TSB-AD-M, `--allow-early-anomaly` 필터)
**Backbone**: iTransformer (`use_norm=True`, `seq_len=192`, `pred_len=96`)

---

## 1. 핵심 요약

다변량 시계열 이상탐지를 위한 **GT-free (ground-truth-free)** 점수를 제안한다. 다중-호라이즌 forecasting 모델의 예측만으로 산출되며, 각 test 시점 $t$ 에 대해 모델이 만든 $H$ 개의 lead-time별 예측 $i \in \{1, \ldots, H\}$ 의 *recency-weighted disagreement* 를 채널별로 측정한 뒤 채널 간 집계한다.

**대표 결과 (VUS-PR, 199개 데이터셋 평균):**

| 변형 (variant) | VUS-PR | $\Delta$ vs raw_max | wins / 199 |
|:--|--:|--:|--:|
| raw_max (baseline)        | 0.2958 |     —    |    —    |
| **z_train_max (no-drop)** (제안, production) | **0.3120** | **+0.0162** | **111** |
| z_train_max (with channel drop, supplementary) | 0.3153 | +0.0195 | 111 |
| raw_median (Exathlon 우세) | 0.2872 | −0.0086 | 89 |

`z_train_max` 가 VUS-PR 평균 **+0.0162** 의 일관된 향상을 보이며, 네 가지 rank 기반 핵심 지표 (VUS-PR / VUS-ROC / AUC-PR / AUC-ROC) 모두에서 raw_max 압승. 추가로 σ ≤ ε 채널을 드랍하면 **+0.0033** 더 향상되어 총 +0.0195 (drop ablation 은 §4.7 참고). 채널-스케일 폭주가 빈발한 **Exathlon (27개)** family 에선 `*_median` 변형이 **VUS-PR ≈ 0.83** 까지 도달 — train/test 스케일 shift 가 만든 폭주 채널을 자동으로 억제.

---

## 2. 방법 개요

### 2.1 설정

Test 시퀀스 인덱싱 (0-indexed): $t \in \{0, 1, \ldots, T-1\}$, 길이 $T$. 모델
$$
f_\theta : \mathbb{R}^{L \times C} \to \mathbb{R}^{H \times C}
$$
은 $C$ 채널, 길이 $L$ 의 lookback 윈도우를 입력받아 $H$-step forecast 를 출력. 정상 데이터로만 학습.

### 2.2 다중-호라이즌 vote 구성

핵심 아이디어 — **하나의 미래 시점 $t$ 가 H 번 예측됨**. 각 anchor $t-i$ 에서 모델이 만든 $i$-step ahead forecast 가 $t$ 에 대한 vote $i$:

$$
\hat y_t^{(t-i)} = \big[\, f_\theta(X_{t-i-L+1:\, t-i}) \,\big]_i \in \mathbb{R}^C, \quad i = 1, \ldots, H
$$

시점 $t$ 의 vote 집합:
$$
\hat{\mathcal Y}_t = \{\, \hat y_t^{(t-i)} \mid i = 1, \ldots, H \,\}
$$

**평가 가능 구간** (0-indexed):
$$
t \in [\,L + H - 1,\ T - 1\,]
$$

### 2.3 Recency-Weighted 채널별 분산

Recency 가중치 $w_i = \lambda^{i-1}$, 정규화 $\tilde w_i = w_i / \sum_i w_i$ ($\lambda = 0.99$, $H = 96$):

$$
\bar v_t[c] = \sum_{i=1}^H \tilde w_i \cdot \hat y_t^{(t-i)}[c]
$$

$$
D_{w,c}(t) = \sum_{i=1}^H \tilde w_i \cdot \big( \hat y_t^{(t-i)}[c] - \bar v_t[c] \big)^2
$$

### 2.4 채널 집계 — 9가지 변형 비교

(정규화 $g_c(\cdot)$, 집계) 9가지 조합 평가:

| 정규화 $g_c(\cdot)$ | 집계 | 변형 이름 |
|:--|:--|:--|
| 없음 (identity) | $\max$ | `raw_max` |
| 없음 | $\mathrm{mean}$ | `raw_mean` |
| 없음 | $\mathrm{median}$ | `raw_median` |
| train baseline z-score | $\max$ | `z_train_max` (제안) |
| train baseline z-score | $\mathrm{mean}$ | `z_train_mean` |
| train baseline z-score | $\mathrm{median}$ | `z_train_median` |
| val baseline z-score | $\max$ | `z_val_max` |
| val baseline z-score | $\mathrm{mean}$ | `z_val_mean` |
| val baseline z-score | $\mathrm{median}$ | `z_val_median` |

z-score 변형은 채널 $c$ 를 *baseline split* (train 또는 val) 에서 산출한 $D_{w,c}$ 분포의 채널별 평균과 표준편차로 정규화:

$$
\mu_c^{(\mathrm{base})} = \mathrm{mean}_t\big[ D_{w,c}(t) \mid t \in \text{base evaluable range} \big]
$$

$$
\sigma_c^{(\mathrm{base})} = \mathrm{std}_t\big[ D_{w,c}(t) \mid t \in \text{base evaluable range} \big]
$$

$$
g_c\big(D_{w,c}(t)\big) = \frac{D_{w,c}(t) - \mu_c^{(\mathrm{base})}}{\sigma_c^{(\mathrm{base})}}
$$

$\sigma_c^{(\mathrm{base})} \le \epsilon$ ($\epsilon = 10^{-8}$) 인 채널은 집계에서 제외.

최종 점수:
$$
\mathrm{score}(t) = \mathrm{aggregate}_c\big[\, g_c\big(D_{w,c}(t)\big) \,\big]
$$

### 2.5 GT-Free 성질

$D_w(t)$ 는 **모델 예측값 $\hat y_t^{(\cdot)}$ 만으로 계산** — 시점 $t$ 의 관측값 $y_t$ 를 절대 사용하지 않음. 배포 시 채널별 baseline 통계 $(\mu_c, \sigma_c)$ 는 **held-out validation split (또는 train split) 에서 한 번만 계산**해 모델 옆에 상수로 저장; 추론 시점엔 가벼운 affine 변환만 적용. 실시간 / 스트리밍 호환성 완전 보장.

---

## 3. 파이프라인 흐름도

```
┌────────────────────────────────────────────────────────────────────────┐
│ OFFLINE — 배포 전                                                      │
└────────────────────────────────────────────────────────────────────────┘
                                  │
  01_data_preparation.py          │
    ├── train CSV 로드                                                   │
    ├── StandardScaler (train 구간으로 fit)                              │
    ├── train / val / test split                                         │
    └── 저장: data/{train,val,test}_data.npy, bundle_meta.json           │
                                  │
                                  ▼
  02_train.py                     │
    ├── iTransformer (use_norm=True, L=192, H=96)                        │
    ├── multi-step forecast 의 MSE loss                                  │
    ├── early-stop (patience=3)                                          │
    └── 저장: models/{key}/checkpoint.pth (BEST snapshot)                │
                                  │
                                  ▼
  03_inference.py                 │
    ├── stride-1 sliding windows on TRAIN, VAL, TEST                     │
    ├── extended length: N = len(data) − L                                │
    │       (전체 평가구간 t ∈ [L+H−1, T−1] 커버)                         │
    └── 저장: predictions_{train,val,test}.npy, shape (N, H, C)          │
                                  │
                                  ▼
  04_score_compute.py             │
    ├── compute_backward_score_per_channel on TEST                       │
    │     → 채널별 D_w_c(t), t ∈ 평가구간                                │
    ├── compute_train_baseline_stats on VAL preds (우선)                 │
    │     → μ_c^val, σ_c^val   (없으면 train fallback)                   │
    ├── apply_channel_zscore_aggregation                                 │
    │     → score(t) = max_c (D_w_c(t) − μ_c)/σ_c     [production]       │
    └── 저장: scores.parquet (t, D_w, D_w_z, base, label)                │
              scores_per_ch.npz (D_w_c, base_c, baseline_mean, baseline_std) │
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ DEPLOYMENT — test 시점 t 마다 실시간                                   │
└────────────────────────────────────────────────────────────────────────┘
   x_{t-1} 관측 → model.forecast(X_{t-L:t-1}) → 저장                     │
                                  │
   매 평가가능 t:                 │
      과거에 만들어진 t에 대한 H개 forecast 수집 (anchor t-1..t-H)       │
      ↓                           │
      D_{w,c}(t) = recency-weighted variance (식 2.3)                    │
      ↓                           │
      z_c(t) = (D_{w,c}(t) − μ_c) / σ_c        ← μ, σ 는 상수            │
      ↓                           │
      score(t) = max_c z_c(t)                                            │
      ↓                           │
   임계값 초과 시 ALARM           │
                                  │
                                  ▼
  05_metrics.py / 06_cross_dataset.py / 07_visualization.py              │
    (오프라인 평가: AUROC, VUS-PR, AUC-PR 등)                            │
```

---

## 4. 상세 결과

### 4.1 전체 평균 (n = 199)

| Metric        | raw_max | raw_mean | raw_median | **z_train_max** | z_train_mean | z_train_median | z_val_max | z_val_mean | z_val_median |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **VUS-PR**    | 0.2958 | 0.3027 | 0.2872 | **0.3153** | 0.3142 | 0.2868 | 0.2761 | 0.2751 | 0.2766 |
| **VUS-ROC**   | 0.7220 | 0.7241 | 0.7066 | **0.7508** | 0.7317 | 0.7037 | 0.7008 | 0.6879 | 0.6772 |
| **AUC-PR**    | 0.2247 | 0.2290 | 0.2206 | **0.2451** | 0.2439 | 0.2202 | 0.2102 | 0.2096 | 0.2112 |
| **AUC-ROC**   | 0.6574 | 0.6617 | 0.6389 | **0.6849** | 0.6668 | 0.6333 | 0.6525 | 0.6389 | 0.6224 |
| Standard-F1   | 0.3092 | 0.3145 | 0.2889 | **0.3301** | 0.3278 | 0.2856 | 0.2871 | 0.2874 | 0.2722 |
| PA-F1         | 0.6028 | 0.6060 | 0.6327 | 0.6176 | 0.6143 | **0.6343** | 0.5099 | 0.5012 | 0.5410 |

`z_train_max` 는 모든 rank 기반 지표 (VUS-PR / VUS-ROC / AUC-PR / AUC-ROC) 의 절대 winner.

### 4.2 raw_max 대비 Δ (199개 데이터셋 평균)

| 변형 | ΔVUS-PR | ΔVUS-ROC | ΔAUC-PR | ΔAUC-ROC |
|:--|--:|--:|--:|--:|
| raw_mean        | +0.0069 | +0.0021 | +0.0044 | +0.0043 |
| raw_median      | −0.0086 | −0.0153 | −0.0041 | −0.0185 |
| **z_train_max**     | **+0.0195** | **+0.0289** | **+0.0204** | **+0.0275** |
| z_train_mean    | +0.0184 | +0.0098 | +0.0192 | +0.0094 |
| z_train_median  | −0.0090 | −0.0183 | −0.0045 | −0.0241 |
| z_val_max       | −0.0197 | −0.0211 | −0.0145 | −0.0049 |
| z_val_mean      | −0.0207 | −0.0341 | −0.0151 | −0.0185 |
| z_val_median    | −0.0192 | −0.0448 | −0.0135 | −0.0350 |

### 4.3 raw_max 대비 승률 (n = 199)

| 변형 | VUS-PR 승 | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|
| raw_mean        | 102 | 88  | 110 | 95  | 117 | 81 |
| raw_median      | 89  | 79  | 96  | 80  | 79  | 70 |
| **z_train_max**     | **111** | **125** | **120** | **120** | **115** | 77 |
| z_train_mean    | 113 | 122 | 122 | 119 | 122 | 73 |
| z_train_median  | 92  | 75  | 96  | 73  | 78  | 71 |
| z_val_max       | 65  | 80  | 73  | 81  | 76  | 60 |
| z_val_mean      | 67  | 76  | 71  | 75  | 76  | 53 |
| z_val_median    | 70  | 60  | 75  | 64  | 67  | 65 |

(승 = 해당 metric 에서 raw_max 보다 *엄밀히* 큰 데이터셋 수, 전체 199 기준.)

### 4.4 Family 별 VUS-PR (변형별 평균)

| family | n | raw_max | raw_mean | raw_median | **z_train_max** | z_train_mean | z_train_med | z_val_max | z_val_mean | z_val_med |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CATSv2       | 6  | 0.271 | 0.234 | 0.059 | **0.369** | 0.309 | 0.054 | 0.317 | 0.283 | 0.054 |
| CreditCard   | 1  | 0.034 | 0.041 | **0.051** | 0.038 | 0.044 | 0.050 | 0.020 | 0.020 | 0.020 |
| Daphnet      | 1  | 0.187 | 0.208 | **0.278** | 0.227 | 0.226 | **0.279** | 0.207 | 0.214 | 0.261 |
| **Exathlon** | 27 | 0.650 | 0.671 | **0.828** | 0.646 | 0.655 | **0.827** | 0.664 | 0.668 | 0.813 |
| GECCO        | 1  | **0.228** | 0.227 | 0.127 | 0.194 | 0.196 | 0.151 | 0.185 | 0.188 | 0.167 |
| GHL          | 25 | 0.010 | 0.010 | 0.012 | 0.012 | 0.010 | 0.011 | **0.012** | 0.011 | 0.011 |
| Genesis      | 1  | 0.016 | 0.026 | 0.015 | 0.087 | 0.117 | 0.012 | **0.392** | 0.389 | 0.027 |
| LTDB         | 5  | 0.388 | **0.391** | 0.391 | 0.386 | 0.390 | 0.390 | 0.351 | 0.355 | 0.355 |
| MITDB        | 13 | 0.143 | 0.145 | 0.145 | 0.154 | 0.154 | 0.154 | **0.156** | 0.155 | 0.155 |
| MSL          | 15 | 0.323 | **0.354** | 0.162 | 0.295 | 0.292 | 0.180 | 0.205 | 0.200 | 0.129 |
| OPPORTUNITY  | 8  | 0.132 | 0.134 | **0.145** | 0.144 | 0.145 | 0.143 | 0.135 | 0.141 | 0.142 |
| PSM          | 1  | 0.160 | 0.170 | 0.174 | 0.138 | 0.146 | 0.179 | 0.188 | **0.194** | 0.179 |
| **SMAP**     | 27 | 0.218 | 0.226 | 0.173 | **0.352** | 0.340 | 0.158 | 0.156 | 0.152 | 0.162 |
| SMD          | 22 | 0.284 | **0.293** | 0.204 | 0.275 | 0.287 | 0.203 | 0.241 | 0.243 | 0.190 |
| SVDB         | 31 | 0.197 | 0.197 | 0.197 | **0.198** | 0.198 | 0.198 | 0.179 | 0.179 | 0.179 |
| SWaT         | 2  | 0.220 | 0.219 | 0.200 | **0.226** | 0.217 | 0.202 | 0.195 | 0.187 | 0.217 |
| TAO          | 13 | **0.804** | 0.805 | 0.802 | 0.804 | 0.804 | 0.802 | 0.803 | 0.803 | 0.803 |

(**bold** = 해당 family 의 최고 변형)

### 4.5 Family 별 — Production 변형 `z_train_max` (전체 metric)

| family | n | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CATSv2      | 6  | 0.369 | 0.793 | 0.383 | 0.770 | 0.448 | 0.639 |
| Exathlon    | 27 | 0.646 | 0.904 | 0.673 | 0.924 | 0.728 | 0.957 |
| LTDB        | 5  | 0.386 | 0.728 | 0.305 | 0.677 | 0.402 | 0.624 |
| MITDB       | 13 | 0.154 | 0.802 | 0.171 | 0.741 | 0.244 | 0.879 |
| MSL         | 15 | 0.295 | 0.735 | 0.178 | 0.626 | 0.301 | 0.626 |
| SMAP        | 27 | 0.352 | 0.844 | 0.234 | 0.770 | 0.363 | 0.749 |
| SMD         | 22 | 0.275 | 0.843 | 0.260 | 0.829 | 0.366 | 0.515 |
| SVDB        | 31 | 0.198 | 0.799 | 0.150 | 0.722 | 0.236 | 0.776 |
| TAO         | 13 | 0.804 | 0.929 | 0.087 | 0.490 | 0.160 | 0.161 |
| OPPORTUNITY | 8  | 0.144 | 0.348 | 0.131 | 0.346 | 0.181 | 0.231 |
| GHL         | 25 | 0.012 | 0.215 | 0.011 | 0.183 | 0.034 | 0.124 |
| Daphnet     | 1  | 0.227 | 0.825 | 0.215 | 0.834 | 0.350 | 0.473 |
| GECCO       | 1  | 0.194 | 0.879 | 0.351 | 0.835 | 0.444 | 0.541 |
| Genesis     | 1  | 0.087 | 0.844 | 0.018 | 0.762 | 0.052 | 0.095 |
| CreditCard  | 1  | 0.038 | 0.679 | 0.005 | 0.594 | 0.026 | 0.022 |
| PSM         | 1  | 0.138 | 0.502 | 0.131 | 0.486 | 0.272 | 0.835 |
| SWaT        | 2  | 0.226 | 0.522 | 0.184 | 0.492 | 0.293 | 0.578 |

### 4.6 Family 별 — Baseline `raw_max` (직접 비교용)

| family | n | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CATSv2      | 6  | 0.271 | 0.748 | 0.300 | 0.726 | 0.347 | 0.737 |
| Exathlon    | 27 | 0.650 | 0.876 | 0.652 | 0.896 | 0.760 | 0.955 |
| LTDB        | 5  | 0.388 | 0.755 | 0.310 | 0.696 | 0.401 | 0.601 |
| MITDB       | 13 | 0.143 | 0.795 | 0.156 | 0.733 | 0.235 | 0.864 |
| MSL         | 15 | 0.323 | 0.812 | 0.193 | 0.741 | 0.336 | 0.760 |
| SMAP        | 27 | 0.218 | 0.785 | 0.157 | 0.706 | 0.245 | 0.762 |
| SMD         | 22 | 0.284 | 0.840 | 0.270 | 0.825 | 0.371 | 0.526 |
| SVDB        | 31 | 0.197 | 0.798 | 0.149 | 0.721 | 0.235 | 0.774 |
| TAO         | 13 | 0.804 | 0.929 | 0.087 | 0.491 | 0.161 | 0.161 |
| OPPORTUNITY | 8  | 0.132 | 0.343 | 0.122 | 0.341 | 0.171 | 0.220 |
| GHL         | 25 | 0.010 | 0.205 | 0.008 | 0.172 | 0.030 | 0.137 |
| Daphnet     | 1  | 0.187 | 0.792 | 0.151 | 0.785 | 0.324 | 0.383 |
| GECCO       | 1  | 0.228 | 0.914 | 0.360 | 0.889 | 0.448 | 0.543 |
| Genesis     | 1  | 0.016 | 0.809 | 0.005 | 0.597 | 0.012 | 0.040 |
| CreditCard  | 1  | 0.034 | 0.716 | 0.005 | 0.610 | 0.022 | 0.023 |
| PSM         | 1  | 0.160 | 0.478 | 0.135 | 0.474 | 0.254 | 0.747 |
| SWaT        | 2  | 0.220 | 0.513 | 0.174 | 0.470 | 0.282 | 0.578 |

### 4.7 채널-드랍 Isolation Ablation (No-Drop 비교)

`z_train_*` 의 +0.020 향상이 **(i) z-score 정규화 자체** 의 효과인지 **(ii) σ ≤ ε 채널 자동 드랍** 의 효과인지 분리. raw 와 z_train 양쪽 모두 *모든 C 채널 사용* (z 의 분모는 σ_safe = max(σ, $\epsilon$) 로 보호).

#### 4.7.1 전체 평균 (n = 199, no-drop)

| Metric        | raw_max | raw_mean | raw_median | **z_train_max** | z_train_mean | z_train_median |
|:--|--:|--:|--:|--:|--:|--:|
| **VUS-PR**    | 0.2958 | 0.3027 | 0.2872 | **0.3120** | 0.3113 | 0.2853 |
| **VUS-ROC**   | 0.7220 | 0.7241 | 0.7066 | **0.7404** | 0.7213 | 0.7027 |
| **AUC-PR**    | 0.2247 | 0.2290 | 0.2206 | **0.2438** | 0.2432 | 0.2210 |
| **AUC-ROC**   | 0.6574 | 0.6617 | 0.6389 | **0.6751** | 0.6558 | 0.6306 |
| Standard-F1   | 0.3092 | 0.3145 | 0.2889 | **0.3292** | 0.3264 | 0.2845 |
| PA-F1         | 0.6028 | 0.6060 | 0.6327 | 0.5925 | 0.5934 | **0.6349** |

#### 4.7.2 Δ (z_train no-drop − raw)

| Metric        | Δ max | Δ mean | Δ median |
|:--|--:|--:|--:|
| VUS-PR        | **+0.0162** | +0.0086 | −0.0019 |
| VUS-ROC       | **+0.0184** | −0.0028 | −0.0039 |
| AUC-PR        | **+0.0191** | +0.0142 | +0.0004 |
| AUC-ROC       | **+0.0177** | −0.0059 | −0.0083 |
| Standard-F1   | **+0.0200** | +0.0119 | −0.0044 |
| PA-F1         | −0.0103 | −0.0126 | +0.0022 |

#### 4.7.3 Family 별 VUS-PR (no-drop, n=199)

| family | n | raw_max | raw_mean | raw_median | **z_train_max** | z_train_mean | z_train_median |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CATSv2       | 6  | 0.271 | 0.234 | 0.059 | **0.369** | 0.309 | 0.054 |
| CreditCard   | 1  | 0.034 | 0.041 | **0.051** | 0.038 | 0.044 | 0.050 |
| Daphnet      | 1  | 0.187 | 0.208 | 0.278 | 0.227 | 0.226 | **0.279** |
| **Exathlon** | 27 | 0.650 | 0.671 | **0.828** | 0.646 | 0.655 | 0.827 |
| GECCO        | 1  | **0.228** | 0.227 | 0.127 | 0.194 | 0.196 | 0.151 |
| GHL          | 25 | 0.010 | 0.010 | **0.012** | **0.012** | 0.010 | 0.011 |
| Genesis      | 1  | 0.016 | 0.026 | 0.015 | 0.087 | **0.117** | 0.012 |
| LTDB         | 5  | 0.388 | **0.391** | 0.391 | 0.386 | 0.390 | 0.390 |
| MITDB        | 13 | 0.143 | 0.145 | 0.145 | **0.154** | 0.154 | 0.154 |
| MSL          | 15 | 0.323 | **0.354** | 0.162 | 0.256 | 0.260 | 0.162 |
| OPPORTUNITY  | 8  | 0.132 | 0.134 | **0.145** | 0.136 | 0.134 | 0.142 |
| PSM          | 1  | 0.160 | 0.170 | 0.174 | 0.138 | 0.146 | **0.179** |
| **SMAP**     | 27 | 0.218 | 0.226 | 0.173 | **0.352** | 0.340 | 0.158 |
| SMD          | 22 | 0.284 | **0.293** | 0.204 | 0.275 | 0.287 | 0.203 |
| SVDB         | 31 | 0.197 | 0.197 | 0.197 | **0.198** | 0.198 | 0.198 |
| SWaT         | 2  | 0.220 | 0.219 | 0.200 | **0.226** | 0.217 | 0.202 |
| TAO          | 13 | **0.804** | 0.805 | 0.802 | 0.804 | 0.804 | 0.802 |

#### 4.7.4 z_train_max no-drop vs raw_max — 데이터셋 단위 승률

| Metric | z_train_max (no-drop) wins | ties | losses |
|:--|--:|--:|--:|
| VUS-PR | **111 / 199** | 1 | 87 |
| VUS-ROC | **123 / 199** | 1 | 75 |
| AUC-PR | **117 / 199** | 1 | 81 |
| AUC-ROC | **117 / 199** | 0 | 82 |
| Standard-F1 | **115 / 199** | 21 | 63 |
| PA-F1 | 73 / 199 | 60 | 66 |

#### 4.7.5 Drop 의 추가 기여 (no-drop → with-drop)

| Variant | no-drop VUS-PR | with-drop VUS-PR | drop 의 추가 기여 |
|:--|--:|--:|--:|
| z_train_max    | 0.3120 | 0.3153 | **+0.0033** |
| z_train_mean   | 0.3113 | 0.3142 | **+0.0029** |
| z_train_median | 0.2853 | 0.2868 | **+0.0015** |

**해석**: `z_train_max` 의 raw_max 대비 +0.0195 향상 중 **+0.0162 (≈ 83 %) 가 z-score 정규화 자체의 효과**, **+0.0033 (≈ 17 %) 가 채널 드랍의 추가 효과**. 즉 **드랍 없이도 z-score 가 raw 를 일관되게 이김** — 채널 드랍은 정밀화를 위한 부가 옵션이지 근본 동력이 아님.

또한 z + max 가 드랍 없이도 catastrophic 폭주를 안 일으킨 이유는, 대부분의 σ ≈ 0 채널은 train 과 test 양쪽에서 비슷하게 거의 상수라 $D_{w,c} - \mu_c \approx 0$ → 분모가 작아도 분자도 작아 z 가 폭주 안 함. 폭주는 train→test 분포 shift 가 큰 채널 (예: Exathlon 의 zero-var-train) 에 한정.

---

## 5. `z_train_max` 가 이긴 이유

**§4.7 ablation 결과 분해**: raw_max 대비 +0.0195 VUS-PR 향상 중

- **+0.0162 (≈ 83 %)** = z-score 정규화 *자체* 의 효과 (no-drop ablation 으로 측정)
- **+0.0033 (≈ 17 %)** = σ ≤ ε 채널 드랍의 추가 효과

→ 헤드라인 향상의 압도적 본체는 **z-score 의 채널 평준화 효과**. 채널 드랍은 미세 정밀화일 뿐 근본 동력이 아님.

### 5.1 왜 z 정규화 자체가 효과적인가

$D_{w,c}(t)$ 의 자릿수는 **그 채널 예측값 $\hat y_t^{(\cdot)}[c]$ 의 자릿수의 제곱**. 채널마다 자연 스케일이 달라 raw max 는 *항상 가장 큰 분산을 가진 채널 한 개* 에 갇힘. z 로 평준화하면 모든 채널이 *자기 baseline 단위 표준편차* 로 측정되어 채널 간 공정 경쟁 가능. 가장 anomaly 다운 채널이 진짜로 max 에 잡힘.

### 5.2 Train baseline 의 overfit-bias 가 max 변형에 도움되는 이유

Train baseline 은 모델이 train 을 외운 상태에서 산출되므로 $\sigma_c^{\mathrm{train}}$ 가 인위적으로 작음 → test 에 z 적용 시 값이 과장됨 → max 가 *대비가 더 첨예해진* 채널을 잡음. Val baseline 은 편향 없는 generalization variance 를 반영하지만, **199개 중 41개 데이터셋에서 모든 채널이 $\sigma_c^{\mathrm{val}} \le \epsilon$ 인 상황** 발생 (모델이 val 도 거의 완벽히 외움; MSL/SMAP 같이 zero-var-train 채널이 많은 family 에서 특히) → z_val 가 0 으로 무너지면서 그 데이터셋들에서 trivial 점수로 처리됨. Train 은 항상 valid σ 를 가진 채널이 일부 살아있어 coverage 가 더 넓음.

### 5.3 Median 변형은 z 정규화 효과 거의 없음

§4.7.2 의 Δ median 행을 보면 ΔVUS-PR ≈ −0.002. **median 자체가 채널 간 outlier 를 무시하므로 z 정규화 추가 이득이 없음**. Exathlon 같이 폭주 채널이 본질적 문제인 family 에선 raw_median 만으로도 0.83 VUS-PR — z 정규화 없이도 자동으로 폭주 채널 억제.

## 6. Family 별 패턴

- **Exathlon (27, polynomial channel-scale 폭주 family)** — `*_median` 변형이 모두 ~0.83 VUS-PR (vs raw_max 0.65). Median 의 outlier 강건성이 zero-var-train 채널 같은 폭주 채널을 자동 억제. 세 median 변형이 동급.
- **CATSv2 (sparse-signal family)** — `z_train_max` 0.369 가 raw_max 0.271 대비 +0.10 압승. 여기선 median 이 나쁨 (0.05) — anomaly 신호가 1–2개 채널에 집중되어 있어 median 이 그 신호 채널을 같이 평균에 묻어버림.
- **SMAP (27)** — z_train_max 의 가장 큰 절대 향상: 0.218 → 0.352 (+0.13).
- **TAO** — 모든 변형이 ~0.80; 변형 선택 무관. 포화 영역.
- **GHL** — 모든 변형이 ~0.01; 이 family 는 forecast-disagreement 기반 탐지 자체가 본질적으로 어려움.

## 7. 한계 및 검증 메모

### 7.1 방법론적 caveat (독립적 코드 리뷰에서 제기)

**A. 변형 간 채널 집합 비대칭 — §4.7 ablation 으로 해소** — 원래 `raw_*` 는 모든 $C$ 채널 사용, `z_*` 는 $\sigma \le \epsilon$ 채널 자동 드랍이라 비교가 비대칭이었음. §4.7 의 *no-drop ablation* 결과 **z-score 정규화 자체로 +0.0162 VUS-PR (≈ 83 %)**, **드랍 추가 효과는 +0.0033 (≈ 17 %)** 로 분리됨. 즉 헤드라인 +0.0195 의 본체는 z-score 자체의 채널 평준화 효과이고, 드랍은 미세 정밀화. 비대칭 confound 정량화 완료.

**B. Train z 와 val z 도 채널 집합이 다름** — `z_train_max` 와 `z_val_max` 가 *서로 다른* baseline 통계 기반으로 채널을 드랍. val 에선 거의 상수지만 train 에선 변하는 채널은 z_train 엔 포함되고 z_val 엔 제외 (반대 케이스도 있음). 두 baseline 의 성능 차이 일부는 overfit 편향만이 아니라 channel-mask 차이에서 기인. per-dataset CSV 의 `n_kept_train_baseline`/`n_kept_val_baseline` 컬럼으로 감사 가능.

**C. Vintage mixing** — 일부 prediction 은 다른 코드 리비전 (legacy `n_pred = T - L - H + 1` vs V13 의 확장된 `n_pred = T - L`) 에서 산출. `get_evaluable_range` 의 `n_pred` 파라미터가 자동 클램프하므로 점수 자체는 정확하지만, baseline-stat row 수가 vintage 간 ~$H$ 행 차이 — 분포는 robust 하나 cross-vintage 절대값엔 약간 영향 가능.

**D. σ-filter mask 도 같이 persist 되어야 함** — 실시간 배포 시 채널별 valid-mask (즉 $\sigma > \epsilon$ 통과 채널 목록) 도 $(\mu_c, \sigma_c)$ 와 함께 저장해야 test 시점 집계가 deterministic / reproducible.

### 7.2 본질적 한계

1. **Empty val baseline (`valid_c = 0`)**: 199개 중 41개 데이터셋이 모든 채널이 $\sigma_c^{\mathrm{val}} \le \epsilon$ (모델이 val 까지 너무 잘 외운 경우, MSL/SMAP 같이 zero-var-train 채널이 많은 family 에서 특히). 이 데이터셋들에서 `z_val_*` 가 0 (trivial 점수). `z_train_*` 은 train 이 더 많은 표본 + 일부 채널은 valid σ 유지로 살아남음 — 이게 family 평균에서 `z_train_max` 가 `z_val_max` 보다 우세한 경험적 이유.

2. **Train 구간 내 anomaly** — 일부 데이터셋은 official train segment 안에 anomaly 포함 (파일 메타에서 `1st_*` < `tr_*`). Train baseline 통계에 오염. 영향: ~15개 (~8%).

3. **Window-leak alignment** — $D_w(t)$ 신호는 실제 anomaly 보다 최대 $H + L_w - 1 = 287$ 스텝 후행. 어떤 forecaster 의 input window 도 $t$ 를 포함하지 않기 때문 (GT-free 설계의 구조적 결과; 2.5 절). 알려진 trade-off.

4. **iTransformer `use_norm=True` 동작** — per-window mean/std 정규화가 입력 스케일을 출력 스케일로 그대로 echo, 따라서 train→test 스케일 shift 가 큰 채널 (예: zero-var-train passthrough) 에서 $D_w$ 값이 자릿수 단위로 폭주 가능. `*_median` 집계가 이걸 자동 억제 (Exathlon 증거: VUS-PR 0.65 → 0.83). `z_*_max` 변형은 채널별 σ 정규화로 부분 보정하지만 baseline σ 가 작으면 small-σ amplification 발생 가능.

---

## 8. 재현성

모든 산출물 `V13/results/` 아래. 핵심 파일:

```
V13/
├── scripts/
│   ├── 01_data_preparation.py       — split + StandardScaler + bundle_meta
│   ├── 02_train.py                  — iTransformer 학습 (use_norm=True, patience=3)
│   ├── 03_inference.py              — train/val/test predictions (확장 길이 T-L)
│   ├── 04_score_compute.py          — D_w + D_w_z (val baseline, max agg) 계산
│   ├── 05_metrics.py                — TSB-AD metric 파이프라인 + edge_fill
│   ├── score_utils.py               — 핵심 수식 (compute_backward_score_per_channel,
│   │                                  compute_train_baseline_stats,
│   │                                  apply_channel_zscore_aggregation)
│   ├── compare_agg_normalize.py     — 9-variant 비교 harness (multiprocess)
│   └── _add_val_inference.py        — standalone val inference (보정용)
├── ablations/results/etc/
│   ├── agg_normalize_per_dataset.csv  — 199 × (9 variant × 6 metric) 전체 (with-drop)
│   ├── agg_normalize_summary.csv      — family 평균 (with-drop)
│   ├── ablation_no_drop_per_dataset.csv  — 199 × (6 variant × 6 metric) no-drop
│   └── ablation_no_drop_summary.csv      — family 평균 (no-drop)
├── results/V13_RESULTS_REPORT.md      — 이 문서
└── models/{key}/checkpoint.pth        — best-snapshot 모델 (데이터셋별 1개)
```

9 variant + ablation 전체 재현:

```bash
cd V13/
# 9 variant (with-drop) — raw + z_train + z_val × {max, mean, median}
python scripts/compare_agg_normalize.py            # 199 데이터셋, 16 코어에서 ~5–10 분
python scripts/compare_agg_normalize.py --family Exathlon   # 한 family 만

# 채널-드랍 isolation ablation (no-drop) — raw + z_train × {max, mean, median}
python scripts/_ablation_no_drop.py                # 199 데이터셋, 16 코어에서 ~3–6 분
```

---

## 9. Production 권장

**기본 점수**: `z_train_max`.

```python
# 배포 전 1회:
mu_c, sigma_c = compute_train_baseline_stats(
    predictions_train, train_values_norm, lookback=L, pred_len=H
)
valid_c = np.isfinite(sigma_c) & (sigma_c > eps)   # 이 마스크도 같이 저장

# 배포 루프 (시점 t 마다):
D_wc_t = recency_weighted_variance(H_predictions_of_t)            # (C,)
z_t = (D_wc_t[valid_c] - mu_c[valid_c]) / sigma_c[valid_c]        # (C',)
score_t = max(z_t)                                                # 스칼라
```

| 속성 | 값 |
|:--|:--|
| 저장 비용 (데이터셋당) | $3C$ float: $\mu_c$, $\sigma_c$, valid mask |
| 시점당 연산 | $O(HC)$ |
| GT-free | ✓ (test value, label 모두 미사용) |
| 실시간 | ✓ ($\mu, \sigma$, mask 가 배포 전 산출된 상수) |
| raw_max baseline 대비 VUS-PR | **+0.0195** (199개 데이터셋 평균) |

## 10. 코드 검토 상태

독립 agent 의 end-to-end 검토 (2026-05-07):
- **BLOCKING 버그 0건** — 핵심 indexing, GT-free 클레임, n_pred clamp, val_start 도출 모두 정합 확인.
- **IMPORTANT 메모 7건** — channel-filter 비대칭 (7.1.A/B), vintage mixing (7.1.C), σ-mask persistence (7.1.D) 등. 대부분 헤드라인 수치를 한정 짓는 caveat 이지 버그 아님.
- **NICE-TO-HAVE 5건** — 정확성에 영향 없는 성능·코드 위생 개선 사항.

검토 대상 파일: `score_utils.py`, `01_data_preparation.py`, `02_train.py`, `03_inference.py`, `04_score_compute.py`, `05_metrics.py`, `compare_agg_normalize.py`.
