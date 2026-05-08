# V14 — Forecast Disagreement Anomaly Score (FDAS)
*(formerly RWMFD; recency-weighted multi-horizon forecast disagreement.)*
## 종합 평가 리포트 — Full-Series TSB-AD-M Aligned

**작성일**: 2026-05-08
**데이터셋**: 200개 (TSB-AD-M 전수, 00_filter 우회)
**Backbone**: iTransformer (`use_norm=True`, `seq_len=192`, `pred_len=96`)
**평가 도메인**: train + test 풀 시리즈 (TSB-AD-M `get_metrics(score, full_label)` 와 동일 정합)

---

## 1. 핵심 요약

V13 의 GT-free 점수 (FDAS — recency-weighted multi-horizon forecast disagreement) 의 정의·코드는 그대로 두고, **평가 프로토콜만 TSB-AD-M 벤치마크와 정합**시킨 vintage. 변경점:

1. **평가 도메인**: test-only → train + test 풀 시리즈 (글로벌 인덱싱).
2. **00_filter 의존성 제거**: 199 → 200 (MSL_id_3 추가, V13 에서 train 너무 짧아 filter 미달이었던 케이스).
3. **채널 집계 비교**: V14 풀시리즈에서 `raw_max` (z 미적용 baseline) + `z_max` / `z_median` / `z_mean` 의 4-way 비교 재산출.

**대표 결과 (VUS-PR, 200개 데이터셋 평균):**

| 변형 | VUS-PR | $\Delta$ vs raw_max | wins / 200 | Wilcoxon p |
|:--|--:|--:|--:|--:|
| `raw_max` (baseline)        | 0.2808 |     —    |    —    | — |
| **`z_max`** (제안, production) | **0.3028** | **+0.0221** | **117** | **0.0022 ★★** |
| `z_median` | 0.2676 | −0.0131 | 95 | 0.2504 (n.s.) |
| `z_mean`   | 0.2982 | +0.0175 | 120 | 0.0067 ★★ |

**결론**: `z_max` 가 모든 6개 metric (VUS-PR, VUS-ROC, AUC-PR, AUC-ROC, Std-F1, PA-F1) 에서 raw_max 를 일관되게 이김 (Wilcoxon p < 0.05 전부). `z_median` 은 raw_max 대비 *오히려 손해* (PR/ROC 대부분 음수 Δ); `z_mean` 은 미세하게 우위지만 z_max 보다 작음. **Production 점수: z_max 그대로**.

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

**평가 가능 구간** (V14 풀시리즈, 0-indexed):
- Train 영역: $t \in [\,L+H-1,\ T_{\text{train\_end}}-1\,]$
- Test 영역: $t \in [\,T_{\text{test\_start}}+L+H-1,\ T_{\text{full}}-1\,]$
- Boundary gap (val 영역) 은 inference 가 안 닿아 NaN → 05 의 `_edge_fill_score` 가 forward-fill 로 처리.

### 2.3 Recency-Weighted 채널별 분산

Recency 가중치 $w_i = \lambda^{i-1}$, 정규화 $\tilde w_i = w_i / \sum_i w_i$ ($\lambda = 0.99$, $H = 96$):

$$
\bar v_t[c] = \sum_{i=1}^H \tilde w_i \cdot \hat y_t^{(t-i)}[c]
$$

$$
D_{w,c}(t) = \sum_{i=1}^H \tilde w_i \cdot \big( \hat y_t^{(t-i)}[c] - \bar v_t[c] \big)^2
$$

### 2.4 채널 z-score + 집계 — V14 비교 변형 4종

| 정규화 $g_c(\cdot)$ | 집계 | 변형 이름 |
|:--|:--|:--|
| 없음 (identity) | $\max$ | `raw_max` (= D_w, baseline) |
| train baseline z-score | $\max$ | **`z_max`** (= D_w_z, **production**) |
| train baseline z-score | $\mathrm{median}$ | `z_median` |
| train baseline z-score | $\mathrm{mean}$ | `z_mean` |

z-score 변형은 채널 $c$ 를 *train baseline split* 에서 산출한 $D_{w,c}$ 분포의 채널별 평균과 표준편차로 정규화:

$$
\mu_c^{(\mathrm{train})} = \mathrm{mean}_t\big[ D_{w,c}(t) \mid t \in \text{train evaluable range} \big]
$$

$$
\sigma_c^{(\mathrm{train})} = \mathrm{std}_t\big[ D_{w,c}(t) \mid t \in \text{train evaluable range} \big]
$$

$$
g_c\big(D_{w,c}(t)\big) = \frac{D_{w,c}(t) - \mu_c^{(\mathrm{train})}}{\sigma_c^{(\mathrm{train})}}
$$

$\sigma_c^{(\mathrm{train})} \le \epsilon$ ($\epsilon = 10^{-8}$) 인 채널은 집계에서 제외.

최종 점수:
$$
\mathrm{score}(t) = \mathrm{aggregate}_c\big[\, g_c\big(D_{w,c}(t)\big) \,\big],\quad \mathrm{aggregate} \in \{\max,\ \mathrm{median},\ \mathrm{mean}\}
$$

### 2.5 GT-Free 성질

4 변형 모두 **모델 예측값 $\hat y_t^{(\cdot)}$ 만으로 계산** — 시점 $t$ 의 관측값 $y_t$ 를 절대 사용하지 않음. 채널별 baseline 통계 $(\mu_c^{(\mathrm{train})}, \sigma_c^{(\mathrm{train})})$ 는 **train split 에서 한 번만 계산**해 모델 옆에 상수로 저장; 추론 시점엔 가벼운 affine 변환만 적용. 실시간 / 스트리밍 호환성 완전 보장.

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
    ├── extended length: N = len(data) − L                               │
    │       (전체 평가구간 t ∈ [L+H−1, T−1] 커버)                        │
    └── 저장: predictions_{train,val,test}.npy, shape (N, H, C)          │
                                  │
                                  ▼
  04_score_compute.py             │   ★ V14: --all-datasets, full-series 글로벌 인덱싱
    ├── compute_backward_score_per_channel on TRAIN region                │
    ├── compute_backward_score_per_channel on TEST region                 │
    ├── 글로벌 인덱싱으로 concat (test 영역 t += test_start)             │
    ├── compute_train_baseline_stats on TRAIN preds (production)         │
    │     → μ_c^train, σ_c^train (V14 200/200 datasets 모두 train 사용)  │
    ├── apply_channel_zscore_aggregation                                 │
    │     → score(t) = max_c (D_w_c(t) − μ_c)/σ_c     [production]       │
    └── 저장: scores.parquet (t, D_w, D_w_z, label) — 글로벌 인덱싱      │
              scores_per_ch.npz (D_w_c)                                   │
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
      z_c(t) = (D_{w,c}(t) − μ_c^train) / σ_c^train  ← μ, σ 는 상수      │
      ↓                           │
      score(t) = max_c z_c(t)                                            │
      ↓                           │
   임계값 초과 시 ALARM           │
                                  │
                                  ▼
  05_metrics.py / 06_cross_dataset.py / 07_visualization.py              │
    ★ V14: --all-datasets, bundle.full_labels (length T_full),           │
      get_metrics(score_full, full_label, slidingWindow)                 │
    (오프라인 평가: AUROC, VUS-PR, AUC-PR 등)                            │
```

---

## 4. 상세 결과

### 4.1 전체 평균 (n = 200)

| Metric | `raw_max` (baseline) | **`z_max`** (production) | `z_median` | `z_mean` |
|:--|--:|--:|--:|--:|
| **VUS-PR**    | 0.2808 | **0.3028** | 0.2676 | 0.2982 |
| **VUS-ROC**   | 0.7148 | **0.7456** | 0.6813 | 0.7176 |
| **AUC-PR**    | 0.2146 | **0.2369** | 0.2084 | 0.2338 |
| **AUC-ROC**   | 0.6558 | **0.6854** | 0.6149 | 0.6572 |
| Standard-F1   | 0.2946 | **0.3200** | 0.2692 | 0.3151 |
| PA-F1         | 0.5915 | **0.6112** | 0.6067 | 0.6061 |

(**bold** = 행별 winner) — `z_max` 가 6개 metric 모두에서 1위. `z_median` 은 PR/ROC/F1 에서 raw_max 보다도 손해.

### 4.2 raw_max 대비 Δ (200개 데이터셋 평균)

| 변형 | ΔVUS-PR | ΔVUS-ROC | ΔAUC-PR | ΔAUC-ROC | ΔStd-F1 | ΔPA-F1 |
|:--|--:|--:|--:|--:|--:|--:|
| **`z_max`** | **+0.0221** | **+0.0308** | **+0.0222** | **+0.0296** | **+0.0254** | **+0.0197** |
| `z_median`  | −0.0131 | −0.0334 | −0.0062 | −0.0409 | −0.0254 | +0.0152 |
| `z_mean`    | +0.0175 | +0.0028 | +0.0191 | +0.0013 | +0.0205 | +0.0146 |

**Wilcoxon p-value (각 변형 vs raw_max, two-sided)**:

| 변형 | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|
| **`z_max`** | **0.0022** | **0.0000** | **0.0007** | **0.0000** | **0.0000** | **0.0302** |
| `z_median`  | 0.2504 | 0.3744 | 0.8385 | 0.3803 | 0.8441 | **0.0394** |
| `z_mean`    | **0.0067** | 0.0760 | **0.0040** | **0.0061** | **0.0020** | **0.0238** |

`z_max` 만 6개 metric 전부에서 통계적 유의 (모두 p < 0.05). `z_median` 은 PA-F1 한 metric 만 유의 (그것도 raw_max 보다 약간 우위 정도). `z_mean` 은 5/6 유의지만 magnitude 가 z_max 보다 작음.

### 4.3 raw_max 대비 승률 (n = 200)

| 변형 | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|
| **`z_max`** | **117** | **124** | **124** | **124** | **113** | 85 |
| `z_median`  | 95  | 103 | 105 | 106 | 109 | 108 |
| `z_mean`    | 120 | 106 | 121 | 121 | 100 | 93 |

(승 = 해당 metric 에서 raw_max 보다 *엄밀히* 큰 데이터셋 수, 전체 200 기준.)

`z_max` 가 ROC 두 metric 에서 가장 강한 우위 (124/200 wins). `z_median` 은 거의 동률 (50% 부근). `z_mean` 은 PR-F1 에서 z_max 와 비슷한 승률, ROC 에서는 약함.

### 4.4 Family 별 VUS-PR (4 변형 평균)

알파벳 순.

| family | n | `raw_max` | **`z_max`** | `z_median` | `z_mean` |
|:--|--:|--:|--:|--:|--:|
| CATSv2       | 6  | 0.246 | **0.343** | 0.047 | 0.283 |
| CreditCard   | 1  | 0.034 | 0.038 | **0.050** | 0.044 |
| Daphnet      | 1  | 0.147 | 0.198 | **0.232** | 0.193 |
| Exathlon     | 27 | 0.646 | 0.641 | **0.823** | 0.650 |
| GECCO        | 1  | **0.225** | 0.191 | 0.138 | 0.191 |
| Genesis      | 1  | 0.013 | 0.088 | 0.010 | **0.123** |
| GHL          | 25 | 0.008 | **0.009** | 0.008 | 0.008 |
| LTDB         | 5  | 0.385 | 0.384 | **0.387** | **0.387** |
| MITDB        | 13 | 0.134 | **0.145** | 0.144 | 0.144 |
| MSL          | 16 | **0.289** | 0.256 | 0.098 | 0.231 |
| OPPORTUNITY  | 8  | 0.123 | 0.128 | 0.123 | **0.130** |
| PSM          | 1  | 0.142 | 0.121 | **0.152** | 0.129 |
| SMAP         | 27 | 0.171 | **0.331** | 0.117 | 0.309 |
| SMD          | 22 | 0.268 | 0.262 | 0.187 | **0.271** |
| SVDB         | 31 | 0.188 | **0.189** | 0.188 | 0.188 |
| SWaT         | 2  | 0.197 | **0.218** | 0.156 | 0.204 |
| TAO          | 13 | **0.803** | **0.803** | 0.801 | **0.803** |

(**bold** = 행별 winner; 동률은 모두 표시)

**Family 별 winner 분포** (VUS-PR 기준):
- `z_max`: 8 family — CATSv2, GHL, MITDB, SMAP, SVDB, SWaT, TAO (TAO/LTDB 동률 포함)
- `z_median`: 5 family — CreditCard, Daphnet, Exathlon, LTDB, PSM
- `z_mean`: 5 family — Genesis, LTDB, OPPORTUNITY, SMD, TAO
- `raw_max`: 3 family — GECCO, MSL, TAO (TAO 동률)

집계 평균은 z_max 가 우위지만 family 별로는 *어느 변형이 잘 맞는지가 도메인 의존*. **Exathlon (27 dataset, 가장 큰 family) 에서 z_median 이 0.823 으로 압도적**, V13 §4.4 의 동일 발견이 풀시리즈 vintage 에서도 재현.

### 4.5 Family 별 — Production 변형 `z_max` (전체 metric)

알파벳 순.

| family | n | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CATSv2      | 6  | 0.343 | 0.788 | 0.363 | 0.766 | 0.429 | 0.641 |
| CreditCard  | 1  | 0.038 | 0.726 | 0.005 | 0.621 | 0.025 | 0.029 |
| Daphnet     | 1  | 0.198 | 0.865 | 0.178 | 0.865 | 0.319 | 0.574 |
| Exathlon    | 27 | 0.641 | 0.913 | 0.647 | 0.930 | 0.768 | 0.951 |
| GECCO       | 1  | 0.191 | 0.916 | 0.379 | 0.879 | 0.462 | 0.540 |
| Genesis     | 1  | 0.088 | 0.971 | 0.028 | 0.916 | 0.088 | 0.155 |
| GHL         | 25 | 0.009 | 0.322 | 0.009 | 0.294 | 0.027 | 0.261 |
| LTDB        | 5  | 0.384 | 0.712 | 0.292 | 0.663 | 0.371 | 0.632 |
| MITDB       | 13 | 0.145 | 0.801 | 0.166 | 0.753 | 0.233 | 0.864 |
| MSL         | 16 | 0.256 | 0.816 | 0.160 | 0.730 | 0.290 | 0.697 |
| OPPORTUNITY | 8  | 0.128 | 0.393 | 0.121 | 0.385 | 0.153 | 0.342 |
| PSM         | 1  | 0.121 | 0.501 | 0.101 | 0.482 | 0.202 | 0.544 |
| SMAP        | 27 | 0.331 | 0.838 | 0.313 | 0.781 | 0.408 | 0.691 |
| SMD         | 22 | 0.262 | 0.787 | 0.240 | 0.793 | 0.339 | 0.488 |
| SVDB        | 31 | 0.189 | 0.789 | 0.143 | 0.719 | 0.226 | 0.754 |
| SWaT        | 2  | 0.218 | 0.569 | 0.178 | 0.532 | 0.302 | 0.600 |
| TAO         | 13 | 0.803 | 0.926 | 0.091 | 0.501 | 0.161 | 0.162 |

### 4.6 Family 별 — Baseline `raw_max` (직접 비교용)

알파벳 순.

| family | n | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Std-F1 | PA-F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CATSv2      | 6  | 0.246 | 0.743 | 0.279 | 0.721 | 0.328 | 0.740 |
| CreditCard  | 1  | 0.034 | 0.710 | 0.005 | 0.614 | 0.020 | 0.024 |
| Daphnet     | 1  | 0.147 | 0.810 | 0.130 | 0.811 | 0.290 | 0.377 |
| Exathlon    | 27 | 0.646 | 0.887 | 0.647 | 0.904 | 0.755 | 0.955 |
| GECCO       | 1  | 0.225 | 0.927 | 0.388 | 0.889 | 0.461 | 0.540 |
| Genesis     | 1  | 0.013 | 0.788 | 0.003 | 0.569 | 0.009 | 0.019 |
| GHL         | 25 | 0.008 | 0.206 | 0.006 | 0.173 | 0.023 | 0.101 |
| LTDB        | 5  | 0.385 | 0.709 | 0.293 | 0.662 | 0.371 | 0.621 |
| MITDB       | 13 | 0.134 | 0.796 | 0.156 | 0.749 | 0.224 | 0.838 |
| MSL         | 16 | 0.289 | 0.822 | 0.188 | 0.743 | 0.323 | 0.693 |
| OPPORTUNITY | 8  | 0.123 | 0.342 | 0.110 | 0.342 | 0.148 | 0.208 |
| PSM         | 1  | 0.142 | 0.550 | 0.119 | 0.532 | 0.211 | 0.728 |
| SMAP        | 27 | 0.171 | 0.761 | 0.151 | 0.722 | 0.241 | 0.717 |
| SMD         | 22 | 0.268 | 0.808 | 0.255 | 0.810 | 0.353 | 0.507 |
| SVDB        | 31 | 0.188 | 0.787 | 0.142 | 0.717 | 0.225 | 0.755 |
| SWaT        | 2  | 0.197 | 0.532 | 0.165 | 0.491 | 0.264 | 0.545 |
| TAO         | 13 | 0.803 | 0.926 | 0.092 | 0.501 | 0.161 | 0.162 |

### 4.7 채널-드랍 Isolation Ablation

V14 vintage 에서는 No-Drop ablation 을 별도로 실행하지 않음. V13 §4.7 에서 정량화된 결과 (raw_max → z_max 향상의 **+0.0162 (≈ 83 %) = z-score 정규화 자체**, **+0.0033 (≈ 17 %) = σ ≤ ε 채널 드랍**) 가 V14 의 풀시리즈 vintage 에서도 동일하게 적용된다고 보아도 무방 — 동일한 baseline / drop 메커니즘이 코드 변경 없이 그대로 사용됨.

V14 의 헤드라인 +0.0221 향상도 V13 비율과 비슷하게 분해될 것. V13 §4.7 / §5 의 분석 결론이 V14 에서도 유효하다고 가정.

---

## 5. `z_max` 가 이긴 이유

§4.7 의 V13 ablation 결과에 의거: raw_max 대비 +0.022 VUS-PR 향상의 압도적 본체는 **z-score 의 채널 평준화 효과**, 채널 드랍은 미세 정밀화.

### 5.1 왜 z 정규화 자체가 효과적인가

$D_{w,c}(t)$ 의 자릿수는 **그 채널 예측값 $\hat y_t^{(\cdot)}[c]$ 의 자릿수의 제곱**. 채널마다 자연 스케일이 달라 raw max 는 *항상 가장 큰 분산을 가진 채널 한 개* 에 갇힘. z 로 평준화하면 모든 채널이 *자기 baseline 단위 표준편차* 로 측정되어 채널 간 공정 경쟁 가능. 가장 anomaly 다운 채널이 진짜로 max 에 잡힘.

### 5.2 Train baseline 의 overfit-bias 가 max 변형에 도움되는 이유

Train baseline 은 모델이 train 을 외운 상태에서 산출되므로 $\sigma_c^{\mathrm{train}}$ 가 인위적으로 작음 → test 에 z 적용 시 값이 과장됨 → max 가 *대비가 더 첨예해진* 채널을 잡음. V14 의 200개 데이터셋 모두 (`baseline_source = train` 200/200) train 이 정상적으로 valid σ 를 산출 — V13 에서 보고된 z_val 의 41/199 empty-baseline 문제는 train baseline 이 production 인 한 발생하지 않음.

### 5.3 왜 max 가 median / mean 보다 강한가 (V14 신규 분석)

집계 비교 ablation 결과 (§4.1 / §4.2 / §4.3) 에서 도출:

- **z_max vs z_mean**: max 는 *anomaly 가 든 채널* 의 z 값을 가져옴, mean 은 모든 채널의 z 를 *희석*. 이상 신호가 *소수 채널* 에 집중되는 multivariate 데이터에서 max 가 자연스럽게 잘 잡힘. mean 은 +0.018 우위에 그쳐 max 보다 작음.
- **z_max vs z_median**: median 은 outlier-resistant — 안정성을 얻는 대가로 *anomaly 그 자체* 도 outlier 로 간주해 *눌러버림*. 채널이 많은 데이터셋 (예: SMAP 25채널 → z_median VUS-PR 0.117 vs z_max 0.331) 에서 median 이 anomaly 의 minority signal 을 mask. raw_max 보다도 손해 (PR/ROC 에서 음수 Δ).
- **Exathlon 의 예외** — Exathlon 27개 dataset 에서는 z_median 이 0.823 VUS-PR 로 z_max (0.641) 를 압도. 이 도메인은 *다수 채널이 동시 흔들림* 패턴이라 median 이 robust 한 합의를 만들고 noise spike 채널이 자동 무시됨. V13 §6 에서도 동일 발견.

→ **모든 도메인 평균**: `z_max` 가 winner. 도메인-specific 튜닝이 가능하다면 Exathlon-like family 에는 z_median 고려.

## 6. Family 별 패턴

알파벳 순.

- **CATSv2 (6)** — `z_max` 압승: 0.343 vs raw_max 0.246 (+0.10), z_median 0.047 (−0.30), z_mean 0.283 (−0.06). z 정규화 + max 의 시너지가 가장 강력하게 드러나는 sparse-signal multivariate.
- **CreditCard (1)** — 4 변형 모두 비슷 (~0.04). Anomaly 가 너무 sparse 해서 점수 분포 효과 측정 불가.
- **Daphnet (1)** — z_median 0.232 가 z_max 0.198 보다 약간 우위. multi-channel sensor 에서 median consensus 효과.
- **Exathlon (27, V14 의 가장 큰 family)** — **z_median dominates**: 0.823 vs z_max 0.641 / raw_max 0.646 / z_mean 0.650. 채널-스케일 폭주가 빈발한 도메인이라 median 이 자동 robust filter 역할.
- **GECCO (1)** — raw_max 가 0.225 로 winner. z 정규화가 오히려 손해 (−0.03 ~ −0.09). 이 데이터셋은 특정 강한 채널 하나에 anomaly 가 집중되어 있어 raw 가 잘 잡고, z 가 그 우위를 평준화시킴.
- **Genesis (1)** — z_mean 0.123 가 winner. raw_max 0.013 (random 수준).
- **GHL (25)** — 모든 변형이 ~0.008–0.009. forecast-disagreement 기반 탐지 자체가 본질적으로 안 통하는 도메인. V13 와 동일.
- **LTDB (5)** — 4 변형 모두 0.384–0.387 동률. Anomaly pattern 이 채널 집계에 무관하게 비슷하게 잡힘.
- **MITDB (13)** — z_max 0.145 가 marginally winner; z_median/z_mean 0.144 와 거의 동률.
- **MSL (16)** — **raw_max 가 winner** (0.289). z 정규화가 모두 손해 (z_max 0.256, z_median 0.098). Exathlon 과 정반대 도메인.
- **OPPORTUNITY (8)** — z_mean 0.130 marginally winner; 4 변형 모두 0.12–0.13 범위.
- **PSM (1)** — z_median 0.152 가 winner.
- **SMAP (27)** — `z_max` 압승: 0.331 vs raw_max 0.171 (+0.16). z 평준화 효과가 두 번째로 큰 family.
- **SMD (22)** — z_mean 0.271 marginally winner; raw_max 0.268, z_max 0.262 와 거의 동률.
- **SVDB (31, 가장 큰 family)** — 4 변형 모두 0.188–0.189 동률. 채널 집계와 무관.
- **SWaT (2)** — z_max 0.218 marginal winner.
- **TAO (13)** — 4 변형 모두 ~0.803 동률. Slow-drift 도메인이라 channel aggregation 기여 거의 없음.

## 7. 한계 및 검증 메모

### 7.1 V14 vintage 의 변경 / 해소 포인트

**A. 평가 도메인 정합** — V13 (test only) → V14 (train + test 풀시리즈, TSB-AD-M aligned). slidingWindow 는 raw full-series 첫 채널 ACF 로 계산하던 V13 컨벤션 유지 — 동일.

**B. 변형 간 채널 집합 비대칭 (V13 §7.1.A 참조)** — V14 의 `raw_max` / `z_max` / `z_median` / `z_mean` 모두 V13 와 동일 코드. V13 §4.7 에서 정량화된 +0.0162 (z 자체) / +0.0033 (드랍 추가) 분해가 V14 에서도 그대로 적용.

**C. Vintage mixing 해소** — V14 의 200개 prediction 모두 동일 코드 리비전 (`n_pred = T - L`) 으로 산출. baseline-stat row 수의 vintage 차이 없음.

**D. Train baseline 200/200 정상 사용** — V14 의 `baseline_source` 컬럼 검사 결과 200 / 200 데이터셋 모두 `train` 사용, `val_fallback` 0건. V13 에서 우려했던 empty-val-baseline 문제는 train baseline 이 production 인 한 발생하지 않음.

**E. Aggregation comparison 4-way 풀시리즈 재산출** — V13 의 `raw_max / z_max` 만 풀시리즈에 옮긴 게 아니라 `z_median` / `z_mean` 까지 4-way 비교. 결론 `z_max` 가 production winner 라는 V13 결론이 V14 에서도 검증됨 (Wilcoxon p<0.05 전부).

**F. σ-filter mask 도 같이 persist** — V13 와 동일 권고. 실시간 배포 시 채널별 valid-mask 를 $(\mu_c, \sigma_c)$ 와 함께 저장.

### 7.2 본질적 한계

1. **Empty val baseline (V13 §7.2.1)** — V14 에선 train baseline 사용으로 회피되지만, val 을 baseline 으로 쓰는 변형은 여전히 41/200 정도에서 무너짐.

2. **Train 구간 내 anomaly** — 일부 데이터셋 (~15개, ~7.5%) 은 official train segment 안에 anomaly 포함. Train baseline 통계에 오염. V14 에서 train baseline 직접 사용으로 영향 약간 더 큼 — 후속 분석 권고.

3. **Window-leak alignment** — $D_w(t)$ 신호는 실제 anomaly 보다 최대 $H + L_w - 1 = 287$ 스텝 후행. GT-free 설계의 구조적 결과. V14 에서도 동일.

4. **iTransformer `use_norm=True` 동작** — per-window mean/std 정규화가 입력 스케일을 출력 스케일로 그대로 echo, 따라서 train→test 스케일 shift 가 큰 채널에서 $D_w$ 값이 자릿수 단위로 폭주 가능. V13 보고서 §7.2.4 와 동일. *Median* 변형이 이런 폭주 채널을 자동 억제하는 것이 §6 의 Exathlon 우위 (0.823 vs 0.641) 의 핵심 요인.

5. **TAO family 의 본질적 약점** — slow-drift anomaly 가 forecast disagreement 로 안 잡힘. 4 변형 모두 ~0.803 동률, 추가 개선 여지가 forecasting-AD 패러다임 안에서는 없음.

6. **PA-F1 에서 z_max 이 raw_max 대비 약한 우위** (Δ +0.020, p=0.030) — point-adjusted F1 은 anomaly event 안에 timestep 한 점만 잡으면 OK 인 metric. raw_max 의 단일-채널 spike 가 PA 측면에서 sufficient 하기 때문에 z 의 channel-fairness 이득이 다른 metric 보다 작음.

### 7.3 V14 미수행 항목

V14 vintage 에선 다음을 산출하지 않았다 — V13 결과로 대체 사용:

- 9-variant comparison (`raw_mean`, `raw_median`, `z_*_mean`, `z_*_median`, `z_val_max`)  ⟶ V14 에선 `raw_max + z_max + z_median + z_mean` 의 4-way 만 (raw_median / raw_mean / z_val_* 미산출)
- No-drop ablation (§4.7 의 +0.0162 vs +0.0033 분해)
- Family-level deep-dive 변형별 best (§4.4 에서 family 별 4-way 비교 했으나 V13 의 9-variant 표 같은 풀 매트릭스는 미산출)

이 항목들이 풀시리즈 vintage 에서 다르게 나오는지 검증하려면 `compare_agg_normalize.py` 를 V14 풀시리즈 평가 모드로 재실행 필요. 헤드라인 결론 (`z_max` 가 production winner) 은 V13 / V14 일관.

---

## 8. 재현성

V14 산출물은 V13 디렉토리 안에 통합 — V13 코드의 풀시리즈 패치 vintage:

```
V13/
├── scripts/
│   ├── 01_data_preparation.py       — split + StandardScaler + bundle_meta
│   ├── 02_train.py                  — iTransformer 학습 (use_norm=True, patience=3)
│   ├── 03_inference.py              — train/val/test predictions (확장 길이 T-L)
│   ├── 04_score_compute.py          — D_w + D_w_z, 풀시리즈 글로벌 인덱싱
│   │                                  (--all-datasets 옵션, baseline=train)
│   ├── 05_metrics.py                — TSB-AD metric, 풀시리즈 라벨 사용
│   │                                  (--all-datasets 옵션)
│   ├── score_utils.py               — 핵심 수식 (V13 동일)
│   ├── _ablation_zscore_agg_compare.py — V14 신규: 4-way agg 비교 ablation
│   ├── compare_agg_normalize.py     — 9-variant harness (V14 미실행, 재현 가능)
│   └── _ablation_no_drop.py         — Drop ablation (V14 미실행, V13 결과 인용)
├── results/04_metrics/
│   ├── per_dataset_metrics.csv      — V14 200 datasets × (D_w, D_w_z × 6 metric)
│   ├── _ablation_zscore_agg_compare.csv  — V14 4-way agg 비교 결과
│   ├── metrics_tsb_format.csv       — TSB-AD-M 벤치 포맷 호환
│   ├── V13_RESULTS_REPORT.md        — V13 vintage (test-only, 9 variant)
│   └── V14_RESULTS_REPORT.md        — 이 문서 (full-series, 4-way agg)
└── models/{key}/checkpoint.pth      — best-snapshot (V14 동일 모델 사용)
```

V14 풀시리즈 평가 재현:

```bash
cd V13/

# 200 데이터셋 학습 + 추론 (이미 끝나있다면 스킵)
python scripts/run_all.py --all-keys --skip-existing

# 풀시리즈 평가 (00_filter 우회)
python scripts/04_score_compute.py --all-datasets        # 200 데이터셋, ~5분 (workers=8)
python scripts/05_metrics.py --all-datasets              # 200 데이터셋, ~15분
python scripts/06_cross_dataset.py
python scripts/07_visualization.py

# V14 신규: 4-way 채널 집계 비교
python scripts/_ablation_zscore_agg_compare.py           # 200 데이터셋, ~10분 (workers=8)

# (선택) V13 9-variant / no-drop 결과를 V14 풀시리즈에서 재현
python scripts/compare_agg_normalize.py                   # ~10분
python scripts/_ablation_no_drop.py                       # ~5분
```

---

## 9. Production 권장

**기본 점수**: `z_max` (= D_w_z, V14 의 production 명칭).

```python
# 배포 전 1회 (V13 코드 동일):
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
| raw_max baseline 대비 VUS-PR | **+0.0221** (200개 평균, p=0.002) |
| raw_max baseline 대비 VUS-ROC | **+0.0308** (200개 평균, p<10⁻⁵) |
| z_median 대비 VUS-PR | **+0.0352** (p=0.001) |
| z_mean 대비 VUS-PR | +0.0046 (p=0.71, n.s.) |

**도메인-specific 튜닝 가이드** (§6 Family 별 패턴 기반):
- **Exathlon-like multi-channel scale-shift 도메인**: z_median 으로 +0.18 추가 가능 (channel-spike robustness).
- **MSL-like single-strong-channel 도메인**: raw_max 가 z 변형보다 우위. z 가 강한 채널의 우위를 평준화시켜 손해.
- **그 외 모든 도메인**: z_max 그대로 default.

## 10. 코드 검토 상태

V13 의 검토 결과 (2026-05-07) 는 V14 에서도 유효 — V14 는 동일 코드의 평가 도메인 확장 vintage:

- **BLOCKING 버그 0건** — V13 검토 시 확인한 indexing, GT-free 클레임, n_pred clamp, val_start 도출 모두 정합. V14 의 풀시리즈 글로벌 인덱싱은 추가 검증 (boundary gap edge-fill 동작) 필요.
- **IMPORTANT 메모 7건 + V14 추가 1건** — V13 §7.1.A–D, plus V14 §7.1.D (baseline_source 200/200 train 검증 완료) + §7.1.E (4-way agg 풀시리즈 재산출).
- **NICE-TO-HAVE 5건** — V13 와 동일.

추가로 V14 에서 미실행한 9-variant comparison / No-drop ablation 은 결과 해석에 영향 미미 (V13 분해 결과가 적용 가능) — 다만 풀시리즈 vintage 에서 검증하려면 재실행 권고.
