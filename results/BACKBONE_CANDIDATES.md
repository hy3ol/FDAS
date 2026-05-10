# Backbone Candidates — FDAS Cross-Backbone Validation Plan

**작성일**: 2026-05-10
**대상 vintage**: V16 (예정, cross-backbone evaluation)
**목적**: V13 framework의 `BACKBONES` registry에 추가할 forecasting backbone 5개의 paper 정보, 공개 코드, 호환성, V13 통합 청사진 정리.

---

## 1. Method Statement (paper-level narrative draft)

**Forecast Disagreement Anomaly Score (FDAS)** — 동일 시점 $t$ 에 대한 multi-horizon forecasting model 의 *서로 다른 lead-time 예측들* 간 가중 분산을 anomaly score 로 사용하는 GT-free 접근법.

### 1.1 핵심 설계
$f_\theta : \mathbb{R}^{L \times C} \to \mathbb{R}^{H \times C}$ 가 normal data 로 학습된 multi-horizon forecaster 일 때, 시점 $t$ 의 anomaly score:

$$
D_w(t) = \max_c \sum_{i=1}^{H} \tilde w_i \cdot \big( \hat y_t^{(t-i)}[c] - \bar v_t[c] \big)^2
$$

여기서 $\hat y_t^{(t-i)}$ 는 anchor $t-i$ 에서 만든 $i$-step ahead forecast 의 시점 $t$ 성분, $\bar v_t$ 는 가중 평균, $\tilde w_i = \lambda^{i-1} / \sum_j \lambda^{j-1}$ ($\lambda=0.99$).

### 1.2 이 method 의 backbone-agnostic 가설

"FDAS 의 효과 (잔차 score 대비 우위) 는 backbone 의 forecasting paradigm 과 무관하게 일관된다."

**검증 design**: 5개의 paradigm-diverse forecasting backbone 에 동일 FDAS scoring 을 적용, TSB-AD-M 200 datasets 에서 잔차 score 와 비교.

### 1.3 GT-free 성질
- $D_w(t)$ 는 **모델 예측 $\hat y_t^{(\cdot)}$ 만 사용** — 시점 $t$ 의 실제 관측값 $y_t$ 를 보지 않음.
- 채널별 baseline $(\mu_c^{train}, \sigma_c^{train})$ 도 training predictions 에서만 산출.
- 실시간 / 스트리밍 호환.

### 1.4 V13 framework 와의 정합

V13 의 `BackboneSpec` 인터페이스를 따르는 모든 backbone 에 대해 FDAS scoring stage (`04_score_compute.py`, `score_utils.py`) 는 코드 변경 없이 적용. backbone 이 stride-1 inference 로 `(N, H, C)` predictions tensor 를 생산하기만 하면 충분.

---

## 2. Backbone Candidates

### 2.1 iTransformer (이미 V13 에 vendored, baseline)

| 속성 | 값 |
|:--|:--|
| 논문 | *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* |
| 저자 | Liu et al. (THUML) |
| 학회 | ICLR 2024 |
| arXiv | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) |
| 공식 코드 | [thuml/iTransformer](https://github.com/thuml/iTransformer) |
| TSL 통합 | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) `models/iTransformer.py` |
| 핵심 아이디어 | Token = *variate* (channel) 단위, attention 이 채널 간 상호작용을 모델링; 시간축은 MLP 로 처리 |
| Long-horizon 강점 | Channel-as-token 구조가 channel correlation 을 잘 잡음 → 96/192/336/720 horizon benchmark 에서 SOTA |

**V13 등록 상태**: ✅ 이미 등록 완료 (`V13/model/iTransformer.py` + `BACKBONES["iTransformer"]`).

---

### 2.2 PatchTST (channel-independent baseline)

| 속성 | 값 |
|:--|:--|
| 논문 | *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* |
| 저자 | Nie, Nguyen, Sinthong, Kalagnanam (IBM Research / Princeton) |
| 학회 | ICLR 2023 |
| arXiv | [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) |
| 공식 코드 | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) |
| TSL 통합 | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) `models/PatchTST.py` |
| 핵심 아이디어 | (1) 각 채널을 *patch* 단위 token sequence 로 변환 (이미지 ViT 차용), (2) 채널 간 weight sharing 으로 channel-independent 모델링 |
| Long-horizon 강점 | Patch tokenization 이 sequence length 압축 → 720 horizon 까지 안정적; channel-independent 가 작은 데이터셋에 robust |

**V13 vendor 난이도**: 🟢 쉬움 (TSL native, V13 `layers/` 와 호환).
**필요 layer**: `Embed.PatchEmbedding` (TSL 에 있음, V13 에 없음 → 추가 필요).

---

### 2.3 TimeXer (exogenous-aware Transformer)

| 속성 | 값 |
|:--|:--|
| 논문 | *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables* |
| 저자 | Wang et al. (THUML) |
| 학회 | NeurIPS 2024 |
| arXiv | [arXiv:2402.19072](https://arxiv.org/abs/2402.19072) |
| 공식 코드 | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) `models/TimeXer.py` (별도 repo 미존재 — TSL 에 통합) |
| 핵심 아이디어 | Endogenous (target) 와 exogenous (covariate) variable 를 *별도 토큰화* → cross-attention 으로 결합 |
| Long-horizon 강점 | Exogenous channel 활용으로 forecasting 신호 보강; endogenous-only 모드도 지원 |

**V13 vendor 난이도**: 🟡 중간 — TimeXer 는 본래 exogenous 분리를 가정하는데 V13 setup 은 모든 channel 을 endogenous 로 다룸. **endogenous-only mode** 로 호출하거나, 임의로 channel split (예: 첫 채널만 endogenous) 으로 우회.

**Paper-grade 의의**: Exogenous-aware backbone 도 FDAS 에 호환됨을 보이면 narrative 의 generality 강화. 다만 V13 의 anomaly detection setup 과 model design assumption 이 약간 mismatch — 결과 해석 시 caveat 명시 필요.

---

### 2.4 TimeMixer (multi-scale MLP-Mixer)

| 속성 | 값 |
|:--|:--|
| 논문 | *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting* |
| 저자 | Wang et al. (THUML) |
| 학회 | ICLR 2024 |
| arXiv | [arXiv:2405.14616](https://arxiv.org/abs/2405.14616) |
| 공식 코드 | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) |
| TSL 통합 | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) `models/TimeMixer.py` |
| 핵심 아이디어 | 시계열을 *multi-scale* (down-sample 여러 단계) 로 분해 → MLP-Mixer 로 scale 간 + scale 내 정보 융합. Transformer 의 attention 대체 |
| Long-horizon 강점 | Multi-scale 구조가 long-horizon 의 trend / seasonality 둘 다 capture; Transformer 보다 가벼움 |

**V13 vendor 난이도**: 🟢 쉬움 (TSL native, MLP 기반이라 추가 layer dependency 적음).

**Paper-grade 의의**: **Non-Transformer paradigm** 추가. "FDAS 가 attention 없이도 작동" 입증.

---

### 2.5 DLinear (simple linear baseline)

| 속성 | 값 |
|:--|:--|
| 논문 | *Are Transformers Effective for Time Series Forecasting?* |
| 저자 | Zeng, Chen, Zhang, Xu (CUHK) |
| 학회 | AAAI 2023 |
| arXiv | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) |
| 공식 코드 | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) |
| TSL 통합 | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) `models/DLinear.py` |
| 핵심 아이디어 | (1) Series decomposition (trend + seasonal), (2) 각 component 에 대해 단일 Linear layer 로 직접 forecast |
| Long-horizon 강점 | Linear projection 이 long-horizon 에 의외로 강력 — 이 paper 가 "Transformer 가 LTSF 에서 정말 필요한가?" 라는 질문을 제기한 milestone |

**V13 vendor 난이도**: 🟢 매우 쉬움 (40~80 줄, layer dependency 없음).

**Paper-grade 의의**: **Simplest possible baseline.** DLinear 에서도 FDAS 가 동작하면 "Transformer 도 MLP-Mixer 도 필요 없이, 단순 linear forecaster 만 있어도 FDAS 가 anomaly detection 으로 작동" → **가장 강한 generality 주장**.

---

## 3. Comparison Matrix

| 모델 | 학회/year | 패러다임 | TSL native | Vendor 난이도 | Long-horizon SOTA 시점 |
|:--|:--:|:--:|:--:|:--:|:--:|
| **iTransformer** | ICLR 2024 | Transformer (channel-token) | ✅ | ✅ done | 2024 SOTA |
| **PatchTST** | ICLR 2023 | Transformer (patch) | ✅ | 🟢 easy | 2023 SOTA |
| **TimeXer** | NeurIPS 2024 | Transformer (exo-aware) | ✅ | 🟡 medium | 2024 SOTA |
| **TimeMixer** | ICLR 2024 | MLP-Mixer (multi-scale) | ✅ | 🟢 easy | 2024 strong |
| **DLinear** | AAAI 2023 | Linear (decomposition) | ✅ | 🟢 trivial | 2022 milestone |

---

## 4. V13 Integration Plan

### 4.1 단일 backbone 추가 절차 (V13 framework 기준)

각 backbone 별로:

**Step 1.** TSL 에서 model 파일 vendor:
```bash
# Time-Series-Library 에서 가져옴
cp Time-Series-Library/models/<Name>.py V13/model/<Name>.py
```

**Step 2.** Layer dependency 점검 — 추가로 필요한 layer 가 있으면 V13/layers/ 로 vendor (예: PatchTST 는 `Embed.PatchEmbedding` 추가 필요).

**Step 3.** `V13/model/__init__.py` 의 `BACKBONES` dict 에 entry 추가:
```python
"PatchTST": BackboneSpec(
    name="PatchTST",
    model_factory=lambda cfg: _patchtst.Model(cfg),
    default_model_hps=dict(
        # paper 권장 HP
        d_model=128, n_heads=16, e_layers=3, d_ff=256,
        dropout=0.2, factor=1, activation="gelu",
        embed="timeF", freq="h",
        patch_len=16, stride=8,        # PatchTST-specific
        output_attention=False,
    ),
    default_training_hps=dict(
        batch_size=128, learning_rate=1e-4,
        num_epochs=10, patience=3,
        optimizer="adam", scheduler="none",
    ),
    extra_config_fields=[
        "d_model", "n_heads", "e_layers", "d_ff",
        "dropout", "factor", "activation", "embed", "freq",
        "patch_len", "stride", "output_attention",
    ],
    forward_signature="tsl",
),
```

**Step 4.** 등록 검증 (no-train smoke test):
```bash
python -c "
import sys; sys.path.insert(0,'.')
from model import get_backbone
from config_factory import build_config
from artifact_paths import load_data_metadata
spec = get_backbone('PatchTST')
cfg = build_config({'lookback':192,'pred_len':96,'num_channels':25}, spec)
m = spec.model_factory(cfg)
import torch
y = m(torch.randn(2,192,25), torch.zeros(2,192,1), None, torch.zeros(2,96,1))
print('output:', tuple(y.shape))   # 기대: (2, 96, 25)
"
```

**Step 5.** 200 데이터셋 학습 + 추론:
```bash
python scripts/run_all.py --all-keys --skip-existing --backbone PatchTST
python scripts/run_all.py --analyze --backbone PatchTST
```

**Step 6.** 결과는 `V13/results/04_metrics/per_dataset_metrics__PatchTST.csv` 에 자동 산출.

### 4.2 권장 구현 순서

| 순서 | 모델 | 이유 |
|:--:|:--|:--|
| 1 | **DLinear** | 가장 가벼움 (40줄) → framework 의 backbone-pluggable 동작 확인용 sanity check; "simplest baseline 도 FDAS 호환" narrative 의 첫 piece |
| 2 | **PatchTST** | TSL native + 가장 인용 많은 forecasting baseline → 외부 reviewer 가 가장 먼저 묻는 backbone |
| 3 | **TimeMixer** | Non-Transformer paradigm 의 SOTA → "FDAS 가 attention 무관하게 작동" 주장 |
| 4 | **TimeXer** | 마지막 — exogenous handling assumption 이 우리 setup 과 약간 mismatch 라 caveat 분석 필요 |

iTransformer 는 이미 baseline 으로 산출돼있으니 위 4개를 위 순서대로 추가.

### 4.3 학습 시간 추정 (200 datasets)

V14 의 iTransformer 200 데이터셋 학습 시간을 baseline 으로 (대략 수 시간 ~ 하루):
- DLinear: iTransformer 대비 **<10%** (linear 라 매우 가벼움)
- PatchTST: iTransformer 대비 **80~120%** (비슷한 capacity)
- TimeMixer: iTransformer 대비 **50~80%**
- TimeXer: iTransformer 대비 **80~120%**

→ 4개 추가로 V14 의 **~3배 GPU-hour** 예상. 한 번에 다 안 돌려도 됨 — `--skip-existing` 으로 incremental 가능.

---

## 5. Risk / Caveat

### 5.1 결과가 backbone 마다 갈릴 수 있음 (V14 dummy 결과로부터의 lesson)

V14 단계에서 CNN/LSTMAD 에 대한 dummy test (PSM/GHL/Genesis 3 데이터셋, 각 backbone 의 best HP) 에서 **Δ VUS-PR 부호가 (backbone, dataset) 조합에 따라 갈렸음** (3 양수 / 3 음수). 5개 forecasting backbone 도 비슷한 패턴 가능.

**대응**:
- "FDAS 가 *모든* backbone 에서 잔차를 능가" 라는 strong claim 은 risky.
- "FDAS 는 forecasting backbone 의 설계 선택과 *무관하게 동작 가능* 한 scoring 방법" 이라는 weak narrative 가 안전.
- backbone 별로 어떤 dataset family 에서 잘 되는지 분석 (V14 의 §6 family-pattern 형식) → 더 풍부한 paper.

### 5.2 학습 HP 통일 vs. backbone-best 통일

V13 framework 의 결정 (형 제안 반영): 각 backbone 은 paper 권장 HP 를 그대로 사용. 이게 narrative 에 더 강함.

다만 reviewer 가 "fair comparison 위해 동일 HP 도 같이 보여달라" 요청 가능. 그럴 때를 대비해 **fixed-HP secondary table** 도 산출 가능하도록 V13 framework 가 이미 지원 (training HP 를 BackboneSpec override 또는 CLI 로 강제 가능).

### 5.3 TimeXer 의 endogenous/exogenous split

TimeXer 는 channel 을 endogenous 와 exogenous 로 분리하는 설계. V13 의 anomaly detection setup 은 모든 채널을 동등하게 다룸. **모든 채널을 endogenous 로 통일** 하는 mode 호출이 가장 안전 — TimeXer paper 의 ablation 에도 endogenous-only mode 가 포함되어 있어 합리적.

대안: 첫 1~3 채널만 endogenous, 나머지를 exogenous 로 보내는 식의 split. 그러나 dataset 마다 channel 의미가 달라 자동 split 기준 잡기 어려움. **endogenous-only 권장**.

---

## 6. Action Items (다음 step)

- [ ] (Phase 0 — 이 문서 검토) ← 지금 단계
- [ ] (Phase 1) DLinear vendor + `BACKBONES` 등록 + Genesis sanity test
- [ ] (Phase 2) PatchTST vendor + `BACKBONES` 등록 + Genesis sanity test
- [ ] (Phase 3) TimeMixer vendor + `BACKBONES` 등록 + Genesis sanity test
- [ ] (Phase 4) TimeXer vendor (endogenous-only mode) + `BACKBONES` 등록 + sanity test
- [ ] (Phase 5) 200 데이터셋 cross-backbone 학습 + 추론 (`--skip-existing` incremental)
- [ ] (Phase 6) Cross-backbone analysis: V14 §1-§9 형식의 V16_RESULTS_REPORT.md 작성
- [ ] (Phase 7) Family-pattern 분석: 각 backbone 이 어떤 dataset family 에서 잘 되는지
- [ ] (Phase 8) Ablation: backbone-best HP vs. 통일 HP secondary comparison

---

## 7. References (one-line for paper bibliography)

```bibtex
@inproceedings{liu2024itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  booktitle={ICLR},
  year={2024}
}
@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={ICLR},
  year={2023}
}
@inproceedings{wang2024timexer,
  title={TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables},
  author={Wang, Yuxuan and Wu, Haixu and Dong, Jiaxiang and Liu, Yong and Long, Mingsheng and Wang, Jianmin},
  booktitle={NeurIPS},
  year={2024}
}
@inproceedings{wang2024timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y and Zhou, Jun},
  booktitle={ICLR},
  year={2024}
}
@inproceedings{zeng2023dlinear,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={AAAI},
  year={2023}
}
```

---

**문서 끝.** 이 문서가 OK면 Phase 1 (DLinear vendor) 부터 진행.
