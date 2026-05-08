"""
score_utils.py — V13 utility layer (Backward Forecast Divergence, GT-free).

Per-timestep score (ground-truth-free by design):
  Y_hat_t[h-1, c] = predictions[t - h - L_w + 1, h-1, c]   (h = 1..H)
                  — H different forecasts of time t, one per lead-time h
  v_bar_t[c]      = Σ_h w̃_h · Y_hat_t[h-1, c]
  D_w^(c)(t)      = Σ_h w̃_h · (Y_hat_t[h-1, c] - v_bar_t[c])²
  D_w(t)          = max_c D_w^(c)(t)

with recency weights w_h = λ^(h-1), w̃_h = w_h / Σ w. Forecast event s = t - h
predicts time s+h via predictions[s + 1 - L_w, h-1, :] under inference's
indexing convention (predictions[idx, h-1] forecasts test[idx + L_w + h - 1]).

Key property: D_w(t) uses only the model's predictions — never test[t] —
so the score remains computable without observing the target value. This
GT-free property is a core contribution of the V12/V13 line.

Evaluable range (within test indexing, 0-indexed):
  t ∈ [L_w + H - 1, test_len - 1]
  — full forward-computable range. Requires inference to store predictions
  for anchors L_w-1 through test_len-2 (n_pred = test_len - L_w). Legacy
  artifacts produced before the V13 inference patch (n_pred = test_len -
  L_w - H + 1) are auto-clamped to the older bound test_len - H by
  get_evaluable_range when n_pred is passed.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from artifact_paths import RESULTS_ROOT, resolve_dataset

# ──────────────────────────────────────────────────────────────
# Constants (v7.0)
# ──────────────────────────────────────────────────────────────
LOOKBACK: int = 192
PRED_LEN: int = 96
LAMBDA_FIXED: float = 0.99
EPS: float = 1e-8
SEED: int = 42

# Dataset filter thresholds
MIN_TEST_LENGTH: int = 200
MIN_ANOMALY_EVENTS: int = 1

# Scaler logic must match 01_data_preparation.py
TRAIN_RATIO_WITHIN_TRAINVAL: float = 0.70


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────
# Recency weights
# ──────────────────────────────────────────────────────────────
def compute_recency_weights(
    pred_len: int = PRED_LEN, lam: float = LAMBDA_FIXED
) -> np.ndarray:
    """w_h = λ^(h-1) for h=1..H, normalized so Σ = 1.

    h=1 (most recent forecast event) gets the largest weight.
    """
    raw = np.array([lam ** (h - 1) for h in range(1, pred_len + 1)], dtype=np.float64)
    w = raw / raw.sum()
    assert abs(w.sum() - 1.0) < 1e-10
    return w


# ──────────────────────────────────────────────────────────────
# Dataset bundle (preserve exact 01_data_preparation.py scaler logic)
# ──────────────────────────────────────────────────────────────
@dataclass
class DatasetBundle:
    dataset_key: str
    family: str
    filename: str
    source_path: Path
    official_train_end: int
    test_start: int
    test_values_norm: np.ndarray   # (test_len, C)
    test_labels: np.ndarray        # (test_len,)
    total_timesteps: int
    full_values_raw: np.ndarray    # (total_timesteps, C) RAW values, train+test concatenated
    full_labels: np.ndarray        # (total_timesteps,)   RAW labels, train+test concatenated
    train_values_norm: np.ndarray  # (scaler_train_end, C) RAW train segment, normalized
    scaler_train_end: int          # length of the model's actual training segment
    val_values_norm: np.ndarray    # (val_end - val_start, C) val segment, normalized
    val_start: int                 # start index of the val segment in full_values_raw
    val_end: int                   # end index (exclusive) of the val segment


def _detect_label_column(df: pd.DataFrame) -> str:
    if "Label" in df.columns:
        return "Label"
    if "is_anomaly" in df.columns:
        return "is_anomaly"
    return df.columns[-1]


def _bundle_meta_path(dataset_key: str) -> Path:
    """Per-dataset persisted split + scaler metadata produced by 01_data_preparation.py."""
    return RESULTS_ROOT / dataset_key / "bundle_meta.json"


def _load_bundle_meta(dataset_key: str) -> dict[str, Any] | None:
    """Read persisted bundle meta if present.

    The file is the authoritative source for split offsets and scaler
    parameters used by 01_data_preparation.py. When present, prepare_dataset_
    bundle uses it verbatim instead of re-deriving the split + scaler — this
    keeps 04/05 in lock-step with whatever 01 actually did, even if the
    preparation logic changes later.
    """
    p = _bundle_meta_path(dataset_key)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def prepare_dataset_bundle(dataset_key: str) -> DatasetBundle:
    info = resolve_dataset(dataset_key)
    df = pd.read_csv(info["path"])
    label_col = _detect_label_column(df)
    value_cols = [c for c in df.columns if c != label_col]

    values = df[value_cols].to_numpy(dtype=np.float64)
    labels = (df[label_col].to_numpy() != 0).astype(np.int64)
    total = int(values.shape[0])

    persisted = _load_bundle_meta(dataset_key)
    val_start_meta: int | None = None
    if persisted is not None:
        official_train_end = int(persisted["official_train_end"])
        scaler_train_end = int(persisted["scaler_train_end"])
        mu = np.asarray(persisted["scaler_mean"], dtype=np.float64)
        sigma = np.asarray(persisted["scaler_scale"], dtype=np.float64)
        sigma = np.where(np.abs(sigma) < EPS, 1.0, sigma)
        if "val_start" in persisted:
            val_start_meta = int(persisted["val_start"])
    else:
        # Fallback: re-derive split + scaler from filename + raw values.
        # Mirrors 01_data_preparation.py exactly so legacy datasets without
        # a persisted bundle_meta.json still work.
        official_train_end = info.get("train_size_from_name")
        if official_train_end is None:
            raise ValueError(f"Missing `tr_` marker in filename: {info['filename']}")
        official_train_end = min(int(official_train_end), total)
        required_span = LOOKBACK + PRED_LEN
        if official_train_end < required_span:
            raise ValueError(f"Train split too short for {dataset_key}: {official_train_end}")

        train_end_nominal = int(official_train_end * TRAIN_RATIO_WITHIN_TRAINVAL)
        train_end_nominal = max(train_end_nominal, required_span)
        uses_overlap_val = (official_train_end - train_end_nominal) < required_span
        scaler_train_end = official_train_end if uses_overlap_val else train_end_nominal

        mu = np.mean(values[:scaler_train_end], axis=0)
        sigma = np.std(values[:scaler_train_end], axis=0)
        sigma = np.where(np.abs(sigma) < EPS, 1.0, sigma)

    # Derive val_start when not persisted (matches 01_data_preparation.py).
    required_span = LOOKBACK + PRED_LEN
    if val_start_meta is None:
        if scaler_train_end == official_train_end:
            # overlap-tail val mode: val takes the last required_span of train.
            val_start_meta = official_train_end - required_span
        else:
            val_start_meta = scaler_train_end
    val_start = max(int(val_start_meta), 0)
    val_end = int(official_train_end)

    test_values = values[official_train_end:]
    test_labels = labels[official_train_end:]

    test_values_norm = (test_values - mu) / sigma
    train_values_norm = (values[:scaler_train_end] - mu) / sigma
    val_values_norm = (values[val_start:val_end] - mu) / sigma

    return DatasetBundle(
        dataset_key=dataset_key,
        family=str(info.get("dataset", dataset_key)),
        filename=str(info["filename"]),
        source_path=Path(info["path"]),
        official_train_end=official_train_end,
        test_start=official_train_end,
        test_values_norm=np.asarray(test_values_norm, dtype=np.float64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        total_timesteps=total,
        full_values_raw=np.asarray(values, dtype=np.float64),
        full_labels=np.asarray(labels, dtype=np.int64),
        train_values_norm=np.asarray(train_values_norm, dtype=np.float64),
        val_values_norm=np.asarray(val_values_norm, dtype=np.float64),
        val_start=val_start,
        val_end=val_end,
        scaler_train_end=int(scaler_train_end),
    )


def has_required_prediction_artifacts(dataset_key: str) -> bool:
    d = RESULTS_ROOT / dataset_key
    return all((d / f).exists() for f in [
        "predictions_test.npy", "test_labels.npy", "inference_metadata.json"
    ])


def load_prediction_artifacts(dataset_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    d = RESULTS_ROOT / dataset_key
    preds = np.load(d / "predictions_test.npy")
    labels = np.load(d / "test_labels.npy")
    with open(d / "inference_metadata.json") as fh:
        meta = json.load(fh)
    return preds, labels, meta


# ──────────────────────────────────────────────────────────────
# Anomaly utilities
# ──────────────────────────────────────────────────────────────
def find_anomaly_regions(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels, dtype=np.int64)
    regions: list[tuple[int, int]] = []
    in_region, start = False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_region:
            start, in_region = i, True
        elif v == 0 and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, int(labels.size)))
    return regions


def anomaly_event_starts_and_durations(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    regions = find_anomaly_regions(labels)
    starts = np.array([s for s, _ in regions], dtype=np.int64)
    durations = np.array([e - s for s, e in regions], dtype=np.int64)
    return starts, durations


def compute_iqr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


# ──────────────────────────────────────────────────────────────
# Score normalization-space check
# ──────────────────────────────────────────────────────────────
def is_normalized_space(arr: np.ndarray, mean_tol: float = 0.5, std_tol: float = 0.5) -> bool:
    arr = np.asarray(arr, dtype=np.float64)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return False
    return abs(float(np.mean(fin))) < mean_tol and abs(float(np.std(fin)) - 1.0) < std_tol


def compute_space_check(predictions: np.ndarray, test_values_norm: np.ndarray) -> dict[str, Any]:
    pf = np.asarray(predictions, dtype=np.float64).reshape(-1)
    gf = np.asarray(test_values_norm, dtype=np.float64).reshape(-1)
    pf = pf[np.isfinite(pf)]
    gf = gf[np.isfinite(gf)]
    pn = is_normalized_space(pf)
    gn = is_normalized_space(gf)

    def _stats(a: np.ndarray) -> dict:
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(a)), "std": float(np.std(a))}

    ps, gs = _stats(pf), _stats(gf)
    return {
        "pred_mean": ps["mean"], "pred_std": ps["std"],
        "gt_mean": gs["mean"], "gt_std": gs["std"],
        "pred_normalized": bool(pn), "gt_normalized": bool(gn),
        "space_consistent": bool(pn == gn),
    }


# ──────────────────────────────────────────────────────────────
# Evaluable range
# ──────────────────────────────────────────────────────────────
def get_evaluable_range(
    test_len: int,
    lookback: int = LOOKBACK,
    pred_len: int = PRED_LEN,
    n_pred: int | None = None,
) -> tuple[int, int] | None:
    """
    For target time t to have a valid backward D_w(t):
      - smallest forecast event i = H needs anchor t-H, input start
        t - H - L_w + 1 ≥ 0 → t ≥ L_w + H - 1
      - largest forecast event i = 1 needs anchor t-1, prediction-array row
        idx = t - L_w ≤ n_pred - 1 → t ≤ n_pred + L_w - 1
      - and t must be ≤ test_len - 1 to have a valid label

    The upper bound on t is min(n_pred + L_w - 1, test_len - 1).

    n_pred is the number of stored prediction rows. V13 inference produces
    n_pred = test_len - L_w (anchors L_w-1 through test_len-2 covered) so the
    upper bound becomes test_len - 1 — i.e., the entire labeled test region
    is evaluable. Legacy artifacts produced before the V13 patch had
    n_pred = test_len - L_w - H + 1, which clamps the upper bound to
    test_len - H. Passing n_pred makes this routine adapt to whichever
    inference vintage produced the predictions.

    n_pred=None falls back to the *theoretical* max (test_len - 1) — used
    by 00_dataset_filter where no prediction artifacts exist yet.
    """
    lo = lookback + pred_len - 1
    if n_pred is None:
        hi = test_len - 1
    else:
        hi = min(n_pred + lookback - 1, test_len - 1)
    return (lo, hi) if hi >= lo else None


# ──────────────────────────────────────────────────────────────
# Train-baseline channel statistics (for per-channel D_w_c z-score)
# ──────────────────────────────────────────────────────────────
def compute_train_baseline_stats(
    predictions_train: np.ndarray,
    train_values_norm: np.ndarray,
    lookback: int = LOOKBACK,
    pred_len: int = PRED_LEN,
    lam: float = LAMBDA_FIXED,
    block_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Compute per-channel D_w_c distribution stats on training predictions.

    Used to z-score per-channel D_w_c at test time so that channels with
    very different intrinsic variability contribute on equal footing under
    median aggregation. The "training baseline" is the model's own forecast
    disagreement on data it was trained on — independent of test-time
    distribution shift, and fully GT-free (no test labels involved).

    Returns:
      mean, std         — (C,) per-channel mean/std of D_w_c on train evaluable range
      median, mad       — (C,) per-channel robust counterparts (MAD scaled by 1.4826)
      n                 — number of finite rows used
    """
    predictions_train = np.asarray(predictions_train, dtype=np.float64)
    train_values_norm = np.asarray(train_values_norm, dtype=np.float64)
    fake_labels = np.zeros(train_values_norm.shape[0], dtype=np.int64)

    _, per_ch, _ = compute_backward_score_per_channel(
        predictions=predictions_train,
        test_values_norm=train_values_norm,
        test_labels=fake_labels,
        block_size=block_size,
        lookback=lookback,
        pred_len=pred_len,
        lam=lam,
    )
    Dwc = per_ch["D_w_c"]                                        # (T_eval, C)
    finite_mask = np.isfinite(Dwc).all(axis=1)
    Dwc_clean = Dwc[finite_mask]
    if Dwc_clean.shape[0] == 0:
        C = Dwc.shape[1]
        nan_c = np.full(C, np.nan, dtype=np.float64)
        return {"mean": nan_c.copy(), "std": nan_c.copy(),
                "median": nan_c.copy(), "mad": nan_c.copy(), "n": 0}
    med = np.median(Dwc_clean, axis=0)
    mad = 1.4826 * np.median(np.abs(Dwc_clean - med[None, :]), axis=0)
    return {
        "mean":   Dwc_clean.mean(axis=0),
        "std":    Dwc_clean.std(axis=0),
        "median": med,
        "mad":    mad,
        "n":      int(Dwc_clean.shape[0]),
    }


def apply_channel_zscore_aggregation(
    D_w_c: np.ndarray,
    baseline: dict[str, np.ndarray],
    *,
    centering: str = "mean",  # "mean" or "median"
    scaling: str = "std",     # "std" or "mad"
    agg: str = "median",      # "median" or "max" or "mean"
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-row channel-aggregated z-score of D_w_c against a training baseline.

    Channels whose baseline scaling stat is below `eps` (i.e. effectively
    constant on train) are dropped — they would otherwise produce huge
    false-positive z-scores on any test variation.
    """
    mu = baseline["mean" if centering == "mean" else "median"]
    sigma = baseline["std" if scaling == "std" else "mad"]
    valid_c = np.isfinite(sigma) & (sigma > eps) & np.isfinite(mu)
    if not valid_c.any():
        return np.full(D_w_c.shape[0], np.nan, dtype=np.float64)
    Dwc_v = D_w_c[:, valid_c]
    z = (Dwc_v - mu[valid_c][None, :]) / sigma[valid_c][None, :]
    if agg == "median":
        return np.nanmedian(z, axis=1)
    if agg == "max":
        return np.nanmax(z, axis=1)
    if agg == "mean":
        return np.nanmean(z, axis=1)
    raise ValueError(f"unknown agg: {agg}")


# ──────────────────────────────────────────────────────────────
# Backward D_w score (vectorized, block-processed)
# ──────────────────────────────────────────────────────────────
def compute_backward_score_per_channel(
    predictions: np.ndarray,
    test_values_norm: np.ndarray,
    test_labels: np.ndarray,
    block_size: int = 4096,
    lookback: int = LOOKBACK,
    pred_len: int = PRED_LEN,
    lam: float = LAMBDA_FIXED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Per-channel variant. Returns (score_frame, per_ch, skipped).

    score_frame: DataFrame [t, D_w, label] — channel-collapsed via max.
    per_ch: dict with arrays
        "D_w_c": (T_eval, C) per-channel weighted variance score.
        "t":     (T_eval,)   evaluable timesteps (aligned with rows of D_w_c).
        "label": (T_eval,)   per-row anomaly label.
    Rows where any participating prediction was NaN have all-channels NaN
    on the corresponding row; downstream metric code filters NaN per channel.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    test_values_norm = np.asarray(test_values_norm, dtype=np.float64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    n_pred, ph, channels = predictions.shape
    if ph != int(pred_len):
        raise ValueError(f"Prediction horizon mismatch: {ph} != {pred_len}")
    test_len = test_values_norm.shape[0]
    if test_labels.shape[0] != test_len:
        raise ValueError(f"Label length mismatch: {test_labels.shape[0]} vs {test_len}")

    sr = get_evaluable_range(
        test_len, lookback=lookback, pred_len=pred_len, n_pred=n_pred
    )
    if sr is None:
        empty_frame = pd.DataFrame(columns=["t", "D_w", "label"])
        empty_per_ch = {
            "D_w_c": np.zeros((0, channels), dtype=np.float64),
            "t": np.zeros((0,), dtype=np.int64),
            "label": np.zeros((0,), dtype=np.int64),
        }
        return empty_frame, empty_per_ch, []
    t_min, t_max = sr

    weights = compute_recency_weights(pred_len, lam)             # (H,)

    out_t: list[np.ndarray] = []
    out_dw: list[np.ndarray] = []
    out_label: list[np.ndarray] = []
    out_dw_c: list[np.ndarray] = []
    skipped: list[dict[str, Any]] = []

    for t_start in range(t_min, t_max + 1, block_size):
        t_end = min(t_start + block_size, t_max + 1)
        ts = np.arange(t_start, t_end, dtype=np.int64)            # (B,)

        # Build Y_hat_block (B, H, C):
        #   Y_hat_block[i, h-1, :] = predictions[ts[i] - h - L_w + 1, h-1, :]
        Y_hat_block = np.empty((ts.size, pred_len, channels), dtype=np.float64)
        for h in range(1, pred_len + 1):
            row_idx = ts - h - lookback + 1                        # (B,)
            Y_hat_block[:, h - 1, :] = predictions[row_idx, h - 1, :]

        finite_yhat = np.isfinite(Y_hat_block).all(axis=(1, 2))     # (B,)

        v_bar = np.tensordot(Y_hat_block, weights, axes=([1], [0]))  # (B, C)
        diff_sq = (Y_hat_block - v_bar[:, None, :]) ** 2             # (B, H, C)
        var_c = np.tensordot(diff_sq, weights, axes=([1], [0]))      # (B, C)  — per-channel D_w_c
        D_w = var_c.max(axis=1)                                      # (B,)

        bad_dw = ~finite_yhat
        D_w[bad_dw] = np.nan
        if bad_dw.any():
            var_c[bad_dw, :] = np.nan

        for bi in np.where(bad_dw)[0]:
            skipped.append({"t": int(ts[bi]), "reason": "NaN_in_Y_hat_t"})

        out_t.append(ts)
        out_dw.append(D_w)
        out_label.append(test_labels[ts])
        out_dw_c.append(var_c)

    score_frame = pd.DataFrame({
        "t": np.concatenate(out_t).astype(np.int64),
        "D_w": np.concatenate(out_dw).astype(np.float64),
        "label": np.concatenate(out_label).astype(np.int64),
    })
    per_ch = {
        "D_w_c": np.concatenate(out_dw_c, axis=0).astype(np.float64),
        "t": np.concatenate(out_t).astype(np.int64),
        "label": np.concatenate(out_label).astype(np.int64),
    }
    return score_frame, per_ch, skipped


# ──────────────────────────────────────────────────────────────
# TSB-AD evaluation — package-direct calls (no custom F1 / VUS code)
# ──────────────────────────────────────────────────────────────
def compute_tsb_sliding_window(full_values_raw: np.ndarray) -> int:
    """slidingWindow per the TSB-AD-M benchmark convention.

    Mirrors `benchmark_exp/Run_Detector_M.py:62` exactly:
        data = df.iloc[:, 0:-1].values.astype(float)   # full series, RAW
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

    Caller MUST pass the FULL series (train + test concatenated) in RAW
    units — `find_length_rank` internally restricts to the first 20000
    timesteps, which under TSB-AD's convention captures the
    train-dominated regime. ACF is invariant to mean-shift / scaling, so
    raw vs StandardScaler-normalized gives the same number, but the
    domain (full vs test-only) does affect the result.
    """
    from TSB_AD.utils.slidingWindows import find_length_rank
    return int(find_length_rank(full_values_raw[:, 0].reshape(-1, 1), rank=1))


def compute_tsb_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    sliding_window: int,
) -> dict[str, float]:
    """Direct call to TSB_AD.evaluation.metrics.get_metrics.

    Returns the FULL TSB-AD metric dict with the original key spelling
    (`AUC-PR`, `VUS-PR`, `Standard-F1`, `PA-F1`, `Event-based-F1`,
    `R-based-F1`, `Affiliation-F`). On any failure (NaN-only score, too
    few labels, library exception) all numeric values are NaN and an
    `error` key is added.
    """
    out: dict[str, float] = {
        "AUC-PR": float("nan"),
        "AUC-ROC": float("nan"),
        "VUS-PR": float("nan"),
        "VUS-ROC": float("nan"),
        "Standard-F1": float("nan"),
        "PA-F1": float("nan"),
        "Event-based-F1": float("nan"),
        "R-based-F1": float("nan"),
        "Affiliation-F": float("nan"),
    }
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if not np.any(np.isfinite(score)):
        out["error"] = "all_score_nan"
        return out
    if labels.size == 0:
        out["error"] = "empty_label"
        return out
    if np.unique(labels).size < 2:
        out["error"] = "single_class"
        return out

    # Replace any residual NaN in score with 0 — TSB-AD expects a clean
    # full-length array (no NaN) and downstream callers already 0-pad the
    # leading/trailing non-evaluable region.
    score = np.where(np.isfinite(score), score, 0.0)

    sw = max(int(sliding_window), 1)
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        res = get_metrics(score, labels, slidingWindow=sw)
        for k in out.keys():
            if k in res:
                out[k] = float(res[k])
        out["sliding_window_used"] = int(sw)
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def compute_normal_errors(
    predictions: np.ndarray,
    test_values_norm: np.ndarray,
    test_labels: np.ndarray,
    lookback: int = LOOKBACK,
    pred_len: int = PRED_LEN,
) -> dict[str, float]:
    """Normal-region 1-step MSE & MAE — both target-only and strict-window variants.

    For each evaluable t inside [L_w + H − 1, test_len − H]:
      err = (test[t, c] − predictions[t − L_w, 0, c])

    Two filters produce two metric pairs:
      - target-only ("normal_mse", "normal_mae"):
          require label[t] == 0
      - strict-window ("normal_mse_strict", "normal_mae_strict"):
          require labels[t − L_w : t + H] all 0
          (i.e., entire 1-step input lookback AND full forecasting horizon clean)

    Both are mean over (t, c). If a filter yields zero qualifying timesteps,
    its metrics are NaN. Counts are returned as n_target / n_strict.
    """
    out = {
        "normal_mse": float("nan"), "normal_mae": float("nan"),
        "normal_mse_median": float("nan"), "normal_mae_median": float("nan"),
        "normal_mse_strict": float("nan"), "normal_mae_strict": float("nan"),
        "normal_mse_strict_median": float("nan"),
        "normal_mae_strict_median": float("nan"),
        "n_normal_target": 0, "n_normal_strict": 0,
    }
    test_len = test_values_norm.shape[0]
    n_pred = int(np.asarray(predictions).shape[0])
    sr = get_evaluable_range(
        test_len, lookback=lookback, pred_len=pred_len, n_pred=n_pred
    )
    if sr is None:
        return out
    t_min, t_max = sr
    ts = np.arange(t_min, t_max + 1, dtype=np.int64)
    if ts.size == 0:
        return out

    # Build cumulative-sum of labels for O(1) window-sum lookups.
    labels = np.asarray(test_labels, dtype=np.int64)
    cs = np.concatenate([[0], np.cumsum(labels)])  # cs[i+1] - cs[a] = sum labels[a..i]

    # The strict-window check at t requires labels[t-L_w : t+H] all 0, which
    # needs t + H - 1 ≤ test_len - 1 → t ≤ test_len - H. With the V13 patch
    # the evaluable range can extend to test_len - 1, so guard the trailing
    # rows where the strict window would read past test_len.
    strict_eligible = ts <= (test_len - pred_len)
    strict_mask = np.zeros_like(ts, dtype=bool)
    if strict_eligible.any():
        ts_se = ts[strict_eligible]
        strict_window_sum = cs[ts_se + pred_len] - cs[ts_se - lookback]
        strict_mask[strict_eligible] = (strict_window_sum == 0)

    target_mask = labels[ts] == 0  # already implied by strict_mask but keep for clarity

    def _aggregate(mask: np.ndarray) -> tuple[float, float, float, float, int]:
        """Return (mse_mean, mae_mean, mse_median, mae_median, n_rows).

        Mean aggregation: flat mean over all (t, c) elements — equivalent to
        averaging per-channel MSEs with equal weight. Sensitive to outlier
        channels (one degenerate channel can dominate the metric).

        Median aggregation: compute per-channel MSE/MAE first (mean over time
        per channel), then take median across channels. Robust to outlier
        channels by construction — invalid channels (e.g. constant during
        training, raw passthrough) move to the tails and don't affect the
        median, so no explicit channel mask is needed.
        """
        if not np.any(mask):
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        ts_n = ts[mask]
        gt = test_values_norm[ts_n, :]
        recent = predictions[ts_n - lookback, 0, :]
        finite = np.isfinite(gt).all(axis=1) & np.isfinite(recent).all(axis=1)
        if not np.any(finite):
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        diff = gt[finite] - recent[finite]                # (N_t, C)
        # Mean aggregation (flat over all elements).
        mse_mean = float(np.mean(diff * diff))
        mae_mean = float(np.mean(np.abs(diff)))
        # Per-channel MSE/MAE → median across channels.
        per_ch_mse = (diff * diff).mean(axis=0)           # (C,)
        per_ch_mae = np.abs(diff).mean(axis=0)
        mse_median = float(np.median(per_ch_mse))
        mae_median = float(np.median(per_ch_mae))
        return mse_mean, mae_mean, mse_median, mae_median, int(np.sum(finite))

    mse_t, mae_t, mse_t_med, mae_t_med, n_t = _aggregate(target_mask)
    mse_s, mae_s, mse_s_med, mae_s_med, n_s = _aggregate(strict_mask)
    out.update({
        "normal_mse": mse_t, "normal_mae": mae_t,
        "normal_mse_median": mse_t_med, "normal_mae_median": mae_t_med,
        "n_normal_target": n_t,
        "normal_mse_strict": mse_s, "normal_mae_strict": mae_s,
        "normal_mse_strict_median": mse_s_med,
        "normal_mae_strict_median": mae_s_med,
        "n_normal_strict": n_s,
    })
    return out


# ──────────────────────────────────────────────────────────────
# Dataset filter (v7.0)
# ──────────────────────────────────────────────────────────────
def summarize_dataset_filter_v7(
    bundle: DatasetBundle,
    require_clean_first_lookback: bool = True,
) -> dict[str, Any]:
    """v7.0 filter:
      1. test_length >= MIN_TEST_LENGTH
      2. (optional) First LOOKBACK steps in test contain no anomaly
      3. n_anomaly_events_in_evaluable_range >= MIN_ANOMALY_EVENTS
      4. has_required_prediction_artifacts (reported separately;
         passes_filter does NOT require it so this can be run pre-training)

    With require_clean_first_lookback=False, the first-192-clean rule is
    bypassed — useful when sample size matters more than early-t lookback
    purity (D_w is still well-defined for t ≥ L_w + H − 1).
    """
    test_labels = bundle.test_labels
    test_len = int(test_labels.shape[0])
    starts, durations = anomaly_event_starts_and_durations(test_labels)
    regions = find_anomaly_regions(test_labels)
    artifacts_ready = has_required_prediction_artifacts(bundle.dataset_key)

    sr = get_evaluable_range(test_len)
    eval_lo = None if sr is None else sr[0]
    eval_hi = None if sr is None else sr[1]

    # Count anomaly events that OVERLAP the evaluable range (not only those
    # whose start is inside). An event [s, e) overlaps [eval_lo, eval_hi] iff
    # e > eval_lo AND s <= eval_hi. This recovers datasets whose lone anomaly
    # starts before the evaluable window but extends into it.
    if sr is not None:
        eval_overlap = [(s, e) for (s, e) in regions
                        if e > eval_lo and s <= eval_hi]
        eval_pos_timesteps = int(test_labels[eval_lo:eval_hi + 1].sum())
    else:
        eval_overlap = []
        eval_pos_timesteps = 0
    n_eval = len(eval_overlap)

    # Filter checks
    skip_reason = ""
    passes = True
    early_lookback_dirty = bool(test_labels[:LOOKBACK].sum() > 0)
    if test_len < MIN_TEST_LENGTH:
        passes = False
        skip_reason = f"test_too_short ({test_len} < {MIN_TEST_LENGTH})"
    elif require_clean_first_lookback and early_lookback_dirty:
        passes = False
        skip_reason = "anomaly_in_first_lookback"
    elif sr is None:
        passes = False
        skip_reason = "no_evaluable_range"
    elif n_eval < MIN_ANOMALY_EVENTS:
        passes = False
        skip_reason = f"too_few_eval_anomalies ({n_eval} < {MIN_ANOMALY_EVENTS})"

    return {
        "dataset_id": bundle.dataset_key,
        "family": bundle.family,
        "dataset_file": bundle.filename,
        "test_length": test_len,
        "n_anomaly_events": int(starts.size),
        "n_eval_anomalies": int(n_eval),
        "n_eval_positive_timesteps": int(eval_pos_timesteps),
        "median_anom_duration": float(np.median(durations)) if durations.size else float("nan"),
        "iqr_anom_duration": compute_iqr(durations),
        "first_anomaly_offset": int(starts[0]) if starts.size else -1,
        "min_eval_t": int(eval_lo) if eval_lo is not None else -1,
        "max_eval_t": int(eval_hi) if eval_hi is not None else -1,
        "early_lookback_dirty": int(early_lookback_dirty),
        "has_prediction_artifacts": int(artifacts_ready),
        "passes_filter": bool(passes),
        "skip_reason": skip_reason,
    }


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────
def save_table_with_fallback(
    frame: pd.DataFrame,
    parquet_path: Path,
    artifact_log: list[dict[str, Any]] | None = None,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=False)
        if artifact_log is not None:
            artifact_log.append({"path": str(parquet_path), "format": "parquet"})
        return parquet_path
    except Exception as exc:
        csv_path = parquet_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        if artifact_log is not None:
            artifact_log.append({"path": str(csv_path), "format": "csv", "reason": str(exc)})
        return csv_path


def save_per_channel_scores(per_ch: dict[str, np.ndarray], dir_path: Path) -> Path:
    """Persist per-channel D_w_c arrays alongside scores.parquet.
    Path returned: {dir_path}/scores_per_ch.npz."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out = dir_path / "scores_per_ch.npz"
    payload: dict[str, np.ndarray] = {
        "D_w_c": np.asarray(per_ch["D_w_c"], dtype=np.float64),
        "t": np.asarray(per_ch["t"], dtype=np.int64),
        "label": np.asarray(per_ch["label"], dtype=np.int64),
    }
    for k in ("baseline_mean", "baseline_std", "baseline_median", "baseline_mad"):
        if k in per_ch and per_ch[k] is not None:
            payload[k] = np.asarray(per_ch[k], dtype=np.float64)
    np.savez_compressed(out, **payload)
    return out


def load_per_channel_scores(dir_path: Path) -> dict[str, np.ndarray]:
    p = Path(dir_path) / "scores_per_ch.npz"
    if not p.exists():
        raise FileNotFoundError(f"per-channel scores missing: {p}")
    z = np.load(p)
    out = {"D_w_c": z["D_w_c"], "t": z["t"], "label": z["label"]}
    return out


def load_score_frame(path: Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == "":
        if p.with_suffix(".parquet").exists():
            return pd.read_parquet(p.with_suffix(".parquet"))
        if p.with_suffix(".csv").exists():
            return pd.read_csv(p.with_suffix(".csv"))
        raise FileNotFoundError(f"No score frame at {p}.parquet or .csv")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported score-frame suffix: {p.suffix}")


# ──────────────────────────────────────────────────────────────
# Toy self-check
# ──────────────────────────────────────────────────────────────
def run_toy_backward_self_check() -> dict[str, Any]:
    """End-to-end check with H=4, C=2, L=3, λ=0.99.

    Verifies:
      - Σ w̃_h = 1
      - Manual D_w(t) at one t matches the vectorized routine
      - Channel max != channel mean (so aggregation differs sensibly)
    """
    H, C, L = 4, 2, 3
    lam = 0.99
    rng = np.random.default_rng(SEED)
    test_len = 30
    n_pred = test_len - L - H + 1                              # = 24
    preds = rng.normal(size=(n_pred, H, C)).astype(np.float64)
    test_vals = rng.normal(size=(test_len, C)).astype(np.float64)
    test_labels = np.zeros(test_len, dtype=np.int64)
    test_labels[15:18] = 1

    weights = compute_recency_weights(pred_len=H, lam=lam)
    assert abs(weights.sum() - 1.0) < 1e-12

    df, _per_ch, skipped = compute_backward_score_per_channel(
        predictions=preds,
        test_values_norm=test_vals,
        test_labels=test_labels,
        block_size=8,
        lookback=L,
        pred_len=H,
        lam=lam,
    )
    # Theoretical range (no n_pred): full forward-computable [L+H-1, T-1].
    sr_theory = get_evaluable_range(test_len, lookback=L, pred_len=H)
    # Practical range when scoring with this `preds` array (legacy n_pred).
    sr = get_evaluable_range(
        test_len, lookback=L, pred_len=H, n_pred=preds.shape[0]
    )
    assert sr is not None and sr_theory is not None
    t_min, t_max = sr
    t_pick = (t_min + t_max) // 2

    # Manual reference
    Y_hat = np.zeros((H, C), dtype=np.float64)
    for h in range(1, H + 1):
        idx = t_pick - h - L + 1
        Y_hat[h - 1, :] = preds[idx, h - 1, :]
    v_bar = (weights[:, None] * Y_hat).sum(axis=0)
    var_c = (weights[:, None] * (Y_hat - v_bar) ** 2).sum(axis=0)
    manual_dw = float(np.max(var_c))
    manual_dw_mean = float(np.mean(var_c))

    row = df[df["t"] == t_pick]
    assert row.shape[0] == 1, f"Expected 1 row at t={t_pick}, got {row.shape[0]}"
    dw_val = float(row.iloc[0]["D_w"])

    assert abs(dw_val - manual_dw) < 1e-12, f"D_w mismatch: {dw_val} vs {manual_dw}"
    # Channel max should differ from channel mean (sanity for aggregation choice)
    assert abs(manual_dw - manual_dw_mean) > 1e-12

    return {
        "checked_t": int(t_pick),
        "manual_D_w": manual_dw,
        "computed_D_w": dw_val,
        "n_skipped": len(skipped),
        "score_frame_rows": int(df.shape[0]),
        "evaluable_range_practical": [int(t_min), int(t_max)],
        "evaluable_range_theoretical": [int(sr_theory[0]), int(sr_theory[1])],
    }


if __name__ == "__main__":
    res = run_toy_backward_self_check()
    print("Toy backward self-check:", res)
