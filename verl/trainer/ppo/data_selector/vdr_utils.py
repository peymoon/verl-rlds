from __future__ import annotations

from typing import Optional

import numpy as np


EPS = 1e-8


def robust_zscore(x: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Median/MAD z-score. NaNs are ignored and returned as 0."""
    arr = np.asarray(x, dtype=np.float32)
    if mask is None:
        valid = np.isfinite(arr)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)

    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[valid]
    if vals.size < 2:
        return out

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(1.4826 * mad, eps)
    out[valid] = (arr[valid] - med) / scale
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def sigmoid_gate(score: np.ndarray, floor: float, threshold: float, temperature: float) -> np.ndarray:
    """Return floor + (1-floor) * sigmoid((score-threshold)/temperature)."""
    floor = float(np.clip(floor, 0.0, 1.0))
    temperature = max(float(temperature), EPS)
    logits = (np.asarray(score, dtype=np.float32) - float(threshold)) / temperature
    logits = np.clip(logits, -50.0, 50.0)
    sig = 1.0 / (1.0 + np.exp(-logits))
    gate = floor + (1.0 - floor) * sig
    return np.clip(gate, floor, 1.0).astype(np.float32)


def time_weighted_scalar(history, current_step: int, decay_rate: float, max_age: int):
    """Aggregate [(step, value), ...] with exponential decay; return None if no fresh entries."""
    if not history:
        return None

    total = 0.0
    total_w = 0.0
    for step, value in history:
        age = max(0, int(current_step) - int(step))
        if max_age > 0 and age > max_age:
            continue
        if not np.isfinite(value):
            continue
        w = float(np.exp(-float(decay_rate) * age))
        total += w * float(value)
        total_w += w

    if total_w <= EPS:
        return None
    return total / total_w


def knn_interpolate_values(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    ref_values: np.ndarray,
    temperature: float = 0.05,
    top_k: int = 64,
    batch_size: int = 1000,
) -> np.ndarray:
    """Cosine KNN interpolation. Returns one value per embedding row."""
    n = embeddings.shape[0]
    ref_indices = np.asarray(ref_indices, dtype=np.int64)
    ref_values = np.asarray(ref_values, dtype=np.float32)

    if ref_indices.size == 0:
        return np.full(n, np.nan, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + EPS
    all_normed = (embeddings / norms).astype(np.float32)
    ref_emb = all_normed[ref_indices]

    k = min(int(top_k), ref_indices.size)
    temp = max(float(temperature), EPS)
    out = np.empty(n, dtype=np.float32)

    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        sims = all_normed[start:end] @ ref_emb.T
        for row_i in range(sims.shape[0]):
            row = sims[row_i]
            if k < row.shape[0]:
                idx = np.argpartition(row, -k)[-k:]
            else:
                idx = np.arange(row.shape[0])
            logits = row[idx] / temp
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum() + EPS
            out[start + row_i] = float(np.dot(weights, ref_values[idx]))

    out[~np.isfinite(out)] = np.nan
    return out


def fill_by_cluster_mean(values: np.ndarray, cluster_ids: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Fill NaNs with per-cluster means, then default."""
    out = np.asarray(values, dtype=np.float32).copy()
    clusters = np.asarray(cluster_ids)
    missing = ~np.isfinite(out)
    if not missing.any():
        return out

    for c_id in np.unique(clusters[missing]):
        cluster_mask = clusters == c_id
        vals = out[cluster_mask & np.isfinite(out)]
        if vals.size > 0:
            out[cluster_mask & missing] = float(vals.mean())

    out[~np.isfinite(out)] = float(default)
    return out.astype(np.float32)
