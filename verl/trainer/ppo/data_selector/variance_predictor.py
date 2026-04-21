"""
Variance predictor framework for online data selection.

Provides a pluggable interface for predicting per-sample reward variance
(and mean reward) from cached embeddings and rollout observations. Three
implementations:

  KNNPredictor   — wraps the existing cosine-KNN / Nadaraya-Watson approach
                   (zero behavior change from the legacy inline code).
  RidgePredictor — closed-form weighted Ridge regression with joint p-hat head
                   and posterior uncertainty for active probe selection.
  MLPPredictor   — 2-layer MLP warm-started across selection rounds; predicts
                   mean reward logit, derives variance as p*(1-p).

All predictors consume raw observation rows (npz_idx, step, rewards_array)
rather than pre-aggregated per-sample scalars, so time decay and n_rollouts
weighting happen inside the fit — not on the labels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PredictionResult:
    """Output of VariancePredictor.predict()."""
    predicted_var: np.ndarray       # (N,) predicted variance
    predicted_mean: np.ndarray      # (N,) predicted mean reward in [0, 1]
    uncertainty: Optional[np.ndarray] = None  # (N,) posterior std (Ridge only)


class VariancePredictor(ABC):
    """Abstract interface for variance prediction over embeddings."""

    @abstractmethod
    def fit(
        self,
        observations: List[Tuple[int, int, np.ndarray]],
        embeddings: np.ndarray,
        current_step: int,
        normalize_variance: bool = True,
    ) -> None:
        """Fit the predictor on rollout observations.

        Args:
            observations: list of (npz_idx, step, rewards_array) tuples.
                Each entry is one observation event — NOT pre-aggregated.
                rewards_array is shape (n_rollouts,) of raw per-rollout rewards.
            embeddings: (N, d) full dataset embeddings.
            current_step: current global training step (for time decay).
            normalize_variance: if True, normalize by mean*(1-mean).
        """

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> PredictionResult:
        """Predict variance + mean reward for all N samples."""

    def get_state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def get_metrics(self) -> Dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# LOO-KNN diagnostics (used by all predictors via get_metrics integration)
# ---------------------------------------------------------------------------

def compute_predictor_diagnostics(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    ref_observed_var: np.ndarray,
    ref_observed_mean: np.ndarray,
    predicted_var_all: np.ndarray,
    predicted_mean_all: np.ndarray,
    dots_temperature: float = 0.05,
    dots_top_k: int = 64,
) -> Dict[str, float]:
    """Compute LOO-KNN R², Spearman rho, and MAE for the current predictor.

    This runs independently of which predictor is active — it always computes
    the LOO-KNN baseline (the embedding ceiling) so you can compare any
    predictor against the best possible non-parametric fit.

    Also computes the predictor's own training-set R² (predicted vs observed
    at the reference points) regardless of predictor type.

    Requires scipy for Spearman correlation. If scipy is not installed,
    Spearman metrics are skipped (R² and MAE still work). Install with:
        pip install scipy
    """
    try:
        from scipy import stats as scipy_stats
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    metrics: Dict[str, float] = {}
    n_refs = len(ref_indices)
    if n_refs < 5:
        return metrics

    # --- Predictor training-set diagnostics ---
    pred_at_refs = predicted_var_all[ref_indices]
    obs = ref_observed_var

    ss_res = float(np.sum((pred_at_refs - obs) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2_train = 1.0 - ss_res / max(ss_tot, 1e-12)
    mae_train = float(np.mean(np.abs(pred_at_refs - obs)))

    metrics["data_selection/predictor_train_r2"] = r2_train
    metrics["data_selection/predictor_train_mae"] = mae_train

    if _has_scipy:
        rho_train, _ = scipy_stats.spearmanr(pred_at_refs, obs)
        if np.isnan(rho_train):
            rho_train = 0.0
        metrics["data_selection/predictor_train_spearman"] = float(rho_train)

    # --- Same for mean reward ---
    if predicted_mean_all is not None and ref_observed_mean is not None:
        pred_mean_refs = predicted_mean_all[ref_indices]
        obs_mean = ref_observed_mean
        ss_res_m = float(np.sum((pred_mean_refs - obs_mean) ** 2))
        ss_tot_m = float(np.sum((obs_mean - obs_mean.mean()) ** 2))
        r2_mean = 1.0 - ss_res_m / max(ss_tot_m, 1e-12)
        metrics["data_selection/predictor_mean_train_r2"] = r2_mean

        # --- Fair variance R² against the empirical-mean-implied variance ---
        # Ridge/MLP predict v̂ = p̂(1−p̂). Comparing that against the empirical
        # 8-rollout sample variance (which is a noisy estimator of p(1−p))
        # penalises them for the noise in the label. The apples-to-apples
        # target is p_emp*(1−p_emp): if the model's p̂ is perfect, v̂ matches
        # this target exactly, and any residual is the model's fault.
        obs_var_implied = obs_mean * (1.0 - obs_mean)
        ss_tot_vi = float(np.sum((obs_var_implied - obs_var_implied.mean()) ** 2))
        ss_res_vi = float(np.sum((pred_at_refs - obs_var_implied) ** 2))
        r2_v_implied = 1.0 - ss_res_vi / max(ss_tot_vi, 1e-12)
        metrics["data_selection/predictor_train_r2_vs_pmean_var"] = r2_v_implied
        metrics["data_selection/predictor_train_mae_vs_pmean_var"] = float(
            np.mean(np.abs(pred_at_refs - obs_var_implied))
        )

    # --- LOO-KNN R² (embedding ceiling) ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    all_normed = (embeddings / norms).astype(np.float32)
    ref_emb = all_normed[ref_indices]

    # ref-ref similarity matrix: (n_refs, n_refs)
    sim_matrix = ref_emb @ ref_emb.T

    top_k = min(dots_top_k, n_refs - 1)
    loo_predictions = np.zeros(n_refs, dtype=np.float32)

    for i in range(n_refs):
        row = sim_matrix[i].copy()
        row[i] = -np.inf  # exclude self
        if top_k < n_refs - 1:
            topk_idx = np.argpartition(row, -top_k)[-top_k:]
        else:
            topk_idx = np.concatenate([np.arange(i), np.arange(i + 1, n_refs)])

        topk_sims = row[topk_idx]
        topk_vals = obs[topk_idx]

        log = topk_sims / dots_temperature
        log -= log.max()
        weights = np.exp(log)
        weights /= weights.sum() + 1e-12

        loo_predictions[i] = float(np.dot(weights, topk_vals))

    ss_res_loo = float(np.sum((loo_predictions - obs) ** 2))
    r2_loo = 1.0 - ss_res_loo / max(ss_tot, 1e-12)
    mae_loo = float(np.mean(np.abs(loo_predictions - obs)))

    metrics["data_selection/loo_knn_r2"] = r2_loo
    metrics["data_selection/loo_knn_mae"] = mae_loo

    if _has_scipy:
        rho_loo, _ = scipy_stats.spearmanr(loo_predictions, obs)
        if np.isnan(rho_loo):
            # NaN = at least one array has zero variance (e.g. kernel smoothing
            # collapsed all loo_predictions to the same value). Silently
            # returning 0.0 masks this pathology as "no correlation" — expose
            # it explicitly via a sentinel flag so downstream triage can tell
            # "rank corr is genuinely zero" from "rank corr is undefined".
            metrics["data_selection/loo_knn_spearman_undefined"] = 1.0
            metrics["data_selection/loo_knn_pred_std"] = float(np.std(loo_predictions))
            metrics["data_selection/loo_knn_obs_std"] = float(np.std(obs))
        else:
            metrics["data_selection/loo_knn_spearman"] = float(rho_loo)
            metrics["data_selection/loo_knn_spearman_undefined"] = 0.0
    metrics["data_selection/n_reference_points"] = float(n_refs)

    return metrics


def make_predictor_scatter_plot(
    ref_indices: np.ndarray,
    ref_observed_var: np.ndarray,
    predicted_var_all: np.ndarray,
    predictor_type: str = "knn",
    selection_round: int = 0,
    ref_observed_mean: Optional[np.ndarray] = None,
):
    """Create predicted-vs-observed scatter plot for wandb logging.

    When ``ref_observed_mean`` is provided, points are colored by the empirical
    mean reward (1.0 = rollouts usually correct, 0.0 = usually wrong, 0.5 =
    ZPD / high-variance). This makes it easy to see whether the predictor
    mis-ranks the hard-but-learnable samples vs the dead-zone samples.

    Returns a wandb.Image or None if dependencies are missing.
    """
    try:
        import wandb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n_refs = len(ref_indices)
    if n_refs < 5:
        return None

    pred = predicted_var_all[ref_indices]
    obs = ref_observed_var

    ss_res = float(np.sum((pred - obs) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    rho = 0.0
    try:
        from scipy import stats as scipy_stats
        rho, _ = scipy_stats.spearmanr(pred, obs)
        if np.isnan(rho):
            rho = 0.0
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(6.5, 5))
    if ref_observed_mean is not None and len(ref_observed_mean) == n_refs:
        sc = ax.scatter(
            obs, pred,
            s=8, alpha=0.6,
            c=ref_observed_mean,
            cmap="RdYlGn",
            vmin=0.0, vmax=1.0,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Empirical mean reward (1=correct, 0=wrong)")
    else:
        ax.scatter(obs, pred, s=4, alpha=0.5, c="steelblue", edgecolors="none")

    lims = [
        min(float(obs.min()), float(pred.min())) - 0.02,
        max(float(obs.max()), float(pred.max())) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed variance (time-weighted)")
    ax.set_ylabel(f"Predicted variance ({predictor_type})")
    ax.set_title(
        f"Round {selection_round}: Predicted vs Observed "
        f"(n={n_refs}, R\u00b2={r2:.3f}, \u03c1={rho:.3f})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# KNNPredictor — wraps the existing DOTS interpolation (zero behavior change)
# ---------------------------------------------------------------------------

class KNNPredictor(VariancePredictor):
    """Cosine-KNN (Nadaraya-Watson) predictor — reproduces legacy behavior."""

    def __init__(
        self,
        dots_temperature: float = 0.05,
        dots_top_k: int = 64,
        decay_rate: float = 0.05,
        max_refs: int = 0,
    ):
        self._dots_temperature = dots_temperature
        self._dots_top_k = dots_top_k
        self._decay_rate = decay_rate
        self._max_refs = max_refs

        # Populated by fit()
        self._ref_indices: Optional[np.ndarray] = None
        self._ref_variances: Optional[np.ndarray] = None
        self._ref_mean_rewards: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        observations: List[Tuple[int, int, np.ndarray]],
        embeddings: np.ndarray,
        current_step: int,
        normalize_variance: bool = True,
    ) -> None:
        if not observations:
            self._fitted = False
            return

        # Group observations by npz_idx — mirrors legacy _compute_time_weighted_*
        from collections import defaultdict
        buffer: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for npz_idx, step, rewards in observations:
            buffer[npz_idx].append((step, rewards))

        # Compute time-weighted variance and mean (same as legacy)
        var_results: Dict[int, float] = {}
        mean_results: Dict[int, float] = {}
        recency: Dict[int, int] = {}

        for npz_idx, history in buffer.items():
            total_w = 0.0
            weighted_var = 0.0
            weighted_mean = 0.0
            latest_step = 0
            for t, rewards in history:
                w = float(np.exp(-self._decay_rate * max(0, current_step - t)))
                v = float(np.var(rewards)) if len(rewards) > 1 else 0.0
                m = float(np.mean(rewards))
                if normalize_variance and len(rewards) > 1:
                    denom = max(m * (1.0 - m), 1e-8)
                    v = min(v / denom, 1.0)
                weighted_var += w * v
                weighted_mean += w * m
                total_w += w
                latest_step = max(latest_step, t)
            if total_w > 1e-12:
                var_results[npz_idx] = weighted_var / total_w
                mean_results[npz_idx] = weighted_mean / total_w
                recency[npz_idx] = latest_step

        # Cap to max_refs
        if self._max_refs > 0 and len(var_results) > self._max_refs:
            top_idxs = sorted(recency, key=lambda i: recency[i], reverse=True)[
                :self._max_refs
            ]
            var_results = {i: var_results[i] for i in top_idxs}
            mean_results = {i: mean_results[i] for i in top_idxs}

        if not var_results:
            self._fitted = False
            return

        self._ref_indices = np.array(list(var_results.keys()))
        self._ref_variances = np.array(
            list(var_results.values()), dtype=np.float32
        )
        self._ref_mean_rewards = np.array(
            [mean_results.get(int(i), 0.0) for i in self._ref_indices],
            dtype=np.float32,
        )
        # Aliases for diagnostic compatibility
        self._ref_observed_var = self._ref_variances
        self._ref_observed_mean = self._ref_mean_rewards
        self._fitted = True

    def predict(self, embeddings: np.ndarray) -> PredictionResult:
        if not self._fitted:
            n = embeddings.shape[0]
            return PredictionResult(
                predicted_var=np.zeros(n, dtype=np.float32),
                predicted_mean=np.full(n, 0.5, dtype=np.float32),
            )

        pred_var = self._interpolate(
            embeddings, self._ref_indices, self._ref_variances
        )
        pred_mean = self._interpolate(
            embeddings, self._ref_indices, self._ref_mean_rewards
        )
        pred_mean = np.clip(pred_mean, 0.0, 1.0)

        return PredictionResult(
            predicted_var=pred_var,
            predicted_mean=pred_mean,
        )

    def _interpolate(
        self,
        embeddings: np.ndarray,
        ref_indices: np.ndarray,
        ref_values: np.ndarray,
    ) -> np.ndarray:
        """Cosine-similarity-weighted average (same as legacy _dots_interpolate)."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (embeddings / norms).astype(np.float32)

        ref_emb = all_normed[ref_indices]
        ref_val = ref_values.astype(np.float32)

        temp = self._dots_temperature
        top_k = min(self._dots_top_k, len(ref_indices))

        batch_size = 1000
        n = len(all_normed)
        predictions = np.zeros(n, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_emb = all_normed[start:end]
            sims = batch_emb @ ref_emb.T

            for i in range(len(batch_emb)):
                row = sims[i]
                if top_k < len(row):
                    topk_idx = np.argpartition(row, -top_k)[-top_k:]
                else:
                    topk_idx = np.arange(len(row))

                topk_sims = row[topk_idx]
                topk_vals = ref_val[topk_idx]

                log = topk_sims / temp
                log -= log.max()
                weights = np.exp(log)
                weights /= weights.sum() + 1e-12

                predictions[start + i] = float(np.dot(weights, topk_vals))

        return predictions

    def get_metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {"data_selection/predictor_type": 0.0}
        if self._fitted and self._ref_indices is not None:
            m["data_selection/predictor_n_refs"] = float(len(self._ref_indices))
        return m


# ---------------------------------------------------------------------------
# RidgePredictor — closed-form weighted Ridge with posterior uncertainty
# ---------------------------------------------------------------------------

class RidgePredictor(VariancePredictor):
    """Two-head weighted Ridge regression on L2-normalized embeddings.

    Head A (mean, per-rollout BCE-style MSE on binary rewards) predicts p_hat.
    Head B (variance, per-sample weighted MSE on normalized variance) predicts
    v_hat directly. Returning the direct v_hat instead of p_hat*(1-p_hat)
    removes the 0.25 cap that p(1-p) imposes and gives the selector a
    ranking signal on the full [0, 1] axis. The p_hat head is still needed
    for asymmetric utility / dead-zone filtering and for the active-probe
    posterior (which is cheapest to derive from the mean head).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        decay_rate: float = 0.05,
        max_refs: int = 0,
    ):
        self._alpha = alpha
        self._decay_rate = decay_rate
        self._max_refs = max_refs

        self._beta: Optional[np.ndarray] = None      # (d,) mean head
        self._beta_var: Optional[np.ndarray] = None  # (d,) variance head
        self._sigma_inv: Optional[np.ndarray] = None  # (d, d) mean-head precision
        self._fitted = False

        # Cached at fit time for diagnostics
        self._ref_indices: Optional[np.ndarray] = None
        self._ref_observed_var: Optional[np.ndarray] = None
        self._ref_observed_mean: Optional[np.ndarray] = None
        self._all_normed: Optional[np.ndarray] = None

    def fit(
        self,
        observations: List[Tuple[int, int, np.ndarray]],
        embeddings: np.ndarray,
        current_step: int,
        normalize_variance: bool = True,
    ) -> None:
        if not observations:
            self._fitted = False
            return

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self._all_normed = (embeddings / norms).astype(np.float32)

        # Build per-observation rows: each (npz_idx, step, rewards) becomes
        # one training point with target = mean(rewards), weight = n_rollouts * time_decay
        from collections import defaultdict
        buffer: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for npz_idx, step, rewards in observations:
            buffer[npz_idx].append((step, rewards))

        # Cap to max_refs by recency
        if self._max_refs > 0 and len(buffer) > self._max_refs:
            recency = {}
            for idx, hist in buffer.items():
                recency[idx] = max(t for t, _ in hist)
            keep = sorted(recency, key=lambda i: recency[i], reverse=True)[
                :self._max_refs
            ]
            buffer = {i: buffer[i] for i in keep}

        if not buffer:
            self._fitted = False
            return

        # Flatten into arrays
        X_rows = []
        y_rows = []
        w_rows = []
        obs_var_per_sample: Dict[int, Tuple[float, float]] = {}

        # Per-rollout expansion: each of the `n_r` rewards in an observation
        # becomes its own training row. Target is the raw (binary) reward,
        # weight is time_w only (n_r falls out because we have n_r rows).
        # This gives the regressor ~8× more supervision than averaging.
        for npz_idx, history in buffer.items():
            total_w = 0.0
            wv = 0.0
            wm = 0.0
            x_row = self._all_normed[npz_idx]
            for t, rewards in history:
                n_r = len(rewards)
                time_w = float(np.exp(-self._decay_rate * max(0, current_step - t)))

                for r_i in rewards:
                    X_rows.append(x_row)
                    y_rows.append(float(r_i))
                    w_rows.append(time_w)

                m = float(np.mean(rewards))
                v = float(np.var(rewards)) if n_r > 1 else 0.0
                if normalize_variance and n_r > 1:
                    denom = max(m * (1.0 - m), 1e-8)
                    v = min(v / denom, 1.0)
                wv += time_w * v
                wm += time_w * m
                total_w += time_w
            if total_w > 1e-12:
                obs_var_per_sample[npz_idx] = (wv / total_w, wm / total_w)

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)
        w = np.array(w_rows, dtype=np.float64)

        n, d = X.shape

        # Store ref info for diagnostics
        self._ref_indices = np.array(list(obs_var_per_sample.keys()))
        self._ref_observed_var = np.array(
            [obs_var_per_sample[i][0] for i in self._ref_indices], dtype=np.float32
        )
        self._ref_observed_mean = np.array(
            [obs_var_per_sample[i][1] for i in self._ref_indices], dtype=np.float32
        )

        # --- Head A: mean-reward regression (per-rollout rows) ---
        W_diag = w  # (n,)
        XtW = X.T * W_diag[np.newaxis, :]  # (d, n)
        XtWX = XtW @ X  # (d, d)
        XtWy = XtW @ y  # (d,)

        A = XtWX + self._alpha * np.eye(d, dtype=np.float64)
        try:
            self._sigma_inv = A
            self._beta = np.linalg.solve(A, XtWy).astype(np.float32)  # (d,)
        except np.linalg.LinAlgError:
            self._beta = (np.linalg.pinv(A) @ XtWy).astype(np.float32)
            self._sigma_inv = A

        # --- Head B: direct variance regression (per-sample rows) ---
        # One row per distinct probed sample, target = time-weighted normalized
        # variance, weight = sum of time weights across observations of that
        # sample (so samples probed more often get more influence). The 0.25
        # cap of head A comes from p(1-p); head B has no such cap and its
        # output is the value that gets ranked by the selector.
        var_X_rows = []
        var_y_rows = []
        var_w_rows = []
        # Re-derive per-sample aggregate weight from obs_var_per_sample + history.
        # obs_var_per_sample stores (weighted_var, weighted_mean) normalized by
        # total_w; we need total_w to reweight. Recompute on the fly.
        for npz_idx, history in buffer.items():
            agg = obs_var_per_sample.get(int(npz_idx))
            if agg is None:
                continue
            v_sample, _ = agg
            total_w = 0.0
            for t, _rewards in history:
                total_w += float(np.exp(-self._decay_rate * max(0, current_step - t)))
            if total_w <= 1e-12:
                continue
            var_X_rows.append(self._all_normed[int(npz_idx)])
            var_y_rows.append(float(v_sample))
            var_w_rows.append(float(total_w))

        if var_X_rows:
            Xv = np.array(var_X_rows, dtype=np.float64)
            yv = np.array(var_y_rows, dtype=np.float64)
            wv = np.array(var_w_rows, dtype=np.float64)
            XvtW = Xv.T * wv[np.newaxis, :]
            XvtWXv = XvtW @ Xv
            XvtWyv = XvtW @ yv
            Av = XvtWXv + self._alpha * np.eye(d, dtype=np.float64)
            try:
                self._beta_var = np.linalg.solve(Av, XvtWyv).astype(np.float32)
            except np.linalg.LinAlgError:
                self._beta_var = (np.linalg.pinv(Av) @ XvtWyv).astype(np.float32)
        else:
            self._beta_var = None

        self._fitted = True

    def predict(self, embeddings: np.ndarray) -> PredictionResult:
        n = embeddings.shape[0]
        if not self._fitted or self._beta is None:
            return PredictionResult(
                predicted_var=np.zeros(n, dtype=np.float32),
                predicted_mean=np.full(n, 0.5, dtype=np.float32),
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (embeddings / norms).astype(np.float32)

        p_hat = (all_normed @ self._beta).astype(np.float32)
        p_hat = np.clip(p_hat, 0.0, 1.0)

        # Variance head: direct regression on normalized variance if available,
        # else fall back to the p(1-p) derivation (cold start, first fit).
        if self._beta_var is not None:
            v_hat = (all_normed @ self._beta_var).astype(np.float32)
            v_hat = np.clip(v_hat, 0.0, 1.0)
        else:
            v_hat = (p_hat * (1.0 - p_hat)).astype(np.float32)

        # Posterior uncertainty: sigma(x) = sqrt(alpha * x^T (X^T W X + alpha I)^{-1} x)
        uncertainty = None
        if self._sigma_inv is not None:
            try:
                A_inv = np.linalg.inv(self._sigma_inv.astype(np.float64))
                # Batch computation: uncertainty_i = sqrt(alpha * x_i^T A_inv x_i)
                # = sqrt(alpha * sum_j (x_i * (A_inv @ x_i^T)))
                XA = all_normed.astype(np.float64) @ A_inv  # (N, d)
                quad = np.sum(XA * all_normed.astype(np.float64), axis=1)  # (N,)
                uncertainty = np.sqrt(
                    np.maximum(self._alpha * quad, 0.0)
                ).astype(np.float32)
            except np.linalg.LinAlgError:
                pass

        return PredictionResult(
            predicted_var=v_hat,
            predicted_mean=p_hat,
            uncertainty=uncertainty,
        )

    def get_state_dict(self) -> dict:
        return {
            "beta": self._beta,
            "beta_var": self._beta_var,
            "sigma_inv": self._sigma_inv,
            "fitted": self._fitted,
            "alpha": self._alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        self._beta = state.get("beta")
        self._beta_var = state.get("beta_var")
        self._sigma_inv = state.get("sigma_inv")
        self._fitted = state.get("fitted", False)
        if "alpha" in state:
            self._alpha = state["alpha"]

    def get_metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {"data_selection/predictor_type": 1.0}
        if self._fitted and self._ref_indices is not None:
            m["data_selection/predictor_n_refs"] = float(len(self._ref_indices))
            if self._beta is not None:
                m["data_selection/predictor_beta_norm"] = float(
                    np.linalg.norm(self._beta)
                )
        return m


# ---------------------------------------------------------------------------
# MLPPredictor — warm-started 2-layer MLP
# ---------------------------------------------------------------------------

class MLPPredictor(VariancePredictor):
    """Two-head MLP, warm-started across rounds.

    Architecture: d -> hidden -> 2 logits. Head A is the mean-reward logit
    (trained with per-rollout BCE), head B is the direct variance logit
    (trained with per-sample weighted MSE on normalized variance, sigmoid
    to keep output in [0, 1]). At predict time, predicted_var comes from
    head B — uncapped by the 0.25 ceiling of p(1-p) — while predicted_mean
    still comes from head A so asymmetric-utility and dead-zone filtering
    keep working.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_steps: int = 10,
        decay_rate: float = 0.05,
        max_refs: int = 0,
    ):
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._weight_decay = weight_decay
        self._n_steps = n_steps
        self._decay_rate = decay_rate
        self._max_refs = max_refs
        self._embed_dim = embed_dim

        self._model = None
        self._optimizer = None
        self._initialized = False
        self._fitted = False

        self._ref_indices: Optional[np.ndarray] = None
        self._ref_observed_var: Optional[np.ndarray] = None
        self._ref_observed_mean: Optional[np.ndarray] = None

    def _ensure_model(self, d: int) -> None:
        """Lazy-init model on first fit (need embedding dim)."""
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("MLPPredictor requires PyTorch")

        self._embed_dim = d

        class _MLP(nn.Module):
            def __init__(self, in_dim, hid_dim):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.ReLU(),
                )
                # Two heads: [0] = mean-reward logit, [1] = variance logit.
                self.head = nn.Linear(hid_dim, 2)

            def forward(self, x):
                h = self.trunk(x)
                return self.head(h)  # (B, 2)

        self._model = _MLP(d, self._hidden_dim)
        self._model.eval()

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        self._initialized = True

    def fit(
        self,
        observations: List[Tuple[int, int, np.ndarray]],
        embeddings: np.ndarray,
        current_step: int,
        normalize_variance: bool = True,
    ) -> None:
        if not observations:
            self._fitted = False
            return

        import torch

        d = embeddings.shape[1]
        self._ensure_model(d)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (embeddings / norms).astype(np.float32)

        # Build per-observation rows
        from collections import defaultdict
        buffer: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for npz_idx, step, rewards in observations:
            buffer[npz_idx].append((step, rewards))

        if self._max_refs > 0 and len(buffer) > self._max_refs:
            recency = {}
            for idx, hist in buffer.items():
                recency[idx] = max(t for t, _ in hist)
            keep = sorted(recency, key=lambda i: recency[i], reverse=True)[
                :self._max_refs
            ]
            buffer = {i: buffer[i] for i in keep}

        X_rows = []
        y_rows = []
        w_rows = []
        obs_var_per_sample: Dict[int, Tuple[float, float]] = {}

        # Per-rollout expansion: one training row per rollout outcome.
        # Target is the raw (binary) reward; we train with BCE on logits so
        # the sigmoid head is directly modelling p(success | x).
        for npz_idx, history in buffer.items():
            total_w = 0.0
            wv = 0.0
            wm = 0.0
            x_row = all_normed[npz_idx]
            for t, rewards in history:
                n_r = len(rewards)
                time_w = float(np.exp(-self._decay_rate * max(0, current_step - t)))

                for r_i in rewards:
                    X_rows.append(x_row)
                    y_rows.append(float(r_i))
                    w_rows.append(time_w)

                m = float(np.mean(rewards))
                v = float(np.var(rewards)) if n_r > 1 else 0.0
                if normalize_variance and n_r > 1:
                    denom = max(m * (1.0 - m), 1e-8)
                    v = min(v / denom, 1.0)
                wv += time_w * v
                wm += time_w * m
                total_w += time_w
            if total_w > 1e-12:
                obs_var_per_sample[npz_idx] = (wv / total_w, wm / total_w)

        self._ref_indices = np.array(list(obs_var_per_sample.keys()))
        self._ref_observed_var = np.array(
            [obs_var_per_sample[i][0] for i in self._ref_indices], dtype=np.float32
        )
        self._ref_observed_mean = np.array(
            [obs_var_per_sample[i][1] for i in self._ref_indices], dtype=np.float32
        )

        X_t = torch.from_numpy(np.array(X_rows, dtype=np.float32))
        y_t = torch.from_numpy(np.array(y_rows, dtype=np.float32))
        w_t = torch.from_numpy(np.array(w_rows, dtype=np.float32))

        # Scale-stable normalization across rounds for the mean head.
        w_sum = float(w_t.sum().item())
        w_sq_sum = float((w_t * w_t).sum().item())
        n_eff = (w_sum * w_sum) / max(w_sq_sum, 1e-12)
        w_t = w_t / max(n_eff, 1.0)

        # --- Per-sample rows for the variance head ---
        # One row per distinct probed sample, target = time-weighted
        # normalized variance, weight = sum of time weights across its
        # observations (so repeatedly-probed samples get more influence).
        var_X_np = []
        var_y_np = []
        var_w_np = []
        for npz_idx, history in buffer.items():
            agg = obs_var_per_sample.get(int(npz_idx))
            if agg is None:
                continue
            v_sample, _ = agg
            total_w = 0.0
            for t, _rewards in history:
                total_w += float(np.exp(-self._decay_rate * max(0, current_step - t)))
            if total_w <= 1e-12:
                continue
            var_X_np.append(all_normed[int(npz_idx)])
            var_y_np.append(float(v_sample))
            var_w_np.append(float(total_w))

        if var_X_np:
            Xv_t = torch.from_numpy(np.array(var_X_np, dtype=np.float32))
            yv_t = torch.from_numpy(np.array(var_y_np, dtype=np.float32))
            wv_t = torch.from_numpy(np.array(var_w_np, dtype=np.float32))
            wv_sum = float(wv_t.sum().item())
            wv_sq = float((wv_t * wv_t).sum().item())
            nv_eff = (wv_sum * wv_sum) / max(wv_sq, 1e-12)
            wv_t = wv_t / max(nv_eff, 1.0)
        else:
            Xv_t = None

        # Warm-start training (do NOT reinitialize model or optimizer).
        import torch.nn.functional as F
        self._model.train()
        for _ in range(self._n_steps):
            self._optimizer.zero_grad()
            logits = self._model(X_t)           # (N_rollouts, 2)
            p_logit = logits[:, 0]
            loss_mean = F.binary_cross_entropy_with_logits(
                p_logit, y_t, weight=w_t, reduction="sum"
            )
            loss = loss_mean
            if Xv_t is not None:
                v_logit = self._model(Xv_t)[:, 1]
                v_pred = torch.sigmoid(v_logit)
                # Weighted MSE on direct normalized variance.
                loss_var = ((v_pred - yv_t) ** 2 * wv_t).sum()
                loss = loss + loss_var
            loss.backward()
            self._optimizer.step()
        self._model.eval()
        self._fitted = True

    def predict(self, embeddings: np.ndarray) -> PredictionResult:
        n = embeddings.shape[0]
        if not self._fitted or self._model is None:
            return PredictionResult(
                predicted_var=np.zeros(n, dtype=np.float32),
                predicted_mean=np.full(n, 0.5, dtype=np.float32),
            )

        import torch

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (embeddings / norms).astype(np.float32)

        with torch.no_grad():
            X_t = torch.from_numpy(all_normed)
            logits = self._model(X_t).numpy()  # (N, 2)
            p_logit = logits[:, 0].astype(np.float64)
            v_logit = logits[:, 1].astype(np.float64)
            p_hat = 1.0 / (1.0 + np.exp(-p_logit))
            p_hat = np.clip(p_hat, 0.0, 1.0).astype(np.float32)
            # Direct variance head: sigmoid keeps v_hat ∈ [0, 1] naturally,
            # not capped at 0.25 like the p(1-p) derivation.
            v_hat = 1.0 / (1.0 + np.exp(-v_logit))
            v_hat = np.clip(v_hat, 0.0, 1.0).astype(np.float32)

        return PredictionResult(
            predicted_var=v_hat,
            predicted_mean=p_hat,
        )

    def get_state_dict(self) -> dict:
        state = {"fitted": self._fitted, "initialized": self._initialized}
        if self._initialized and self._model is not None:
            import torch
            state["model_state"] = self._model.state_dict()
            state["optimizer_state"] = self._optimizer.state_dict()
            state["embed_dim"] = self._embed_dim
            state["hidden_dim"] = self._hidden_dim
        return state

    def load_state_dict(self, state: dict) -> None:
        self._fitted = state.get("fitted", False)
        if state.get("initialized", False) and "model_state" in state:
            d = state.get("embed_dim", self._embed_dim)
            self._hidden_dim = state.get("hidden_dim", self._hidden_dim)
            self._ensure_model(d)
            # Old checkpoints have a single-head linear layer (shape (1, H)),
            # new architecture has a two-head layer (shape (2, H)). Skip
            # loading rather than crash — the predictor will cold-start.
            try:
                self._model.load_state_dict(state["model_state"])
                self._optimizer.load_state_dict(state["optimizer_state"])
            except (RuntimeError, ValueError) as e:
                print(f"[MLPPredictor] Skipping state load (shape mismatch, "
                      f"likely pre-two-head checkpoint): {e}")
                self._fitted = False
            self._model.eval()

    def get_metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {"data_selection/predictor_type": 2.0}
        if self._fitted and self._ref_indices is not None:
            m["data_selection/predictor_n_refs"] = float(len(self._ref_indices))
        return m


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_predictor(cluster_config) -> VariancePredictor:
    """Build a VariancePredictor from ClusterSelectorConfig."""
    ptype = getattr(cluster_config, "predictor_type", "knn")

    if ptype == "knn":
        return KNNPredictor(
            dots_temperature=cluster_config.dots_temperature,
            dots_top_k=cluster_config.dots_top_k,
            decay_rate=cluster_config.rollout_history_decay_rate,
            max_refs=cluster_config.rollout_history_max_refs,
        )
    elif ptype == "ridge":
        return RidgePredictor(
            alpha=getattr(cluster_config, "predictor_alpha", 1.0),
            decay_rate=cluster_config.rollout_history_decay_rate,
            max_refs=cluster_config.rollout_history_max_refs,
        )
    elif ptype == "mlp":
        return MLPPredictor(
            hidden_dim=getattr(cluster_config, "predictor_mlp_hidden", 256),
            lr=getattr(cluster_config, "predictor_mlp_lr", 1e-3),
            weight_decay=getattr(cluster_config, "predictor_mlp_weight_decay", 1e-3),
            n_steps=getattr(cluster_config, "predictor_mlp_steps", 10),
            decay_rate=cluster_config.rollout_history_decay_rate,
            max_refs=cluster_config.rollout_history_max_refs,
        )
    else:
        raise ValueError(f"Unknown predictor_type: {ptype!r}. "
                         f"Valid: 'knn', 'ridge', 'mlp'")
