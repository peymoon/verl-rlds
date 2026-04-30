"""
Online cluster-based data selector.

Uses pre-computed embeddings and FAISS clustering to select training samples
adaptively during training. At each selection round:

  1. Roll out on cluster representatives (medoids) to get a capability signal
  2. Compute per-cluster reward variance from representative rollouts
  3. Select training samples using one of several strategies:
     - top_clusters: greedily include samples from highest-variance clusters
     - weighted: allocate budget proportionally to cluster variance
     - scored: use composite scores (variance * transferability * 1/density)
     - interpolated: DOTS-style per-sample variance prediction via embeddings
  4. Within-cluster selection via centroid_nearest or MMD coreset

The key difference from offline cluster selection: this re-runs step 1 every
N training steps using the current policy, so the selection adapts as the
model improves.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import DataSelectionConfig, DataSelector
from .variance_predictor import (
    VariancePredictor,
    PredictionResult,
    build_predictor,
    compute_predictor_diagnostics,
    make_predictor_scatter_plot,
)


@dataclass
class ClusterSelectorConfig:
    """Cluster-specific configuration for online selection."""

    embeddings_file: Optional[str] = None
    cluster_arrays_file: Optional[str] = None
    n_clusters: int = 50
    n_reps: int = 5
    strategy: str = "scored"
    within_cluster_method: str = "centroid_nearest"
    representative_method: str = "medoid"

    # --- density_diverse representative selection ---
    # Density-aware diverse selection: scores each point with
    #   score = density_norm^alpha * centroid_sim^beta
    # then greedily picks top scorers while suppressing neighbours within
    # a cosine-distance radius.  Naturally supports n_reps > 1 with
    # built-in diversity enforcement.
    density_diverse_k: int = 10            # k for k-NN density estimation
    density_diverse_alpha: float = 1.0     # exponent on normalised density
    density_diverse_beta: float = 1.0      # exponent on centroid proximity
    density_diverse_radius: float = 0.15   # cosine-distance suppression radius (0 = disable)

    dots_temperature: float = 0.05
    dots_top_k: int = 64

    # --- dots_diversity: per-cluster allocation for the interpolated strategy ---
    # When False (default): global top-k by predicted variance.
    # When True: allocate budget across clusters (min 1 each), then take the
    # highest predicted-variance samples within each cluster.
    dots_diversity: bool = False

    # When dots_diversity=True, use composite cluster score for allocation:
    #   score[c] = mean(predicted_var[c]) * transferability[c] * (1/density[c])
    # When False, allocation is proportional to mean predicted variance only.
    dots_diversity_use_composite_score: bool = False

    # Softmax temperature for dots_diversity allocation (controls concentration).
    # High temperature (~1.0) → near-uniform allocation across clusters.
    # Low temperature (~0.05) → allocation dominated by top-scoring clusters.
    dots_diversity_temperature: float = 0.1

    # Temperature annealing for dots_diversity allocation.
    # When enabled, temperature decays exponentially from _start to _end
    # at rate _decay per selection round:
    #   temp(t) = temp_end + (temp_start - temp_end) * exp(-decay * t)
    dots_diversity_anneal: bool = False
    dots_diversity_temperature_start: float = 1.0
    dots_diversity_temperature_end: float = 0.05
    dots_diversity_temperature_decay: float = 0.1

    # --- Exploration rollouts ---
    # Periodically roll out on random un-selected samples to break the
    # feedback loop where the history buffer only contains previously-selected
    # samples.  Exploration rollouts are fed into the history buffer alongside
    # training rollouts, giving DOTS visibility into regions of the data space
    # that the selector has been ignoring.
    exploration_enabled: bool = False
    exploration_pct: float = 5.0        # % of budget to explore each round
    exploration_interval: int = 1       # explore every N selection rounds
    # Base used by exploration_pct:
    #  - "dataset": percent of full dataset size (default; backward compatible)
    #  - "representatives": percent of current reference-set size (n_reps * n_clusters)
    exploration_pct_base: str = "representatives"

    # --- Asymmetric utility (hard-side bias) ---
    # Variance alone is symmetric around p=0.5, so a sample with 1/8 successes
    # (barely passed) and a sample with 7/8 successes (almost mastered) look
    # identical to the interpolated strategy.  Empirically the 1/8 case is more
    # informative — it lives at the policy's capability frontier and solving it
    # expands capability rather than reinforcing it.  When enabled, we predict
    # both per-sample variance AND per-sample mean reward via DOTS, then form:
    #
    #   utility = predicted_var * (1 + hard_side_bias * (0.5 - predicted_mean))
    #
    # so at the same variance, harder samples (lower mean reward) score higher.
    # Samples with predicted mean outside [dead_zone_low, dead_zone_high] are
    # zeroed out — under GRPO they produce no gradient (advantage collapses to
    # 0 when all rollouts agree), so selecting them wastes budget.
    asymmetric_utility_enabled: bool = False
    hard_side_bias: float = 0.5          # α in (1 + α*(0.5 - mean))
    asymmetric_dead_zone_low: float = 0.05
    asymmetric_dead_zone_high: float = 0.95

    # --- Observed dead-zone (leakage suppression) ---
    # When observed_deadzone_high < 1.0, samples whose *observed* mean reward
    # (from the rollout-history buffer) exceeds this threshold are zeroed out of
    # the utility score.  Unlike asymmetric_dead_zone_high, which uses the
    # KNN-predicted mean (loo R² < 0, unreliable), this uses actual rollout
    # outcomes and therefore reliably suppresses saturated and text-leaky samples.
    # Samples not yet in the rollout buffer default to 0.5 (neutral, not killed).
    # Recommended value: 0.80 (matches the text-leaky threshold from the 2K audit).
    # Set to 1.0 (default) to disable.
    observed_deadzone_high: float = 1.0

    # --- Static-VDR: true no-image audit prior, CPU-built ---
    vdr_enabled: bool = False
    vdr_static_prior_file: Optional[str] = None
    vdr_gate_floor: float = 0.7
    vdr_gate_threshold: float = 0.10
    vdr_gate_temperature: float = 0.10
    vdr_apply_to_sample_utility: bool = True
    vdr_apply_to_cluster_allocation: bool = False
    vdr_cluster_blend: float = 0.25

    # --- CAVA-VDR: Counterfactual-Attention Visual Attribution, v1 ---
    cava_vdr_enabled: bool = False

    # Static VDR prior from offline audit / cluster smoothing.
    cava_static_prior_file: Optional[str] = None
    cava_weight_static_prior: float = 1.0

    # Online null-image log-prob contrast.
    cava_use_logp_contrast: bool = False
    cava_weight_logp_contrast: float = 1.0
    cava_logp_decay_rate: float = 0.05
    cava_logp_max_age: int = 500
    cava_logp_min_points_for_interp: int = 5

    # Soft gate.
    cava_gate_floor: float = 0.7
    cava_gate_temperature: float = 1.0
    cava_gate_threshold: float = 0.0
    cava_robust_normalize: bool = True

    # Where to apply the gate.
    cava_apply_to_sample_utility: bool = True
    cava_apply_to_cluster_allocation: bool = False
    cava_cluster_gate_blend: float = 0.25

    # Trainer-side CAVA log-prob settings, read through cluster_config.
    cava_logp_every_n_steps: int = 1
    cava_logp_max_samples_per_step: int = 32
    cava_null_image_mode: str = "drop_vision"
    cava_logp_sample_policy: str = "first"

    # Attention and sparse null-image rollout probes are deferred to future work.

    # --- Image Grounding Score (IGS) ---
    # Measures multimodal dependency: ratio of reward variance WITH image to
    # variance WITHOUT image.  High IGS means the image is essential for
    # answering (genuinely multimodal).  Low IGS means the question can be
    # answered from text alone.  Used as a multiplier in composite scoring.
    igs_enabled: bool = False
    igs_weight: float = 1.0             # exponent on IGS in composite score

    score_temperature: float = 0.1
    transferability_sim_threshold: float = 0.9
    density_gamma: float = 1.0
    mmd_gamma: float = 1.0

    faiss_nredo: int = 10
    faiss_niter: int = 50
    faiss_seed: int = 42
    use_gpu_faiss: bool = True

    # --- Time-weighted rollout history buffer ---
    # When enabled, training rollouts are accumulated alongside the REPR rollouts
    # and used as additional reference points for DOTS interpolation. Each entry
    # in the buffer carries the training step at which it was observed; at
    # selection time, variance estimates are weighted by recency:
    #   w(t) = exp(-decay_rate * (current_step - t))
    # so stale observations (many steps ago) contribute far less than fresh ones.
    #
    # Round 0 seeds the buffer with REPR rollouts. Subsequent training steps
    # continuously add new entries. By round N the reference set has grown
    # from ~600 fixed medoids to potentially thousands of trained-on samples,
    # all decayed appropriately.
    use_rollout_history: bool = False
    rollout_history_decay_rate: float = 0.05   # λ in exp(-λ * Δstep)
    rollout_history_max_age: int = 500          # discard entries older than this
    # Cap on reference-set size for DOTS.  Set to 0 to disable the cap entirely
    # and use the full age-pruned buffer (recommended — there is no reason to
    # throw away recent observations the trainer already paid to compute).
    rollout_history_max_refs: int = 0

    # --- Discovery / freeze behaviour ---
    # When True, samples already in the cumulative `_ever_selected_set` are
    # excluded from the candidate pool inside _select_interpolated.  This makes
    # each pre-freeze round add `budget` *new* unique samples instead of mostly
    # reselecting the same top-K.  Combined with a small per-round budget +
    # global_budget_pct, this implements selection-without-replacement: the
    # cumulative unique count rises by exactly `budget` per round until the
    # global cap freezes the pool.
    exclude_already_selected: bool = True

    # When False, medoid (REPR) reference rollouts only fire on round 0.
    # Subsequent rounds reuse the rollout-history buffer accumulated from
    # training-step rewards instead of re-rolling the same fixed medoids.
    # Saves a substantial amount of inference compute per round.
    reroll_medoids: bool = False

    # Path to the JSONL/JSON source file that was used to build the cluster
    # embeddings (e.g. VLAA-Thinking-GRPO-25K_train_90_100.json).  Required
    # when the cluster_arrays.npz was built from this JSON file and the
    # training parquet has a *different* row ordering — which is the common
    # case when the parquet was produced via a separate conversion pipeline.
    # When provided, initialize() builds a bidirectional npz↔parquet alignment
    # map so that:
    #   - REPR rollouts reference the correct parquet rows
    #   - select() returns correct parquet indices for _rebuild_dataloader
    #   - update_rollout_history() stores rewards at the right NPZ positions
    # If not provided and the orderings differ, selection is silently incorrect.
    dataset_json_file: Optional[str] = None

    # --- Variance normalization ---
    # When True, normalize per-sample variance by mean*(1-mean) so that
    # binary-reward tasks (math/mcq/digit, max raw var = 0.25) and
    # continuous-reward tasks (IoU grounding, typically much lower raw var)
    # are scored on a comparable [0, 1] scale.  A value of 1.0 means
    # "maximally uncertain given the observed mean reward."
    normalize_variance: bool = True

    # Reward-extra field used by the selector for reference/training rollout
    # labels. Keep PPO/GRPO's shaped reward in "score", but use true task
    # accuracy/IoU ("acc") for normalized variance and dead-zone logic.
    selection_reward_key: str = "acc"

    # --- Medoid budget accounting ---
    # When False (default), medoid/representative rollouts do NOT count
    # against the global_budget_pct cap — only training and exploration
    # samples count.  Set True to restore legacy behaviour where medoid
    # probes consume part of the annotation budget.
    count_medoids_in_budget: bool = False

    # --- Budget schedule (phased discovery) ---
    # When set, overrides the flat selection_budget_pct and reselect_interval
    # with a multi-phase schedule that tapers discovery rate as the budget
    # fills.  Each entry is a dict with:
    #   until_budget_pct: float  — transition to next phase when cumulative
    #                              unique *training* samples reach this % of
    #                              the global budget cap
    #   per_round_pct: float     — per-round selection budget (% of dataset)
    #   interval: int             — reselect every N training steps
    # Phases are evaluated in order; the first phase whose until_budget_pct
    # has NOT been reached is the active phase.  When all phases are
    # exhausted, the pool freezes as before.
    # Example:
    #   [{"until_budget_pct": 50, "per_round_pct": 1.0, "interval": 4},
    #    {"until_budget_pct": 85, "per_round_pct": 0.3, "interval": 10},
    #    {"until_budget_pct": 100, "per_round_pct": 0.15, "interval": 16}]
    budget_schedule: Optional[list] = None

    # --- Variance predictor ---
    # Controls which predictor backs the DOTS interpolation.
    #   "knn"   — legacy cosine-KNN / Nadaraya-Watson (default, zero behavior change)
    #   "ridge" — weighted Ridge regression with joint p-hat head + uncertainty
    #   "mlp"   — 2-layer MLP warm-started across rounds
    predictor_type: str = "knn"

    # Ridge-specific: regularization strength
    predictor_alpha: float = 1.0

    # MLP-specific
    predictor_mlp_hidden: int = 256
    predictor_mlp_lr: float = 1e-3
    predictor_mlp_steps: int = 10
    predictor_mlp_weight_decay: float = 1e-3

    # Active probe selection (requires predictor with uncertainty, e.g. "ridge")
    predictor_active_probes: bool = False
    predictor_ucb_beta: float = 1.0
    predictor_active_probe_suppress_radius: float = 0.1


class ClusterSelector(DataSelector):
    """Online cluster-based data selector.

    Maintains pre-computed clusters and representatives. At each selection
    round, rolls out on representatives, computes cluster-level variance,
    and selects training data accordingly.
    """

    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)
        cluster_cfg = config.cluster if isinstance(config.cluster, dict) else {}
        self.cluster_config = ClusterSelectorConfig(**cluster_cfg)

        self._dataset = None
        self._dataset_size = 0

        self._embeddings: Optional[np.ndarray] = None
        self._cluster_ids: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._rep_indices: List[int] = []
        self._rep_cluster_ids: List[int] = []

        self._transferability: Optional[np.ndarray] = None
        self._density: Optional[np.ndarray] = None

        self._cluster_variances: Dict[int, float] = {}
        self._cluster_mean_rewards: Dict[int, float] = {}
        self._rep_variances: Dict[int, float] = {}  # global_idx -> individual rollout variance
        self._rep_mean_rewards: Dict[int, float] = {}  # global_idx -> individual rollout mean reward
        self._rep_reward_rows: Dict[int, np.ndarray] = {}  # global_idx -> raw rollout rewards
        self._last_selected_indices: List[int] = []
        self._selection_round: int = 0  # incremented each call to select()

        # --- Selection overlap tracking ---
        self._prev_selected_set: set = set()  # NPZ indices from previous round
        self._selection_jaccard: float = 0.0
        self._selection_jaccard_vs_random: float = 0.0  # per-round, overwritten
        self._selection_history: List[set] = []  # all rounds' NPZ index sets
        self._per_cluster_coverage: Dict[int, int] = {}  # cluster_id -> times selected

        # --- Annotation budget tracking ---
        # _ever_selected_set is the union of EVERY sample for which the trainer
        # has consumed ground-truth (rollout reward computation).  It is the
        # quantity that the global_budget_pct cap is enforced against.  Three
        # disjoint sources contribute, tracked separately so we can report a
        # per-source breakdown to wandb:
        #   training    — samples returned from select() and trained on
        #   medoid      — REPR rollouts run via update_rewards() (round 0 only
        #                 when reroll_medoids=False)
        #   exploration — random rollouts run via update_exploration_rewards()
        # _ever_selected_set is maintained as the union of these three.
        self._ever_selected_train: set = set()
        self._ever_selected_medoid: set = set()
        self._ever_selected_exploration: set = set()
        self._ever_selected_set: set = set()
        # Per-round snapshot of (round, train, medoid, exploration) for the
        # cumulative budget breakdown stacked-area chart in wandb.
        self._budget_history: List[Tuple[int, int, int, int]] = []

        # Cache of the most recent per-sample DOTS-predicted mean reward
        # (only populated when asymmetric_utility_enabled=True).  Used by the
        # wandb visualization to colour selected samples by predicted difficulty.
        self._last_predicted_mean: Optional[np.ndarray] = None
        self._last_predicted_var: Optional[np.ndarray] = None
        # 2D PCA projection of the embeddings, computed once at initialize().
        # Used for the selection-overlay scatter plot.
        self._embedding_2d: Optional[np.ndarray] = None

        # --- Exploration rollouts ---
        self._exploration_indices: List[int] = []  # NPZ indices for next exploration
        self._exploration_rewards: Optional[np.ndarray] = None

        # --- Image Grounding Score (IGS) ---
        # Per-sample IGS: var_with_image / var_without_image.
        # Computed externally and loaded, or computed on-the-fly from rollouts.
        self._igs_scores: Optional[np.ndarray] = None  # shape (N,), per-sample
        self._cluster_igs: Dict[int, float] = {}  # cluster_id -> mean IGS

        # --- Static-VDR ---
        self._vdr_sample_delta_prior: Optional[np.ndarray] = None
        self._vdr_sample_confidence: Optional[np.ndarray] = None
        self._vdr_cluster_delta_mean: Optional[np.ndarray] = None
        self._vdr_cluster_delta_count: Optional[np.ndarray] = None
        self._last_vdr_gate: Optional[np.ndarray] = None
        self._last_vdr_metrics: Dict[str, float] = {}

        # --- CAVA-VDR v1 ---
        # Static prior and online null-image log-prob contrast buffers are in
        # NPZ/embedding index space.
        self._cava_static_prior: Optional[np.ndarray] = None
        self._cava_cluster_prior: Optional[np.ndarray] = None
        self._cava_logp_buffer: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self._last_cava_logp_interp: Optional[np.ndarray] = None
        self._last_cava_score: Optional[np.ndarray] = None
        self._last_cava_gate: Optional[np.ndarray] = None
        self._last_cava_metrics: Dict[str, float] = {}

        # Time-weighted rollout history buffer.
        # Keys are NPZ positions (0..N-1). Seeded with REPR rollouts;
        # grows with training rollouts when alignment map is available.
        self._rollout_buffer: Dict[int, List] = {}
        # Maps string UID -> integer position in embeddings array.
        # Built from cluster_arrays.npz uids field during initialize().
        self._uid_to_dataset_idx: Dict[str, int] = {}

        # Alignment maps between NPZ row order and training-parquet row order.
        # These are different orderings of the same underlying samples.
        # _npz_to_dataset[npz_i] = parquet_i  (used to remap select() output)
        # _dataset_to_npz[parquet_i] = npz_i  (used in update_rollout_history)
        # Built during initialize() when dataset_json_file is provided.
        self._npz_to_dataset: Optional[np.ndarray] = None   # shape (N_npz,)
        self._dataset_to_npz: Optional[Dict[int, int]] = None

        self._current_training_step: int = 0

        # NPZ indices of the frozen pool (set when global budget cap triggers).
        self._frozen_pool_npz: Optional[List[int]] = None

        self._rng = np.random.RandomState(42)

        # Variance predictor (KNN / Ridge / MLP)
        self._predictor: VariancePredictor = build_predictor(self.cluster_config)
        # Cached diagnostics from the last predictor round (LOO-KNN R², etc.)
        self._last_predictor_diagnostics: Dict[str, float] = {}
        # Per-round cache so fit+predict runs at most once per select() call
        # even when both _get_active_probe_indices and _select_interpolated
        # would otherwise invoke the predictor on the same observations.
        self._predictor_cache_round: int = -1
        self._predictor_cache_result: Optional[PredictionResult] = None

    # ------------------------------------------------------------------
    # Budget schedule helpers
    # ------------------------------------------------------------------

    def _budget_counted_set(self) -> set:
        """NPZ indices that count against the global selection budget."""
        if self.cluster_config.count_medoids_in_budget:
            return set(self._ever_selected_set)
        return set(self._ever_selected_train) | set(self._ever_selected_exploration)

    def _get_current_phase(self):
        """Return (per_round_pct, interval) for the active budget phase.

        If ``budget_schedule`` is None or empty, falls back to the flat
        config values (selection_budget_pct, reselect_interval).
        """
        schedule = self.cluster_config.budget_schedule
        if not schedule:
            return (self.config.selection_budget_pct, self.config.reselect_interval)

        # Compute current budget utilisation.
        n_used = len(self._budget_counted_set())

        # Global max from config
        n_embeddings = len(self._embeddings) if self._embeddings is not None else self._dataset_size
        global_budget_pct = self.config.global_budget_pct
        if global_budget_pct is None:
            # No global cap — schedule doesn't apply
            return (self.config.selection_budget_pct, self.config.reselect_interval)
        global_max = int(n_embeddings * global_budget_pct / 100.0)
        if global_max <= 0:
            return (self.config.selection_budget_pct, self.config.reselect_interval)

        current_pct = (n_used / global_max) * 100.0

        for phase in schedule:
            if current_pct < phase["until_budget_pct"]:
                return (phase["per_round_pct"], phase["interval"])

        # All phases exhausted — return the last phase's values
        last = schedule[-1]
        return (last["per_round_pct"], last["interval"])

    def should_reselect_step(self, global_step: int) -> bool:
        """Override base class to use the budget schedule's interval."""
        if self.config.reselect_schedule != "step":
            return False
        _, interval = self._get_current_phase()
        if interval <= 0:
            return False
        return global_step % interval == 0

    def get_selection_budget(self, dataset_size: int) -> int:
        """Override base class to use the budget schedule's per_round_pct.

        When the pool is frozen, returns the frozen pool size instead of
        the per-round discovery budget — this ensures the trainer always
        gets at least one full batch worth of samples for training.
        """
        if self.config.selection_budget is not None:
            return min(self.config.selection_budget, dataset_size)
        # Frozen reweight: return the pool size so the trainer gets a full
        # training window from the frozen pool rather than the tiny
        # per-round discovery budget (which may be < train_batch_size).
        if self._selection_frozen and self._frozen_pool_npz is not None:
            return len(self._frozen_pool_npz)
        per_round_pct, _ = self._get_current_phase()
        return max(1, int(dataset_size * per_round_pct / 100.0))

    # ------------------------------------------------------------------
    # Checkpoint serialization
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Return all mutable selector state for checkpoint saving.

        Only runtime state is saved — static data (embeddings, cluster
        arrays, alignment maps) is always reloaded from disk on resume.
        """
        return {
            # Base class
            "_step_count": self._step_count,
            "_selection_frozen": self._selection_frozen,
            # Budget tracking
            "_ever_selected_set": self._ever_selected_set,
            "_ever_selected_train": self._ever_selected_train,
            "_ever_selected_medoid": self._ever_selected_medoid,
            "_ever_selected_exploration": self._ever_selected_exploration,
            "_budget_history": self._budget_history,
            # Frozen pool
            "_frozen_pool_npz": self._frozen_pool_npz,
            # Rollout history buffer (core DOTS signal)
            "_rollout_buffer": dict(self._rollout_buffer),
            # CAVA-VDR online signal buffers
            "_cava_logp_buffer": dict(self._cava_logp_buffer),
            # Cluster-level signals
            "_cluster_variances": dict(self._cluster_variances),
            "_cluster_mean_rewards": dict(self._cluster_mean_rewards),
            "_rep_variances": dict(self._rep_variances),
            "_rep_mean_rewards": dict(self._rep_mean_rewards),
            "_rep_reward_rows": dict(self._rep_reward_rows),
            # Selection tracking
            "_selection_round": self._selection_round,
            "_current_training_step": self._current_training_step,
            "_prev_selected_set": self._prev_selected_set,
            "_selection_jaccard": self._selection_jaccard,
            "_per_cluster_coverage": dict(self._per_cluster_coverage),
            # RNG state for reproducibility
            "_rng_state": self._rng.get_state(),
            # Variance predictor state (model weights for MLP/Ridge)
            "_predictor_state": self._predictor.get_state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore mutable selector state from a checkpoint.

        Called after initialize() so that static data (embeddings,
        alignment maps) is already loaded before we restore runtime state.
        """
        self._step_count = state["_step_count"]
        self._selection_frozen = state["_selection_frozen"]

        self._ever_selected_set = state["_ever_selected_set"]
        self._ever_selected_train = state["_ever_selected_train"]
        self._ever_selected_medoid = state["_ever_selected_medoid"]
        self._ever_selected_exploration = state["_ever_selected_exploration"]
        self._budget_history = state["_budget_history"]

        self._frozen_pool_npz = state["_frozen_pool_npz"]
        self._rollout_buffer = state["_rollout_buffer"]
        self._cava_logp_buffer = defaultdict(list, state.get("_cava_logp_buffer", {}))

        self._cluster_variances = state["_cluster_variances"]
        self._cluster_mean_rewards = state["_cluster_mean_rewards"]
        self._rep_variances = state["_rep_variances"]
        self._rep_mean_rewards = state["_rep_mean_rewards"]
        self._rep_reward_rows = state.get("_rep_reward_rows", {})

        self._selection_round = state["_selection_round"]
        self._current_training_step = state["_current_training_step"]
        self._prev_selected_set = state["_prev_selected_set"]
        self._selection_jaccard = state["_selection_jaccard"]
        self._per_cluster_coverage = state["_per_cluster_coverage"]

        self._rng.set_state(state["_rng_state"])

        if "_predictor_state" in state:
            self._predictor.load_state_dict(state["_predictor_state"])

        frozen_str = " (pool FROZEN)" if self._selection_frozen else ""
        print(f"[ClusterSelector] Restored state from checkpoint{frozen_str}: "
              f"round={self._selection_round}, step={self._current_training_step}, "
              f"ever_selected_train={len(self._ever_selected_train)}, "
              f"ever_selected_medoid={len(self._ever_selected_medoid)}, "
              f"rollout_buffer={len(self._rollout_buffer)} unique samples, "
              f"frozen_pool={len(self._frozen_pool_npz) if self._frozen_pool_npz else 0}")

    def initialize(self, dataset, collate_fn=None) -> None:
        self._dataset = dataset
        self._dataset_size = len(dataset)

        print(f"[ClusterSelector] Initializing with {self._dataset_size} samples, "
              f"strategy={self.cluster_config.strategy}")

        if self.cluster_config.cluster_arrays_file:
            self._load_precomputed_clusters()
        elif self.cluster_config.embeddings_file:
            self._load_embeddings_and_cluster()
        else:
            raise ValueError(
                "ClusterSelector requires either cluster_arrays_file or "
                "embeddings_file to be specified in config"
            )

        # Build NPZ↔parquet alignment map from the source JSON if provided.
        # This is required when the cluster_arrays.npz was built from a JSON file
        # whose row ordering differs from the training parquet row ordering.
        if self.cluster_config.dataset_json_file:
            self._build_alignment_from_json(dataset, self.cluster_config.dataset_json_file)
        else:
            print("[ClusterSelector] WARNING: dataset_json_file not set. "
                  "Assuming NPZ row order == parquet row order. If they differ, "
                  "REPR rollouts and select() will reference wrong parquet rows. "
                  "Set data_selection.cluster.dataset_json_file to the JSON source "
                  "used to build the embeddings.")

        self._load_vdr_static_prior()
        self._load_cava_static_prior()
        self._select_representatives()

        if self.cluster_config.strategy in ("scored", "interpolated"):
            self._compute_static_scores()

        # Cheap one-shot 2D PCA projection of the embeddings for the wandb
        # selection-overlay scatter plot.  Done once because the embeddings are
        # static; subsequent rounds just index into _embedding_2d.
        try:
            X = self._embeddings.astype(np.float32)
            mu = X.mean(axis=0, keepdims=True)
            Xc = X - mu
            # Use SVD on a row sample if the dataset is huge — fitting on 5k
            # rows is more than enough for a 2D projection.
            if Xc.shape[0] > 5000:
                idx = self._rng.choice(Xc.shape[0], size=5000, replace=False)
                _, _, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
            else:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._embedding_2d = (Xc @ Vt[:2].T).astype(np.float32)
        except Exception as e:
            print(f"[ClusterSelector] PCA projection failed ({e}); "
                  f"selection scatter plot will be disabled.")
            self._embedding_2d = None

        print(f"[ClusterSelector] Ready: {len(set(self._cluster_ids))} clusters, "
              f"{len(self._rep_indices)} representatives")

    def _load_precomputed_clusters(self) -> None:
        """Load pre-computed cluster arrays (embeddings, centroids, assignments)."""
        path = self.cluster_config.cluster_arrays_file
        print(f"[ClusterSelector] Loading pre-computed clusters from {path}")
        data = np.load(path, allow_pickle=True)

        self._embeddings = data["embeddings"].astype(np.float32)
        self._centroids = data["centroids"].astype(np.float32)
        self._cluster_ids = data["assignments"].astype(np.int64)

        n_active = len(np.unique(self._cluster_ids))
        print(f"[ClusterSelector] Loaded {len(self._embeddings)} embeddings, "
              f"{self._centroids.shape[0]} centroids, {n_active} active clusters")

        if len(self._embeddings) != self._dataset_size:
            print(f"[ClusterSelector] WARNING: embeddings size ({len(self._embeddings)}) "
                  f"!= dataset size ({self._dataset_size}). Using min.")
            min_size = min(len(self._embeddings), self._dataset_size)
            self._embeddings = self._embeddings[:min_size]
            self._cluster_ids = self._cluster_ids[:min_size]

        if self.cluster_config.use_rollout_history:
            if "uids" in data:
                raw_uids = data["uids"]
                self._uid_to_dataset_idx = {
                    str(u): i for i, u in enumerate(raw_uids)
                    if i < len(self._embeddings)
                }
                print(f"[ClusterSelector] Built UID→idx map: "
                      f"{len(self._uid_to_dataset_idx)} entries")
            else:
                print("[ClusterSelector] WARNING: use_rollout_history=True but "
                      "cluster_arrays.npz has no 'uids' field — training-batch rollouts "
                      "will NOT be accumulated. Re-run Stage 1 (01_cluster.py) to "
                      "regenerate cluster_arrays.npz with the uids field.")

    def _load_embeddings_and_cluster(self) -> None:
        """Load embeddings from file and run FAISS clustering."""
        path = self.cluster_config.embeddings_file
        print(f"[ClusterSelector] Loading embeddings from {path}")
        data = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"].astype(np.float32)

        if len(self._embeddings) != self._dataset_size:
            print(f"[ClusterSelector] WARNING: embeddings ({len(self._embeddings)}) "
                  f"!= dataset ({self._dataset_size})")
            min_size = min(len(self._embeddings), self._dataset_size)
            self._embeddings = self._embeddings[:min_size]

        print(f"[ClusterSelector] Running FAISS spherical KMeans "
              f"(K={self.cluster_config.n_clusters})...")

        import faiss

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
        self._embeddings = (self._embeddings / norms).astype(np.float32)

        n, d = self._embeddings.shape
        kmeans = faiss.Kmeans(
            d=d,
            k=self.cluster_config.n_clusters,
            niter=self.cluster_config.faiss_niter,
            nredo=self.cluster_config.faiss_nredo,
            spherical=True,
            seed=self.cluster_config.faiss_seed,
            verbose=True,
            gpu=self.cluster_config.use_gpu_faiss and faiss.get_num_gpus() > 0,
        )
        kmeans.train(self._embeddings)

        distances, assignments = kmeans.index.search(self._embeddings, 1)
        self._cluster_ids = assignments.squeeze(1).astype(np.int64)
        self._centroids = kmeans.centroids.astype(np.float32)

        print(f"[ClusterSelector] Clustering complete: "
              f"{len(np.unique(self._cluster_ids))} active clusters")

        # Build UID→idx map from embeddings file if UIDs are stored there.
        if self.cluster_config.use_rollout_history:
            for key in ("uids", "ids"):
                if key in data:
                    raw_uids = data[key]
                    self._uid_to_dataset_idx = {
                        str(u): i for i, u in enumerate(raw_uids)
                        if i < len(self._embeddings)
                    }
                    print(f"[ClusterSelector] Built UID→idx map from embeddings file "
                          f"(key='{key}'): {len(self._uid_to_dataset_idx)} entries")
                    break
            else:
                print("[ClusterSelector] WARNING: use_rollout_history=True but embeddings "
                      "file has no 'uids'/'ids' field — training-batch rollouts will NOT "
                      "be accumulated. Use cluster_arrays_file instead of embeddings_file, "
                      "or run Stage 1 (01_cluster.py) to produce cluster_arrays.npz.")

    def _build_alignment_from_json(self, dataset, json_path: str) -> None:
        """Build npz↔parquet index alignment maps.

        Produces:
            _npz_to_dataset[npz_i]   = parquet_i   (shape N_npz, fill -1 if no match)
            _dataset_to_npz[parquet_i] = npz_i     (only for matched rows)

        Strategy (in priority order):
        1. **QID-based** (preferred): if the NPZ has a 'uids' array of qids AND the
           parquet stores 'qid' in extra_info, match directly on qid.  This is
           content-based and immune to differences in row ordering between the JSON
           source file (e.g. all.jsonl) and the file the NPZ was actually built from
           (e.g. train_90_100.jsonl).
        2. **Image-path-based via JSONL**: for each NPZ row, look up the qid in the
           JSON source to get the image path, then match against extra_info['image']
           in the parquet.  Handles cases where the parquet has no 'qid' field.
        3. **Positional fallback** (legacy): assume JSONL row i == NPZ row i.  Only
           correct when the JSONL and NPZ share the exact same row ordering.

        Once built, get_reference_indices() and select() remap their NPZ-order
        outputs to parquet-order indices via _npz_to_dataset.
        """
        import json as _json

        print(f"[ClusterSelector] Building NPZ↔parquet alignment from {json_path} ...")

        n_npz = len(self._embeddings)

        # --- Step 1: scan parquet once to build lookup maps ---
        parquet_img_to_idx: Dict[str, int] = {}   # image_path  → parquet_i
        parquet_qid_to_idx: Dict[str, int] = {}   # qid string  → parquet_i
        for parquet_i in range(len(dataset)):
            try:
                item = dataset.dataframe[parquet_i]  # direct pandas access, no preprocessing
            except Exception:
                try:
                    item = dataset[parquet_i]
                except Exception:
                    continue
            ei = item.get("extra_info") or {}
            img = ei.get("image", "")
            qid = str(ei.get("qid", ""))
            if img:
                parquet_img_to_idx[img] = parquet_i
            if qid:
                parquet_qid_to_idx[qid] = parquet_i

        # --- Step 2: load JSON records, building {qid: image_path} lookup ---
        # We read the full JSONL regardless of strategy so we have the image paths
        # available for the image-path fallback.  'image_rel' (new dataset_prep
        # pipeline) and 'image' (legacy VLAA-style) are both supported.
        json_qid_to_img: Dict[str, str] = {}      # qid → image path (from JSONL)
        json_image_paths_positional: List[str] = []  # for legacy positional fallback
        with open(json_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    img = rec.get("image", "") or rec.get("image_rel", "")
                    qid = str(rec.get("qid", ""))
                    if qid:
                        json_qid_to_img[qid] = img
                    json_image_paths_positional.append(img)
                except _json.JSONDecodeError:
                    json_image_paths_positional.append("")

        json_image_paths_positional = json_image_paths_positional[:n_npz]

        # --- Step 3: build alignment ---
        self._npz_to_dataset = np.full(n_npz, -1, dtype=np.int64)
        self._dataset_to_npz = {}
        n_matched = 0

        if self._uid_to_dataset_idx and parquet_qid_to_idx:
            # Strategy 1: QID-based — content-safe, ordering-independent.
            # _uid_to_dataset_idx = {qid: npz_idx} (built from NPZ 'uids' in
            # _load_precomputed_clusters / _load_embeddings_and_cluster).
            for qid, npz_idx in self._uid_to_dataset_idx.items():
                if npz_idx >= n_npz:
                    continue
                parquet_i = parquet_qid_to_idx.get(qid, -1)
                if parquet_i < 0 and json_qid_to_img:
                    # Try image-path fallback for this qid (Strategy 2).
                    img = json_qid_to_img.get(qid, "")
                    if img:
                        parquet_i = parquet_img_to_idx.get(img, -1)
                self._npz_to_dataset[npz_idx] = parquet_i
                if parquet_i >= 0:
                    self._dataset_to_npz[parquet_i] = npz_idx
                    n_matched += 1
            method = "qid"
        elif json_image_paths_positional and parquet_img_to_idx:
            # Strategy 3: positional fallback — only correct when JSONL and NPZ
            # share the exact same row ordering.
            for npz_i, img_path in enumerate(json_image_paths_positional):
                parquet_i = parquet_img_to_idx.get(img_path, -1)
                self._npz_to_dataset[npz_i] = parquet_i
                if parquet_i >= 0:
                    self._dataset_to_npz[parquet_i] = npz_i
                    n_matched += 1
            method = "positional-image-path"
        else:
            method = "none"

        n_unmatched = n_npz - n_matched
        print(f"[ClusterSelector] Alignment ({method}): {n_matched}/{n_npz} NPZ rows matched "
              f"to parquet rows ({n_unmatched} unmatched — typically test-set samples "
              f"not present in the training parquet).")

    def _remap_npz_to_dataset(self, npz_indices: List[int]) -> List[int]:
        """Map NPZ positions → parquet positions, dropping any without a match."""
        if self._npz_to_dataset is None:
            return npz_indices  # assume aligned (no JSON provided)
        result = []
        for i in npz_indices:
            p = int(self._npz_to_dataset[i])
            if p >= 0:
                result.append(p)
        return result

    def _dataset_index_to_npz(self, dataset_idx: int) -> Optional[int]:
        """Map a parquet/dataloader index to NPZ index space."""
        if self._dataset_to_npz is not None:
            npz_idx = self._dataset_to_npz.get(int(dataset_idx), -1)
            return int(npz_idx) if npz_idx >= 0 else None
        return int(dataset_idx)

    # ------------------------------------------------------------------
    # Static-VDR: true no-image audit prior
    # ------------------------------------------------------------------

    def _load_vdr_static_prior(self) -> None:
        """Load CPU-built static VDR prior in NPZ/embedding order."""
        cfg = self.cluster_config
        path = cfg.vdr_static_prior_file
        self._vdr_sample_delta_prior = None
        self._vdr_sample_confidence = None
        self._vdr_cluster_delta_mean = None
        self._vdr_cluster_delta_count = None

        if not cfg.vdr_enabled or not path:
            return

        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[StaticVDR] Failed to load prior from {path}: {e}")
            return

        n = len(self._embeddings)
        k = self._centroids.shape[0] if self._centroids is not None else 0

        sample_prior = None
        for key in ("sample_delta_prior", "sample_prior", "prior"):
            if key in data:
                sample_prior = data[key].astype(np.float32)
                break
        if sample_prior is not None:
            if len(sample_prior) != n:
                print(
                    f"[StaticVDR] sample prior length mismatch: "
                    f"{len(sample_prior)} vs embeddings {n}; disabling Static-VDR."
                )
                sample_prior = None
            else:
                self._vdr_sample_delta_prior = sample_prior

        if "sample_prior_confidence" in data:
            confidence = data["sample_prior_confidence"].astype(np.float32)
            if len(confidence) == n:
                self._vdr_sample_confidence = confidence

        cluster_prior = None
        for key in ("cluster_delta_mean", "cluster_prior"):
            if key in data:
                cluster_prior = data[key].astype(np.float32)
                break
        if cluster_prior is not None:
            if len(cluster_prior) != k:
                print(
                    f"[StaticVDR] cluster prior length mismatch: "
                    f"{len(cluster_prior)} vs clusters {k}; ignoring cluster prior."
                )
            else:
                self._vdr_cluster_delta_mean = cluster_prior
                if self._vdr_sample_delta_prior is None:
                    self._vdr_sample_delta_prior = cluster_prior[self._cluster_ids].astype(np.float32)

        if "cluster_delta_count" in data:
            counts = data["cluster_delta_count"].astype(np.float32)
            if len(counts) == k:
                self._vdr_cluster_delta_count = counts

        if self._vdr_sample_delta_prior is None:
            print(f"[StaticVDR] No usable static prior found in {path}")
            return

        finite = self._vdr_sample_delta_prior[np.isfinite(self._vdr_sample_delta_prior)]
        msg = (
            f"mean={float(finite.mean()):.4f}, std={float(finite.std()):.4f}, "
            f"min={float(finite.min()):.4f}, max={float(finite.max()):.4f}"
            if finite.size
            else "no finite values"
        )
        print(f"[StaticVDR] Loaded prior from {path}: {msg}")

    def _compute_vdr_gate(self) -> np.ndarray:
        """Direct soft gate from raw true-no-image delta prior."""
        cfg = self.cluster_config
        n = len(self._embeddings)
        if not cfg.vdr_enabled or self._vdr_sample_delta_prior is None:
            gate = np.ones(n, dtype=np.float32)
            self._last_vdr_gate = gate
            self._last_vdr_metrics = {
                "data_selection/vdr_gate_mean_all": 1.0,
                "data_selection/vdr_score_mean_all": 0.0,
            }
            return gate

        score = np.asarray(self._vdr_sample_delta_prior, dtype=np.float32)
        score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)
        floor = float(np.clip(cfg.vdr_gate_floor, 0.0, 1.0))
        temp = max(float(cfg.vdr_gate_temperature), 1e-8)
        logits = np.clip((score - float(cfg.vdr_gate_threshold)) / temp, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-logits))
        gate = (floor + (1.0 - floor) * sig).astype(np.float32)
        gate = np.clip(gate, floor, 1.0)

        self._last_vdr_gate = gate
        self._last_vdr_metrics = {
            "data_selection/vdr_gate_mean_all": float(gate.mean()),
            "data_selection/vdr_gate_std_all": float(gate.std()),
            "data_selection/vdr_gate_min_all": float(gate.min()),
            "data_selection/vdr_gate_max_all": float(gate.max()),
            "data_selection/vdr_score_mean_all": float(score.mean()),
            "data_selection/vdr_score_std_all": float(score.std()),
        }
        return gate

    def _record_vdr_selection_metrics(self, selected_npz: List[int], gate: np.ndarray) -> None:
        if not selected_npz or self._vdr_sample_delta_prior is None:
            return
        sel = np.asarray(selected_npz, dtype=np.int64)
        sel = sel[(sel >= 0) & (sel < len(gate))]
        if sel.size == 0:
            return

        score_sel = self._vdr_sample_delta_prior[sel].astype(np.float64)
        gate_sel = gate[sel].astype(np.float64)
        low_gate = gate_sel <= (float(self.cluster_config.vdr_gate_floor) + 1e-6)
        metrics = dict(self._last_vdr_metrics)
        metrics.update(
            {
                "data_selection/vdr_gate_mean_selected": float(gate_sel.mean()),
                "data_selection/vdr_gate_min_selected": float(gate_sel.min()),
                "data_selection/vdr_gate_max_selected": float(gate_sel.max()),
                "data_selection/vdr_score_mean_selected": float(score_sel.mean()),
                "data_selection/vdr_score_std_selected": float(score_sel.std()),
                "data_selection/vdr_low_gate_selected_frac": float(low_gate.mean()),
            }
        )
        self._last_vdr_metrics = metrics

    # ------------------------------------------------------------------
    # CAVA-VDR v1: static prior + null-image log-prob contrast
    # ------------------------------------------------------------------

    def _load_cava_static_prior(self) -> None:
        """Load optional CAVA static prior in NPZ/embedding order."""
        cfg = self.cluster_config
        path = cfg.cava_static_prior_file
        self._cava_static_prior = None
        self._cava_cluster_prior = None

        if not cfg.cava_vdr_enabled or not path:
            return

        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[CAVA] Failed to load static prior from {path}: {e}")
            return

        n = len(self._embeddings)
        k = self._centroids.shape[0] if self._centroids is not None else 0

        sample_prior = None
        for key in ("sample_prior", "prior", "cava_prior", "delta_prior"):
            if key in data:
                sample_prior = data[key].astype(np.float32)
                break

        if sample_prior is not None:
            if len(sample_prior) != n:
                print(
                    f"[CAVA] Static sample prior length mismatch: "
                    f"{len(sample_prior)} vs embeddings {n}; disabling sample prior."
                )
            else:
                self._cava_static_prior = sample_prior

        if "cluster_prior" in data:
            cluster_prior = data["cluster_prior"].astype(np.float32)
            if len(cluster_prior) != k:
                print(
                    f"[CAVA] Static cluster prior length mismatch: "
                    f"{len(cluster_prior)} vs clusters {k}; ignoring cluster prior."
                )
            else:
                self._cava_cluster_prior = cluster_prior
                if self._cava_static_prior is None:
                    self._cava_static_prior = cluster_prior[self._cluster_ids].astype(np.float32)

        if self._cava_static_prior is not None:
            finite = self._cava_static_prior[np.isfinite(self._cava_static_prior)]
            msg = (
                f"mean={float(finite.mean()):.4f}, std={float(finite.std()):.4f}"
                if finite.size
                else "no finite values"
            )
            print(f"[CAVA] Loaded static prior from {path}: {msg}")
        else:
            print(f"[CAVA] No usable static prior found in {path}")

    def update_cava_logp_contrast(
        self,
        dataset_indices: List[int],
        contrasts: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """Store online null-image log-prob contrasts.

        dataset_indices are parquet/dataloader indices. Contrasts are scalar
        per original sample: mean_response_tokens(logp_real - logp_null).
        """
        cfg = self.cluster_config
        if not cfg.cava_vdr_enabled or not cfg.cava_use_logp_contrast:
            return

        current_step = self._current_training_step if step is None else int(step)
        self._current_training_step = current_step
        max_age = int(cfg.cava_logp_max_age)
        n_added = 0

        for dataset_idx, contrast in zip(dataset_indices, np.asarray(contrasts, dtype=np.float32)):
            if not np.isfinite(contrast):
                continue
            npz_idx = self._dataset_index_to_npz(int(dataset_idx))
            if npz_idx is None or npz_idx < 0 or npz_idx >= len(self._embeddings):
                continue
            hist = self._cava_logp_buffer[int(npz_idx)]
            hist.append((current_step, float(contrast)))
            if max_age > 0:
                self._cava_logp_buffer[int(npz_idx)] = [
                    (t, v) for t, v in hist if current_step - int(t) <= max_age
                ]
            n_added += 1

        if n_added > 0:
            print(
                f"[CAVA] Added {n_added} log-prob contrasts "
                f"(observed_npz={len(self._cava_logp_buffer)})"
            )

    def _aggregate_and_interpolate_cava_logp(self) -> Optional[np.ndarray]:
        cfg = self.cluster_config
        if not self._cava_logp_buffer:
            self._last_cava_logp_interp = None
            return None

        from .vdr_utils import fill_by_cluster_mean, knn_interpolate_values, time_weighted_scalar

        ref_idx = []
        ref_val = []
        for npz_idx, history in self._cava_logp_buffer.items():
            val = time_weighted_scalar(
                history,
                current_step=self._current_training_step,
                decay_rate=cfg.cava_logp_decay_rate,
                max_age=cfg.cava_logp_max_age,
            )
            if val is None or not np.isfinite(val):
                continue
            ref_idx.append(int(npz_idx))
            ref_val.append(float(val))

        if len(ref_idx) < int(cfg.cava_logp_min_points_for_interp):
            self._last_cava_logp_interp = None
            return None

        interp = knn_interpolate_values(
            self._embeddings,
            np.array(ref_idx, dtype=np.int64),
            np.array(ref_val, dtype=np.float32),
            temperature=self.cluster_config.dots_temperature,
            top_k=self.cluster_config.dots_top_k,
        )
        interp = fill_by_cluster_mean(interp, self._cluster_ids, default=0.0)
        self._last_cava_logp_interp = interp
        return interp

    def _compute_cava_gate(self, base_utility: np.ndarray) -> np.ndarray:
        cfg = self.cluster_config
        n = len(self._embeddings)
        if not cfg.cava_vdr_enabled:
            return np.ones(n, dtype=np.float32)

        from .vdr_utils import robust_zscore, sigmoid_gate

        channels = []
        weights = []
        channel_names = []

        if self._cava_static_prior is not None and float(cfg.cava_weight_static_prior) != 0.0:
            channels.append(robust_zscore(self._cava_static_prior))
            weights.append(float(cfg.cava_weight_static_prior))
            channel_names.append("static")

        if cfg.cava_use_logp_contrast and float(cfg.cava_weight_logp_contrast) != 0.0:
            logp_values = self._aggregate_and_interpolate_cava_logp()
            if logp_values is not None:
                channels.append(robust_zscore(logp_values))
                weights.append(float(cfg.cava_weight_logp_contrast))
                channel_names.append("logp")

        if not channels:
            gate = np.ones(n, dtype=np.float32)
            self._last_cava_gate = gate
            self._last_cava_score = None
            self._last_cava_metrics = {
                "data_selection/cava_gate_mean_all": 1.0,
                "data_selection/cava_gate_std_all": 0.0,
            }
            return gate

        stack = np.stack(channels, axis=0)
        weight_arr = np.asarray(weights, dtype=np.float32)
        weight_arr /= max(float(weight_arr.sum()), 1e-8)
        score = np.sum(stack * weight_arr[:, None], axis=0).astype(np.float32)
        if cfg.cava_robust_normalize:
            score = robust_zscore(score)

        gate = sigmoid_gate(
            score,
            floor=cfg.cava_gate_floor,
            threshold=cfg.cava_gate_threshold,
            temperature=cfg.cava_gate_temperature,
        )

        finite_score = score[np.isfinite(score)]
        self._last_cava_gate = gate
        self._last_cava_score = score
        self._last_cava_metrics = {
            "data_selection/cava_gate_mean_all": float(gate.mean()),
            "data_selection/cava_gate_std_all": float(gate.std()),
            "data_selection/cava_score_mean_all": float(finite_score.mean()) if finite_score.size else 0.0,
            "data_selection/cava_score_std_all": float(finite_score.std()) if finite_score.size else 0.0,
            "data_selection/cava_channels": float(len(channel_names)),
        }
        return gate

    def _record_cava_selection_metrics(
        self,
        selected_npz: List[int],
        utility_before: np.ndarray,
        utility_after: np.ndarray,
        gate: np.ndarray,
    ) -> None:
        if not selected_npz:
            return
        sel = np.asarray(selected_npz, dtype=np.int64)
        sel = sel[(sel >= 0) & (sel < len(gate))]
        if sel.size == 0:
            return

        before = utility_before[sel].astype(np.float64)
        after = utility_after[sel].astype(np.float64)
        gate_sel = gate[sel].astype(np.float64)
        ratio = after / np.maximum(before, 1e-12)

        metrics = dict(self._last_cava_metrics)
        metrics.update(
            {
                "data_selection/cava_gate_mean_selected": float(gate_sel.mean()),
                "data_selection/cava_gate_min_selected": float(gate_sel.min()),
                "data_selection/cava_gate_max_selected": float(gate_sel.max()),
                "data_selection/cava_utility_before_mean_selected": float(before.mean()),
                "data_selection/cava_utility_after_mean_selected": float(after.mean()),
                "data_selection/cava_utility_ratio_mean_selected": float(ratio.mean()),
            }
        )
        if self._last_cava_score is not None:
            score_sel = self._last_cava_score[sel]
            finite_score = score_sel[np.isfinite(score_sel)]
            if finite_score.size:
                metrics["data_selection/cava_score_mean_selected"] = float(finite_score.mean())
        self._last_cava_metrics = metrics

    @staticmethod
    def _knn_density(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
        """Estimate local density for each point via k-NN cosine distance.

        density_i = 1 / (mean cosine distance to k nearest neighbors + eps)
        Higher value = denser region.  Uses only numpy (no sklearn needed).
        """
        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        normed = (embeddings / norms).astype(np.float32)
        # Cosine similarity matrix  (N, N)
        sim = normed @ normed.T
        # Cosine distance = 1 - cosine_similarity
        cos_dist = 1.0 - sim
        np.fill_diagonal(cos_dist, np.inf)  # exclude self
        # For each point, take k smallest cosine distances
        # argpartition is O(n) per row vs O(n log n) for full sort
        M = len(embeddings)
        k_eff = min(k, M - 1)
        if k_eff < 1:
            return np.ones(M, dtype=np.float32)
        knn_idx = np.argpartition(cos_dist, k_eff, axis=1)[:, :k_eff]
        knn_dists = np.take_along_axis(cos_dist, knn_idx, axis=1)
        mean_dist = knn_dists.mean(axis=1)
        return (1.0 / (mean_dist + 1e-8)).astype(np.float32)

    @staticmethod
    def _density_diverse_select(
        cluster_embeddings: np.ndarray,
        centroid: np.ndarray,
        n_select: int,
        k_density: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        diversity_radius: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select N representatives using density-weighted scoring with greedy
        diversity enforcement.

        Algorithm:
          1. k-NN density estimation per point (cosine metric)
          2. score_i = density_norm^alpha * centroid_sim^beta
          3. Greedy loop: pick top scorer, suppress all points within
             ``diversity_radius`` cosine distance, repeat

        Returns:
            (selected_indices, scores, densities) — arrays of length n_select.
        """
        M = len(cluster_embeddings)
        n_select = min(n_select, M)

        # Helper: cosine similarity between a matrix and a single vector (pure numpy)
        def _cosine_sim_to_vec(X: np.ndarray, v: np.ndarray) -> np.ndarray:
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            x_norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
            return ((X / x_norms) @ v_norm).astype(np.float32)

        if M <= n_select:
            k_eff = min(k_density, M - 1)
            dens = (ClusterSelector._knn_density(cluster_embeddings, k=k_eff)
                    if k_eff >= 1 else np.ones(M))
            return np.arange(M), np.ones(M), dens

        # 1. Density estimation
        k_eff = min(k_density, M - 1)
        if k_eff < 1:
            return (np.arange(min(n_select, M)),
                    np.ones(min(n_select, M)),
                    np.ones(min(n_select, M)))
        density = ClusterSelector._knn_density(cluster_embeddings, k=k_eff)

        # 2. Centroid proximity (cosine similarity, clipped to [0, 1])
        centroid_sim = np.clip(_cosine_sim_to_vec(cluster_embeddings, centroid), 0.0, 1.0)

        # 3. Composite score with normalised density
        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            density_norm = (density - d_min) / (d_max - d_min)
        else:
            density_norm = np.ones(M)
        scores = (density_norm ** alpha) * (centroid_sim ** beta)

        # 4. Greedy diverse selection with suppression
        selected = []
        sel_scores = []
        available = np.ones(M, dtype=bool)

        for _ in range(n_select):
            masked_scores = scores.copy()
            masked_scores[~available] = -1.0

            if masked_scores.max() <= 0:
                remaining = np.where(available)[0]
                if len(remaining) == 0:
                    break
                best = remaining[density[remaining].argmax()]
            else:
                best = int(masked_scores.argmax())

            selected.append(best)
            sel_scores.append(float(scores[best]))
            available[best] = False

            if diversity_radius > 0 and len(selected) < n_select:
                sims_to_best = _cosine_sim_to_vec(
                    cluster_embeddings,
                    cluster_embeddings[best],
                )
                too_close = sims_to_best > (1.0 - diversity_radius)
                available[too_close] = False

        sel_arr = np.array(selected)
        return sel_arr, np.array(sel_scores), density[sel_arr]

    def _select_representatives(self) -> None:
        """Select representative samples (medoids) for each cluster.

        Supported methods (via ``representative_method`` config):
          - ``medoid``: highest mean cosine similarity to all cluster members
          - ``centroid_nearest``: closest L2 distance to centroid
          - ``density_diverse``: density × centroid-proximity scoring with
            greedy diversity suppression (supports n_reps > 1 natively)
        """
        n_clusters = self._centroids.shape[0]
        n_reps = self.cluster_config.n_reps
        method = self.cluster_config.representative_method

        self._rep_indices = []
        self._rep_cluster_ids = []

        for c_id in range(n_clusters):
            mask = self._cluster_ids == c_id
            if not mask.any():
                continue

            cluster_indices = np.where(mask)[0]
            cluster_embs = self._embeddings[mask]

            if method == "density_diverse":
                # Density-aware diverse selection — handles n_reps natively
                sel_local, _, _ = self._density_diverse_select(
                    cluster_embs,
                    self._centroids[c_id],
                    n_select=n_reps,
                    k_density=self.cluster_config.density_diverse_k,
                    alpha=self.cluster_config.density_diverse_alpha,
                    beta=self.cluster_config.density_diverse_beta,
                    diversity_radius=self.cluster_config.density_diverse_radius,
                )
                for li in sel_local:
                    self._rep_indices.append(int(cluster_indices[li]))
                    self._rep_cluster_ids.append(c_id)
                continue

            if method == "centroid_nearest":
                dists = np.linalg.norm(
                    cluster_embs - self._centroids[c_id][np.newaxis, :], axis=1
                )
                sorted_local = np.argsort(dists)
            elif method == "medoid":
                if len(cluster_embs) <= 1:
                    sorted_local = np.array([0])
                else:
                    sims = cluster_embs @ cluster_embs.T
                    avg_sim = sims.mean(axis=1)
                    sorted_local = np.argsort(-avg_sim)
            else:
                dists = np.linalg.norm(
                    cluster_embs - self._centroids[c_id][np.newaxis, :], axis=1
                )
                sorted_local = np.argsort(dists)

            n_take = min(n_reps, len(cluster_indices))
            for i in range(n_take):
                global_idx = int(cluster_indices[sorted_local[i]])
                self._rep_indices.append(global_idx)
                self._rep_cluster_ids.append(c_id)

        print(f"[ClusterSelector] Selected {len(self._rep_indices)} representatives "
              f"across {n_clusters} clusters (method={method}, n_reps={n_reps})")

    def _compute_static_scores(self) -> None:
        """Compute geometry-based scores that don't depend on the policy."""
        print("[ClusterSelector] Computing static cluster scores "
              "(transferability, density)...")

        n_clusters = self._centroids.shape[0]

        norms = np.linalg.norm(self._centroids, axis=1, keepdims=True).clip(min=1e-8)
        normed = self._centroids / norms
        cos_sim = normed @ normed.T

        threshold = self.cluster_config.transferability_sim_threshold
        high_sim_mask = cos_sim > threshold
        cos_sim_masked = cos_sim.copy()
        cos_sim_masked[high_sim_mask] = 0.0
        counts = (~high_sim_mask).sum(axis=1).clip(min=1)
        self._transferability = cos_sim_masked.sum(axis=1) / counts

        gamma = self.cluster_config.density_gamma
        self._density = np.zeros(n_clusters, dtype=np.float64)

        for c_id in range(n_clusters):
            mask = self._cluster_ids == c_id
            if mask.sum() <= 1:
                self._density[c_id] = 1.0
                continue

            c_embs = self._embeddings[mask].astype(np.float64)
            if len(c_embs) > 2000:
                idx = self._rng.choice(len(c_embs), size=2000, replace=False)
                c_embs = c_embs[idx]

            norms_sq = (c_embs ** 2).sum(axis=1)
            pairwise_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * c_embs @ c_embs.T
            pairwise_sq = np.maximum(pairwise_sq, 0.0)
            K = np.exp(-pairwise_sq / (2.0 * gamma ** 2))
            self._density[c_id] = K.mean()

        print(f"[ClusterSelector] Transferability: mean={self._transferability.mean():.4f}, "
              f"Density: mean={self._density[self._density > 0].mean():.4f}")

    def get_reference_indices(self) -> List[int]:
        if self._selection_frozen:
            return []

        # Active probe selection: when the predictor provides uncertainty
        # (e.g. RidgePredictor), pick probe points that maximize information
        # gain via UCB instead of using fixed medoids after round 0.
        if (self._selection_round > 0
                and self.cluster_config.predictor_active_probes
                and not self.cluster_config.reroll_medoids):
            return self._get_active_probe_indices()

        # When reroll_medoids is False, only roll out the fixed medoids on
        # round 0 (cold start).  Subsequent rounds rely on the rollout-history
        # buffer accumulated from training-step rewards.
        if (self._selection_round > 0
                and not self.cluster_config.reroll_medoids):
            return []
        return self._remap_npz_to_dataset(self._rep_indices)

    def _get_active_probe_indices(self) -> List[int]:
        """Select probe points via UCB using the predictor's posterior uncertainty.

        acq(x) = utility(x) + beta * sigma(x)

        Where utility is the current predicted variance (or asymmetric utility)
        and sigma is the predictor's posterior standard deviation.  Points are
        selected greedily with diversity suppression (nearby points within
        cosine distance `predictor_active_probe_suppress_radius` are masked
        after each selection).
        """
        result = self._fit_predict_cached()
        if result is None:
            return self._remap_npz_to_dataset(self._rep_indices)

        if result.uncertainty is None:
            return []

        n_probes = len(self._rep_indices)
        utility = result.predicted_var.copy()
        sigma = result.uncertainty

        beta = self.cluster_config.predictor_ucb_beta
        acq = utility + beta * sigma

        # Exclude already-selected samples from probing
        exclude = self._discovery_exclude_set()
        if exclude:
            ever = np.fromiter(exclude, dtype=np.int64, count=len(exclude))
            ever = ever[(ever >= 0) & (ever < len(acq))]
            acq[ever] = -np.inf

        # Greedy selection with diversity suppression
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (self._embeddings / norms).astype(np.float32)
        suppress_radius = self.cluster_config.predictor_active_probe_suppress_radius

        selected_npz: List[int] = []
        mask = np.ones(len(acq), dtype=bool)

        for _ in range(min(n_probes, int(mask.sum()))):
            candidates = np.where(mask & (acq > -np.inf))[0]
            if len(candidates) == 0:
                break
            best = candidates[np.argmax(acq[candidates])]
            selected_npz.append(int(best))
            mask[best] = False

            # Suppress nearby points
            if suppress_radius > 0:
                sims = all_normed @ all_normed[best]
                too_close = sims > (1.0 - suppress_radius)
                mask[too_close] = False

        print(f"[ClusterSelector] Active probes: selected {len(selected_npz)} "
              f"(UCB beta={beta:.2f}, suppress_r={suppress_radius:.2f})")

        return self._remap_npz_to_dataset(selected_npz)

    def update_rewards(self, ref_indices, ref_rewards) -> None:
        """Compute per-cluster variance from representative rollout rewards.

        Args:
            ref_indices: list of dataset indices (the representatives)
            ref_rewards: (n_ref, n_rollouts) array
        """
        ref_rewards = np.asarray(ref_rewards)
        if ref_rewards.ndim == 1:
            ref_rewards = ref_rewards.reshape(-1, 1)

        cluster_rewards = defaultdict(list)
        cluster_all_rewards = defaultdict(list)

        # Look up cluster membership from the full per-sample assignments
        # array, not a rep-only dict.  Active probes (and any other non-medoid
        # ref selection) must also contribute to cluster variance — otherwise
        # _cluster_variances stays empty and select() short-circuits to random.
        n_cluster_ids = len(self._cluster_ids) if self._cluster_ids is not None else 0

        # ref_indices are parquet positions (from get_reference_indices which
        # already remapped via _npz_to_dataset). Convert back to NPZ positions
        # so we can look up cluster membership and store at the right NPZ slot.
        def to_npz(parquet_i: int) -> int:
            if self._dataset_to_npz is not None:
                return self._dataset_to_npz.get(parquet_i, -1)
            return parquet_i  # assume aligned when no JSON was provided

        self._rep_variances = {}
        self._rep_mean_rewards = {}
        self._rep_reward_rows = {}
        medoid_npz_used: List[int] = []
        for i, parquet_ref_idx in enumerate(ref_indices):
            global_idx = to_npz(parquet_ref_idx)  # NPZ position
            if global_idx < 0:
                continue  # parquet row has no NPZ counterpart (filtered sample)
            if n_cluster_ids == 0 or global_idx >= n_cluster_ids:
                continue
            c_id = int(self._cluster_ids[global_idx])
            mean_r = float(ref_rewards[i].mean())
            var_r = float(ref_rewards[i].var())
            # Normalize variance by mean*(1-mean) so binary and continuous
            # rewards are comparable.  A score of 1.0 = maximum uncertainty.
            if self.cluster_config.normalize_variance:
                denom = max(mean_r * (1.0 - mean_r), 1e-8)
                var_r = min(var_r / denom, 1.0)
            self._rep_variances[global_idx] = var_r  # individual variance per rep
            self._rep_mean_rewards[global_idx] = mean_r  # individual mean reward per rep
            self._rep_reward_rows[global_idx] = ref_rewards[i].copy()
            cluster_rewards[c_id].append(var_r)
            cluster_all_rewards[c_id].append(mean_r)
            medoid_npz_used.append(int(global_idx))

            # Seed the rollout buffer with REPR rollouts so round 0 already has
            # reference points. Subsequent REPR rounds update with fresh entries.
            if self.cluster_config.use_rollout_history:
                if global_idx not in self._rollout_buffer:
                    self._rollout_buffer[global_idx] = []
                self._rollout_buffer[global_idx].append(
                    (self._current_training_step, ref_rewards[i].copy())
                )

        # Track medoid rollouts separately from training/exploration budget.
        # They count only when count_medoids_in_budget=True.
        if medoid_npz_used:
            new_medoid = set(medoid_npz_used) - self._ever_selected_medoid
            self._ever_selected_medoid |= new_medoid
            self._ever_selected_set |= set(medoid_npz_used)

        self._cluster_variances = {}
        self._cluster_mean_rewards = {}
        for c_id, vars_list in cluster_rewards.items():
            self._cluster_variances[c_id] = float(np.mean(vars_list))
            self._cluster_mean_rewards[c_id] = float(np.mean(cluster_all_rewards[c_id]))

        n_zero_var = sum(1 for v in self._cluster_variances.values() if v == 0)
        n_high_var = sum(1 for v in self._cluster_variances.values() if v > 0.1)
        print(f"[ClusterSelector] Cluster variances updated: "
              f"{len(self._cluster_variances)} clusters, "
              f"{n_zero_var} zero-variance, {n_high_var} high-variance (>0.1)")

        top_n = 15
        sorted_by_var = sorted(self._cluster_variances.items(), key=lambda x: -x[1])
        header = f"  {'CID':>5} {'Size':>6} {'MeanRew':>9} {'Variance':>9}"
        rows = []
        for c_id, var in sorted_by_var[:top_n]:
            size = int((self._cluster_ids == c_id).sum())
            mean_r = self._cluster_mean_rewards.get(c_id, 0.0)
            rows.append(f"  {c_id:>5} {size:>6} {mean_r:>9.4f} {var:>9.4f}")
        print(f"[ClusterSelector] Top-{top_n} clusters by variance:\n{header}\n" + "\n".join(rows))

    def update_rollout_history(
        self,
        step: int,
        uids: np.ndarray,
        rewards_per_rollout: np.ndarray,
        dataset_indices: np.ndarray = None,
    ) -> None:
        """Record training-batch rollout rewards into the time-weighted buffer.

        Called from ray_trainer after each training step. The batch contains
        n_rollouts copies of each original sample (interleaved), so UIDs repeat.
        We group by original sample, compute variance across rollouts, and append
        to the buffer with the current training step as the timestamp.

        Entries older than rollout_history_max_age steps are pruned on the fly.

        Args:
            step: current global training step (used as timestamp).
            uids: (batch_size * n_rollouts,) session UID strings per rollout entry.
            rewards_per_rollout: (batch_size * n_rollouts,) summed reward per
                rollout (i.e. reward_tensor.sum(dim=-1)).
            dataset_indices: (batch_size * n_rollouts,) integer full-dataset
                positions for each rollout entry, passed through from
                RLHFDataset.__getitem__ via batch.non_tensor_batch["dataset_idx"].
                When provided, used directly for buffer insertion (bypasses the
                string-UID lookup that was previously always failing because
                training batches use uuid4 session IDs, not dataset-derived UIDs).
        """
        if not self.cluster_config.use_rollout_history:
            return

        self._current_training_step = step
        max_age = self.cluster_config.rollout_history_max_age
        n_new = 0

        if dataset_indices is not None and self._dataset_to_npz is not None:
            # Fast path: convert parquet dataset positions → NPZ positions using
            # the alignment map built from the JSON source file during initialize().
            # dataset_indices is (batch_size * n_rollouts,) with each original
            # index repeated n_rollouts consecutive times (interleave=True).
            idx_to_rewards: Dict[int, List[float]] = defaultdict(list)
            for parquet_idx, r in zip(dataset_indices, rewards_per_rollout):
                npz_idx = self._dataset_to_npz.get(int(parquet_idx), -1)
                if npz_idx >= 0:
                    idx_to_rewards[npz_idx].append(float(r))

            for npz_idx, rewards in idx_to_rewards.items():
                if npz_idx not in self._rollout_buffer:
                    self._rollout_buffer[npz_idx] = []
                self._rollout_buffer[npz_idx].append(
                    (step, np.array(rewards, dtype=np.float32))
                )
                self._rollout_buffer[npz_idx] = [
                    (t, r) for t, r in self._rollout_buffer[npz_idx]
                    if step - t <= max_age
                ]
                n_new += 1
        else:
            # Fallback: string-UID lookup.  Works only when _uid_to_dataset_idx
            # is keyed on the same UID scheme as the incoming batch UIDs.
            if not self._uid_to_dataset_idx:
                if not getattr(self, "_warned_no_uid_map", False):
                    print("[ClusterSelector] WARNING: use_rollout_history=True but "
                          "no alignment map (_dataset_to_npz) and _uid_to_dataset_idx "
                          "is empty — training-batch rollouts will NOT be accumulated. "
                          "Set cluster.dataset_json_file to enable alignment.")
                    self._warned_no_uid_map = True
                return

            uid_to_rewards: Dict[str, List[float]] = defaultdict(list)
            for uid, r in zip(uids, rewards_per_rollout):
                uid_to_rewards[str(uid)].append(float(r))

            for uid, rewards in uid_to_rewards.items():
                dataset_idx = self._uid_to_dataset_idx.get(uid)
                if dataset_idx is None:
                    continue
                if dataset_idx not in self._rollout_buffer:
                    self._rollout_buffer[dataset_idx] = []
                self._rollout_buffer[dataset_idx].append(
                    (step, np.array(rewards, dtype=np.float32))
                )
                self._rollout_buffer[dataset_idx] = [
                    (t, r) for t, r in self._rollout_buffer[dataset_idx]
                    if step - t <= max_age
                ]
                n_new += 1

        # Diagnostic: log match rate and buffer state on every step.
        # After alignment is working, n_new should equal batch_size at each step
        # and buffer_unique should grow toward rollout_history_max_refs.
        n_unique_in_batch = (len(np.unique(dataset_indices)) if dataset_indices is not None
                             else len(np.unique(uids)))
        if n_new < n_unique_in_batch or step % 10 == 0:
            print(f"[ClusterSelector] update_rollout_history step={step}: "
                  f"matched {n_new}/{n_unique_in_batch} samples, "
                  f"buffer_unique={len(self._rollout_buffer)}")

    def _compute_time_weighted_variances(self, current_step: int) -> Dict[int, float]:
        """Compute time-decayed effective variance for all samples in the buffer.

        For each sample with history [(t1, r1), (t2, r2), ...]:
            w_i = exp(-decay_rate * (current_step - t_i))
            effective_var = Σ(w_i * var(r_i)) / Σ(w_i)

        Returns a dict of {dataset_idx: effective_variance} with at most
        rollout_history_max_refs entries (keeping the most recently observed
        samples to maximise relevance to the current policy).
        """
        decay = self.cluster_config.rollout_history_decay_rate
        max_refs = self.cluster_config.rollout_history_max_refs

        results: Dict[int, float] = {}
        # Track the most recent step seen per sample for sorting.
        recency: Dict[int, int] = {}

        for dataset_idx, history in self._rollout_buffer.items():
            if not history:
                continue
            total_w = 0.0
            weighted_var = 0.0
            latest_step = 0
            for t, rewards in history:
                w = float(np.exp(-decay * max(0, current_step - t)))
                v = float(np.var(rewards)) if len(rewards) > 1 else 0.0
                # Normalize variance by mean*(1-mean) for cross-task comparability
                if self.cluster_config.normalize_variance and len(rewards) > 1:
                    m = float(np.mean(rewards))
                    denom = max(m * (1.0 - m), 1e-8)
                    v = min(v / denom, 1.0)
                weighted_var += w * v
                total_w += w
                latest_step = max(latest_step, t)
            if total_w > 1e-12:
                results[dataset_idx] = weighted_var / total_w
                recency[dataset_idx] = latest_step

        # Cap to max_refs by keeping the most recently observed samples.
        # max_refs <= 0 disables the cap entirely (use the full age-pruned buffer).
        if max_refs > 0 and len(results) > max_refs:
            top_idxs = sorted(recency, key=lambda i: recency[i], reverse=True)[:max_refs]
            results = {i: results[i] for i in top_idxs}

        return results

    def _compute_time_weighted_mean_rewards(self, current_step: int) -> Dict[int, float]:
        """Compute time-decayed effective mean reward for all samples in the buffer.

        Parallel to _compute_time_weighted_variances but averages raw rewards
        instead of computing per-observation variance.  Used by the asymmetric
        utility path in _select_interpolated so that DOTS can predict a
        continuous "capability" signal (p ∈ [0, 1]) for every sample alongside
        predicted variance.  Returned dict is keyed on the same NPZ indices and
        capped to the same rollout_history_max_refs most-recent samples so it
        aligns exactly with _compute_time_weighted_variances.
        """
        decay = self.cluster_config.rollout_history_decay_rate
        max_refs = self.cluster_config.rollout_history_max_refs

        results: Dict[int, float] = {}
        recency: Dict[int, int] = {}

        for dataset_idx, history in self._rollout_buffer.items():
            if not history:
                continue
            total_w = 0.0
            weighted_mean = 0.0
            latest_step = 0
            for t, rewards in history:
                w = float(np.exp(-decay * max(0, current_step - t)))
                m = float(np.mean(rewards))
                weighted_mean += w * m
                total_w += w
                latest_step = max(latest_step, t)
            if total_w > 1e-12:
                results[dataset_idx] = weighted_mean / total_w
                recency[dataset_idx] = latest_step

        if max_refs > 0 and len(results) > max_refs:
            top_idxs = sorted(recency, key=lambda i: recency[i], reverse=True)[:max_refs]
            results = {i: results[i] for i in top_idxs}

        return results

    def _get_observed_mean_array(self, size: int) -> Optional[np.ndarray]:
        """Per-NPZ-index time-weighted observed mean reward, shape (size,).

        Used by the observed dead-zone to kill saturated/text-leaky samples using
        actual rollout outcomes rather than the KNN-predicted mean (which has
        loo R² < 0 and therefore cannot reliably identify the high-mean tail).

        Returns None when the rollout buffer is empty (e.g. round 0 before any
        training rollouts are recorded).  Indices not present in the buffer
        receive 0.5 (neutral — they are neither killed nor boosted).
        """
        if not self._rollout_buffer:
            return None
        obs = self._compute_time_weighted_mean_rewards(self._current_training_step)
        if not obs:
            return None
        arr = np.full(size, 0.5, dtype=np.float32)
        for npz_idx, mean_r in obs.items():
            if 0 <= npz_idx < size:
                arr[npz_idx] = float(mean_r)
        return arr

    # ------------------------------------------------------------------
    # Observation rows for predictor
    # ------------------------------------------------------------------

    def _build_observation_rows(self) -> List[Tuple[int, int, np.ndarray]]:
        """Extract raw observation rows from the rollout buffer.

        Returns a list of (npz_idx, step, rewards_array) tuples — one row per
        observation event, NOT pre-aggregated.  The predictor decides internally
        how to weight by time and n_rollouts inside its fit() method.

        Also includes raw REPR medoid rollout rows when the buffer is empty
        (cold-start fallback).
        """
        rows: List[Tuple[int, int, np.ndarray]] = []

        if self._rollout_buffer:
            for npz_idx, history in self._rollout_buffer.items():
                for t, rewards in history:
                    rows.append((int(npz_idx), int(t), rewards))

        # Cold-start: if buffer is empty but we have REPR medoid rollouts, use
        # those exact reward rows. Do not fabricate binary rows from the mean;
        # doing so changes both the variance scale and the empirical mean.
        if not rows and self._rep_reward_rows:
            for idx, c_id in zip(self._rep_indices, self._rep_cluster_ids):
                rewards = self._rep_reward_rows.get(idx)
                if rewards is not None:
                    rows.append((int(idx), 0, np.asarray(rewards, dtype=np.float32)))

        return rows

    def _discovery_exclude_set(self) -> set:
        """Samples to exclude from discovery-mode selection.

        Policy follows ``count_medoids_in_budget``:
          - True  → everything ever touched (train ∪ exploration ∪ medoid),
                    i.e. ``_ever_selected_set``.
          - False → only train ∪ exploration; medoid-only samples stay
                    eligible so they can still win naturally by predicted
                    variance in later rounds.
        """
        if self.cluster_config.count_medoids_in_budget:
            return self._ever_selected_set
        return self._ever_selected_train | self._ever_selected_exploration

    def _fit_predict_cached(self) -> Optional[PredictionResult]:
        """Fit the predictor on the current observations and return predictions.

        Results are cached for the duration of the current selection round,
        so multiple call sites (active probe selection, _select_interpolated,
        _select_frozen_reweight) share a single fit+predict pass. The cache
        key is ``self._selection_round``; it is reset implicitly when that
        counter advances.
        """
        if (self._predictor_cache_round == self._selection_round
                and self._predictor_cache_result is not None):
            return self._predictor_cache_result

        obs = self._build_observation_rows()
        if not obs:
            self._predictor_cache_round = self._selection_round
            self._predictor_cache_result = None
            return None

        self._predictor.fit(
            obs, self._embeddings, self._current_training_step,
            normalize_variance=self.cluster_config.normalize_variance,
        )
        # For the random_score ablation: advance the per-round seed so scores
        # differ across rounds but are stable within a round. No-op for
        # predictors that don't define set_round().
        if hasattr(self._predictor, "set_round"):
            self._predictor.set_round(self._selection_round)
        result = self._predictor.predict(self._embeddings)

        self._last_predicted_var = result.predicted_var.copy()
        if result.predicted_mean is not None:
            self._last_predicted_mean = result.predicted_mean.copy()

        self._predictor_cache_round = self._selection_round
        self._predictor_cache_result = result

        self._run_predictor_diagnostics(result)
        return result

    def _run_predictor_diagnostics(self, result: PredictionResult) -> None:
        """Refresh self._last_predictor_diagnostics from the current predictor."""
        ref_indices = getattr(self._predictor, "_ref_indices", None)
        ref_observed_var = getattr(self._predictor, "_ref_observed_var", None)
        ref_observed_mean = getattr(self._predictor, "_ref_observed_mean", None)
        if ref_indices is None or ref_observed_var is None:
            return

        self._last_predictor_diagnostics = compute_predictor_diagnostics(
            embeddings=self._embeddings,
            ref_indices=ref_indices,
            ref_observed_var=ref_observed_var,
            ref_observed_mean=ref_observed_mean,
            predicted_var_all=result.predicted_var,
            predicted_mean_all=result.predicted_mean,
            dots_temperature=self.cluster_config.dots_temperature,
            dots_top_k=self.cluster_config.dots_top_k,
            cluster_ids=self._cluster_ids,
            variance_normalized=self.cluster_config.normalize_variance,
        )
        d = self._last_predictor_diagnostics
        print(f"[ClusterSelector] Predictor diagnostics: "
              f"type={self.cluster_config.predictor_type}, "
              f"n_refs={len(ref_indices)}, "
              f"train_R²={d.get('data_selection/predictor_train_r2', 0):.4f}, "
              f"train_R²_vs_pmean={d.get('data_selection/predictor_train_r2_vs_pmean_var', 0):.4f}, "
              f"LOO_R²={d.get('data_selection/loo_knn_r2', 0):.4f}, "
              f"LOO_ρ="
              + ("undef" if d.get('data_selection/loo_knn_spearman_undefined', 0) > 0.5
                 else f"{d.get('data_selection/loo_knn_spearman', 0):.4f}")
              + f", cluster_LOO_R²={d.get('data_selection/loo_cluster_r2', float('nan')):.4f}"
              + f", cluster_LOO_ρ={d.get('data_selection/loo_cluster_spearman', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # Exploration rollouts
    # ------------------------------------------------------------------

    def get_exploration_indices(self) -> List[int]:
        """Return random un-selected sample indices for exploration rollouts.

        Called by the trainer alongside get_reference_indices() when
        exploration is enabled and this is an exploration round.  The
        returned indices are rolled out and fed back via
        update_exploration_rewards().

        Returns parquet-order indices (like get_reference_indices).
        """
        if self._selection_frozen:
            return []
        if not self.cluster_config.exploration_enabled:
            return []
        if self._selection_round % self.cluster_config.exploration_interval != 0:
            return []

        budget = self.get_selection_budget_for_exploration()
        if budget <= 0:
            return []

        # Pick samples NOT in the current selection to break the feedback loop
        all_indices = set(range(len(self._embeddings)))
        selected_set = self._prev_selected_set if self._prev_selected_set else set()
        candidates = list(all_indices - selected_set)

        if len(candidates) == 0:
            candidates = list(all_indices)

        n_explore = min(budget, len(candidates))
        explore_npz = self._rng.choice(candidates, size=n_explore, replace=False).tolist()
        self._exploration_indices = explore_npz

        parquet_indices = self._remap_npz_to_dataset(explore_npz)
        print(f"[ClusterSelector] Exploration: {len(parquet_indices)} random "
              f"un-selected samples for rollout (round={self._selection_round})")
        return parquet_indices

    def get_selection_budget_for_exploration(self) -> int:
        """Compute number of exploration samples from config."""
        pct = self.cluster_config.exploration_pct
        base = str(getattr(self.cluster_config, "exploration_pct_base", "dataset")).lower()

        if base in ("representatives", "reps", "reference", "reference_set"):
            n = len(self._rep_indices) if self._rep_indices else 0
        else:
            # Backward-compatible default: percentage of full dataset.
            n = len(self._embeddings)

        if n <= 0 or pct <= 0:
            return 0

        return max(1, int(n * pct / 100.0))

    def update_exploration_rewards(
        self, explore_indices: List[int], explore_rewards: np.ndarray
    ) -> None:
        """Feed exploration rollout results into the history buffer.

        Args:
            explore_indices: parquet-order indices that were rolled out
            explore_rewards: (n_explore, n_rollouts) reward array
        """
        if not self.cluster_config.exploration_enabled:
            return
        if not self.cluster_config.use_rollout_history:
            return

        explore_rewards = np.asarray(explore_rewards)
        if explore_rewards.ndim == 1:
            explore_rewards = explore_rewards.reshape(-1, 1)

        n_added = 0
        explore_npz_used: List[int] = []
        for i, parquet_idx in enumerate(explore_indices):
            npz_idx = (self._dataset_to_npz.get(int(parquet_idx), -1)
                       if self._dataset_to_npz is not None else parquet_idx)
            if npz_idx < 0:
                continue
            if npz_idx not in self._rollout_buffer:
                self._rollout_buffer[npz_idx] = []
            self._rollout_buffer[npz_idx].append(
                (self._current_training_step, explore_rewards[i].copy())
            )
            n_added += 1
            explore_npz_used.append(int(npz_idx))

        # Account exploration rollouts against the global annotation budget.
        if explore_npz_used:
            new_explore = set(explore_npz_used) - self._ever_selected_exploration
            self._ever_selected_exploration |= new_explore
            self._ever_selected_set |= set(explore_npz_used)

        print(f"[ClusterSelector] Exploration: added {n_added} samples to history buffer "
              f"({len(explore_npz_used)} new uniques counted against global budget)")

    # ------------------------------------------------------------------
    # Image Grounding Score (IGS)
    # ------------------------------------------------------------------

    def update_igs_from_rollouts(
        self,
        indices: List[int],
        rewards_with_image: np.ndarray,
        rewards_without_image: np.ndarray,
    ) -> None:
        """Compute per-sample IGS from paired rollouts (with/without image).

        IGS = var(rewards_with_image) / max(var(rewards_without_image), epsilon)

        High IGS → image is essential (high multimodal dependency).
        Low IGS → text-only shortcut exists.

        Args:
            indices: parquet-order sample indices
            rewards_with_image: (n, n_rollouts) rewards from normal rollouts
            rewards_without_image: (n, n_rollouts) rewards with image removed/blanked
        """
        rewards_with = np.asarray(rewards_with_image)
        rewards_without = np.asarray(rewards_without_image)
        if rewards_with.ndim == 1:
            rewards_with = rewards_with.reshape(-1, 1)
        if rewards_without.ndim == 1:
            rewards_without = rewards_without.reshape(-1, 1)

        if self._igs_scores is None:
            self._igs_scores = np.ones(len(self._embeddings), dtype=np.float32)

        for i, parquet_idx in enumerate(indices):
            npz_idx = (self._dataset_to_npz.get(int(parquet_idx), -1)
                       if self._dataset_to_npz is not None else parquet_idx)
            if npz_idx < 0 or npz_idx >= len(self._igs_scores):
                continue
            var_with = float(np.var(rewards_with[i]))
            var_without = float(np.var(rewards_without[i]))
            # IGS: ratio of variance with image to variance without
            self._igs_scores[npz_idx] = var_with / max(var_without, 1e-6)

        # Aggregate per-cluster IGS
        self._cluster_igs = {}
        for c_id in range(self._centroids.shape[0]):
            mask = self._cluster_ids == c_id
            if not mask.any():
                continue
            cluster_igs = self._igs_scores[mask]
            self._cluster_igs[c_id] = float(cluster_igs.mean())

        n_multimodal = sum(1 for v in self._cluster_igs.values() if v > 1.5)
        print(f"[ClusterSelector] IGS updated: {len(self._cluster_igs)} clusters, "
              f"{n_multimodal} strongly multimodal (IGS > 1.5)")

    def load_igs_scores(self, path: str) -> None:
        """Load pre-computed per-sample IGS scores from a numpy file.

        Expected format: .npz with key 'igs_scores' of shape (N,).
        """
        data = np.load(path)
        self._igs_scores = data["igs_scores"].astype(np.float32)
        if len(self._igs_scores) != len(self._embeddings):
            print(f"[ClusterSelector] WARNING: IGS scores length ({len(self._igs_scores)}) "
                  f"!= embeddings length ({len(self._embeddings)}). Truncating.")
            min_len = min(len(self._igs_scores), len(self._embeddings))
            self._igs_scores = self._igs_scores[:min_len]

        # Aggregate per-cluster
        self._cluster_igs = {}
        for c_id in range(self._centroids.shape[0]):
            mask = self._cluster_ids == c_id
            if not mask.any():
                continue
            self._cluster_igs[c_id] = float(self._igs_scores[mask].mean())

        print(f"[ClusterSelector] Loaded IGS scores from {path}: "
              f"{len(self._igs_scores)} samples")

    def select(self, budget: int) -> List[int]:
        self._selection_round += 1

        was_frozen = self._selection_frozen
        if was_frozen:
            strategy = "frozen_reweight"
            npz_indices = self._select_frozen_reweight(budget)
        else:
            strategy = self.cluster_config.strategy

            # Cold-start fallback: only resort to random when neither the
            # cluster-level signal NOR the sample-level rollout buffer has any
            # data yet.  The `interpolated` strategy reads from the rollout
            # buffer via _fit_predict_cached, so an empty _cluster_variances
            # alone is NOT a reason to go random — that used to silently
            # degrade runs where active probes populated the buffer but not
            # the rep-keyed variance dict.
            has_cluster_signal = bool(self._cluster_variances)
            has_sample_signal = bool(self._rollout_buffer) or bool(self._rep_reward_rows)
            if not has_cluster_signal and not (strategy == "interpolated" and has_sample_signal):
                print("[ClusterSelector] No variance data yet, selecting random")
                return self._rng.choice(
                    self._dataset_size, size=min(budget, self._dataset_size), replace=False
                ).tolist()

            if strategy == "top_clusters":
                npz_indices = self._select_top_clusters(budget)
            elif strategy == "weighted":
                npz_indices = self._select_weighted(budget)
            elif strategy == "scored":
                npz_indices = self._select_scored(budget)
            elif strategy == "interpolated":
                npz_indices = self._select_interpolated(budget)
            else:
                raise ValueError(f"Unknown cluster selection strategy: {strategy}")

        # --- Selection overlap tracking ---
        current_set = set(npz_indices)
        if self._prev_selected_set:
            intersection = current_set & self._prev_selected_set
            union = current_set | self._prev_selected_set
            self._selection_jaccard = len(intersection) / max(len(union), 1)
        else:
            self._selection_jaccard = 0.0

        # --- Jaccard between current selection and a matched-size random
        # draw from the SAME eligible pool (discovery: unseen; frozen
        # reweight: full pool). Near 0 → method picks systematically
        # differently from random. Near (|S|/|pool|) → indistinguishable
        # from random. Used by the 2026-04-21 supervisor experiment to
        # test the diversity-vs-difficulty reframing of the KNN predictor.
        self._selection_jaccard_vs_random = 0.0
        if len(current_set) > 0:
            n_embeddings = len(self._embeddings)
            if was_frozen:
                pool = self._frozen_pool_npz or sorted(self._budget_counted_set())
                eligible = np.asarray(pool, dtype=np.int64)
            elif (getattr(self.cluster_config, "exclude_already_selected", True)
                    and self._discovery_exclude_set()):
                exclude = self._discovery_exclude_set()
                eligible = np.setdiff1d(
                    np.arange(n_embeddings, dtype=np.int64),
                    np.fromiter(exclude, dtype=np.int64, count=len(exclude)),
                    assume_unique=True,
                )
            else:
                eligible = np.arange(n_embeddings, dtype=np.int64)
            if len(eligible) >= len(current_set):
                rng = np.random.default_rng(
                    (0x85EBCA77 ^ int(self._selection_round)) & 0xFFFFFFFF
                )
                rand_pick = set(
                    rng.choice(eligible, size=len(current_set), replace=False).tolist()
                )
                inter_r = len(current_set & rand_pick)
                union_r = len(current_set | rand_pick)
                self._selection_jaccard_vs_random = inter_r / max(union_r, 1)

        self._prev_selected_set = current_set
        self._selection_history.append(current_set)
        # Account training samples against the global annotation budget.
        new_train = current_set - self._ever_selected_train
        self._ever_selected_train |= new_train
        self._ever_selected_set |= current_set

        # Track per-cluster selection frequency
        for idx in npz_indices:
            c_id = int(self._cluster_ids[idx])
            self._per_cluster_coverage[c_id] = self._per_cluster_coverage.get(c_id, 0) + 1

        n_embeddings = len(self._embeddings)
        coverage_pct = 100.0 * len(self._ever_selected_set) / max(n_embeddings, 1)

        print(f"[ClusterSelector] Overlap: jaccard={self._selection_jaccard:.3f}, "
              f"cumulative_coverage={coverage_pct:.1f}% ({len(self._ever_selected_set)}/{n_embeddings})")

        # Remap NPZ positions → parquet positions so the trainer's
        # Subset(train_dataset, indices) accesses the correct rows.
        indices = self._remap_npz_to_dataset(npz_indices)

        # --- Budget schedule phase logging ---
        if self.cluster_config.budget_schedule:
            per_round_pct, interval = self._get_current_phase()
            print(f"[ClusterSelector] Budget schedule: active phase "
                  f"per_round_pct={per_round_pct}, interval={interval}")

        # --- Global budget enforcement ---
        # When global_budget_pct is set, freeze selection once cumulative unique
        # samples reach the cap.  By default medoid probes are excluded from
        # the count (they're measurement cost, not training data).
        if self.config.global_budget_pct is not None and not was_frozen:
            global_max = max(1, int(n_embeddings * self.config.global_budget_pct / 100.0))
            n_budget_used = len(self._budget_counted_set())
            if n_budget_used >= global_max:
                self._selection_frozen = True
                self._frozen_pool_npz = sorted(self._budget_counted_set())
                print(f"[ClusterSelector] Global budget cap reached: "
                      f"{n_budget_used} budget samples >= {global_max} "
                      f"({self.config.global_budget_pct}% of {n_embeddings}, "
                      f"medoids_in_budget={self.cluster_config.count_medoids_in_budget}). "
                      f"Pool frozen — subsequent rounds will reweight within "
                      f"this pool using training reward variance.")

        # --- Pad to minimum training-pool size ---
        # The per-round discovery budget only counts NEW unique samples the
        # selector wants rolled out; the dataloader, however, needs enough
        # rows to form at least one full batch (drop_last=True).  When the
        # budget schedule tapers per_round_pct below train_batch_size, we
        # backfill with samples already counted in the training/exploration
        # budget. These are free — they already have rollout history and have
        # already been counted against the global annotation budget — so
        # padding keeps training alive without inflating the annotation cost.
        floor = int(self._min_training_pool_size)
        if floor > 0 and len(indices) < floor:
            selected_set = set(indices)
            # Previously annotated parquet indices, minus what we already have.
            pool_npz = self._budget_counted_set() - set(npz_indices)
            pool_parquet: List[int] = []
            if pool_npz:
                remapped = self._remap_npz_to_dataset(list(pool_npz))
                pool_parquet = [p for p in remapped if p not in selected_set]

            need = floor - len(indices)
            if len(pool_parquet) >= need:
                pad = self._rng.choice(pool_parquet, size=need, replace=False).tolist()
                source = "pool"
            elif pool_parquet:
                pad = list(pool_parquet)
                source = "pool(exhausted)"
            else:
                pad = []
                source = "none"

            # If the already-annotated pool is still too small (very early
            # rounds), fall back to random draws from the full parquet
            # dataset so the trainer always gets a batch.  These extras are
            # NOT added to _ever_selected_* — they're padding, not new
            # annotations.
            remaining = floor - (len(indices) + len(pad))
            if remaining > 0:
                full = set(range(self._dataset_size)) - selected_set - set(pad)
                if full:
                    extra_n = min(remaining, len(full))
                    extra = self._rng.choice(
                        list(full), size=extra_n, replace=False
                    ).tolist()
                    pad.extend(extra)
                    source = source + "+random"

            if pad:
                print(f"[ClusterSelector] Padding training pool "
                      f"{len(indices)} -> {len(indices) + len(pad)} "
                      f"(floor={floor}, source={source})")
                indices = list(indices) + list(pad)

        self._last_selected_indices = indices
        self._budget_history.append((
            self._selection_round,
            len(self._ever_selected_train),
            len(self._ever_selected_medoid),
            len(self._ever_selected_exploration),
        ))
        print(f"[ClusterSelector] Selected {len(indices)} samples "
              f"(strategy={strategy}, budget={budget})")
        return indices

    def _select_frozen_reweight(self, budget: int) -> List[int]:
        """DOTS-interpolated variance-weighted sampling within the frozen pool.

        After the global budget cap freezes the pool composition, this method
        replaces the normal strategy-based selection.  It reuses the same DOTS
        interpolation machinery as _select_interpolated but restricts output
        to the frozen pool:

        1. Build reference set from rollout history buffer (training-batch
           rewards accumulated every step + decayed medoid rollouts from round 0)
        2. DOTS interpolation: predict variance for ALL embeddings using
           cosine-similarity-weighted averaging of reference variances
        3. Extract predicted variances for pool samples only
        4. Sample pool indices with replacement, weighted by predicted variance

        This means samples the model hasn't trained on recently still get good
        variance predictions (from similar samples that WERE recently observed),
        instead of falling back to a blind floor weight.

        Falls back to uniform sampling when no variance data is available
        (e.g. use_rollout_history=False or first round after freeze).
        """
        from collections import Counter

        pool = self._frozen_pool_npz
        if pool is None or len(pool) == 0:
            pool = sorted(self._budget_counted_set()) or sorted(self._ever_selected_set)
        n_pool = len(pool)

        if n_pool == 0:
            print("[ClusterSelector] Frozen reweight: empty pool, selecting random")
            n_embeddings = len(self._embeddings) if self._embeddings is not None else self._dataset_size
            return self._rng.choice(
                n_embeddings, size=min(budget, n_embeddings), replace=False
            ).tolist()

        asym = self.cluster_config.asymmetric_utility_enabled

        # Fit predictor from rollout history, same machinery as _select_interpolated.
        # _fit_predict_cached also refreshes _last_predictor_diagnostics so the
        # frozen-pool phase (which covers ~80% of training) keeps logging R².
        result = self._fit_predict_cached()
        if result is None:
            chosen_npz = list(pool) if n_pool <= budget else \
                self._rng.choice(pool, size=budget, replace=False).tolist()
            print(f"[ClusterSelector] Frozen reweight round {self._selection_round}: "
                  f"no variance data yet, uniform {len(chosen_npz)} samples")
            return chosen_npz

        pool_arr = np.array(pool, dtype=np.int64)
        variances = result.predicted_var[pool_arr]

        if variances.sum() == 0:
            chosen_npz = list(pool) if n_pool <= budget else \
                self._rng.choice(pool, size=budget, replace=False).tolist()
            print(f"[ClusterSelector] Frozen reweight round {self._selection_round}: "
                  f"all predicted variances zero, uniform {len(chosen_npz)} samples")
            return chosen_npz

        dead_mask = np.zeros(n_pool, dtype=bool)

        if asym and result.predicted_mean is not None:
            predicted_mean = np.clip(result.predicted_mean[pool_arr], 0.0, 1.0)
            alpha = float(self.cluster_config.hard_side_bias)
            utility = variances * (1.0 + alpha * (0.5 - predicted_mean))
            low = float(self.cluster_config.asymmetric_dead_zone_low)
            high = float(self.cluster_config.asymmetric_dead_zone_high)
            dead_mask |= predicted_mean < low
            dead_mask |= predicted_mean > high
            utility[dead_mask] = 0.0
            utility = np.maximum(utility, 0.0).astype(np.float32)
            score = utility
        else:
            score = variances.copy()

        # Observed dead-zone: kill saturated/text-leaky pool entries by observed R_with.
        # Applied after the asymmetric utility score so it acts as a final filter.
        obs_high = float(self.cluster_config.observed_deadzone_high)
        if obs_high < 1.0:
            obs_means_full = self._get_observed_mean_array(
                int(pool_arr.max()) + 1 if len(pool_arr) else 0
            )
            if obs_means_full is not None:
                obs_means_pool = obs_means_full[pool_arr]
                n_before = int((score > 0).sum())
                score = score.copy()
                obs_dead = obs_means_pool > obs_high
                dead_mask |= obs_dead
                score[obs_dead] = 0.0
                n_obs_killed = n_before - int((score > 0).sum())
                print(
                    f"[ClusterSelector] frozen reweight observed dead-zone: "
                    f"high={obs_high:.2f}, killed={n_obs_killed}/{len(pool_arr)} "
                    f"pool entries with observed R_with > {obs_high:.2f}"
                )

        # Floor: 5% of max eligible score. Prevents starvation among eligible
        # samples while preserving hard dead-zone exclusions.
        eligible = ~dead_mask
        if not eligible.any():
            eligible = np.ones(n_pool, dtype=bool)
            score = variances.copy()
        max_score = float(score[eligible].max()) if eligible.any() else 0.0
        if max_score <= 0.0:
            weights = eligible.astype(np.float64)
        else:
            floor = max(max_score * 0.05, 1e-8)
            weights = np.where(eligible, np.maximum(score, floor), 0.0).astype(np.float64)
        weights /= weights.sum()

        # Sample with replacement — high-variance samples appear multiple times.
        chosen_positions = self._rng.choice(n_pool, size=budget, replace=True, p=weights)
        chosen_npz = [pool[pos] for pos in chosen_positions]

        n_unique = len(set(chosen_npz))
        counts = Counter(chosen_npz)
        max_reps = max(counts.values())
        n_zero_pred = int(np.sum(variances < 1e-8))
        print(f"[ClusterSelector] Frozen reweight round {self._selection_round}: "
              f"{n_unique} unique/{budget} total, max_reps={max_reps}, "
              f"predicted_var=[{variances.min():.4f}, {variances.max():.4f}], "
              f"zero_pred={n_zero_pred}/{n_pool}")
        return chosen_npz

    def _select_top_clusters(self, budget: int) -> List[int]:
        """Greedily include all samples from highest-variance clusters."""
        sorted_clusters = sorted(
            self._cluster_variances.items(), key=lambda x: x[1], reverse=True
        )

        selected = []
        for c_id, var in sorted_clusters:
            if len(selected) >= budget:
                break
            cluster_indices = np.where(self._cluster_ids == c_id)[0]
            remaining = budget - len(selected)
            take = self._within_cluster_select(
                cluster_indices, min(len(cluster_indices), remaining), c_id
            )
            selected.extend(take)
        return selected

    def _select_weighted(self, budget: int) -> List[int]:
        """Allocate budget proportionally to cluster variance."""
        active = {c: v for c, v in self._cluster_variances.items() if v > 0}
        if not active:
            return self._rng.choice(
                self._dataset_size, size=min(budget, self._dataset_size), replace=False
            ).tolist()

        total_var = sum(active.values())
        allocations = {}
        for c_id, var in active.items():
            cluster_size = int((self._cluster_ids == c_id).sum())
            raw = var / total_var * budget
            allocations[c_id] = min(max(1, int(round(raw))), cluster_size)

        allocations = self._adjust_allocations(allocations, budget, active)
        return self._execute_allocations(allocations)

    def _select_scored(self, budget: int) -> List[int]:
        """Allocate budget using composite scores (COINCIDE-inspired)."""
        if self._transferability is None or self._density is None:
            print("[ClusterSelector] Static scores not computed, falling back to weighted")
            return self._select_weighted(budget)

        scores = {}
        for c_id, var in self._cluster_variances.items():
            trans = float(self._transferability[c_id])
            dens = float(self._density[c_id])
            inv_density = 1.0 / max(dens, 1e-8)
            scores[c_id] = var * trans * inv_density

        active = {c: s for c, s in scores.items() if s > 0}
        if not active:
            return self._select_weighted(budget)

        temp = self.cluster_config.score_temperature
        score_values = np.array(list(active.values()))
        logits = score_values / temp
        logits -= logits.max()
        exp_logits = np.exp(logits)
        ratios = exp_logits / exp_logits.sum()

        allocations = {}
        for (c_id, _), ratio in zip(active.items(), ratios):
            cluster_size = int((self._cluster_ids == c_id).sum())
            raw = ratio * budget
            allocations[c_id] = min(max(1, int(round(raw))), cluster_size)

        allocations = self._adjust_allocations(allocations, budget, active)

        top_n = 15
        sorted_by_score = sorted(active.items(), key=lambda x: -x[1])[:top_n]
        header = f"  {'CID':>5} {'Size':>6} {'Var':>8} {'Trans':>7} {'1/Dens':>8} {'Score':>10} {'Alloc':>6}"
        rows = []
        for c_id, score in sorted_by_score:
            var = self._cluster_variances.get(c_id, 0.0)
            trans = float(self._transferability[c_id])
            dens = float(self._density[c_id])
            alloc = allocations.get(c_id, 0)
            size = int((self._cluster_ids == c_id).sum())
            rows.append(
                f"  {c_id:>5} {size:>6} {var:>8.4f} {trans:>7.4f} {1.0/max(dens,1e-8):>8.4f} {score:>10.6f} {alloc:>6}"
            )
        n_alloc = sum(1 for v in allocations.values() if v > 0)
        print(
            f"[ClusterSelector] Scored allocation — {n_alloc} clusters selected, "
            f"top-{top_n} by score:\n{header}\n" + "\n".join(rows)
        )

        return self._execute_allocations(allocations)

    def _select_interpolated(self, budget: int) -> List[int]:
        """DOTS-style per-sample variance prediction via embedding similarity.

        Two modes controlled by dots_diversity:

        False (default) — global top-k:
            Rank all samples by predicted variance and return the top-budget.
            Maximum focus on the highest-variance region; may concentrate
            selection in one part of the embedding space.

        True — diversity-aware per-cluster allocation:
            Compute cluster-mean predicted variance, allocate budget across
            clusters proportionally (min 1 per cluster), then within each
            cluster take samples with the highest predicted variance.
            Guarantees every cluster contributes at least one sample regardless
            of whether it dominates the global variance ranking.
        """
        asym = self.cluster_config.asymmetric_utility_enabled

        # --- Fit and predict via the pluggable VariancePredictor ---
        result = self._fit_predict_cached()
        if result is None:
            return self._select_weighted(budget)
        predicted_var = result.predicted_var
        predicted_mean = result.predicted_mean

        if predicted_var.sum() == 0:
            return self._select_weighted(budget)

        # Asymmetric utility: reweight by predicted mean reward so that at equal
        # variance, harder (lower-mean) samples are preferred.
        if asym and predicted_mean is not None:
            alpha = float(self.cluster_config.hard_side_bias)
            utility = predicted_var * (1.0 + alpha * (0.5 - predicted_mean))
            low = float(self.cluster_config.asymmetric_dead_zone_low)
            high = float(self.cluster_config.asymmetric_dead_zone_high)
            utility[predicted_mean < low] = 0.0
            utility[predicted_mean > high] = 0.0
            utility = np.maximum(utility, 0.0).astype(np.float32)

            n_alive = int((utility > 0).sum())
            mean_p_alive = (
                float(predicted_mean[utility > 0].mean()) if n_alive > 0 else 0.0
            )
            print(
                f"[ClusterSelector] asymmetric utility: α={alpha:.2f}, "
                f"dead_zone=[{low:.2f},{high:.2f}], "
                f"{n_alive}/{len(utility)} samples alive, "
                f"mean p̂ (alive)={mean_p_alive:.3f}"
            )

            if utility.sum() == 0:
                print("[ClusterSelector] asymmetric utility: no alive samples, "
                      "falling back to predicted_var")
            else:
                predicted_var = utility

        # --- Observed dead-zone: kill saturated/text-leaky samples by observed R_with ---
        obs_high = float(self.cluster_config.observed_deadzone_high)
        if obs_high < 1.0:
            obs_means = self._get_observed_mean_array(len(predicted_var))
            if obs_means is not None:
                predicted_var = predicted_var.copy()
                n_before = int((predicted_var > 0).sum())
                predicted_var[obs_means > obs_high] = 0.0
                n_obs_killed = n_before - int((predicted_var > 0).sum())
                n_buffer = int((obs_means != 0.5).sum())
                print(
                    f"[ClusterSelector] observed dead-zone: high={obs_high:.2f}, "
                    f"buffer_size={n_buffer}, killed={n_obs_killed} "
                    f"(observed R_with > {obs_high:.2f})"
                )

        # --- CAVA soft VDR gate ---
        utility_before_cava = predicted_var.copy()
        utility_after_cava = predicted_var
        cava_gate = None
        if self.cluster_config.cava_vdr_enabled:
            cava_gate = self._compute_cava_gate(predicted_var)
            if self.cluster_config.cava_apply_to_sample_utility:
                predicted_var = (predicted_var * cava_gate).astype(np.float32)
            utility_after_cava = predicted_var.copy()

        # --- Static-VDR soft gate from true no-image audit prior ---
        vdr_gate = None
        if self.cluster_config.vdr_enabled:
            vdr_gate = self._compute_vdr_gate()
            if self.cluster_config.vdr_apply_to_sample_utility:
                predicted_var = (predicted_var * vdr_gate).astype(np.float32)

        # --- Exclude already-selected samples ---
        # Without this, top-K selection keeps re-picking the same high-variance
        # cluster every round, so cumulative unique grows very slowly and the
        # per-round budget is mostly wasted on re-selecting the same prompts.
        # Masking already-selected samples turns each pre-freeze round into a
        # discovery step that adds `budget` brand-new unique samples to the
        # pool — i.e. selection without replacement.  After global_budget_pct
        # is hit, _select_frozen_reweight takes over and is allowed to pick
        # repeats inside the frozen pool.
        exclude = self._discovery_exclude_set() if self.cluster_config.exclude_already_selected else set()
        if exclude:
            mask = np.zeros(predicted_var.shape[0], dtype=bool)
            ever = np.fromiter(exclude, dtype=np.int64, count=len(exclude))
            ever = ever[(ever >= 0) & (ever < mask.shape[0])]
            mask[ever] = True
            n_excluded = int(mask.sum())
            if n_excluded > 0:
                # Use a copy so we don't mutate any caller-shared array.
                predicted_var = predicted_var.copy()
                predicted_var[mask] = 0.0
                n_remaining = int((predicted_var > 0).sum())
                print(f"[ClusterSelector] discovery mask: excluded "
                      f"{n_excluded} already-selected, "
                      f"{n_remaining} candidates remaining")
                if n_remaining == 0:
                    print("[ClusterSelector] discovery mask: pool exhausted "
                          "before global cap; returning empty selection")
                    return []

        if not self.cluster_config.dots_diversity:
            # Global top-k: pure ranking by predicted variance
            top_indices = np.argsort(-predicted_var)[:budget]
            if self.cluster_config.cava_vdr_enabled and cava_gate is not None:
                self._record_cava_selection_metrics(
                    top_indices.tolist(), utility_before_cava, utility_after_cava, cava_gate
                )
            if self.cluster_config.vdr_enabled and vdr_gate is not None:
                self._record_vdr_selection_metrics(top_indices.tolist(), vdr_gate)
            return top_indices.tolist()

        # Diversity mode: allocate budget across clusters via softmax, then
        # within each cluster take the highest predicted-variance samples.

        # Step 1: compute cluster-mean predicted variance for active clusters.
        cluster_pred_var = {}
        for c_id in range(self._centroids.shape[0]):
            mask = self._cluster_ids == c_id
            if not mask.any():
                continue
            cpv = float(predicted_var[mask].mean())
            if cpv > 0:
                cluster_pred_var[c_id] = cpv

        if not cluster_pred_var:
            top_indices = np.argsort(-predicted_var)[:budget]
            if self.cluster_config.vdr_enabled and vdr_gate is not None:
                self._record_vdr_selection_metrics(top_indices.tolist(), vdr_gate)
            return top_indices.tolist()

        # Step 2: compute per-cluster allocation scores.
        # Optionally multiply by transferability and inverse density (COINCIDE-style)
        # to reward clusters that are diverse and whose skills transfer broadly.
        use_composite = (
            self.cluster_config.dots_diversity_use_composite_score
            and self._transferability is not None
            and self._density is not None
        )
        # Check if IGS is available and enabled for composite scoring
        use_igs = (
            self.cluster_config.igs_enabled
            and self._cluster_igs
        )

        cluster_scores = {}
        for c_id, cpv in cluster_pred_var.items():
            if use_composite:
                trans = float(self._transferability[c_id])
                inv_density = 1.0 / max(float(self._density[c_id]), 1e-8)
                score = cpv * trans * inv_density
                # Multiply by IGS if available: rewards genuinely multimodal clusters
                if use_igs and c_id in self._cluster_igs:
                    igs = max(float(self._cluster_igs[c_id]), 1e-3)
                    score *= igs ** self.cluster_config.igs_weight
                cluster_scores[c_id] = score
            else:
                cluster_scores[c_id] = cpv

        if (
            self.cluster_config.cava_vdr_enabled
            and self.cluster_config.cava_apply_to_cluster_allocation
            and cava_gate is not None
        ):
            blend = float(np.clip(self.cluster_config.cava_cluster_gate_blend, 0.0, 1.0))
            for c_id in list(cluster_scores.keys()):
                mask = self._cluster_ids == c_id
                if mask.any():
                    mean_gate = float(np.mean(cava_gate[mask]))
                    cluster_scores[c_id] *= (1.0 - blend) + blend * mean_gate

        if (
            self.cluster_config.vdr_enabled
            and self.cluster_config.vdr_apply_to_cluster_allocation
            and vdr_gate is not None
        ):
            blend = float(np.clip(self.cluster_config.vdr_cluster_blend, 0.0, 1.0))
            for c_id in list(cluster_scores.keys()):
                mask = self._cluster_ids == c_id
                if mask.any():
                    mean_gate = float(np.mean(vdr_gate[mask]))
                    cluster_scores[c_id] *= (1.0 - blend) + blend * mean_gate

        # Step 3: compute current temperature (fixed or annealed).
        if self.cluster_config.dots_diversity_anneal:
            t_start = self.cluster_config.dots_diversity_temperature_start
            t_end = self.cluster_config.dots_diversity_temperature_end
            decay = self.cluster_config.dots_diversity_temperature_decay
            temperature = t_end + (t_start - t_end) * np.exp(
                -decay * self._selection_round
            )
        else:
            temperature = self.cluster_config.dots_diversity_temperature

        # Step 4: softmax over scores to get allocation ratios.
        c_ids = list(cluster_scores.keys())
        score_arr = np.array([cluster_scores[c] for c in c_ids], dtype=np.float64)
        logits = score_arr / max(temperature, 1e-8)
        logits -= logits.max()
        ratios = np.exp(logits)
        ratios /= ratios.sum()

        allocations = {}
        for c_id, ratio in zip(c_ids, ratios):
            cluster_size = int((self._cluster_ids == c_id).sum())
            allocations[c_id] = min(max(1, int(round(ratio * budget))), cluster_size)

        allocations = self._adjust_allocations(allocations, budget, cluster_scores)

        # Step 5: within each cluster take top-n by predicted variance.
        selected = []
        for c_id, n_take in allocations.items():
            if n_take <= 0:
                continue
            cluster_indices = np.where(self._cluster_ids == c_id)[0]
            sorted_local = np.argsort(-predicted_var[cluster_indices])[:n_take]
            selected.extend(cluster_indices[sorted_local].tolist())

        n_clusters_used = sum(1 for v in allocations.values() if v > 0)
        print(f"[ClusterSelector] interpolated+diversity: "
              f"{n_clusters_used} clusters, {len(selected)} samples, "
              f"temp={temperature:.4f} (round={self._selection_round}), "
              f"composite={'on' if use_composite else 'off'}")
        if self.cluster_config.cava_vdr_enabled and cava_gate is not None:
            self._record_cava_selection_metrics(
                selected, utility_before_cava, utility_after_cava, cava_gate
            )
        if self.cluster_config.vdr_enabled and vdr_gate is not None:
            self._record_vdr_selection_metrics(selected, vdr_gate)
        return selected

    def _dots_interpolate(
        self, ref_indices: np.ndarray, ref_variances: np.ndarray
    ) -> np.ndarray:
        """Predict variance for all samples via cosine-similarity-weighted average."""
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
        all_normed = (self._embeddings / norms).astype(np.float32)

        ref_emb = all_normed[ref_indices]
        ref_var = ref_variances.astype(np.float32)

        temp = self.cluster_config.dots_temperature
        top_k = min(self.cluster_config.dots_top_k, len(ref_indices))

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
                topk_vars = ref_var[topk_idx]

                log = topk_sims / temp
                log -= log.max()
                weights = np.exp(log)
                weights /= weights.sum() + 1e-12

                predictions[start + i] = float(np.dot(weights, topk_vars))

        return predictions

    def _within_cluster_select(
        self, cluster_indices: np.ndarray, n_take: int, cluster_id: int
    ) -> List[int]:
        """Select samples within a cluster."""
        if n_take >= len(cluster_indices):
            return cluster_indices.tolist()

        method = self.cluster_config.within_cluster_method
        if method == "mmd":
            return self._mmd_select(cluster_indices, n_take)
        else:
            cluster_embs = self._embeddings[cluster_indices]
            centroid = self._centroids[cluster_id]
            dists = np.linalg.norm(cluster_embs - centroid[np.newaxis, :], axis=1)
            sorted_local = np.argsort(dists)[:n_take]
            return cluster_indices[sorted_local].tolist()

    def _mmd_select(self, cluster_indices: np.ndarray, n_take: int) -> List[int]:
        """Greedy MMD coreset selection within a cluster."""
        cluster_embs = self._embeddings[cluster_indices].astype(np.float64)

        max_kernel_size = 2000
        if len(cluster_embs) > max_kernel_size:
            sub_idx = self._rng.choice(
                len(cluster_embs), size=max_kernel_size, replace=False
            )
            cluster_embs_sub = cluster_embs[sub_idx]
            local_indices = sub_idx
        else:
            cluster_embs_sub = cluster_embs
            local_indices = np.arange(len(cluster_embs))

        gamma = self.cluster_config.mmd_gamma
        norms_sq = (cluster_embs_sub ** 2).sum(axis=1)
        pairwise_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * cluster_embs_sub @ cluster_embs_sub.T
        pairwise_sq = np.maximum(pairwise_sq, 0.0)
        K = np.exp(-pairwise_sq / (2.0 * gamma ** 2))

        n = len(K)
        M = min(n_take, n)
        selected_local = []

        for _ in range(M):
            candidates = np.setdiff1d(np.arange(n), selected_local)
            best_mmd = np.inf
            best_idx = candidates[0]

            for c in candidates:
                trial = selected_local + [c]
                trial_arr = np.array(trial)
                K_XY = K[:, trial_arr].mean()
                K_YY = K[np.ix_(trial_arr, trial_arr)].mean()
                mmd = K.mean() + K_YY - 2.0 * K_XY
                if mmd < best_mmd:
                    best_mmd = mmd
                    best_idx = c

            selected_local.append(best_idx)

        global_selected = cluster_indices[local_indices[np.array(selected_local)]]
        return global_selected.tolist()

    def _adjust_allocations(
        self,
        allocations: Dict[int, int],
        budget: int,
        score_lookup: Optional[Dict[int, float]] = None,
    ) -> Dict[int, int]:
        """Adjust allocations to match budget exactly."""
        score_lookup = score_lookup or self._cluster_variances
        total = sum(allocations.values())
        if total > budget:
            sorted_keys = sorted(allocations.keys(),
                                 key=lambda c: score_lookup.get(c, 0))
            for c_id in sorted_keys:
                if total <= budget:
                    break
                reduction = min(allocations[c_id] - 1, total - budget)
                if reduction > 0:
                    allocations[c_id] -= reduction
                    total -= reduction
        elif total < budget:
            sorted_keys = sorted(allocations.keys(),
                                 key=lambda c: score_lookup.get(c, 0),
                                 reverse=True)
            for c_id in sorted_keys:
                if total >= budget:
                    break
                cluster_size = int((self._cluster_ids == c_id).sum())
                can_add = cluster_size - allocations[c_id]
                add = min(can_add, budget - total)
                if add > 0:
                    allocations[c_id] += add
                    total += add
        return allocations

    def _execute_allocations(self, allocations: Dict[int, int]) -> List[int]:
        """Execute allocated selections across clusters."""
        selected = []
        for c_id, n_take in allocations.items():
            if n_take <= 0:
                continue
            cluster_indices = np.where(self._cluster_ids == c_id)[0]
            take = self._within_cluster_select(cluster_indices, n_take, c_id)
            selected.extend(take)
        return selected

    def get_metrics(self) -> Dict[str, float]:
        metrics = {"data_selection/method": 2.0}
        if self._cluster_variances:
            vars_list = list(self._cluster_variances.values())
            metrics["data_selection/cluster_var_mean"] = float(np.mean(vars_list))
            metrics["data_selection/cluster_var_std"] = float(np.std(vars_list))
            metrics["data_selection/cluster_var_max"] = float(np.max(vars_list))
            metrics["data_selection/n_zero_var_clusters"] = sum(1 for v in vars_list if v == 0)
            metrics["data_selection/n_active_clusters"] = len(vars_list)
        if self._cluster_mean_rewards:
            rewards = list(self._cluster_mean_rewards.values())
            metrics["data_selection/cluster_reward_mean"] = float(np.mean(rewards))
        metrics["data_selection/n_selected"] = len(self._last_selected_indices)

        # Rollout history buffer diagnostics
        if self.cluster_config.use_rollout_history:
            n_unique = len(self._rollout_buffer)
            n_total_entries = sum(len(v) for v in self._rollout_buffer.values())
            metrics["data_selection/history_buffer_unique"] = float(n_unique)
            metrics["data_selection/history_buffer_total_entries"] = float(n_total_entries)

        # --- Selection overlap metrics ---
        metrics["data_selection/overlap_jaccard"] = self._selection_jaccard
        metrics["data_selection/overlap_jaccard_vs_random"] = (
            self._selection_jaccard_vs_random
        )
        if self._ever_selected_set:
            metrics["data_selection/cumulative_coverage_pct"] = (
                100.0 * len(self._ever_selected_set) / max(len(self._embeddings), 1)
            )
            metrics["data_selection/cumulative_unique_selected"] = float(len(self._ever_selected_set))
            metrics["data_selection/selection_round"] = float(self._selection_round)

            # Per-source budget breakdown. A sample can be both a medoid probe
            # and a later training sample; the training/exploration union is
            # what counts when medoids are excluded from the budget.
            metrics["data_selection/budget_breakdown/training"] = float(
                len(self._ever_selected_train))
            metrics["data_selection/budget_breakdown/medoid"] = float(
                len(self._ever_selected_medoid))
            metrics["data_selection/budget_breakdown/exploration"] = float(
                len(self._ever_selected_exploration))
            n_total = max(len(self._ever_selected_set), 1)
            metrics["data_selection/budget_breakdown/training_pct"] = (
                100.0 * len(self._ever_selected_train) / n_total)
            metrics["data_selection/budget_breakdown/medoid_pct"] = (
                100.0 * len(self._ever_selected_medoid) / n_total)
            metrics["data_selection/budget_breakdown/exploration_pct"] = (
                100.0 * len(self._ever_selected_exploration) / n_total)

            # Report global budget cap utilization if enabled
            if self.config.global_budget_pct is not None:
                global_max = max(1, int(len(self._embeddings) * self.config.global_budget_pct / 100.0))
                metrics["data_selection/global_budget"] = float(global_max)
                n_budget_used = len(self._budget_counted_set())
                metrics["data_selection/global_budget_utilization_pct"] = (
                    100.0 * n_budget_used / global_max
                )
                metrics["data_selection/selection_frozen"] = float(self._selection_frozen)
                # Budget schedule phase info
                if self.cluster_config.budget_schedule:
                    per_round_pct, interval = self._get_current_phase()
                    metrics["data_selection/phase_per_round_pct"] = per_round_pct
                    metrics["data_selection/phase_interval"] = float(interval)

                # Reweight stats: predicted variance within the frozen pool
                if (self._selection_frozen
                        and self._last_predicted_var is not None):
                    pool = self._frozen_pool_npz or sorted(self._budget_counted_set())
                    pool_arr = np.array(pool)
                    if len(pool_arr) > 0:
                        pv = self._last_predicted_var[pool_arr]
                        metrics["data_selection/frozen_pool_var_mean"] = float(pv.mean())
                        metrics["data_selection/frozen_pool_var_max"] = float(pv.max())
                        metrics["data_selection/frozen_pool_n_zero_var"] = float(
                            np.sum(pv < 1e-8))

        # Per-cluster selection frequency stats (how evenly distributed is selection?)
        if self._per_cluster_coverage:
            freq_vals = list(self._per_cluster_coverage.values())
            n_clusters_total = len(set(self._cluster_ids.tolist()))
            n_clusters_ever_selected = len(self._per_cluster_coverage)
            metrics["data_selection/n_clusters_ever_selected"] = float(n_clusters_ever_selected)
            metrics["data_selection/cluster_selection_freq_mean"] = float(np.mean(freq_vals))
            metrics["data_selection/cluster_selection_freq_std"] = float(np.std(freq_vals))
            # Gini coefficient: 0=perfectly equal, 1=all budget to one cluster
            sorted_freq = np.sort(freq_vals).astype(np.float64)
            n = len(sorted_freq)
            if n > 0 and sorted_freq.sum() > 0:
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_freq) / (n * sorted_freq.sum())) - (n + 1) / n
                metrics["data_selection/cluster_selection_gini"] = float(gini)

        # --- IGS metrics ---
        if self._cluster_igs:
            igs_vals = list(self._cluster_igs.values())
            metrics["data_selection/igs_mean"] = float(np.mean(igs_vals))
            metrics["data_selection/igs_std"] = float(np.std(igs_vals))
            metrics["data_selection/n_multimodal_clusters"] = sum(
                1 for v in igs_vals if v > 1.5
            )

        # --- Static-VDR metrics ---
        if self.cluster_config.vdr_enabled:
            metrics["data_selection/vdr_enabled"] = 1.0
            metrics["data_selection/vdr_prior_loaded"] = float(
                self._vdr_sample_delta_prior is not None
            )
            if self._vdr_cluster_delta_count is not None:
                covered = self._vdr_cluster_delta_count > 0
                metrics["data_selection/vdr_clusters_with_audit"] = float(covered.sum())
                metrics["data_selection/vdr_cluster_coverage_frac"] = float(
                    covered.sum() / max(len(covered), 1)
                )
            if self._last_vdr_gate is not None:
                gate = self._last_vdr_gate
                metrics["data_selection/vdr_gate_mean_all"] = float(np.mean(gate))
                metrics["data_selection/vdr_gate_std_all"] = float(np.std(gate))
                metrics["data_selection/vdr_gate_min_all"] = float(np.min(gate))
                metrics["data_selection/vdr_gate_max_all"] = float(np.max(gate))
            metrics.update(self._last_vdr_metrics)
        else:
            metrics["data_selection/vdr_enabled"] = 0.0

        # --- CAVA-VDR metrics ---
        if self.cluster_config.cava_vdr_enabled:
            metrics["data_selection/cava_enabled"] = 1.0
            metrics["data_selection/cava_static_prior_loaded"] = float(
                self._cava_static_prior is not None
            )
            metrics["data_selection/cava_logp_buffer_size"] = float(
                sum(len(v) for v in self._cava_logp_buffer.values())
            )
            metrics["data_selection/cava_logp_observed_npz"] = float(
                len(self._cava_logp_buffer)
            )
            metrics["data_selection/cava_static_weight"] = float(
                self.cluster_config.cava_weight_static_prior
            )
            metrics["data_selection/cava_logp_weight"] = float(
                self.cluster_config.cava_weight_logp_contrast
            )
            if self._last_cava_gate is not None:
                gate = self._last_cava_gate
                metrics["data_selection/cava_gate_mean_all"] = float(np.mean(gate))
                metrics["data_selection/cava_gate_std_all"] = float(np.std(gate))
                metrics["data_selection/cava_gate_min_all"] = float(np.min(gate))
                metrics["data_selection/cava_gate_max_all"] = float(np.max(gate))
            metrics.update(self._last_cava_metrics)
        else:
            metrics["data_selection/cava_enabled"] = 0.0

        # --- Variance predictor metrics ---
        metrics.update(self._predictor.get_metrics())
        metrics.update(self._last_predictor_diagnostics)

        return metrics

    def get_wandb_images(self) -> Dict[str, "object"]:
        """Generate wandb.Image objects for selection visualizations.

        Returns a dict of {metric_name: wandb.Image} that can be passed
        to wandb.log() alongside scalar metrics. Returns empty dict if
        wandb or matplotlib are not available.

        Called by ray_trainer after get_metrics() at each selection round.
        """
        try:
            import wandb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        images = {}

        # --- Plot 1: Per-cluster selection count for this round ---
        if self._prev_selected_set and self._cluster_ids is not None:
            current_npz = self._prev_selected_set
            n_clusters = self._centroids.shape[0]
            cluster_counts = np.zeros(n_clusters, dtype=np.int32)
            for idx in current_npz:
                c_id = int(self._cluster_ids[idx])
                cluster_counts[c_id] += 1

            # Sort by count, show top 40
            active = np.where(cluster_counts > 0)[0]
            sorted_clusters = active[np.argsort(-cluster_counts[active])]
            top_n = min(40, len(sorted_clusters))
            show = sorted_clusters[:top_n]

            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(top_n)
            bars = ax.bar(x, cluster_counts[show], color="steelblue", alpha=0.8)

            # Color bars by cluster variance
            if self._cluster_variances:
                max_var = max(self._cluster_variances.values()) or 1.0
                for i, c_id in enumerate(show):
                    v = self._cluster_variances.get(int(c_id), 0.0)
                    intensity = min(v / max_var, 1.0)
                    bars[i].set_color(plt.cm.RdYlGn(intensity))

            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in show], fontsize=6, rotation=45)
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("# Samples Selected")
            ax.set_title(f"Round {self._selection_round}: Per-Cluster Selection "
                         f"(top {top_n}, colored by variance)")
            fig.tight_layout()
            images["data_selection/cluster_allocation"] = wandb.Image(fig)
            plt.close(fig)

        # --- Plot: Difficulty histogram of the most recently selected samples ---
        # Uses the cached DOTS-predicted mean reward (only populated when
        # asymmetric_utility_enabled=True), so this plot only appears in that mode.
        if (self._prev_selected_set
                and self._last_predicted_mean is not None):
            sel = np.fromiter(self._prev_selected_set, dtype=np.int64,
                              count=len(self._prev_selected_set))
            sel = sel[(sel >= 0) & (sel < self._last_predicted_mean.shape[0])]
            if sel.size > 0:
                pm = self._last_predicted_mean[sel]
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(pm, bins=30, range=(0.0, 1.0),
                        color="darkorange", alpha=0.85, edgecolor="black",
                        linewidth=0.3)
                ax.axvline(0.5, color="k", linestyle="--", alpha=0.4,
                           label="frontier (0.5)")
                low = self.cluster_config.asymmetric_dead_zone_low
                high = self.cluster_config.asymmetric_dead_zone_high
                ax.axvspan(0.0, low, color="grey", alpha=0.2,
                           label="dead zone")
                ax.axvspan(high, 1.0, color="grey", alpha=0.2)
                ax.set_xlabel("DOTS-predicted mean reward (0=hard, 1=easy)")
                ax.set_ylabel("# Samples")
                ax.set_title(f"Round {self._selection_round}: Difficulty "
                             f"distribution of selected samples (n={sel.size})")
                ax.set_xlim(0, 1)
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout()
                images["data_selection/selected_difficulty"] = wandb.Image(fig)
                plt.close(fig)

        # --- Plot: 2D PCA cluster map (one-shot, but re-logged each round so
        # it's visible on every wandb panel). Points are colored by their
        # cluster assignment; selected samples are overlaid with black edges.
        # Per-round overhead is trivial (no recomputation; the PCA projection
        # is cached from initialize()).
        if (self._embedding_2d is not None
                and self._cluster_ids is not None):
            E = self._embedding_2d
            labels = self._cluster_ids
            n_bg = min(12000, E.shape[0])
            if n_bg < E.shape[0]:
                bg_idx = self._rng.choice(E.shape[0], size=n_bg, replace=False)
            else:
                bg_idx = np.arange(E.shape[0])

            fig, ax = plt.subplots(figsize=(7, 6))
            # tab20 cycles every 20 clusters — fine for visual grouping even
            # when K >> 20 (adjacent cluster ids share colors but spatial
            # separation still reads).
            ax.scatter(E[bg_idx, 0], E[bg_idx, 1],
                       c=labels[bg_idx] % 20, cmap="tab20",
                       s=2, alpha=0.5, edgecolors="none")

            if self._prev_selected_set:
                sel = np.fromiter(self._prev_selected_set, dtype=np.int64,
                                  count=len(self._prev_selected_set))
                sel = sel[(sel >= 0) & (sel < E.shape[0])]
                if sel.size > 0:
                    ax.scatter(E[sel, 0], E[sel, 1], s=12, facecolors="none",
                               edgecolors="black", linewidths=0.4,
                               label=f"selected (n={sel.size})")
                    ax.legend(fontsize=8, loc="best")

            n_clusters = int(len(np.unique(labels)))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(
                f"Cluster map (K={n_clusters}, 2D PCA) — round {self._selection_round}"
            )
            fig.tight_layout()
            images["data_selection/cluster_map_pca"] = wandb.Image(fig)
            plt.close(fig)

        # --- Plot: 2D PCA scatter of selected vs unselected samples ---
        # Uses the cached PCA projection (computed once at initialize()) and
        # colours selected points by predicted mean reward when available, else
        # by predicted variance.  Background is downsampled for legibility.
        if (self._embedding_2d is not None
                and self._prev_selected_set):
            E = self._embedding_2d
            sel = np.fromiter(self._prev_selected_set, dtype=np.int64,
                              count=len(self._prev_selected_set))
            sel = sel[(sel >= 0) & (sel < E.shape[0])]
            n_bg = min(8000, E.shape[0])
            if n_bg < E.shape[0]:
                bg_idx = self._rng.choice(E.shape[0], size=n_bg, replace=False)
            else:
                bg_idx = np.arange(E.shape[0])

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(E[bg_idx, 0], E[bg_idx, 1], s=2, c="lightgrey",
                       alpha=0.4, label="unselected (bg sample)")

            if sel.size > 0:
                color_vals = None
                cbar_label = None
                if self._last_predicted_mean is not None:
                    color_vals = self._last_predicted_mean[sel]
                    cmap = "RdYlGn"
                    cbar_label = "predicted mean reward (1=easy)"
                elif self._last_predicted_var is not None:
                    color_vals = self._last_predicted_var[sel]
                    cmap = "viridis"
                    cbar_label = "predicted variance"
                if color_vals is not None:
                    sc = ax.scatter(E[sel, 0], E[sel, 1], s=10, c=color_vals,
                                    cmap=cmap, edgecolors="black",
                                    linewidths=0.2)
                    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                    cb.set_label(cbar_label, fontsize=8)
                else:
                    ax.scatter(E[sel, 0], E[sel, 1], s=10, c="crimson",
                               edgecolors="black", linewidths=0.2,
                               label="selected")
                    ax.legend(fontsize=8, loc="best")

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Round {self._selection_round}: Selected samples in "
                         f"embedding space (n_selected={sel.size})")
            fig.tight_layout()
            images["data_selection/selection_scatter"] = wandb.Image(fig)
            plt.close(fig)

        # --- Plot: Cumulative annotation budget by source over time ---
        if len(self._budget_history) >= 1:
            hist = np.array(self._budget_history, dtype=np.int64)
            rounds = hist[:, 0]
            train = hist[:, 1]
            medoid = hist[:, 2]
            explore = hist[:, 3]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.stackplot(rounds, train, medoid, explore,
                         labels=["training", "medoid", "exploration"],
                         colors=["#4c72b0", "#dd8452", "#55a868"],
                         alpha=0.85)

            if self.config.global_budget_pct is not None:
                global_max = max(1, int(
                    len(self._embeddings) * self.config.global_budget_pct / 100.0))
                ax.axhline(global_max, color="red", linestyle="--",
                           linewidth=1.0,
                           label=f"global cap ({self.config.global_budget_pct:g}%)")

            ax.set_xlabel("Selection round")
            ax.set_ylabel("Cumulative unique samples")
            ax.set_title("Annotation budget breakdown by source")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            images["data_selection/budget_breakdown"] = wandb.Image(fig)
            plt.close(fig)

        # --- Plot: Predicted vs Observed variance scatter (predictor diagnostics) ---
        ref_indices = getattr(self._predictor, "_ref_indices", None)
        ref_observed_var = getattr(self._predictor, "_ref_observed_var", None)
        ref_observed_mean = getattr(self._predictor, "_ref_observed_mean", None)
        if (ref_indices is not None and ref_observed_var is not None
                and self._last_predicted_var is not None):
            scatter_img = make_predictor_scatter_plot(
                ref_indices=ref_indices,
                ref_observed_var=ref_observed_var,
                predicted_var_all=self._last_predicted_var,
                predictor_type=self.cluster_config.predictor_type,
                selection_round=self._selection_round,
                ref_observed_mean=ref_observed_mean,
            )
            if scatter_img is not None:
                images["data_selection/pred_vs_obs_scatter"] = scatter_img

        # --- Plot 2: Overlap Jaccard history ---
        if len(self._selection_history) > 2:
            jaccards = []
            for i in range(1, len(self._selection_history)):
                prev = self._selection_history[i - 1]
                curr = self._selection_history[i]
                inter = len(curr & prev)
                union = len(curr | prev)
                jaccards.append(inter / max(union, 1))

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(1, len(jaccards) + 1), jaccards, "o-", markersize=3)
            ax.set_xlabel("Selection Round")
            ax.set_ylabel("Jaccard with Previous")
            ax.set_title("Selection Overlap Over Training")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.4)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            images["data_selection/overlap_history"] = wandb.Image(fig)
            plt.close(fig)

        return images
