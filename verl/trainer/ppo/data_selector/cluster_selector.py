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
from typing import Dict, List, Optional

import numpy as np

from .base import DataSelectionConfig, DataSelector


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
    rollout_history_max_refs: int = 2000        # cap reference set size for DOTS

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
        self._last_selected_indices: List[int] = []
        self._selection_round: int = 0  # incremented each call to select()

        # --- Selection overlap tracking ---
        self._prev_selected_set: set = set()  # NPZ indices from previous round
        self._selection_jaccard: float = 0.0
        self._selection_history: List[set] = []  # all rounds' NPZ index sets
        self._per_cluster_coverage: Dict[int, int] = {}  # cluster_id -> times selected

        # --- Exploration rollouts ---
        self._exploration_indices: List[int] = []  # NPZ indices for next exploration
        self._exploration_rewards: Optional[np.ndarray] = None

        # --- Image Grounding Score (IGS) ---
        # Per-sample IGS: var_with_image / var_without_image.
        # Computed externally and loaded, or computed on-the-fly from rollouts.
        self._igs_scores: Optional[np.ndarray] = None  # shape (N,), per-sample
        self._cluster_igs: Dict[int, float] = {}  # cluster_id -> mean IGS

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

        self._rng = np.random.RandomState(42)

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

        self._select_representatives()

        if self.cluster_config.strategy in ("scored", "interpolated"):
            self._compute_static_scores()

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
        """Build npz↔parquet index alignment maps from the JSON source file.

        The cluster_arrays.npz is built from a JSON/JSONL where each line has
        an 'image' field (e.g. 'clevr_math/CLEVR_train_026670.png').  The
        training parquet stores the same path in extra_info['image'].  We match
        on this field to produce:
            _npz_to_dataset[npz_i]   = parquet_i   (shape N_npz, fill -1 if no match)
            _dataset_to_npz[parquet_i] = npz_i     (only for matched rows)

        Once built, get_reference_indices() and select() remap their NPZ-order
        outputs to parquet-order indices via _npz_to_dataset.
        """
        import json as _json

        print(f"[ClusterSelector] Building NPZ↔parquet alignment from {json_path} ...")

        # --- Step 1: load JSON records and extract image paths ---
        json_image_paths: List[str] = []
        with open(json_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    json_image_paths.append(rec.get("image", ""))
                except _json.JSONDecodeError:
                    json_image_paths.append("")

        # Truncate to embeddings size in case JSON has more records.
        n_npz = len(self._embeddings)
        json_image_paths = json_image_paths[:n_npz]

        # --- Step 2: build parquet image_path → parquet_idx map ---
        parquet_img_to_idx: Dict[str, int] = {}
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
            if img:
                parquet_img_to_idx[img] = parquet_i

        # --- Step 3: cross-reference to build alignment arrays ---
        self._npz_to_dataset = np.full(n_npz, -1, dtype=np.int64)
        self._dataset_to_npz = {}

        n_matched = 0
        for npz_i, img_path in enumerate(json_image_paths):
            parquet_i = parquet_img_to_idx.get(img_path, -1)
            self._npz_to_dataset[npz_i] = parquet_i
            if parquet_i >= 0:
                self._dataset_to_npz[parquet_i] = npz_i
                n_matched += 1

        n_unmatched = n_npz - n_matched
        print(f"[ClusterSelector] Alignment: {n_matched}/{n_npz} NPZ rows matched to parquet rows "
              f"({n_unmatched} NPZ rows have no parquet counterpart and will be excluded from selection).")
        if n_unmatched > 0:
            print(f"[ClusterSelector] Unmatched rows are typically samples in the JSON that were "
                  f"filtered out during parquet creation (missing images, preprocessing failures, etc.).")

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

    def _select_representatives(self) -> None:
        """Select representative samples (medoids) for each cluster."""
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
        # _rep_indices are in NPZ order. Remap to parquet order before returning
        # so the trainer's Subset(dataset, ref_indices) accesses the correct rows.
        return self._remap_npz_to_dataset(self._rep_indices)

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

        idx_to_cluster = dict(zip(self._rep_indices, self._rep_cluster_ids))

        # ref_indices are parquet positions (from get_reference_indices which
        # already remapped via _npz_to_dataset). Convert back to NPZ positions
        # so we can look up cluster membership and store at the right NPZ slot.
        def to_npz(parquet_i: int) -> int:
            if self._dataset_to_npz is not None:
                return self._dataset_to_npz.get(parquet_i, -1)
            return parquet_i  # assume aligned when no JSON was provided

        self._rep_variances = {}
        for i, parquet_ref_idx in enumerate(ref_indices):
            global_idx = to_npz(parquet_ref_idx)  # NPZ position
            if global_idx < 0:
                continue  # parquet row has no NPZ counterpart (filtered sample)
            c_id = idx_to_cluster.get(global_idx)
            if c_id is None:
                continue
            mean_r = float(ref_rewards[i].mean())
            var_r = float(ref_rewards[i].var())
            self._rep_variances[global_idx] = var_r  # individual variance per rep
            cluster_rewards[c_id].append(var_r)
            cluster_all_rewards[c_id].append(mean_r)

            # Seed the rollout buffer with REPR rollouts so round 0 already has
            # reference points. Subsequent REPR rounds update with fresh entries.
            if self.cluster_config.use_rollout_history:
                if global_idx not in self._rollout_buffer:
                    self._rollout_buffer[global_idx] = []
                self._rollout_buffer[global_idx].append(
                    (self._current_training_step, ref_rewards[i].copy())
                )

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
                weighted_var += w * v
                total_w += w
                latest_step = max(latest_step, t)
            if total_w > 1e-12:
                results[dataset_idx] = weighted_var / total_w
                recency[dataset_idx] = latest_step

        # Cap to max_refs by keeping the most recently observed samples.
        if len(results) > max_refs:
            top_idxs = sorted(recency, key=lambda i: recency[i], reverse=True)[:max_refs]
            results = {i: results[i] for i in top_idxs}

        return results

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

        print(f"[ClusterSelector] Exploration: added {n_added} samples to history buffer")

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
        if not self._cluster_variances:
            print("[ClusterSelector] No variance data yet, selecting random")
            return self._rng.choice(
                self._dataset_size, size=min(budget, self._dataset_size), replace=False
            ).tolist()

        strategy = self.cluster_config.strategy
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
        self._prev_selected_set = current_set
        self._selection_history.append(current_set)

        # Track per-cluster selection frequency
        for idx in npz_indices:
            c_id = int(self._cluster_ids[idx])
            self._per_cluster_coverage[c_id] = self._per_cluster_coverage.get(c_id, 0) + 1

        # Count how many unique samples have EVER been selected across all rounds
        all_ever_selected = set()
        for s in self._selection_history:
            all_ever_selected |= s
        coverage_pct = 100.0 * len(all_ever_selected) / max(len(self._embeddings), 1)

        print(f"[ClusterSelector] Overlap: jaccard={self._selection_jaccard:.3f}, "
              f"cumulative_coverage={coverage_pct:.1f}% ({len(all_ever_selected)}/{len(self._embeddings)})")

        # Remap NPZ positions → parquet positions so the trainer's
        # Subset(train_dataset, indices) accesses the correct rows.
        indices = self._remap_npz_to_dataset(npz_indices)

        self._last_selected_indices = indices
        print(f"[ClusterSelector] Selected {len(indices)} samples "
              f"(strategy={strategy}, budget={budget})")
        return indices

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

        allocations = self._adjust_allocations(allocations, budget)
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

        allocations = self._adjust_allocations(allocations, budget)

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
        # Build reference set for DOTS.
        # When rollout history is enabled and the buffer has entries, use the
        # full time-weighted buffer as reference points — this covers the whole
        # training trajectory, not just the fixed REPR medoids.
        # Fall back to REPR medoids when the buffer is empty (first round before
        # any training steps have been recorded).
        if (self.cluster_config.use_rollout_history and self._rollout_buffer):
            tw_vars = self._compute_time_weighted_variances(self._current_training_step)
            if tw_vars:
                ref_indices = np.array(list(tw_vars.keys()))
                ref_variances = np.array(list(tw_vars.values()), dtype=np.float32)
                print(f"[ClusterSelector] DOTS reference: {len(ref_indices)} samples "
                      f"from rollout history (step={self._current_training_step})")
            else:
                ref_indices = np.array(self._rep_indices)
                ref_variances = np.array([
                    self._rep_variances.get(idx, self._cluster_variances.get(c_id, 0.0))
                    for idx, c_id in zip(self._rep_indices, self._rep_cluster_ids)
                ], dtype=np.float32)
        else:
            # Default path: fixed REPR medoids with individual per-rep variances.
            ref_indices = np.array(self._rep_indices)
            ref_variances = np.array([
                self._rep_variances.get(global_idx, self._cluster_variances.get(c_id, 0.0))
                for global_idx, c_id in zip(self._rep_indices, self._rep_cluster_ids)
            ], dtype=np.float32)

        if ref_variances.sum() == 0:
            return self._select_weighted(budget)

        predicted_var = self._dots_interpolate(ref_indices, ref_variances)

        if not self.cluster_config.dots_diversity:
            # Global top-k: pure ranking by predicted variance
            top_indices = np.argsort(-predicted_var)[:budget]
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

        allocations = self._adjust_allocations(allocations, budget)

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

    def _adjust_allocations(self, allocations: Dict[int, int], budget: int) -> Dict[int, int]:
        """Adjust allocations to match budget exactly."""
        total = sum(allocations.values())
        if total > budget:
            sorted_keys = sorted(allocations.keys(),
                                 key=lambda c: self._cluster_variances.get(c, 0))
            for c_id in sorted_keys:
                if total <= budget:
                    break
                reduction = min(allocations[c_id] - 1, total - budget)
                if reduction > 0:
                    allocations[c_id] -= reduction
                    total -= reduction
        elif total < budget:
            sorted_keys = sorted(allocations.keys(),
                                 key=lambda c: self._cluster_variances.get(c, 0),
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
        if self._selection_history:
            all_ever = set()
            for s in self._selection_history:
                all_ever |= s
            metrics["data_selection/cumulative_coverage_pct"] = (
                100.0 * len(all_ever) / max(len(self._embeddings), 1)
            )
            metrics["data_selection/cumulative_unique_selected"] = float(len(all_ever))
            metrics["data_selection/selection_round"] = float(self._selection_round)

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
