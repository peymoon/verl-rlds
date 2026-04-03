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

    score_temperature: float = 0.1
    transferability_sim_threshold: float = 0.9
    density_gamma: float = 1.0
    mmd_gamma: float = 1.0

    faiss_nredo: int = 10
    faiss_niter: int = 50
    faiss_seed: int = 42
    use_gpu_faiss: bool = True


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
        self._last_selected_indices: List[int] = []

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
        return self._rep_indices

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

        for i, global_idx in enumerate(ref_indices):
            c_id = idx_to_cluster.get(global_idx)
            if c_id is None:
                continue
            mean_r = float(ref_rewards[i].mean())
            var_r = float(ref_rewards[i].var())
            cluster_rewards[c_id].append(var_r)
            cluster_all_rewards[c_id].append(mean_r)

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

    def select(self, budget: int) -> List[int]:
        if not self._cluster_variances:
            print("[ClusterSelector] No variance data yet, selecting random")
            return self._rng.choice(
                self._dataset_size, size=min(budget, self._dataset_size), replace=False
            ).tolist()

        strategy = self.cluster_config.strategy
        if strategy == "top_clusters":
            indices = self._select_top_clusters(budget)
        elif strategy == "weighted":
            indices = self._select_weighted(budget)
        elif strategy == "scored":
            indices = self._select_scored(budget)
        elif strategy == "interpolated":
            indices = self._select_interpolated(budget)
        else:
            raise ValueError(f"Unknown cluster selection strategy: {strategy}")

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
        """DOTS-style per-sample variance prediction via embedding similarity."""
        ref_indices = np.array(self._rep_indices)
        ref_variances = np.array([
            self._cluster_variances.get(c_id, 0.0)
            for c_id in self._rep_cluster_ids
        ], dtype=np.float32)

        if ref_variances.sum() == 0:
            return self._select_weighted(budget)

        predicted_var = self._dots_interpolate(ref_indices, ref_variances)

        top_indices = np.argsort(-predicted_var)[:budget]
        return top_indices.tolist()

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
        return metrics
