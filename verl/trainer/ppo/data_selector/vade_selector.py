"""
VADE data selector adapted to this VERL data-selection interface.

This keeps VADE's beta-count state, Thompson/UCB/direct weighting rules, and
its three a/b update strategies.  The only dataset adaptation is that our
stable parquet row id (``dataset_idx``) is used as VADE's ``sample_index``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import DataSelectionConfig, DataSelector


@dataclass
class VadeConfig:
    """VADE-specific configuration."""

    # Initial base-policy rollout data.  VADE expects a JSON dict:
    #   {"sample_index": {"total_rollouts": G, "correct_count": k}}
    # This adapter also accepts our dumped JSONL rollout files/directories and
    # converts rows grouped by "index" or "dataset_idx" into that same state.
    initial_rollout_data_path: Optional[str] = None
    log_state_dir: Optional[str] = None

    # VADE sampler hyperparameters.
    T: float = 1.0
    lam1: float = 0.2
    lam2: float = 0.999
    G: int = 8
    replacement: bool = False
    sample_method: str = "thompson"
    p_function: int = 1
    alpha: float = 0.95
    seed: int = 42
    a_b_update_strategy: int = 1
    tau: float = 1.0
    k: float = 100.0

    # Match the cluster selector's phased discovery schedule.  When this is
    # set together with data_selection.global_budget_pct, each phase controls
    # how many NEW unique training samples VADE may add per round.
    budget_schedule: Optional[list] = None

    # Which reward stream the trainer should pass to VADE.  For ViRL39K/VLAA,
    # "acc" is the closest match to VADE's correct rollout count.
    selection_reward_key: str = "acc"
    correctness_threshold: float = 0.5
    selection_reward_threshold: Optional[float] = None


class VadeSelector(DataSelector):
    """VADE DynamicWeightedSampler as an online DataSelector."""

    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)
        vade_cfg = config.vade if isinstance(config.vade, dict) else {}
        self.vade_config = VadeConfig(**vade_cfg)
        if self.vade_config.selection_reward_threshold is not None:
            self.vade_config.correctness_threshold = float(self.vade_config.selection_reward_threshold)
        if str(self.vade_config.sample_method).lower() == "ucb":
            self.vade_config.sample_method = "UCB"

        self._dataset = None
        self._dataset_size = 0
        self.sample_to_dataset_map: Dict[int, int] = {}
        self.dataset_to_sample_map: Dict[int, int] = {}

        self.a_dict: Dict[int, float] = {}
        self.b_dict: Dict[int, float] = {}
        self.last_updated_steps: Dict[int, int] = {}
        self.last_updated_values: Dict[int, float] = {}
        self.weights: np.ndarray = np.zeros(0, dtype=np.float64)

        self.global_steps = 0
        self.correct_num_list: List[int] = []
        self.selected_sample_index_list: List[int] = []
        self._last_selected_indices: List[int] = []
        self._selection_round = 0
        self._ever_selected_set: set[int] = set()
        self._last_new_indices: List[int] = []
        self._frozen_pool_indices: List[int] = []
        self._rng = np.random.RandomState(int(self.vade_config.seed))

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle / state
    # ------------------------------------------------------------------

    def initialize(self, dataset, collate_fn=None) -> None:
        self._dataset = dataset
        self._dataset_size = len(dataset)
        self._build_index_mapping()

        if not self.vade_config.initial_rollout_data_path:
            raise ValueError("VadeSelector requires data_selection.vade.initial_rollout_data_path.")
        self._initialize_from_rollout(self.vade_config.initial_rollout_data_path)

        self.weights = np.zeros(self._dataset_size, dtype=np.float64)
        if self.vade_config.log_state_dir:
            os.makedirs(self.vade_config.log_state_dir, exist_ok=True)
            self._log_state()

        print(
            f"[VadeSelector] Ready: dataset={self._dataset_size}, "
            f"initial_rollouts={self.vade_config.initial_rollout_data_path}, "
            f"method={self.vade_config.sample_method}, p_function={self.vade_config.p_function}, "
            f"strategy={self.vade_config.a_b_update_strategy}, G={self.vade_config.G}"
        )

    def get_state_dict(self) -> dict:
        return {
            "a_dict": self.a_dict.copy(),
            "b_dict": self.b_dict.copy(),
            "weights": self.weights.tolist(),
            "sample_to_dataset_map": self.sample_to_dataset_map.copy(),
            "dataset_to_sample_map": self.dataset_to_sample_map.copy(),
            "selected_sample_index_list": list(self.selected_sample_index_list),
            "correct_num_list": list(self.correct_num_list),
            "last_updated_steps": self.last_updated_steps.copy(),
            "last_updated_values": self.last_updated_values.copy(),
            "global_steps": self.global_steps,
            "_last_selected_indices": list(self._last_selected_indices),
            "_selection_round": self._selection_round,
            "_ever_selected_set": sorted(self._ever_selected_set),
            "_last_new_indices": list(self._last_new_indices),
            "_frozen_pool_indices": list(self._frozen_pool_indices),
            "_rng_state": self._rng.get_state(),
            "_selection_frozen": self._selection_frozen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.a_dict = {int(k): float(v) for k, v in state.get("a_dict", {}).items()}
        self.b_dict = {int(k): float(v) for k, v in state.get("b_dict", {}).items()}
        self.sample_to_dataset_map = {
            int(k): int(v) for k, v in state.get("sample_to_dataset_map", {}).items()
        }
        self.dataset_to_sample_map = {
            int(k): int(v) for k, v in state.get("dataset_to_sample_map", {}).items()
        }
        self.last_updated_steps = {
            int(k): int(v) for k, v in state.get("last_updated_steps", {}).items()
        }
        self.last_updated_values = {
            int(k): float(v) for k, v in state.get("last_updated_values", {}).items()
        }
        self.weights = np.asarray(state.get("weights", self.weights), dtype=np.float64)
        self.selected_sample_index_list = [
            int(i) for i in state.get("selected_sample_index_list", self.selected_sample_index_list)
        ]
        self.correct_num_list = [int(i) for i in state.get("correct_num_list", self.correct_num_list)]
        self.global_steps = int(state.get("global_steps", self.global_steps))
        self._last_selected_indices = [int(i) for i in state.get("_last_selected_indices", [])]
        self._selection_round = int(state.get("_selection_round", 0))
        self._ever_selected_set = set(int(i) for i in state.get("_ever_selected_set", []))
        self._last_new_indices = [int(i) for i in state.get("_last_new_indices", [])]
        self._frozen_pool_indices = [
            int(i) for i in state.get("_frozen_pool_indices", [])
        ]
        self._selection_frozen = bool(state.get("_selection_frozen", False))
        if self._selection_frozen and not self._frozen_pool_indices:
            self._frozen_pool_indices = sorted(self._ever_selected_set)
        if "_rng_state" in state:
            self._rng.set_state(state["_rng_state"])

        if not self.sample_to_dataset_map or not self.dataset_to_sample_map:
            print("[VadeSelector] Legacy selector state missing index maps; rebuilding identity maps.")
            self._build_index_mapping()

        print(
            f"[VadeSelector] Restored state: round={self._selection_round}, "
            f"global_step={self.global_steps}, train_uniques={len(self._ever_selected_set)}"
        )

    # ------------------------------------------------------------------
    # Trainer API
    # ------------------------------------------------------------------

    def get_reference_indices(self) -> List[int]:
        return []

    def update_rewards(self, ref_indices: Sequence[int], ref_rewards: np.ndarray) -> None:
        return None

    def should_reselect_step(self, global_step: int) -> bool:
        """Use the VADE budget schedule interval when a schedule is configured."""
        if self.config.reselect_schedule != "step":
            return False
        _, interval = self._get_current_phase()
        if interval <= 0:
            return False
        return global_step % interval == 0

    def get_selection_budget(self, dataset_size: int) -> int:
        """Return VADE's per-round NEW unique discovery budget.

        With a budget schedule enabled, this intentionally overrides the flat
        ``selection_budget`` value that older VADE launchers pass by default.
        The selector pads the actual training window from already-admitted
        samples when this budget falls below the trainer's batch-size floor.
        """
        if self._schedule_active():
            per_round_pct, _ = self._get_current_phase()
            return max(1, int(dataset_size * float(per_round_pct) / 100.0))
        return super().get_selection_budget(dataset_size)

    def select(self, budget: int) -> List[int]:
        if self._dataset_size <= 0:
            return []

        self._compute_weights()
        target_size = min(self._dataset_size, max(int(budget), int(self._min_training_pool_size)))
        if target_size <= 0:
            return []

        sorted_indices = [
            int(i) for i in np.argsort(-self.weights, kind="mergesort").tolist()
        ]
        cap = self._global_budget_cap()
        if cap is None:
            selected_indices = sorted_indices[:target_size]
            new_indices = [
                int(i) for i in selected_indices if int(i) not in self._ever_selected_set
            ]
            self._ever_selected_set.update(selected_indices)
        else:
            selected_indices, new_indices = self._select_with_global_budget(
                sorted_indices=sorted_indices,
                target_size=target_size,
                new_budget=int(budget),
                global_cap=cap,
            )

        self._last_selected_indices = selected_indices
        self._last_new_indices = new_indices
        self._selection_round += 1

        if self.vade_config.budget_schedule:
            per_round_pct, interval = self._get_current_phase()
            print(
                f"[VadeSelector] Budget schedule: active phase "
                f"per_round_pct={float(per_round_pct):g}, interval={int(interval)}"
            )

        if not selected_indices:
            print(
                f"[VadeSelector] Round {self._selection_round}: selected 0 rows "
                f"(budget={budget}, floor={self._min_training_pool_size}); "
                f"new_uniques=0, train_uniques={len(self._ever_selected_set)}"
            )
            return []

        selected_weights = self.weights[np.asarray(selected_indices, dtype=np.int64)]
        cap_msg = ""
        if cap is not None:
            cap_msg = (
                f", new_uniques={len(new_indices)}, "
                f"train_uniques={len(self._ever_selected_set)}/{cap}, "
                f"frozen={self._selection_frozen}"
            )
        print(
            f"[VadeSelector] Round {self._selection_round}: selected {len(selected_indices)} "
            f"rows (budget={budget}, floor={self._min_training_pool_size}); "
            f"weight_mean={selected_weights.mean():.6f}, weight_max={selected_weights.max():.6f}"
            f"{cap_msg}"
        )
        return selected_indices

    def update_rollout_history(
        self,
        step: int,
        uids,
        rewards_per_rollout,
        dataset_indices=None,
    ) -> None:
        if dataset_indices is None:
            if not getattr(self, "_warned_missing_dataset_idx", False):
                print("[VadeSelector] WARNING: missing dataset_idx; skipping VADE a/b update.")
                self._warned_missing_dataset_idx = True
            return

        rewards = np.asarray(rewards_per_rollout, dtype=np.float32).reshape(-1)
        indices = np.asarray(dataset_indices).reshape(-1)
        n = min(len(rewards), len(indices))
        if n == 0:
            return
        if len(rewards) != len(indices) and not getattr(self, "_warned_shape_mismatch", False):
            print(
                f"[VadeSelector] WARNING: rewards length {len(rewards)} != dataset_idx length {len(indices)}; "
                f"using first {n} rows."
            )
            self._warned_shape_mismatch = True

        self.global_steps = int(step)
        sample_order: List[int] = []
        grouped: Dict[int, List[float]] = {}
        for raw_idx, reward in zip(indices[:n], rewards[:n]):
            try:
                dataset_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            sample_index = self.dataset_to_sample_map.get(dataset_idx, dataset_idx)
            if sample_index not in grouped:
                grouped[sample_index] = []
                sample_order.append(sample_index)
            grouped[sample_index].append(float(reward))

        sample_index_list: List[int] = []
        correct_num_list: List[int] = []
        threshold = float(self.vade_config.correctness_threshold)
        for sample_index in sample_order:
            vals = grouped[sample_index]
            if len(vals) != int(self.vade_config.G) and not getattr(self, "_warned_rollout_group_size", False):
                print(
                    f"[VadeSelector] WARNING: observed {len(vals)} rollouts for sample {sample_index}; "
                    f"VADE is configured with G={self.vade_config.G}."
                )
                self._warned_rollout_group_size = True
            correct = int((np.asarray(vals, dtype=np.float32) >= threshold).sum())
            correct = min(max(correct, 0), int(self.vade_config.G))
            sample_index_list.append(int(sample_index))
            correct_num_list.append(correct)

        if sample_index_list:
            self.update_weights(sample_index_list, correct_num_list)

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "data_selection/vade_round": float(self._selection_round),
            "data_selection/vade_global_step": float(self.global_steps),
            "data_selection/vade_train_uniques": float(len(self._ever_selected_set)),
            "data_selection/vade_selected_count": float(len(self._last_selected_indices)),
            "data_selection/vade_new_uniques_last_round": float(len(self._last_new_indices)),
            "data_selection/cumulative_unique_selected": float(len(self._ever_selected_set)),
        }
        if self._dataset_size > 0:
            metrics["data_selection/cumulative_coverage_pct"] = (
                100.0 * len(self._ever_selected_set) / self._dataset_size
            )
        global_cap = self._global_budget_cap()
        if global_cap is not None:
            metrics["data_selection/global_budget"] = float(global_cap)
            metrics["data_selection/global_budget_utilization_pct"] = (
                100.0 * len(self._budget_counted_set()) / global_cap
            )
            metrics["data_selection/selection_frozen"] = float(self._selection_frozen)
            metrics["data_selection/frozen_pool_size"] = float(
                len(self._frozen_pool_indices or sorted(self._ever_selected_set))
            )
            if self.vade_config.budget_schedule:
                per_round_pct, interval = self._get_current_phase()
                metrics["data_selection/phase_per_round_pct"] = float(per_round_pct)
                metrics["data_selection/phase_interval"] = float(interval)
        if self.weights.size > 0:
            metrics.update({
                "data_selection/vade_weight_mean": float(np.mean(self.weights)),
                "data_selection/vade_weight_std": float(np.std(self.weights)),
                "data_selection/vade_weight_min": float(np.min(self.weights)),
                "data_selection/vade_weight_max": float(np.max(self.weights)),
            })
        if self._last_selected_indices:
            vals = self.weights[np.asarray(self._last_selected_indices, dtype=np.int64)]
            metrics.update({
                "data_selection/vade_selected_weight_mean": float(np.mean(vals)),
                "data_selection/vade_selected_weight_min": float(np.min(vals)),
                "data_selection/vade_selected_weight_max": float(np.max(vals)),
            })
        if self.correct_num_list:
            arr = np.asarray(self.correct_num_list, dtype=np.float32)
            metrics["data_selection/vade_last_correct_mean"] = float(np.mean(arr))
        return metrics

    # ------------------------------------------------------------------
    # VADE algorithm
    # ------------------------------------------------------------------

    def update_weights(self, sample_index_list: List[int], correct_num_list: List[int]) -> None:
        if len(sample_index_list) != len(correct_num_list):
            raise ValueError(
                f"index numbers ({len(sample_index_list)}) and k_t numbers "
                f"({len(correct_num_list)}) do not match."
            )

        self.selected_sample_index_list = [int(i) for i in sample_index_list]
        self.correct_num_list = [int(i) for i in correct_num_list]

        lambda1_update_count = 0
        lambda2_update_count = 0
        strategy = int(self.vade_config.a_b_update_strategy)
        G = int(self.vade_config.G)

        if strategy == 1:
            for sample_index, k_t in zip(self.selected_sample_index_list, self.correct_num_list):
                if sample_index not in self.a_dict:
                    print(f"[VadeSelector] WARNING: sample_index {sample_index} not in a_dict; skipping.")
                    continue
                current_a = self.a_dict[sample_index]
                current_b = self.b_dict[sample_index]
                if k_t == 0 or k_t == G:
                    self.a_dict[sample_index] = float(self.vade_config.lam2) * current_a + k_t
                    self.b_dict[sample_index] = float(self.vade_config.lam2) * current_b + (G - k_t)
                    lambda2_update_count += 1
                else:
                    self.a_dict[sample_index] = float(self.vade_config.lam1) * current_a + k_t
                    self.b_dict[sample_index] = float(self.vade_config.lam1) * current_b + (G - k_t)
                    lambda1_update_count += 1

            selected = set(self.selected_sample_index_list)
            for dataset_idx in range(self._dataset_size):
                sample_index = self.dataset_to_sample_map.get(dataset_idx, dataset_idx)
                if sample_index not in selected:
                    self.a_dict[sample_index] = float(self.vade_config.lam2) * self.a_dict[sample_index]
                    self.b_dict[sample_index] = float(self.vade_config.lam2) * self.b_dict[sample_index]
                    lambda2_update_count += 1

        elif strategy == 2:
            for sample_index, k_t in zip(self.selected_sample_index_list, self.correct_num_list):
                if sample_index not in self.a_dict:
                    print(f"[VadeSelector] WARNING: sample_index {sample_index} not in a_dict; skipping.")
                    continue
                self.last_updated_steps[sample_index] = int(self.global_steps)
                self.last_updated_values[sample_index] = float(k_t)

        elif strategy == 3:
            for sample_index, k_t in zip(self.selected_sample_index_list, self.correct_num_list):
                if sample_index not in self.a_dict:
                    print(f"[VadeSelector] WARNING: sample_index {sample_index} not in a_dict; skipping.")
                    continue
                self.a_dict[sample_index] = float(k_t)
                self.b_dict[sample_index] = float(G - k_t)

        else:
            raise ValueError(f"update strategy {strategy} is not supported.")

        print(
            f"[VadeSelector] Updated VADE state at step={self.global_steps}: "
            f"lambda1_updates={lambda1_update_count}, lambda2_updates={lambda2_update_count}, "
            f"selected={len(self.selected_sample_index_list)}"
        )
        self._log_state()

    def _compute_weights(self) -> None:
        weights = np.zeros(self._dataset_size, dtype=np.float64)
        strategy = int(self.vade_config.a_b_update_strategy)
        G = int(self.vade_config.G)

        if strategy == 2 and self.global_steps != 1:
            for dataset_idx in range(self._dataset_size):
                sample_index = self.dataset_to_sample_map.get(dataset_idx, dataset_idx)
                last_update_step = self.last_updated_steps[sample_index]
                last_update_value = self.last_updated_values[sample_index]
                decay_coef = 1 / (1 + (self.global_steps - last_update_step) / float(self.vade_config.tau))
                self.a_dict[sample_index] = float(self.vade_config.k) * decay_coef * last_update_value
                self.b_dict[sample_index] = float(self.vade_config.k) * decay_coef * (G - last_update_value)

        method = str(self.vade_config.sample_method)
        for dataset_idx in range(self._dataset_size):
            sample_index = self.dataset_to_sample_map.get(dataset_idx, dataset_idx)
            a = self.a_dict.get(sample_index, 0.0) + 1
            b = self.b_dict.get(sample_index, 0.0) + 1

            if method == "thompson":
                p_hat = self._rng.beta(a / float(self.vade_config.T), b / float(self.vade_config.T))
                weight = self._p_function(p_hat)
            elif method == "UCB":
                if int(self.vade_config.p_function) == 1:
                    weight = self._quantile_y_function1(a, b, float(self.vade_config.alpha))
                elif int(self.vade_config.p_function) == 2:
                    weight = self._quantile_y_function2(a, b, float(self.vade_config.alpha))
                else:
                    raise ValueError("Unsupported p_function form.")
            elif method == "direct":
                p = a / (a + b)
                weight = self._p_function(p)
            else:
                raise ValueError(f"Unsupported Sample Method: {method}")
            weights[dataset_idx] = weight

        self.weights = weights

    def _p_function(self, p: float) -> float:
        if int(self.vade_config.p_function) == 1:
            return float(p * (1 - p))
        if int(self.vade_config.p_function) == 2:
            return float(p * (1 - p) ** 2)
        raise ValueError("Unsupported p_function form.")

    @staticmethod
    def _quantile_y_function1(a: float, b: float, alpha_val: float) -> float:
        from scipy.optimize import brentq
        from scipy.stats import beta

        if alpha_val <= 0:
            return 0.0
        if alpha_val >= 1:
            return 0.25

        def cdf_y(q: float) -> float:
            if q < 0:
                return 0.0
            if q >= 0.25:
                return 1.0
            sqrt_term = np.sqrt(1 - 4 * q)
            p1 = (1 - sqrt_term) / 2
            p2 = (1 + sqrt_term) / 2
            return float(beta.cdf(p1, a, b) + (1 - beta.cdf(p2, a, b)))

        return float(brentq(lambda q: cdf_y(q) - alpha_val, 0.0, 0.25))

    @staticmethod
    def _quantile_y_function2(a: float, b: float, alpha: float) -> float:
        from scipy.optimize import brentq
        from scipy.stats import beta

        def solve_p_one_minus_p_squared(q: float) -> Tuple[float, float]:
            coeffs = [1, -2, 1, -q]
            roots = np.roots(coeffs)
            real_roots = []
            for root in roots:
                if np.isreal(root):
                    root = float(np.real(root))
                    if 0 <= root <= 1:
                        real_roots.append(root)
            real_roots = sorted(real_roots)
            if len(real_roots) < 2:
                return real_roots[0], real_roots[0]
            return real_roots[0], real_roots[-1]

        def cdf_y(q: float) -> float:
            if q <= 0:
                return 0.0
            if q >= 4 / 27:
                return 1.0
            p1, p2 = solve_p_one_minus_p_squared(q)
            return float(beta.cdf(p1, a, b) + (1 - beta.cdf(p2, a, b)))

        if alpha <= 0:
            return 0.0
        if alpha >= 1:
            return 4 / 27
        return float(brentq(lambda q: cdf_y(q) - alpha, 0.0, 4 / 27))

    # ------------------------------------------------------------------
    # Data loading / utility
    # ------------------------------------------------------------------

    def _schedule_active(self) -> bool:
        return bool(self.vade_config.budget_schedule) and self._global_budget_cap() is not None

    def _budget_counted_set(self) -> set[int]:
        """Dataset indices that count against VADE's global training budget."""
        return set(self._ever_selected_set)

    def _global_budget_cap(self) -> Optional[int]:
        if self.config.global_budget_pct is None or self._dataset_size <= 0:
            return None
        return max(1, int(self._dataset_size * float(self.config.global_budget_pct) / 100.0))

    def _get_current_phase(self) -> Tuple[float, int]:
        """Return (per_round_pct, interval) for the active VADE budget phase."""
        schedule = self.vade_config.budget_schedule
        global_cap = self._global_budget_cap()
        if not schedule or global_cap is None:
            return (float(self.config.selection_budget_pct), int(self.config.reselect_interval))

        current_pct = 100.0 * len(self._budget_counted_set()) / global_cap
        for phase in schedule:
            until_budget_pct = float(phase["until_budget_pct"])
            if current_pct < until_budget_pct:
                return (float(phase["per_round_pct"]), int(phase["interval"]))

        last = schedule[-1]
        return (float(last["per_round_pct"]), int(last["interval"]))

    def _top_from_candidates(self, candidates: Iterable[int], target_size: int) -> List[int]:
        if target_size <= 0:
            return []
        clean: List[int] = []
        seen: set[int] = set()
        for raw_idx in candidates:
            idx = int(raw_idx)
            if idx in seen or idx < 0 or idx >= self._dataset_size:
                continue
            seen.add(idx)
            clean.append(idx)
        if not clean:
            return []

        arr = np.asarray(clean, dtype=np.int64)
        order = np.argsort(-self.weights[arr], kind="mergesort")
        return [int(i) for i in arr[order[:target_size]].tolist()]

    def _select_with_global_budget(
        self,
        sorted_indices: List[int],
        target_size: int,
        new_budget: int,
        global_cap: int,
    ) -> Tuple[List[int], List[int]]:
        """Select rows while capping cumulative NEW unique training samples.

        ``new_budget`` controls discovery spend for this round.  ``target_size``
        controls the dataloader size, so rows already inside the admitted pool
        may be reused as no-cost padding.
        """
        if self._selection_frozen:
            pool = self._frozen_pool_indices or sorted(self._ever_selected_set)
            return self._top_from_candidates(pool, target_size), []

        seen_before = set(self._ever_selected_set)
        remaining_cap = max(0, int(global_cap) - len(seen_before))
        round_new_budget = min(max(0, int(new_budget)), remaining_cap)

        selected_indices: List[int] = []
        selected_set: set[int] = set()
        new_indices: List[int] = []

        for idx in sorted_indices:
            if len(new_indices) >= round_new_budget:
                break
            if idx in seen_before:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            new_indices.append(idx)

        for idx in sorted_indices:
            if len(selected_indices) >= target_size:
                break
            if idx in selected_set or idx not in seen_before:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)

        if len(selected_indices) < target_size:
            # Very early/tiny-budget runs may not yet have enough admitted
            # rows to satisfy the training floor.  Admit extra unseen rows only
            # up to the global cap, and count them as budget spend.
            for idx in sorted_indices:
                if len(selected_indices) >= target_size:
                    break
                if idx in selected_set or idx in seen_before:
                    continue
                if len(seen_before) + len(new_indices) >= global_cap:
                    break
                selected_indices.append(idx)
                selected_set.add(idx)
                new_indices.append(idx)

        self._ever_selected_set.update(new_indices)
        if len(self._ever_selected_set) >= global_cap:
            self._selection_frozen = True
            self._frozen_pool_indices = sorted(self._ever_selected_set)
            print(
                f"[VadeSelector] Global budget cap reached: "
                f"{len(self._ever_selected_set)} train samples >= {global_cap} "
                f"({self.config.global_budget_pct}% of {self._dataset_size}). "
                f"Pool frozen; subsequent rounds will reweight within this pool."
            )

        if len(selected_indices) < int(self._min_training_pool_size):
            print(
                f"[VadeSelector] WARNING: selected {len(selected_indices)} rows below "
                f"training floor {self._min_training_pool_size}. Increase "
                f"data_selection.global_budget_pct or train_batch_size for this run."
            )

        return selected_indices, new_indices

    def _validate_config(self) -> None:
        if float(self.vade_config.T) <= 0:
            raise ValueError("data_selection.vade.T must be positive.")
        if int(self.vade_config.G) <= 0:
            raise ValueError("data_selection.vade.G must be positive.")
        if float(self.vade_config.tau) <= 0:
            raise ValueError("data_selection.vade.tau must be positive.")
        if str(self.vade_config.sample_method) not in {"thompson", "UCB", "direct"}:
            raise ValueError("data_selection.vade.sample_method must be thompson, UCB, or direct.")
        if int(self.vade_config.p_function) not in {1, 2}:
            raise ValueError("data_selection.vade.p_function must be 1 or 2.")
        if int(self.vade_config.a_b_update_strategy) not in {1, 2, 3}:
            raise ValueError("data_selection.vade.a_b_update_strategy must be 1, 2, or 3.")
        if self.vade_config.budget_schedule:
            for phase in self.vade_config.budget_schedule:
                for key in ("until_budget_pct", "per_round_pct", "interval"):
                    if key not in phase:
                        raise ValueError(f"data_selection.vade.budget_schedule phase missing {key}.")
                if float(phase["until_budget_pct"]) <= 0:
                    raise ValueError("data_selection.vade.budget_schedule until_budget_pct must be positive.")
                if float(phase["per_round_pct"]) <= 0:
                    raise ValueError("data_selection.vade.budget_schedule per_round_pct must be positive.")
                if int(phase["interval"]) <= 0:
                    raise ValueError("data_selection.vade.budget_schedule interval must be positive.")

    def _build_index_mapping(self) -> None:
        self.sample_to_dataset_map = {}
        self.dataset_to_sample_map = {}
        for dataset_idx in range(self._dataset_size):
            sample_index = int(dataset_idx)
            self.sample_to_dataset_map[sample_index] = int(dataset_idx)
            self.dataset_to_sample_map[int(dataset_idx)] = sample_index
        print(f"[VadeSelector] Built identity dataset_idx<->sample_index map for {self._dataset_size} rows.")

    def _initialize_from_rollout(self, rollout_data_path: str) -> None:
        rollout_data = self._load_rollout_data(rollout_data_path)

        valid_count = 0
        invalid_count = 0
        for idx_str, info in rollout_data.items():
            try:
                sample_index = int(idx_str)
                if sample_index not in self.sample_to_dataset_map:
                    invalid_count += 1
                    continue
                correct_count = float(info.get("correct_count", 0))
                total_rollouts = float(info.get("total_rollouts", self.vade_config.G))
                self.a_dict[sample_index] = correct_count
                self.b_dict[sample_index] = total_rollouts - correct_count
                self.last_updated_values[sample_index] = correct_count
                valid_count += 1
            except (AttributeError, TypeError, ValueError):
                invalid_count += 1

        default_count = 0
        for sample_index in self.sample_to_dataset_map:
            if sample_index not in self.a_dict:
                self.a_dict[sample_index] = 1.0
                self.b_dict[sample_index] = 1.0
                self.last_updated_values[sample_index] = 1.0
                default_count += 1
            self.last_updated_steps[sample_index] = 0

        print(
            f"[VadeSelector] Rollout data loaded: {valid_count} valid, "
            f"{default_count} defaults used, {invalid_count} invalid/out-of-range."
        )

    def _load_rollout_data(self, path: str) -> Dict[str, dict]:
        if not isinstance(path, str):
            raise ValueError("data_selection.vade.initial_rollout_data_path must be a string.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"VADE initial rollout path does not exist: {path}")

        files: List[str]
        if os.path.isdir(path):
            files = []
            for root, _, names in os.walk(path):
                for name in names:
                    if name.endswith((".jsonl", ".json")):
                        files.append(os.path.join(root, name))
            files = sorted(files)
            if not files:
                raise FileNotFoundError(f"No .json/.jsonl rollout files found under {path}")
        else:
            files = [path]

        merged: Dict[str, dict] = {}
        jsonl_files: List[str] = []
        for file_path in files:
            if file_path.endswith(".jsonl"):
                jsonl_files.append(file_path)
                continue
            with open(file_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"VADE JSON rollout file must contain a dict: {file_path}")
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                total_rollouts = int(value.get("total_rollouts", self.vade_config.G))
                correct_count = float(value.get("correct_count", 0))
                if key not in merged:
                    merged[str(key)] = {
                        "total_rollouts": total_rollouts,
                        "correct_count": correct_count,
                    }
                else:
                    merged[str(key)]["total_rollouts"] += total_rollouts
                    merged[str(key)]["correct_count"] += correct_count

        if jsonl_files:
            grouped = self._rollout_jsonl_to_counts(jsonl_files)
            for key, value in grouped.items():
                if key not in merged:
                    merged[key] = value
                else:
                    merged[key]["total_rollouts"] += value["total_rollouts"]
                    merged[key]["correct_count"] += value["correct_count"]

        return merged

    def _rollout_jsonl_to_counts(self, files: Iterable[str]) -> Dict[str, dict]:
        grouped: Dict[str, dict] = {}
        reward_keys = [
            self.vade_config.selection_reward_key,
            "acc",
            "score",
            "reward",
        ]
        threshold = float(self.vade_config.correctness_threshold)

        for file_path in files:
            with open(file_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample_index = (
                        rec.get("sample_index")
                        if rec.get("sample_index") is not None
                        else rec.get("dataset_idx", rec.get("index"))
                    )
                    if sample_index is None:
                        continue
                    reward = None
                    for key in reward_keys:
                        if key in rec and rec[key] is not None:
                            reward = rec[key]
                            break
                    if reward is None:
                        continue
                    try:
                        idx_key = str(int(sample_index))
                        value = float(reward)
                    except (TypeError, ValueError):
                        continue
                    if idx_key not in grouped:
                        grouped[idx_key] = {"total_rollouts": 0, "correct_count": 0.0}
                    grouped[idx_key]["total_rollouts"] += 1
                    grouped[idx_key]["correct_count"] += float(value >= threshold)

        return grouped

    def _log_state(self) -> None:
        if not self.vade_config.log_state_dir:
            return
        os.makedirs(self.vade_config.log_state_dir, exist_ok=True)
        log_path = os.path.join(self.vade_config.log_state_dir, f"global_step_{self.global_steps}.json")
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(self._json_state_dict(), fh, indent=2)

    def _json_state_dict(self) -> dict:
        state = self.get_state_dict().copy()
        state.pop("_rng_state", None)
        return state
