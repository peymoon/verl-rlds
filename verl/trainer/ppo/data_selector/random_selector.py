"""
Random data selector — baseline that selects training samples uniformly at random.

No reference rollouts are needed; select() just returns a random subset.
"""

from typing import Dict, List

import numpy as np

from .base import DataSelectionConfig, DataSelector


class RandomSelector(DataSelector):
    """Trivial baseline: uniform random sampling each selection round."""

    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)
        self._dataset_size = 0
        self._rng = np.random.RandomState(int(config.seed))
        self._selection_round = 0

    def initialize(self, dataset, collate_fn=None) -> None:
        self._dataset_size = len(dataset)
        print(f"[RandomSelector] Initialized with {self._dataset_size} samples")

    def get_reference_indices(self) -> List[int]:
        return []

    def update_rewards(self, ref_indices, ref_rewards) -> None:
        pass

    def select(self, budget: int) -> List[int]:
        budget = min(budget, self._dataset_size)
        indices = self._rng.choice(self._dataset_size, size=budget, replace=False).tolist()
        self._last_selected_indices = [int(i) for i in indices]
        self._selection_round += 1
        print(f"[RandomSelector] Selected {len(indices)} random samples")
        return list(self._last_selected_indices)

    def get_state_dict(self) -> dict:
        """Serialize enough state to resume the exact subset and RNG stream."""
        return {
            "state_version": 2,
            "_dataset_size": self._dataset_size,
            "_step_count": self._step_count,
            "_selection_frozen": self._selection_frozen,
            "_selection_round": self._selection_round,
            "_last_selected_indices": list(self._last_selected_indices),
            "_rng_state": self._rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_size = state.get("_dataset_size")
        if saved_size is not None and int(saved_size) != self._dataset_size:
            raise RuntimeError(
                "RandomSelector checkpoint dataset size mismatch: "
                f"checkpoint={int(saved_size)}, current={self._dataset_size}."
            )

        self._step_count = int(state.get("_step_count", 0))
        self._selection_frozen = bool(state.get("_selection_frozen", False))
        self._selection_round = int(state.get("_selection_round", 0))
        self._last_selected_indices = [
            int(i) for i in state.get("_last_selected_indices", [])
        ]
        if "_rng_state" in state:
            self._rng.set_state(state["_rng_state"])

    def get_metrics(self) -> Dict[str, float]:
        return {"data_selection/method": 0.0}
