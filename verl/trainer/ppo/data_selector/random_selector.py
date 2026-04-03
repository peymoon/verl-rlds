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
        self._rng = np.random.RandomState(42)

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
        print(f"[RandomSelector] Selected {len(indices)} random samples")
        return indices

    def get_metrics(self) -> Dict[str, float]:
        return {"data_selection/method": 0.0}
