"""
Abstract base class for online data selection during GRPO/PPO training.

The training loop calls these methods in order:
  1. initialize(dataset)         — once, at training start
  2. get_reference_indices()     — which prompts to roll out for the capability signal
  3. update_rewards(...)         — feed back the rollout results
  4. select(budget)              — return dataset indices for the next training window
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DataSelectionConfig:
    """Top-level config for data selection, shared by all methods."""

    method: str = "none"
    # "epoch": reselect_interval = every N epochs (at epoch start).
    # "step": reselect_interval = every N training steps (mid-epoch ok; dataloader iterator is refreshed).
    reselect_schedule: str = "epoch"
    reselect_interval: int = 1
    selection_budget: Optional[int] = None
    selection_budget_pct: float = 100.0

    # Cumulative budget cap: limit total UNIQUE samples used across ALL
    # selection rounds.  When set (not None), the selector will track every
    # sample that has ever been selected and refuse to introduce new samples
    # once the cumulative unique count reaches this percentage of the dataset.
    # The per-round budget (selection_budget / selection_budget_pct) controls
    # how many samples are used in each training window; this controls the
    # global cap on unique samples across the entire training run.
    # Example: selection_budget_pct=10, cumulative_budget_pct=10 → the first
    # round picks the best 10%, then all subsequent rounds train on subsets
    # of those same samples (no new data introduced).
    # Example: selection_budget_pct=5, cumulative_budget_pct=10 → the pool
    # grows across rounds up to 10%, with each round using 5%.
    cumulative_budget_pct: Optional[float] = None

    cluster: dict = field(default_factory=dict)
    dots: dict = field(default_factory=dict)


class DataSelector(ABC):
    """Abstract interface for online data selection during RL training.

    The training loop calls:
      1. initialize(dataset)         — once before training starts
      2. get_reference_indices()     — which prompts to roll out for the signal
      3. update_rewards(ref_indices, ref_rewards)  — feed back results
      4. select(budget)              — return training indices
    """

    def __init__(self, config: DataSelectionConfig):
        self.config = config
        self._step_count = 0

    @abstractmethod
    def initialize(self, dataset, collate_fn=None) -> None:
        """Called once before training starts.

        Use for expensive one-time setup: computing embeddings, clustering,
        loading teacher models, etc.
        """

    @abstractmethod
    def get_reference_indices(self) -> List[int]:
        """Return the indices of samples to roll out for the capability signal.

        Called before each select() call so the trainer knows what to generate.
        Returns empty list if no reference rollouts are needed.
        """

    @abstractmethod
    def update_rewards(
        self,
        ref_indices: List[int],
        ref_rewards: np.ndarray,
    ) -> None:
        """Feed back the rollout results for the reference samples.

        Args:
            ref_indices: which dataset indices were rolled out
            ref_rewards: shape (n_ref, n_rollouts) — raw reward per rollout
        """

    @abstractmethod
    def select(self, budget: int) -> List[int]:
        """Return dataset indices for the next training window.

        Args:
            budget: how many training samples to return

        Returns:
            List of dataset indices to use for the next training window.
        """

    def should_reselect_epoch(self, epoch: int) -> bool:
        """True at epoch start when using reselect_schedule='epoch'."""
        if self.config.reselect_schedule != "epoch":
            return False
        if self.config.reselect_interval <= 0:
            return False
        return epoch % self.config.reselect_interval == 0

    def should_reselect_step(self, global_step: int) -> bool:
        """True after a training step when using reselect_schedule='step'.

        Called with the trainer's global_steps value *after* that step's update
        (i.e. after increment). Reselect when global_step % interval == 0.
        """
        if self.config.reselect_schedule != "step":
            return False
        if self.config.reselect_interval <= 0:
            return False
        return global_step % self.config.reselect_interval == 0

    def should_reselect(self, epoch: int) -> bool:
        """Backward-compatible alias for should_reselect_epoch."""
        return self.should_reselect_epoch(epoch)

    def get_selection_budget(self, dataset_size: int) -> int:
        """Compute the selection budget from config."""
        if self.config.selection_budget is not None:
            return min(self.config.selection_budget, dataset_size)
        return max(1, int(dataset_size * self.config.selection_budget_pct / 100.0))

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics about the last selection for logging."""
        return {}
