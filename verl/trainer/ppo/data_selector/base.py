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

    # Cap on cumulative unique samples seen over the entire run.
    # When set, the selector freezes the pool once the union of all selected
    # samples reaches this percentage of the full dataset.  After freeze:
    #   - Reference and exploration rollouts are skipped (saves compute)
    #   - Training-batch rollout rewards continue accumulating (if use_rollout_history)
    #   - Periodic reselection rounds reweight within the frozen pool based on
    #     accumulated per-sample variance (high-variance samples appear more often)
    # Set equal to selection_budget_pct for a fair comparison with a fixed
    # random baseline (e.g., both at 10% → exactly 10% unique samples).
    global_budget_pct: Optional[float] = None

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
        self._selection_frozen = False
        # Minimum number of training samples the trainer must receive per
        # selection round, regardless of per-round annotation budget. The
        # trainer sets this to `train_batch_size` so that a dataloader
        # rebuilt with `drop_last=True` is never empty. The selector is
        # expected to pad below-floor selections from its already-annotated
        # pool (reusing samples already counted against the global budget).
        self._min_training_pool_size: int = 0

    def set_min_training_pool_size(self, n: int) -> None:
        self._min_training_pool_size = int(max(0, n))

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
        """True at epoch start when using reselect_schedule='epoch'.

        Still fires when the pool is frozen (for variance-weighted resampling
        within the frozen pool).  Reference rollouts are skipped by the
        selector's get_reference_indices() returning [].
        """
        if self.config.reselect_schedule != "epoch":
            return False
        if self.config.reselect_interval <= 0:
            return False
        return epoch % self.config.reselect_interval == 0

    def should_reselect_step(self, global_step: int) -> bool:
        """True after a training step when using reselect_schedule='step'.

        Called with the trainer's global_steps value *after* that step's update
        (i.e. after increment). Reselect when global_step % interval == 0.

        Still fires when the pool is frozen (for variance-weighted resampling
        within the frozen pool).  Reference rollouts are skipped by the
        selector's get_reference_indices() returning [].
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
