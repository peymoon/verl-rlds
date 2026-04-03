"""
DOTS-style data selector.

Implements the Data-efficient Online Training Selection (DOTS) approach:
  1. Roll out on a random reference set to measure current model capability
  2. Use a few-shot regression teacher to predict difficulty for ALL samples
  3. Sample proportionally to a difficulty target (alpha)
  4. Optional replay buffer to avoid catastrophic forgetting

This is extracted and generalized from the data-efficient-llm-rl codebase
(ray_trainer.py L800-925).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .base import DataSelectionConfig, DataSelector


@dataclass
class DotsConfig:
    """DOTS-specific configuration."""

    ref_size: int = 256
    alpha: float = 0.5
    tau: float = 1e-3
    teacher_checkpoint: Optional[str] = None
    teacher_model_name: Optional[str] = None
    teacher_batch_size: int = 32
    use_replay: bool = False
    replay_sigma: float = 1.0


class DotsSelector(DataSelector):
    """DOTS: predict difficulty for all samples, then sample by target difficulty.

    Steps per selection round:
      1. get_reference_indices() -> random subset of ref_size samples
      2. update_rewards() -> compute mean reward per reference sample
      3. select() -> use teacher to predict difficulty for all, softmax sample
    """

    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)
        dots_cfg = config.dots if isinstance(config.dots, dict) else {}
        self.dots_config = DotsConfig(**dots_cfg)
        self._dataset_size = 0
        self._dataset = None
        self._rng = np.random.RandomState(42)
        self._ref_indices: List[int] = []
        self._ref_mean_rewards: Optional[np.ndarray] = None
        self._teacher_wg = None
        self._last_predicted_labels = None
        self._replay_buffer_indices: List[int] = []

    def initialize(self, dataset, collate_fn=None) -> None:
        self._dataset = dataset
        self._dataset_size = len(dataset)
        print(f"[DotsSelector] Initialized with {self._dataset_size} samples, "
              f"ref_size={self.dots_config.ref_size}, alpha={self.dots_config.alpha}")

    def set_teacher_wg(self, teacher_wg):
        """Set the teacher worker group (called by the trainer after init_workers)."""
        self._teacher_wg = teacher_wg

    def get_reference_indices(self) -> List[int]:
        ref_size = min(self.dots_config.ref_size, self._dataset_size)
        self._ref_indices = self._rng.choice(
            self._dataset_size, size=ref_size, replace=False
        ).tolist()
        return self._ref_indices

    def update_rewards(self, ref_indices, ref_rewards) -> None:
        """Store mean reward per reference sample.

        Args:
            ref_indices: list of dataset indices
            ref_rewards: (n_ref, n_rollouts) array of rewards
        """
        ref_rewards = np.asarray(ref_rewards)
        if ref_rewards.ndim == 2:
            self._ref_mean_rewards = ref_rewards.mean(axis=1)
        else:
            self._ref_mean_rewards = ref_rewards

    def select(self, budget: int) -> List[int]:
        if self._ref_mean_rewards is None or len(self._ref_indices) == 0:
            print("[DotsSelector] No reference data available, falling back to random")
            return self._rng.choice(self._dataset_size, size=min(budget, self._dataset_size),
                                    replace=False).tolist()

        if self._teacher_wg is not None:
            predicted_labels = self._predict_with_teacher()
        else:
            predicted_labels = self._predict_with_interpolation()

        self._last_predicted_labels = predicted_labels

        alpha = self.dots_config.alpha
        tau = self.dots_config.tau

        scores = -torch.abs(predicted_labels - alpha)
        logits = scores / tau
        logits -= logits.max()
        probabilities = torch.softmax(logits, dim=0)

        effective_budget = budget
        if self.dots_config.use_replay and len(self._replay_buffer_indices) > 0:
            replay_size = int(budget * self.dots_config.replay_sigma)
            effective_budget = budget + replay_size

        effective_budget = min(effective_budget, self._dataset_size)
        selected = torch.multinomial(probabilities, effective_budget, replacement=False)
        selected = selected[torch.randperm(len(selected))]
        selected_indices = selected.tolist()

        solve_none = int((self._ref_mean_rewards == 0).sum())
        solve_all = int((self._ref_mean_rewards == 1).sum())
        informative = len(self._ref_mean_rewards) - solve_none - solve_all
        print(f"[DotsSelector] Reference signal: {solve_none} unsolvable, "
              f"{solve_all} trivial, {informative} informative out of {len(self._ref_indices)}")
        print(f"[DotsSelector] Selected {len(selected_indices)} samples "
              f"(alpha={alpha}, tau={tau})")

        return selected_indices[:budget]

    def _predict_with_teacher(self) -> torch.Tensor:
        """Use the teacher model to predict difficulty for all samples."""
        all_questions = []
        for i in range(self._dataset_size):
            item = self._dataset[i]
            if isinstance(item, dict):
                q = item.get("extra_info", {}).get("question", "")
                if not q:
                    q = item.get("question", f"sample_{i}")
            else:
                q = f"sample_{i}"
            all_questions.append(q)

        ref_questions = [all_questions[i] for i in self._ref_indices]
        ref_labels = self._ref_mean_rewards.tolist()

        predicted = self._teacher_wg.predict(
            all_questions, ref_questions, ref_labels,
            self._ref_indices,
            batch_size=self.dots_config.teacher_batch_size,
        )
        if not isinstance(predicted, torch.Tensor):
            predicted = torch.tensor(predicted, dtype=torch.float32)
        return predicted

    def _predict_with_interpolation(self) -> torch.Tensor:
        """Fallback: use simple interpolation when no teacher is available.

        Uses the reference rewards directly — samples close to reference
        samples (by index) get similar predicted difficulty.
        """
        predicted = torch.full((self._dataset_size,), 0.5, dtype=torch.float32)
        for idx, reward in zip(self._ref_indices, self._ref_mean_rewards):
            predicted[idx] = float(reward)
        return predicted

    def get_metrics(self) -> Dict[str, float]:
        metrics = {"data_selection/method": 1.0}
        if self._ref_mean_rewards is not None:
            metrics["data_selection/ref_mean_reward"] = float(self._ref_mean_rewards.mean())
            metrics["data_selection/ref_std_reward"] = float(self._ref_mean_rewards.std())
            metrics["data_selection/ref_solve_none"] = int((self._ref_mean_rewards == 0).sum())
            metrics["data_selection/ref_solve_all"] = int((self._ref_mean_rewards == 1).sum())
        if self._last_predicted_labels is not None:
            metrics["data_selection/predicted_mean"] = float(self._last_predicted_labels.mean())
            metrics["data_selection/predicted_std"] = float(self._last_predicted_labels.std())
        return metrics
