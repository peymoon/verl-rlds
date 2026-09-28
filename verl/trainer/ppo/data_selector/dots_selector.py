"""
DOTS data selector adapted to the multimodal VERL data path.

This ports the runtime algorithm from data-efficient-llm-rl:

  1. sample a random reference set
  2. roll out the current policy on those references
  3. predict every prompt's current solve probability from cached embeddings
  4. sample prompts with probability softmax(-|p_hat - alpha| / tau)

The original repo used a Ray teacher worker over text-only questions.  This
version keeps the same learned few-shot regressor and sampling rule, but uses
cached multimodal embeddings plus NPZ<->parquet alignment so the selected rows
can still flow through Qwen-VL image generation.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import DataSelectionConfig, DataSelector


def _build_projection(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    use_layernorm: bool = True,
    dropout: float = 0.1,
) -> nn.Sequential:
    """Projection block copied from the DOTS teacher implementation."""
    layers: List[nn.Module] = []
    if use_layernorm:
        layers.append(nn.LayerNorm(input_size))

    for i in range(num_layers):
        layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        if num_layers > 1 and i < num_layers - 1:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

    layers.append(nn.LayerNorm(hidden_size))
    return nn.Sequential(*layers)


class _DotsRegressionHead(nn.Module):
    """Embedding attention head from data-efficient-llm-rl."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, top_k: Optional[int] = None):
        super().__init__()
        self.top_k = top_k
        self.query_proj = _build_projection(input_size, hidden_size, num_layers=num_layers)
        self.ref_proj = _build_projection(input_size, hidden_size, num_layers=num_layers)

    def forward(self, query_repr: torch.Tensor, ref_repr: torch.Tensor, ref_values: torch.Tensor, tau: float = 1.0):
        q_proj = self.query_proj(query_repr)
        r_proj = self.ref_proj(ref_repr)
        scores = torch.bmm(r_proj, q_proj.unsqueeze(-1)).squeeze(-1) / math.sqrt(r_proj.size(-1))

        if self.top_k is not None and 0 < self.top_k < scores.size(1):
            vals, idx = torch.topk(scores, self.top_k, dim=1)
            mask = scores.new_full(scores.shape, float("-inf"))
            mask.scatter_(1, idx, vals)
            scores = mask

        weights = scores / tau
        weights -= torch.max(weights, dim=-1, keepdim=True).values
        weights = F.softmax(weights, dim=-1)
        return (weights * ref_values).sum(dim=-1)


class _DotsResidualHead(nn.Module):
    """Residual calibration head compatible with both DOTS checkpoint variants."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        scaling: str = "group_logit_temp",
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.sim_head = _DotsRegressionHead(input_size, hidden_size, num_layers=num_layers, top_k=top_k)
        self.scaling = scaling

        if scaling == "platt":
            self.res_scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones(1))
        elif scaling == "temperature":
            self.scale = nn.Parameter(torch.ones(1))
        elif scaling == "group_logit_temp":
            self.mlp = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
        elif scaling == "group_logit_temp_dual":
            # Older released DOTS checkpoints used two scalar MLPs named
            # mlp1/mlp2 rather than one 2-output MLP.
            self.mlp1 = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
            self.mlp2 = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
        elif scaling == "plain":
            pass
        else:
            raise ValueError(f"Invalid DOTS scaling method: {scaling}")

    def forward(self, q: torch.Tensor, r: torch.Tensor, ref_vals: torch.Tensor, tau: float = 1.0):
        base = self.sim_head(q, r, ref_vals, tau=tau)
        base_logit = torch.logit(base.clamp(1e-4, 1 - 1e-4))

        if self.scaling == "platt":
            return torch.sigmoid(self.scale * base_logit + self.res_scale)
        if self.scaling == "temperature":
            return torch.sigmoid(base_logit / self.scale)
        if self.scaling in {"group_logit_temp", "group_logit_temp_dual"}:
            mean_vec = torch.mean(ref_vals, dim=-1, keepdim=True)
            std_vec = torch.std(ref_vals, dim=-1, keepdim=True)
            concat_vec = torch.cat((mean_vec, std_vec), dim=-1)
            if self.scaling == "group_logit_temp":
                temp_bias = self.mlp(concat_vec)
                temp = F.softplus(temp_bias[:, 0])
                bias = torch.tanh(temp_bias[:, 1])
            else:
                temp = F.softplus(self.mlp1(concat_vec).squeeze(-1))
                bias = torch.tanh(self.mlp2(concat_vec).squeeze(-1))
            return torch.sigmoid(base_logit / temp.clamp(1e-2, 10) + bias)
        return base


class _DotsFewShotRegressor(nn.Module):
    """Few-shot embedding regressor with the same parameter names as DOTS."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        scaling: str = "group_logit_temp",
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.regressor = _DotsResidualHead(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            scaling=scaling,
            top_k=top_k,
        )

    def forward(
        self,
        query_input: torch.Tensor,
        ref_input: torch.Tensor,
        ref_values: torch.Tensor,
        tau: float = 1.0,
    ):
        batch_size = query_input.size(0)
        n_refs = ref_values.size(1)
        ref_repr = ref_input.unsqueeze(0).expand(batch_size, n_refs, -1)
        return self.regressor(query_input, ref_repr, ref_values, tau=tau)


@dataclass
class DotsConfig:
    """DOTS-specific configuration."""

    # Data / alignment.
    embeddings_file: Optional[str] = None
    cluster_arrays_file: Optional[str] = None
    dataset_json_file: Optional[str] = None
    normalize_embeddings: bool = False

    # DOTS sampling rule.
    ref_size: int = 256
    alpha: float = 0.5
    tau: float = 1e-3
    seed: int = 42
    exclude_already_selected: bool = False
    skip_reference_when_frozen: bool = True
    # Fair budget-matched DOTS-MM: count reference probes as verified prompts.
    # Original DOTS usually treats the teacher/reference signal as extra
    # machinery; for our unique-prompt-budget comparison, this should be true.
    count_references_in_budget: bool = False

    # Teacher / predictor.
    teacher_checkpoint: Optional[str] = None
    teacher_model_name: Optional[str] = None
    teacher_batch_size: int = 32
    teacher_device: str = "auto"
    teacher_hidden_size: int = 896
    teacher_num_layers: int = 3
    teacher_scaling: str = "group_logit_temp"
    teacher_top_k: Optional[int] = None
    teacher_tau: float = 1.0
    calibrate_predictions: bool = True
    clip_predictions: bool = False
    require_teacher_checkpoint: bool = False
    fallback_predictor: str = "identity_attention"

    # Trainer reward stream.  The trainer reads this when available.
    selection_reward_key: str = "reward"

    # Kept for backwards compatibility with the previous scaffold.
    use_replay: bool = False
    replay_sigma: float = 1.0


class DotsSelector(DataSelector):
    """Exact DOTS scorer/sampler with multimodal dataset indexing."""

    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)
        dots_cfg = config.dots if isinstance(config.dots, dict) else {}
        self.dots_config = DotsConfig(**dots_cfg)

        self._dataset = None
        self._dataset_size = 0
        self._embeddings_np: Optional[np.ndarray] = None
        self._embedding_tensor: Optional[torch.Tensor] = None
        self._embedding_dim = 0

        self._uid_to_npz: Dict[str, int] = {}
        self._npz_to_dataset: Optional[np.ndarray] = None
        self._dataset_to_npz: Optional[Dict[int, int]] = None
        self._valid_dataset_indices: List[int] = []

        self._rng = np.random.RandomState(self.dots_config.seed)
        self._selection_round = 0
        self._ref_indices: List[int] = []
        self._ref_mean_rewards: Optional[np.ndarray] = None
        self._teacher_model: Optional[nn.Module] = None
        self._teacher_loaded = False
        self._teacher_arch: Dict[str, object] = {}
        self._last_predicted_labels: Optional[torch.Tensor] = None
        self._last_probabilities: Optional[torch.Tensor] = None
        self._last_selected_indices: List[int] = []
        self._last_calibration: Tuple[float, float] = (1.0, 0.0)
        self._last_prediction_mode = "uninitialized"

        # The trainer checks for this attribute when resuming.  For matched
        # DOTS-MM, this is the budget-counted verified set: training selections
        # plus, optionally, reference probes.  Keep a separate training-only
        # set so metrics make the accounting explicit.
        self._ever_selected_set: set[int] = set()
        self._train_selected_set: set[int] = set()
        self._reference_seen_set: set[int] = set()
        self._frozen_pool: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Lifecycle / state
    # ------------------------------------------------------------------

    def initialize(self, dataset, collate_fn=None) -> None:
        self._dataset = dataset
        self._dataset_size = len(dataset)
        self._load_embeddings()
        self._build_alignment(dataset, self.dots_config.dataset_json_file)
        self._load_teacher_model()

        mode = "learned_teacher" if self._teacher_loaded else self.dots_config.fallback_predictor
        print(
            f"[DotsSelector] Ready: dataset={self._dataset_size}, "
            f"embeddings={len(self._embeddings_np)}, dim={self._embedding_dim}, "
            f"valid_rows={len(self._valid_dataset_indices)}, ref_size={self.dots_config.ref_size}, "
            f"alpha={self.dots_config.alpha}, selection_tau={self.dots_config.tau}, mode={mode}"
        )

    def get_state_dict(self) -> dict:
        return {
            "_selection_round": self._selection_round,
            "_ref_indices": self._ref_indices,
            "_ref_mean_rewards": self._ref_mean_rewards,
            "_last_predicted_labels": self._last_predicted_labels,
            "_last_selected_indices": self._last_selected_indices,
            "_last_calibration": self._last_calibration,
            "_last_prediction_mode": self._last_prediction_mode,
            "_ever_selected_set": sorted(self._ever_selected_set),
            "_train_selected_set": sorted(self._train_selected_set),
            "_reference_seen_set": sorted(self._reference_seen_set),
            "_frozen_pool": self._frozen_pool,
            "_selection_frozen": self._selection_frozen,
            "_rng_state": self._rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._selection_round = int(state.get("_selection_round", 0))
        self._ref_indices = list(state.get("_ref_indices", []))
        self._ref_mean_rewards = state.get("_ref_mean_rewards", None)
        self._last_predicted_labels = state.get("_last_predicted_labels", None)
        self._last_selected_indices = list(state.get("_last_selected_indices", []))
        self._last_calibration = tuple(state.get("_last_calibration", (1.0, 0.0)))
        self._last_prediction_mode = state.get("_last_prediction_mode", "uninitialized")
        self._ever_selected_set = set(int(i) for i in state.get("_ever_selected_set", []))
        self._train_selected_set = set(
            int(i) for i in state.get("_train_selected_set", state.get("_ever_selected_set", []))
        )
        self._reference_seen_set = set(int(i) for i in state.get("_reference_seen_set", []))
        frozen_pool = state.get("_frozen_pool", None)
        self._frozen_pool = None if frozen_pool is None else [int(i) for i in frozen_pool]
        self._selection_frozen = bool(state.get("_selection_frozen", False))
        if "_rng_state" in state:
            self._rng.set_state(state["_rng_state"])
        print(
            f"[DotsSelector] Restored state: round={self._selection_round}, "
            f"verified_uniques={len(self._ever_selected_set)}, "
            f"train_uniques={len(self._train_selected_set)}, "
            f"frozen={self._selection_frozen}"
        )

    # ------------------------------------------------------------------
    # Trainer API
    # ------------------------------------------------------------------

    def get_reference_indices(self) -> List[int]:
        if self._selection_frozen and self.dots_config.skip_reference_when_frozen:
            print("[DotsSelector] Pool frozen; skipping reference rollouts.")
            self._ref_indices = []
            return []

        candidates = self._valid_dataset_indices or list(range(self._dataset_size))
        if self.dots_config.count_references_in_budget:
            if self.config.global_budget_pct is not None:
                global_max = self._global_budget_count()
                remaining = max(0, global_max - len(self._ever_selected_set))
                if remaining <= 0:
                    self._selection_frozen = True
                    self._frozen_pool = sorted(self._ever_selected_set)
                    print("[DotsSelector] Global budget already exhausted; skipping reference rollouts.")
                    self._ref_indices = []
                    return []
                ref_size = min(int(self.dots_config.ref_size), remaining)
            else:
                ref_size = int(self.dots_config.ref_size)
            if self.dots_config.exclude_already_selected and self._ever_selected_set:
                seen = self._ever_selected_set
                candidates = [i for i in candidates if int(i) not in seen]
        else:
            ref_size = int(self.dots_config.ref_size)

        ref_size = min(ref_size, len(candidates))
        if ref_size <= 0:
            self._ref_indices = []
            return []

        chosen = self._rng.choice(np.asarray(candidates, dtype=np.int64), size=ref_size, replace=False)
        self._ref_indices = [int(i) for i in chosen.tolist()]
        self._reference_seen_set.update(self._ref_indices)
        if self.dots_config.count_references_in_budget:
            self._ever_selected_set.update(self._ref_indices)
            if self.config.global_budget_pct is not None:
                global_max = self._global_budget_count()
                if len(self._ever_selected_set) >= global_max:
                    self._selection_frozen = True
                    self._frozen_pool = sorted(self._ever_selected_set)
                    print(
                        f"[DotsSelector] Global unique budget reached by reference probes: "
                        f"{len(self._ever_selected_set)}/{global_max}. Freezing DOTS pool after this signal."
                    )
        return self._ref_indices

    def update_rewards(self, ref_indices: Sequence[int], ref_rewards: np.ndarray) -> None:
        ref_rewards = np.asarray(ref_rewards, dtype=np.float32)
        if ref_rewards.ndim == 2:
            means = ref_rewards.mean(axis=1)
        else:
            means = ref_rewards.reshape(-1)

        if len(means) != len(ref_indices):
            n = min(len(means), len(ref_indices))
            print(
                f"[DotsSelector] WARNING: got {len(means)} reward rows for "
                f"{len(ref_indices)} references; truncating to {n}."
            )
            self._ref_indices = [int(i) for i in ref_indices[:n]]
            means = means[:n]
        else:
            self._ref_indices = [int(i) for i in ref_indices]

        self._ref_mean_rewards = means.astype(np.float32)

    def select(self, budget: int) -> List[int]:
        if self._dataset_size <= 0:
            return []

        if self._ref_mean_rewards is None or len(self._ref_indices) == 0:
            if self._last_predicted_labels is not None:
                predicted_labels = self._last_predicted_labels
                print("[DotsSelector] Reusing previous predictions (no fresh reference data).")
            else:
                print("[DotsSelector] No reference signal yet; falling back to random selection.")
                return self._random_selection(budget)
        else:
            predicted_labels = self._predict_all()

        self._last_predicted_labels = predicted_labels.detach().cpu().float()

        probabilities = self._dots_probabilities(self._last_predicted_labels)
        target_size = min(self._dataset_size, max(int(budget), int(self._min_training_pool_size)))
        selected_indices = self._sample_with_budget(probabilities, int(budget), target_size)
        self._last_probabilities = probabilities.detach().cpu()
        self._last_selected_indices = selected_indices
        self._selection_round += 1

        self._print_round_summary(selected_indices, budget, target_size)
        return selected_indices

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "data_selection/dots_round": float(self._selection_round),
            "data_selection/dots_teacher_loaded": float(self._teacher_loaded),
            "data_selection/dots_verified_uniques": float(len(self._ever_selected_set)),
            "data_selection/dots_train_uniques": float(len(self._train_selected_set)),
            # Backward-compatible name; now explicitly mirrors train_uniques.
            "data_selection/dots_selected_uniques": float(len(self._train_selected_set)),
            "data_selection/dots_reference_uniques": float(len(self._reference_seen_set)),
            "data_selection/dots_count_references_in_budget": float(self.dots_config.count_references_in_budget),
            "data_selection/dots_frozen": float(self._selection_frozen),
        }

        if self.config.global_budget_pct is not None and self._dataset_size > 0:
            global_max = self._global_budget_count()
            metrics["data_selection/dots_global_budget_count"] = float(global_max)
            metrics["data_selection/dots_global_budget_used_pct"] = (
                100.0 * len(self._ever_selected_set) / max(1, global_max)
            )

        if self._ref_mean_rewards is not None and len(self._ref_mean_rewards) > 0:
            ref = self._ref_mean_rewards
            metrics.update({
                "data_selection/ref_mean_reward": float(ref.mean()),
                "data_selection/ref_std_reward": float(ref.std()),
                "data_selection/ref_solve_none": float((ref == 0).sum()),
                "data_selection/ref_solve_all": float((ref == 1).sum()),
            })

        if self._last_predicted_labels is not None:
            preds = self._last_predicted_labels.float()
            finite = preds[torch.isfinite(preds)]
            if len(finite) > 0:
                metrics.update({
                    "data_selection/predicted_mean": float(finite.mean()),
                    "data_selection/predicted_std": float(finite.std(unbiased=False)),
                    "data_selection/predicted_min": float(finite.min()),
                    "data_selection/predicted_max": float(finite.max()),
                    "data_selection/dots_calibration_slope": float(self._last_calibration[0]),
                    "data_selection/dots_calibration_intercept": float(self._last_calibration[1]),
                })

        if self._last_probabilities is not None:
            probs = self._last_probabilities.float()
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
            metrics["data_selection/dots_sampling_entropy"] = float(entropy)

        if self._last_selected_indices and self._last_predicted_labels is not None:
            idx = torch.tensor(self._last_selected_indices, dtype=torch.long)
            vals = self._last_predicted_labels.index_select(0, idx)
            metrics["data_selection/dots_selected_pred_mean"] = float(vals.mean())
            metrics["data_selection/dots_selected_target_abs_err"] = float(
                torch.abs(vals - float(self.dots_config.alpha)).mean()
            )

        return metrics

    # ------------------------------------------------------------------
    # Embeddings / alignment
    # ------------------------------------------------------------------

    def _load_embeddings(self) -> None:
        path = self.dots_config.embeddings_file or self.dots_config.cluster_arrays_file
        if not path:
            raise ValueError(
                "DotsSelector requires data_selection.dots.embeddings_file "
                "or data_selection.dots.cluster_arrays_file."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"DOTS embeddings file not found: {path}")

        print(f"[DotsSelector] Loading embeddings from {path}")
        data = np.load(path, allow_pickle=True)
        if "embeddings" not in data:
            raise KeyError(f"DOTS embeddings file has keys {data.files}, expected 'embeddings'.")

        embeddings = data["embeddings"].astype(np.float32)
        if self.dots_config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = (embeddings / norms).astype(np.float32)

        self._embeddings_np = embeddings
        self._embedding_tensor = torch.from_numpy(embeddings)
        self._embedding_dim = int(embeddings.shape[1])

        for key in ("uids", "ids", "qids", "qid"):
            if key in data:
                raw_ids = data[key]
                self._uid_to_npz = {str(uid): i for i, uid in enumerate(raw_ids)}
                print(f"[DotsSelector] Built id->NPZ map from key='{key}' ({len(self._uid_to_npz)} entries).")
                break

    def _build_alignment(self, dataset, json_path: Optional[str]) -> None:
        """Build maps between embedding NPZ rows and training parquet rows."""
        n_npz = len(self._embeddings_np)
        parquet_qid_to_idx: Dict[str, int] = {}
        parquet_img_to_idx: Dict[str, int] = {}

        for parquet_i in range(len(dataset)):
            item = self._get_raw_dataset_item(dataset, parquet_i)
            extra = self._extra_info(item)
            qid = str(extra.get("qid", "") or "")
            img = str(extra.get("image", "") or extra.get("image_rel", "") or "")
            if qid:
                parquet_qid_to_idx[qid] = parquet_i
            if img:
                parquet_img_to_idx[img] = parquet_i

        json_qid_to_img: Dict[str, str] = {}
        json_image_paths_positional: List[str] = []
        if json_path:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"DOTS dataset_json_file not found: {json_path}")
            with open(json_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        rec = {}
                    qid = str(rec.get("qid", "") or "")
                    img = str(rec.get("image", "") or rec.get("image_rel", "") or "")
                    if qid:
                        json_qid_to_img[qid] = img
                    json_image_paths_positional.append(img)

        npz_to_dataset = np.full(n_npz, -1, dtype=np.int64)
        dataset_to_npz: Dict[int, int] = {}
        n_matched = 0
        method = "none"

        if self._uid_to_npz and parquet_qid_to_idx:
            method = "qid"
            matched_dataset_rows = set()
            for qid, npz_i in self._uid_to_npz.items():
                parquet_i = parquet_qid_to_idx.get(qid, -1)
                if 0 <= npz_i < n_npz:
                    npz_to_dataset[npz_i] = parquet_i
                    if parquet_i >= 0:
                        dataset_to_npz[parquet_i] = npz_i
                        matched_dataset_rows.add(parquet_i)
                        n_matched += 1
            if json_qid_to_img and len(matched_dataset_rows) < len(dataset):
                method = "qid+image"
                for qid, npz_i in self._uid_to_npz.items():
                    if not (0 <= npz_i < n_npz) or npz_to_dataset[npz_i] >= 0:
                        continue
                    img = json_qid_to_img.get(qid, "")
                    parquet_i = parquet_img_to_idx.get(img, -1)
                    if parquet_i >= 0 and parquet_i not in matched_dataset_rows:
                        npz_to_dataset[npz_i] = parquet_i
                        dataset_to_npz[parquet_i] = npz_i
                        matched_dataset_rows.add(parquet_i)
                        n_matched += 1
        elif json_image_paths_positional and parquet_img_to_idx:
            method = "positional-image-path"
            for npz_i, img in enumerate(json_image_paths_positional[:n_npz]):
                parquet_i = parquet_img_to_idx.get(img, -1)
                npz_to_dataset[npz_i] = parquet_i
                if parquet_i >= 0:
                    dataset_to_npz[parquet_i] = npz_i
                    n_matched += 1

        if n_matched == 0:
            n_direct = min(n_npz, len(dataset))
            method = "direct-row-order"
            npz_to_dataset[:n_direct] = np.arange(n_direct, dtype=np.int64)
            dataset_to_npz = {i: i for i in range(n_direct)}
            n_matched = n_direct

        self._npz_to_dataset = npz_to_dataset
        self._dataset_to_npz = dataset_to_npz
        self._valid_dataset_indices = sorted(dataset_to_npz.keys())
        print(
            f"[DotsSelector] Alignment ({method}): {n_matched}/{n_npz} embeddings "
            f"matched to {len(dataset)} parquet rows."
        )

    @staticmethod
    def _get_raw_dataset_item(dataset, idx: int):
        dataframe = getattr(dataset, "dataframe", None)
        if dataframe is not None:
            try:
                if hasattr(dataframe, "iloc"):
                    row = dataframe.iloc[idx]
                    return row.to_dict() if hasattr(row, "to_dict") else row
                return dataframe[idx]
            except Exception:
                pass
        try:
            return dataset[idx]
        except Exception:
            return {}

    @staticmethod
    def _extra_info(item) -> Dict[str, object]:
        if not isinstance(item, dict):
            return {}
        extra = item.get("extra_info") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        return extra if isinstance(extra, dict) else {}

    def _dataset_index_to_npz(self, dataset_idx: int) -> Optional[int]:
        if self._dataset_to_npz is None:
            if 0 <= dataset_idx < len(self._embeddings_np):
                return int(dataset_idx)
            return None
        npz_idx = self._dataset_to_npz.get(int(dataset_idx), -1)
        return int(npz_idx) if npz_idx >= 0 else None

    # ------------------------------------------------------------------
    # Teacher inference
    # ------------------------------------------------------------------

    def _load_teacher_model(self) -> None:
        ckpt = self.dots_config.teacher_checkpoint
        if ckpt in {"", "null", "None"}:
            ckpt = None

        if not ckpt:
            if self.dots_config.require_teacher_checkpoint:
                raise ValueError("DOTS require_teacher_checkpoint=true but teacher_checkpoint is empty.")
            print(
                "[DotsSelector] No DOTS teacher checkpoint supplied; using "
                f"fallback_predictor={self.dots_config.fallback_predictor!r}."
            )
            return

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"DOTS teacher checkpoint not found: {ckpt}")

        raw_state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state_dict = raw_state.get("model_state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
        state_dict = {self._strip_module_prefix(k): v for k, v in state_dict.items()}
        state_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith("regressor.")
        }
        if not state_dict:
            raise ValueError(f"DOTS checkpoint {ckpt} did not contain regressor.* weights.")

        input_size, hidden_size, num_layers, scaling = self._infer_checkpoint_arch(state_dict)
        if input_size != self._embedding_dim:
            raise ValueError(
                f"DOTS teacher checkpoint input dim is {input_size}, but embeddings dim is "
                f"{self._embedding_dim}. For Qwen3-VL VIRL embeddings, train/export a "
                "DOTS teacher on the same embeddings, or leave teacher_checkpoint=null "
                "to run the identity_attention fallback baseline."
            )

        model = _DotsFewShotRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            scaling=scaling,
            top_k=self.dots_config.teacher_top_k,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        device = self._resolve_teacher_device()
        model.to(device)

        self._teacher_model = model
        self._teacher_loaded = True
        self._teacher_arch = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "scaling": scaling,
            "device": str(device),
        }
        print(f"[DotsSelector] Loaded DOTS teacher from {ckpt}: {self._teacher_arch}")

    @staticmethod
    def _strip_module_prefix(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module."):]
        return key

    def _infer_checkpoint_arch(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int, str]:
        linear_weights = []
        prefix = "regressor.sim_head.query_proj."
        for key, tensor in state_dict.items():
            if key.startswith(prefix) and key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2:
                layer_idx = int(key[len(prefix):].split(".")[0])
                linear_weights.append((layer_idx, tensor))
        if not linear_weights:
            raise ValueError("Could not infer DOTS projection architecture from checkpoint.")
        linear_weights.sort(key=lambda x: x[0])
        first_weight = linear_weights[0][1]
        input_size = int(first_weight.shape[1])
        hidden_size = int(first_weight.shape[0])
        num_layers = len(linear_weights)

        if any(k.startswith("regressor.mlp1.") for k in state_dict):
            scaling = "group_logit_temp_dual"
        elif any(k.startswith("regressor.mlp.") for k in state_dict):
            scaling = "group_logit_temp"
        elif "regressor.res_scale" in state_dict:
            scaling = "platt"
        elif "regressor.scale" in state_dict:
            scaling = "temperature"
        else:
            scaling = self.dots_config.teacher_scaling
        return input_size, hidden_size, num_layers, scaling

    def _resolve_teacher_device(self) -> torch.device:
        requested = (self.dots_config.teacher_device or "auto").lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _predict_all(self) -> torch.Tensor:
        ref_npz, ref_labels = self._valid_reference_arrays()
        if len(ref_npz) == 0:
            print("[DotsSelector] No references have matching embeddings; falling back to random predictions.")
            return torch.full((self._dataset_size,), float(self.dots_config.alpha), dtype=torch.float32)

        if self._teacher_loaded and self._teacher_model is not None:
            preds = self._predict_with_teacher(ref_npz, ref_labels)
            self._last_prediction_mode = "learned_teacher"
        else:
            preds = self._predict_with_fallback(ref_npz, ref_labels)
            self._last_prediction_mode = self.dots_config.fallback_predictor

        if self.dots_config.calibrate_predictions:
            preds = self._calibrate_predictions(preds, ref_labels)
        else:
            self._last_calibration = (1.0, 0.0)

        if self.dots_config.clip_predictions:
            preds = preds.clamp(0.0, 1.0)
        return preds.float()

    def _valid_reference_arrays(self) -> Tuple[List[int], np.ndarray]:
        assert self._ref_mean_rewards is not None
        ref_npz: List[int] = []
        ref_labels: List[float] = []
        for dataset_idx, reward in zip(self._ref_indices, self._ref_mean_rewards):
            npz_idx = self._dataset_index_to_npz(int(dataset_idx))
            if npz_idx is None:
                continue
            ref_npz.append(npz_idx)
            ref_labels.append(float(reward))
        return ref_npz, np.asarray(ref_labels, dtype=np.float32)

    def _predict_with_teacher(self, ref_npz: Sequence[int], ref_labels: np.ndarray) -> torch.Tensor:
        assert self._teacher_model is not None
        device = next(self._teacher_model.parameters()).device
        batch_size = max(1, int(self.dots_config.teacher_batch_size))
        ref_embeddings = self._embedding_tensor[list(ref_npz)].to(device=device, dtype=torch.float32)
        ref_values = torch.tensor([ref_labels.tolist()], dtype=torch.float32, device=device)
        output = torch.full((self._dataset_size,), float("nan"), dtype=torch.float32)

        with torch.no_grad():
            for dataset_batch, npz_batch in self._iter_query_batches(batch_size):
                query_embeddings = self._embedding_tensor[npz_batch].to(device=device, dtype=torch.float32)
                preds = self._teacher_model(
                    query_embeddings,
                    ref_embeddings,
                    ref_values,
                    tau=float(self.dots_config.teacher_tau),
                )
                output[torch.tensor(dataset_batch, dtype=torch.long)] = preds.detach().cpu().float()
        return output

    def _predict_with_fallback(self, ref_npz: Sequence[int], ref_labels: np.ndarray) -> torch.Tensor:
        fallback = self.dots_config.fallback_predictor
        if fallback != "identity_attention":
            raise ValueError(
                f"Unsupported DOTS fallback_predictor={fallback!r}. "
                "Supported: 'identity_attention'."
            )

        device = self._resolve_teacher_device()
        batch_size = max(1, int(self.dots_config.teacher_batch_size))
        ref_embeddings = self._embedding_tensor[list(ref_npz)].to(device=device, dtype=torch.float32)
        ref_values = torch.tensor(ref_labels, dtype=torch.float32, device=device)
        output = torch.full((self._dataset_size,), float("nan"), dtype=torch.float32)

        with torch.no_grad():
            for dataset_batch, npz_batch in self._iter_query_batches(batch_size):
                query_embeddings = self._embedding_tensor[npz_batch].to(device=device, dtype=torch.float32)
                scores = query_embeddings @ ref_embeddings.t()
                scores = scores / math.sqrt(query_embeddings.size(-1))
                if self.dots_config.teacher_top_k is not None and 0 < self.dots_config.teacher_top_k < scores.size(1):
                    vals, idx = torch.topk(scores, int(self.dots_config.teacher_top_k), dim=1)
                    mask = scores.new_full(scores.shape, float("-inf"))
                    mask.scatter_(1, idx, vals)
                    scores = mask
                weights = scores / float(self.dots_config.teacher_tau)
                weights -= weights.max(dim=-1, keepdim=True).values
                weights = torch.softmax(weights, dim=-1)
                preds = weights @ ref_values
                output[torch.tensor(dataset_batch, dtype=torch.long)] = preds.detach().cpu().float()
        return output

    def _iter_query_batches(self, batch_size: int):
        valid = self._valid_dataset_indices or list(range(min(self._dataset_size, len(self._embeddings_np))))
        for start in range(0, len(valid), batch_size):
            dataset_batch = valid[start:start + batch_size]
            npz_batch = [self._dataset_index_to_npz(i) for i in dataset_batch]
            kept_dataset: List[int] = []
            kept_npz: List[int] = []
            for dataset_idx, npz_idx in zip(dataset_batch, npz_batch):
                if npz_idx is not None:
                    kept_dataset.append(int(dataset_idx))
                    kept_npz.append(int(npz_idx))
            if kept_dataset:
                yield kept_dataset, kept_npz

    def _calibrate_predictions(self, preds: torch.Tensor, ref_labels: np.ndarray) -> torch.Tensor:
        pairs: List[Tuple[float, float]] = []
        for dataset_idx, label in zip(self._ref_indices, ref_labels):
            val = preds[int(dataset_idx)]
            if torch.isfinite(val):
                pairs.append((float(val), float(label)))
        if not pairs:
            self._last_calibration = (1.0, 0.0)
            return preds

        x = np.asarray([p[0] for p in pairs], dtype=np.float64)
        y = np.asarray([p[1] for p in pairs], dtype=np.float64)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        var_x = float(((x - x_mean) ** 2).mean())
        if var_x <= 1e-12:
            slope = 0.0
            intercept = y_mean
        else:
            cov_xy = float(((x - x_mean) * (y - y_mean)).mean())
            slope = cov_xy / var_x
            intercept = y_mean - slope * x_mean

        calibrated = preds * float(slope) + float(intercept)
        self._last_calibration = (float(slope), float(intercept))
        return calibrated

    # ------------------------------------------------------------------
    # Sampling / budget accounting
    # ------------------------------------------------------------------

    def _dots_probabilities(self, predicted_labels: torch.Tensor) -> torch.Tensor:
        alpha = float(self.dots_config.alpha)
        tau = max(float(self.dots_config.tau), 1e-12)
        scores = -torch.abs(predicted_labels.float() - alpha)
        valid_mask = torch.isfinite(scores)

        if self._valid_dataset_indices:
            aligned_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            aligned_mask[torch.tensor(self._valid_dataset_indices, dtype=torch.long)] = True
            valid_mask &= aligned_mask

        logits = torch.full_like(scores, float("-inf"))
        logits[valid_mask] = scores[valid_mask] / tau
        if torch.isfinite(logits).sum() == 0:
            logits = torch.zeros(self._dataset_size, dtype=torch.float32)
        else:
            logits -= torch.nan_to_num(logits[valid_mask].max(), nan=0.0)
        probs = torch.softmax(logits, dim=0)
        if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
            probs = torch.zeros(self._dataset_size, dtype=torch.float32)
            candidates = self._valid_dataset_indices or list(range(self._dataset_size))
            probs[torch.tensor(candidates, dtype=torch.long)] = 1.0 / max(1, len(candidates))
        return probs.cpu()

    def _sample_with_budget(self, probs: torch.Tensor, requested_budget: int, target_size: int) -> List[int]:
        if self.config.global_budget_pct is None:
            return self._sample_indices(probs, target_size)

        global_max = self._global_budget_count()
        if global_max <= 0:
            return []

        if not self._selection_frozen:
            remaining = max(0, global_max - len(self._ever_selected_set))
            need_for_train_floor = max(0, target_size - len(self._ever_selected_set))
            new_budget = min(max(max(0, requested_budget), need_for_train_floor), remaining)
            if new_budget > 0:
                if self.dots_config.exclude_already_selected:
                    new_probs = probs.clone()
                    if self._ever_selected_set:
                        new_probs[torch.tensor(sorted(self._ever_selected_set), dtype=torch.long)] = 0.0
                    new_indices = self._sample_indices(new_probs, new_budget)
                else:
                    new_indices = self._sample_indices(probs, new_budget)
                self._train_selected_set.update(int(i) for i in new_indices)
                self._ever_selected_set.update(int(i) for i in new_indices)

            if len(self._ever_selected_set) >= global_max:
                self._selection_frozen = True
                self._frozen_pool = sorted(self._ever_selected_set)
                print(
                    f"[DotsSelector] Global unique budget reached: "
                    f"{len(self._ever_selected_set)}/{global_max}. Freezing DOTS pool."
                )

        pool = self._frozen_pool if self._selection_frozen and self._frozen_pool is not None else sorted(self._ever_selected_set)
        if not pool:
            pool = self._sample_indices(probs, min(target_size, self._dataset_size))
            self._train_selected_set.update(pool)
            self._ever_selected_set.update(pool)
        pool_probs = probs.clone()
        mask = torch.zeros_like(pool_probs)
        mask[torch.tensor(pool, dtype=torch.long)] = 1.0
        pool_probs *= mask

        n_select = min(target_size, len(pool))
        return self._sample_indices(pool_probs, n_select)

    def _sample_indices(self, probs: torch.Tensor, n: int) -> List[int]:
        n = min(int(n), self._dataset_size)
        if n <= 0:
            return []
        probs = probs.detach().cpu().float().clone()
        probs[~torch.isfinite(probs)] = 0.0
        if float(probs.sum()) <= 0:
            candidates = self._valid_dataset_indices or list(range(self._dataset_size))
            probs.zero_()
            probs[torch.tensor(candidates, dtype=torch.long)] = 1.0
        probs /= probs.sum()
        nonzero = int((probs > 0).sum().item())
        n = min(n, nonzero)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.dots_config.seed) + 1009 * int(self._selection_round) + 17)
        selected = torch.multinomial(probs, n, replacement=False, generator=generator)
        if len(selected) > 1:
            perm = torch.randperm(len(selected), generator=generator)
            selected = selected[perm]
        return [int(i) for i in selected.tolist()]

    def _random_selection(self, budget: int) -> List[int]:
        candidates = self._valid_dataset_indices or list(range(self._dataset_size))
        n = min(max(int(budget), int(self._min_training_pool_size)), len(candidates))
        chosen = self._rng.choice(np.asarray(candidates, dtype=np.int64), size=n, replace=False)
        result = [int(i) for i in chosen.tolist()]
        self._train_selected_set.update(result[:max(0, int(budget))])
        if self.config.global_budget_pct is not None:
            self._ever_selected_set.update(result[:max(0, int(budget))])
        self._last_selected_indices = result
        return result

    def _global_budget_count(self) -> int:
        if self.config.global_budget_pct is None:
            return self._dataset_size
        return max(1, int(self._dataset_size * float(self.config.global_budget_pct) / 100.0))

    def _print_round_summary(self, selected_indices: Sequence[int], requested_budget: int, target_size: int) -> None:
        if self._ref_mean_rewards is not None and len(self._ref_mean_rewards) > 0:
            solve_none = int((self._ref_mean_rewards == 0).sum())
            solve_all = int((self._ref_mean_rewards == 1).sum())
            informative = len(self._ref_mean_rewards) - solve_none - solve_all
            ref_msg = (
                f"refs: none={solve_none}, all={solve_all}, "
                f"informative={informative}/{len(self._ref_mean_rewards)}"
            )
        else:
            ref_msg = "refs: reused/none"

        if self._last_predicted_labels is not None and selected_indices:
            idx = torch.tensor(selected_indices, dtype=torch.long)
            selected_pred = self._last_predicted_labels.index_select(0, idx)
            pred_msg = (
                f"selected_pred_mean={float(selected_pred.mean()):.4f}, "
                f"target_abs_err={float(torch.abs(selected_pred - self.dots_config.alpha).mean()):.4f}"
            )
        else:
            pred_msg = "selected_pred_mean=n/a"

        print(
            f"[DotsSelector] Round {self._selection_round}: selected {len(selected_indices)} "
            f"(requested_budget={requested_budget}, train_floor={target_size}, "
            f"verified_unique={len(self._ever_selected_set)}, "
            f"train_unique={len(self._train_selected_set)}, "
            f"mode={self._last_prediction_mode}); "
            f"{ref_msg}; {pred_msg}"
        )
