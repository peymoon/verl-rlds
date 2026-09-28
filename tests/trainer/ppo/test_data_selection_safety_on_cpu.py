from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from verl.trainer.ppo.data_selector.base import DataSelectionConfig
from verl.trainer.ppo.data_selector.cluster_selector import ClusterSelector
from verl.trainer.ppo.data_selector.random_selector import RandomSelector
from verl.trainer.ppo.data_selector.variance_predictor import KNNPredictor


class _IdDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return index


class _AlignmentDataset:
    def __init__(self, qids: list[str]):
        self.dataframe = [
            {"extra_info": {"qid": qid, "image": f"{qid}.png"}}
            for qid in qids
        ]

    def __len__(self):
        return len(self.dataframe)


def _selected_loader(indices: list[int], seed: int) -> StatefulDataLoader:
    subset = Subset(_IdDataset(12), indices)
    generator = torch.Generator().manual_seed(seed)
    return StatefulDataLoader(
        subset,
        batch_size=2,
        sampler=RandomSampler(subset, generator=generator),
        num_workers=0,
    )


def test_selected_loader_state_restores_exact_order_and_duplicates():
    indices = [7, 2, 9, 2, 5, 1]
    loader = _selected_loader(indices, seed=17)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    uninterrupted = [batch.tolist() for batch in iterator]

    restored = _selected_loader(indices, seed=17)
    restored.load_state_dict(state)
    resumed = [batch.tolist() for batch in restored]

    assert resumed == uninterrupted
    assert all(value in set(indices) for batch in resumed for value in batch)


def test_random_selector_state_round_trip_preserves_subset_and_rng():
    cfg = DataSelectionConfig(method="random", seed=123)
    original = RandomSelector(cfg)
    original.initialize(_IdDataset(20))
    first = original.select(6)
    state = original.get_state_dict()
    expected_next = original.select(6)

    restored = RandomSelector(cfg)
    restored.initialize(_IdDataset(20))
    restored.load_state_dict(state)

    assert restored.get_current_training_indices() == first
    assert restored.select(6) == expected_next


def test_fixed_max_binary_variance_preserves_uncertainty_levels():
    observations = []
    for successes in range(9):
        rewards = np.array(
            [1.0] * successes + [0.0] * (8 - successes), dtype=np.float32
        )
        observations.append((successes, 0, rewards))

    predictor = KNNPredictor()
    predictor.fit(
        observations,
        embeddings=np.eye(9, dtype=np.float32),
        current_step=0,
        normalize_variance=True,
    )

    expected = np.array(
        [4.0 * (k / 8.0) * (1.0 - k / 8.0) for k in range(9)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(predictor._ref_variances, expected, atol=1e-7)
    assert len(set(np.round(predictor._ref_variances, 6))) == 5


def test_cluster_alignment_is_qid_based_and_bijective(tmp_path):
    arrays_path = tmp_path / "clusters.npz"
    np.savez(
        arrays_path,
        embeddings=np.eye(3, dtype=np.float32),
        centroids=np.eye(3, dtype=np.float32),
        assignments=np.arange(3, dtype=np.int64),
        uids=np.array(["c", "a", "b"], dtype=object),
    )
    json_path = tmp_path / "records.jsonl"
    json_path.write_text(
        "".join(json.dumps({"qid": q, "image": f"{q}.png"}) + "\n" for q in ["a", "b", "c"]),
        encoding="utf-8",
    )

    cfg = DataSelectionConfig(
        method="cluster",
        cluster={"cluster_arrays_file": str(arrays_path), "dataset_json_file": str(json_path)},
    )
    selector = ClusterSelector(cfg)
    dataset = _AlignmentDataset(["a", "b", "c"])
    selector._dataset_size = len(dataset)
    selector._load_precomputed_clusters()
    selector._build_alignment_from_json(dataset, str(json_path))

    assert selector._npz_to_dataset.tolist() == [2, 0, 1]
    assert selector._dataset_to_npz == {0: 1, 1: 2, 2: 0}


def test_cluster_artifacts_without_stable_uids_fail_closed(tmp_path):
    arrays_path = tmp_path / "clusters.npz"
    np.savez(
        arrays_path,
        embeddings=np.eye(2, dtype=np.float32),
        centroids=np.eye(2, dtype=np.float32),
        assignments=np.arange(2, dtype=np.int64),
    )
    selector = ClusterSelector(
        DataSelectionConfig(method="cluster", cluster={"cluster_arrays_file": str(arrays_path)})
    )
    selector._dataset_size = 2

    with pytest.raises(ValueError, match="stable 'uids'"):
        selector._load_precomputed_clusters()


def _minimal_cluster_selector(config: DataSelectionConfig, size: int = 20):
    selector = ClusterSelector(config)
    selector._dataset_size = size
    selector._embeddings = np.eye(size, dtype=np.float32)
    selector._cluster_ids = np.arange(size) % 2
    selector._centroids = np.eye(2, size, dtype=np.float32)
    selector._npz_to_dataset = np.arange(size)
    selector._dataset_to_npz = {i: i for i in range(size)}
    return selector


def test_global_budget_caps_unique_rows_and_padding_only_reuses_pool():
    selector = _minimal_cluster_selector(
        DataSelectionConfig(
            method="cluster",
            seed=9,
            global_budget_pct=90.0,
            global_budget_count=4,
            cluster={
                "n_clusters": 2,
                "count_medoids_in_budget": True,
                "strategy": "interpolated",
            },
        )
    )
    selector.set_min_training_pool_size(6)

    chosen = selector.select(10)

    assert selector._selection_frozen
    assert len(selector._budget_counted_set()) == 4
    assert len(chosen) == 6
    assert len(set(chosen)) <= 4
    assert set(chosen) <= selector._budget_counted_set()


def test_exploration_is_clipped_and_counted_without_history_storage():
    selector = _minimal_cluster_selector(
        DataSelectionConfig(
            method="cluster",
            seed=4,
            global_budget_pct=20.0,
            cluster={
                "n_clusters": 2,
                "exploration_enabled": True,
                "exploration_pct": 50.0,
                "exploration_pct_base": "dataset",
                "use_rollout_history": False,
            },
        )
    )
    selector._ever_selected_train = {0, 1, 2}
    selector._ever_selected_set = {0, 1, 2}

    explore = selector.get_exploration_indices()
    assert len(explore) == 1

    selector.update_exploration_rewards(
        explore, np.zeros((1, 8), dtype=np.float32)
    )
    assert len(selector._budget_counted_set()) == 4
    assert not selector._rollout_buffer
