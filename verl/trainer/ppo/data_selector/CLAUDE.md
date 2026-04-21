# Online Data Selection — Deep Pipeline Notes for Claude

> Loaded automatically when working inside `online_data_selection/`.
> Pairs with the root `CLAUDE.md` and `README.md`.

## Module map

| File                       | Role                                                                            |
|----------------------------|---------------------------------------------------------------------------------|
| `cluster_selector.py`      | Main `DataSelector` impl. Owns the pool state, allocation, and reselect logic. |
| `variance_predictor.py`    | `VariancePredictor` ABC, `KNNPredictor`, `RidgePredictor`, `MLPPredictor`, `build_predictor()`. |
| `rollout_history.py`       | Time-decayed buffer of (sample_idx, mean_reward, per_rollout_rewards).         |
| `geometry.py`              | Static cluster geometry: transferability matrix, density estimates.            |
| `cluster_arrays.npz`       | Pre-computed: 22K × 2048 Qwen3-VL embeddings, K=200 centroids, assignments.    |

## The state machine (always reason about which phase you're in)

```
            +------------------+
START ----> | Round 0          |
            | • Medoid probes  |
            | • Seed history   |
            | • Discovery sel  |
            +---------+--------+
                      |
                      v
            +------------------+    every reselect_interval
            | Discovery rounds | <----+
            | • No medoids     |     |
            | • History as ref |     |
            | • Mask + top-k   |     |
            | • Add uniques    | ----+
            +---------+--------+
                      |  cap reached
                      v
            +------------------+
            | Frozen reweight  | <----+
            | • Pool fixed     |     |
            | • Multinomial    |     |
            |   resampling     | ----+
            +------------------+
```

- **Mode is decided by `len(_ever_selected_set) >= global_budget_pct * N`.**
  The dispatch is in `_run_selection_round`. If you see "stuck in frozen
  reweight" symptoms, check whether `selection_budget_pct` × `N_rounds_until_cap`
  burned the budget too fast.
- **`reselect_interval` is the only knob that controls discovery cadence in
  `step` schedule.** With `selection_budget_pct=0.58%` and budget cap=10%,
  you get ~17 discovery rounds. If `reselect_interval=1` you burn the cap in
  17 steps; with `reselect_interval=4` you spread it over ~68 steps. Check
  what the user means before tuning.

## Common gotchas

1. **NPZ ↔ parquet alignment.** `cluster_arrays.npz` indexes the *original*
   dataset order. The dataloader may shuffle. The map is built once in
   `initialize()` from `dataset_json_file`. If indices look "off by some
   permutation" after a refactor, check this map.
2. **Predictor warm-up.** MLP `predicted_var` starts at 0.5 for the first 3–5
   rounds (sigmoid output, untrained). KNN doesn't have this issue. If a run
   shows uniform top-k for the first few rounds, this is expected for MLP and
   *not* a bug.
3. **The `predictor_train_r2 ≈ 0.9` for KNN is memorization.** Each ref point
   is essentially its own nearest neighbor. The honest signal is `loo_knn_r2`.
4. **`rollout_history_max_age` vs `_max_refs`.** Pruning by *age* preserves the
   recency-weighted target; pruning by *count* throws away recent observations
   the trainer paid for. Prefer age-based pruning.
5. **Asymmetric utility dead zones must be checked at the right
   `predicted_mean`.** Head A produces `predicted_mean`. If you accidentally
   wire Head B in, dead-zone clipping will silently malfunction (Head B is
   variance-direct, not probability).
6. **`exclude_already_selected=true` + `exploration_enabled=false` is the
   intended combo.** Random exploration is largely redundant under the
   discovery mask. If you re-enable exploration, expect overlap with the
   discovery pool.

## Adding a new selection method

1. Subclass `DataSelector` in a new file.
2. Implement: `initialize`, `get_reference_indices`, `update_rewards`,
   `select`, `should_reselect_*`, `get_selection_budget`, `get_metrics`.
3. Register in `build_selector()` factory.
4. Add the corresponding config block to `ClusterSelectorConfig` (or your
   own config dataclass) and a Hydra group entry.
5. **Do not touch the training loop.** The plugin contract is the whole point.

## Adding a new predictor

1. Subclass `VariancePredictor` in `variance_predictor.py`.
2. Implement `fit(X, y_per_rollout, y_per_sample, obs_var)` and
   `predict(X) -> PredictionResult(predicted_var, predicted_mean, ...)`.
3. Register in `build_predictor()`.
4. Add diagnostic state (e.g., MLP optimizer state) to the predictor's
   `state_dict()` / `load_state_dict()` so warm-start across resume works.
5. Confirm `loo_knn_r2` still computes against the *same* observed-variance
   target your predictor was trained on. The honest baseline is the
   embedding ceiling.

## Wandb metric cheatsheet (in `data_selection/` namespace)

- `predictor_train_r2` — predictor R² at probed refs (training error).
- `loo_knn_r2` — embedding-ceiling R² (the honest number).
- `predictor_train_r2_vs_pmean_var` — legacy: how much Head B diverges from
  the Bernoulli-derived `p̂(1−p̂)`. Large divergence is informative.
- `pred_vs_obs_scatter` — wandb image, generated each selection round.
- `n_reference_points` — size of the rollout history buffer at this round.
- `predictor_type` — 0=knn, 1=ridge, 2=mlp.
- `selection/n_unique_pool` — `len(_ever_selected_set)`.
- `selection/budget_used_frac` — `n_unique_pool / (global_budget_pct × N)`.
