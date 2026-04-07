# Online Data Selection for GRPO/PPO Training

## Motivation

Standard verl GRPO training uses a static dataset: every epoch trains on the same data (or a random shuffle). But the model's capability changes dramatically during training — what was challenging at step 100 is trivial at step 1000. The training distribution never adapts to this.

**Online data selection** makes the training distribution adaptive: at regular intervals during training, the system measures what the current policy can and cannot solve, then re-selects the training subset to focus on "informative" samples — not too easy (model already solves them), not too hard (model never solves them), but in the zone where the reward signal has variance and the policy can actually learn.

## Architecture

The system uses a **plugin architecture** with a stable interface that the training loop calls, while the selection strategy is fully encapsulated behind it.

```
Training Loop (ray_trainer.py)
    │
    ├── _init_data_selector()      ← once at startup
    │
    └── for each epoch:
            │
            ├── _maybe_reselect_data(epoch)     ← only if reselect_schedule=epoch
            │
            └── while batches (manual iterator):
                    ├── (step schedule) initial / every N steps → _run_data_selection_round()
                    │       → rebuild dataloader + iter(self.train_dataloader) again
                    │
                    └── standard step: generate → reward → advantage → update
```

### Reselection cadence: `epoch` vs `step`

Config: `data_selection.reselect_schedule` and `data_selection.reselect_interval`.

| Schedule | Meaning of `reselect_interval` | When it runs |
|----------|-------------------------------|--------------|
| **`epoch`** (default) | Every N **epochs** | At the **start** of each matching epoch (before the batch loop). |
| **`step`** | Every N **completed training steps** | Once **before the first batch** of training (initial subset), then whenever `global_steps % N == 0` after each step’s increment. The dataloader is rebuilt and the **iterator is refreshed** so the new subset is used immediately (a plain `for batch in dataloader` would keep the old iterator). |

**Cost:** `step` mode triggers reference rollouts more often (each reselect round). Use a larger `reselect_interval` (e.g. 10–50) if rollouts are expensive.

Example (Hydra):

```yaml
data_selection:
  method: cluster
  reselect_schedule: step
  reselect_interval: 10
```

```bash
data_selection.reselect_schedule=step data_selection.reselect_interval=10
```

### The DataSelector Interface

```python
class DataSelector(ABC):
    def initialize(self, dataset, collate_fn=None) -> None: ...
    def get_reference_indices(self) -> List[int]: ...
    def update_rewards(self, ref_indices, ref_rewards) -> None: ...
    def select(self, budget: int) -> List[int]: ...
    def should_reselect_epoch(self, epoch: int) -> bool: ...   # schedule=epoch
    def should_reselect_step(self, global_step: int) -> bool: ...  # schedule=step
    def get_selection_budget(self, dataset_size: int) -> int: ...
    def get_metrics(self) -> Dict[str, float]: ...
```

To add a new selection method, implement this interface and register it in the `build_selector()` factory. No changes to the training loop are needed.

## Quick Reference: Bash Parameters → Code

Every `data_selection.*` key in the bash script maps directly to a config field. Here is the complete mapping with where each is implemented:

| Bash parameter | Default | Implemented in | What it controls |
|---|---|---|---|
| `data_selection.method` | `none` | `__init__.py: build_selector()` | Which selector class to use |
| `data_selection.reselect_schedule` | `epoch` | `base.py: should_reselect_epoch/step()` | When to trigger re-selection |
| `data_selection.reselect_interval` | `1` | `base.py: should_reselect_epoch/step()` | Every N epochs or steps |
| `data_selection.selection_budget_pct` | `100.0` | `base.py: get_selection_budget()` | % of full dataset to select |
| `data_selection.selection_budget` | `None` | `base.py: get_selection_budget()` | Absolute sample count (overrides pct) |
| `data_selection.cluster.cluster_arrays_file` | `None` | `cluster_selector.py: _load_precomputed_clusters()` | Path to `.npz` with embeddings+centroids+assignments |
| `data_selection.cluster.embeddings_file` | `None` | `cluster_selector.py: _load_embeddings_and_cluster()` | Path to raw embeddings (runs FAISS at startup) |
| `data_selection.cluster.n_clusters` | `50` | `cluster_selector.py: _load_embeddings_and_cluster()` | K for KMeans (ignored if cluster_arrays_file given) |
| `data_selection.cluster.n_reps` | `5` | `cluster_selector.py: _select_representatives()` | Representatives per cluster → reference rollout count = K × n_reps |
| `data_selection.cluster.strategy` | `scored` | `cluster_selector.py: select()` | How budget is split across clusters (see below) |
| `data_selection.cluster.within_cluster_method` | `centroid_nearest` | `cluster_selector.py: _within_cluster_select()` | How samples are picked within a cluster |
| `data_selection.cluster.representative_method` | `medoid` | `cluster_selector.py: _select_representatives()` | How representatives are chosen |
| `data_selection.cluster.score_temperature` | `0.1` | `cluster_selector.py: _select_scored()` | Softmax sharpness for scored strategy |
| `data_selection.cluster.transferability_sim_threshold` | `0.9` | `cluster_selector.py: _compute_static_scores()` | Cosine threshold above which clusters are "too similar" |
| `data_selection.cluster.density_gamma` | `1.0` | `cluster_selector.py: _compute_static_scores()` | Gaussian kernel bandwidth for density computation |
| `data_selection.cluster.dots_temperature` | `0.05` | `cluster_selector.py: _dots_interpolate()` | Softmax temperature for interpolated strategy only |
| `data_selection.cluster.dots_top_k` | `64` | `cluster_selector.py: _dots_interpolate()` | Neighbours used for per-sample variance prediction in interpolated strategy |
| `data_selection.cluster.dots_diversity` | `False` | `cluster_selector.py: _select_interpolated()` | When `True`, allocates budget per cluster (softmax over cluster scores, min 1 per cluster) then takes top-n within each cluster by predicted variance. When `False`, pure global top-k. |
| `data_selection.cluster.dots_diversity_use_composite_score` | `False` | `cluster_selector.py: _select_interpolated()` | When `True` (requires `dots_diversity=True`), weights cluster allocation by `mean_predicted_var × transferability × (1/density)` instead of predicted variance alone. Penalises tight redundant clusters and rewards transferable ones. |
| `data_selection.cluster.dots_diversity_temperature` | `0.1` | `cluster_selector.py: _select_interpolated()` | Softmax temperature for cluster allocation when `dots_diversity=True`. High (1.0) → near-uniform. Low (0.01) → only top clusters get budget. |
| `data_selection.cluster.dots_diversity_anneal` | `False` | `cluster_selector.py: _select_interpolated()` | When `True`, decays `dots_diversity_temperature` exponentially over selection rounds. |
| `data_selection.cluster.dots_diversity_temperature_start` | `1.0` | `cluster_selector.py: _select_interpolated()` | Starting temperature when annealing is enabled (hot = near-uniform allocation). |
| `data_selection.cluster.dots_diversity_temperature_end` | `0.05` | `cluster_selector.py: _select_interpolated()` | Minimum temperature when annealing is enabled (cold = concentrated allocation). |
| `data_selection.cluster.dots_diversity_temperature_decay` | `0.1` | `cluster_selector.py: _select_interpolated()` | Exponential decay rate per selection round: `temp(t) = end + (start−end) × exp(−decay × t)`. |
| `data_selection.cluster.use_rollout_history` | `False` | `cluster_selector.py: update_rollout_history()` | When `True`, training-batch rollouts are accumulated into a time-weighted buffer and used as additional DOTS reference points alongside REPR medoids. See below. |
| `data_selection.cluster.rollout_history_decay_rate` | `0.05` | `cluster_selector.py: _compute_time_weighted_variances()` | λ in `exp(-λ * Δstep)`. Higher = faster decay of old observations. |
| `data_selection.cluster.rollout_history_max_age` | `500` | `cluster_selector.py: update_rollout_history()` | Entries older than this many training steps are discarded from the buffer. |
| `data_selection.cluster.rollout_history_max_refs` | `2000` | `cluster_selector.py: _compute_time_weighted_variances()` | Maximum number of reference points passed to DOTS (keeps most recent). |
| `data_selection.cluster.dataset_json_file` | `None` | `cluster_selector.py: _build_alignment_from_json()` | **Required when NPZ and parquet have different row orderings.** Path to the JSON/JSONL used to build the cluster embeddings. The selector matches by `image` field to produce a bidirectional NPZ↔parquet index map. Without this, REPR rollouts and `select()` reference wrong parquet rows (silently incorrect selection). |
| `data_selection.cluster.exploration_enabled` | `false` | `cluster_selector.py: get_exploration_indices()` | Enable exploration rollouts on random un-selected samples to break feedback loops |
| `data_selection.cluster.exploration_pct` | `5.0` | `cluster_selector.py: get_selection_budget_for_exploration()` | Percentage of full dataset to explore each round |
| `data_selection.cluster.exploration_interval` | `1` | `cluster_selector.py: get_exploration_indices()` | Explore every N selection rounds |
| `data_selection.cluster.igs_enabled` | `false` | `cluster_selector.py: _select_interpolated()` | Enable Image Grounding Score in composite cluster scoring |
| `data_selection.cluster.igs_weight` | `1.0` | `cluster_selector.py: _select_interpolated()` | Exponent on IGS in composite score. Higher = stronger preference for multimodal clusters |

> **Note on experiment names**: The WandB `experiment_name` string (e.g. `"…interpolated_centroid…"`) is just a human label — it does **not** control any algorithm. The actual strategy is set by `data_selection.cluster.strategy`. Double-check the strategy param, not the experiment name.

---

## Selection Methods

### 1. `none` (default)
No data selection. The full dataset is used every epoch, identical to standard verl training. This is the default — existing training runs are completely unaffected.

### 2. `random`
Uniform random sampling each epoch. Useful as a baseline to measure whether adaptive selection actually helps. No reference rollouts needed.

### 3. `dots` (Data-efficient Online Training Selection)
Adapted from the [data-efficient-llm-rl](https://github.com/data-efficient-llm-rl) project.

**How it works:**
1. **Reference signal**: Sample ~256 random questions from the dataset. Roll out the current policy on them (n generations each). Compute mean reward per question.
2. **Difficulty prediction**: Use a teacher model (few-shot regression over text) to predict difficulty for every question in the full dataset based on the reference signal.
3. **Target sampling**: Score each sample by `-|predicted_difficulty - alpha|` where `alpha` is the target difficulty (default 0.5 = frontier). Apply softmax with temperature `tau` and sample the training batch.

**Key parameters:**
- `ref_size`: Number of random reference samples (default: 256)
- `alpha`: Target difficulty (0.0=easy, 1.0=hard, 0.5=frontier)
- `tau`: Softmax temperature (lower = sharper selection)
- `teacher_checkpoint`: Path to teacher model (optional; falls back to simple interpolation)

### 4. `cluster` (Cluster-based Selection) ★ New
Uses embedding geometry and cluster structure for selection. This is the main contribution.

**How it works:**

1. **One-time setup** (`initialize`):
   - Load pre-computed embeddings (Qwen3-VL 2048-dim, L2-normalized)
   - Run FAISS spherical KMeans clustering (or load pre-computed clusters)
   - Select representative samples (medoids) per cluster
   - Compute static geometry scores: transferability (inter-cluster cosine similarity) and density (intra-cluster Gaussian kernel)

2. **Reference signal** (`get_reference_indices`):
   - Return the cluster medoids as the reference set
   - These are structurally spread across the full data distribution by construction
   - K clusters × n_reps representatives = ~250 probes (similar cost to DOTS's random 256)

3. **Capability measurement** (`update_rewards`):
   - Receive rollout rewards for each representative
   - Compute per-cluster reward variance: `var(rewards)` for each cluster's representatives
   - High variance = the model is uncertain on this cluster → good for learning
   - Zero variance = model either always solves or never solves → little gradient signal

4. **Selection** (`select`):
   - Score clusters using one of four strategies (see below)
   - Allocate budget across clusters proportionally to scores
   - Within each cluster, select samples via centroid-nearest or MMD coreset

### All Options Reference

#### `data_selection.cluster.strategy` — how budget is split across clusters

| Value | Bash param | Offline equivalent (`04_select_samples.py`) | Budget allocation formula | Extra params | Speed |
|---|---|---|---|---|---|
| `top_clusters` | `data_selection.cluster.strategy=top_clusters` | `--strategy top_clusters` | Sort clusters by variance desc, greedily take all samples from each until budget full | none | fastest |
| `weighted` | `data_selection.cluster.strategy=weighted` | `--strategy weighted` | `alloc[c] = (var[c] / Σvar) × budget` | none | fast |
| `scored` ★ | `data_selection.cluster.strategy=scored` | `--strategy scored` | `score[c] = var[c] × trans[c] × (1/density[c])`, then `alloc[c] = softmax(score/temp)[c] × budget` | `score_temperature`, `transferability_sim_threshold`, `density_gamma` | fast |
| `interpolated` | `data_selection.cluster.strategy=interpolated` | `--strategy interpolated` | Predict per-sample variance via embedding similarity to reps, globally rank all samples, return top-budget | `dots_temperature`, `dots_top_k` | slow (full-dataset pass) |

★ current script uses `scored`

**Detailed explanation of each:**

- **`top_clusters`**: Greedy, sharp. The top 1–5 highest-variance clusters consume the entire budget. All other clusters get zero samples. Use if you want maximum focus on the current learning frontier. Risk: unstable if the top cluster is noisy.

- **`weighted`**: Proportional to variance. All non-zero-variance clusters get *some* samples. Softer and more stable than `top_clusters` but still purely variance-driven — ignores whether learning in one cluster helps others.

- **`scored`** (recommended): Composite score combines three signals:
  - **variance** (dynamic, re-measured each round from live policy rollouts): is the model uncertain here?
  - **transferability** (static, computed once from centroid cosine similarities): does learning here help other clusters?
  - **1/density** (static, computed once from intra-cluster Gaussian kernel): is the cluster internally diverse rather than a tight redundant ball?

  Budget allocated via softmax over scores — `score_temperature=0.1` gives a relatively sharp distribution (a few clusters dominate but none monopolises). Corresponds to the COINCIDE algorithm.

- **`interpolated`**: Works at per-sample granularity instead of per-cluster. For every sample in the full dataset, predicts its variance by embedding-similarity-weighted average of the **individual** representative rollout variances (not cluster averages):
  ```
  predicted_var[i] = Σ softmax(cosine_sim(emb[i], rep_embs) / dots_temperature) × rep_variances
  ```
  Supports two final selection modes controlled by `dots_diversity`:

  | `dots_diversity` | Behaviour | Risk |
  |---|---|---|
  | `False` (default) | Global top-k: return the `budget` samples with highest predicted variance | May concentrate all selections in one high-variance region |
  | `True` | Per-cluster softmax allocation (min 1 per cluster), then top-n within each cluster by predicted variance | Guarantees coverage; trades some variance focus for diversity |

  When `dots_diversity=True`, three additional knobs control the allocation:

  - **`dots_diversity_use_composite_score`** — instead of allocating proportional to mean predicted variance, weight each cluster by `mean_predicted_var × transferability × (1/density)`. This penalises tight redundant clusters and rewards clusters whose learned skills transfer broadly. Requires `_compute_static_scores()` which runs automatically for the `interpolated` strategy.

  - **`dots_diversity_temperature`** — softmax temperature over cluster scores. High (e.g. 1.0) gives near-uniform allocation (diverse, like random). Low (e.g. 0.05) concentrates budget on top-scoring clusters. Default 0.1 is a moderate focus.

  - **`dots_diversity_anneal`** — when enabled, temperature decays exponentially from `dots_diversity_temperature_start` to `dots_diversity_temperature_end` at rate `dots_diversity_temperature_decay` per selection round:
    ```
    temp(t) = temp_end + (temp_start - temp_end) × exp(-decay × t)
    ```
    Intuition: start warm (diverse exploration early in training) and cool down as the model's learning frontier becomes clearer.

  Expensive: requires a full-dataset embedding pass per selection round. The offline `04_select_samples.py --strategy interpolated` is the same algorithm with pre-computed static variances.

#### Time-weighted rollout history buffer (`use_rollout_history`)

By default, `interpolated` (and all other strategies) use only the **REPR medoids** as reference points — K×n_reps samples (e.g. 50×5=250). These are structurally spread but their variance signal is limited to a small, fixed probe set.

When `use_rollout_history=True`, every training batch's rollouts are also accumulated into a **rolling buffer** alongside the REPR rollouts. The effective variance for each buffered sample is computed with exponential time-weighting:

```
w(t) = exp(-λ × (current_step - t))
effective_var[uid] = Σ_t w(t) × var(rewards_at_step_t) / Σ_t w(t)
```

This means:
- **Recent rollouts count most** — rewards from 10 steps ago have more weight than rewards from 100 steps ago
- **Old observations decay** — entries older than `rollout_history_max_age` steps are discarded entirely
- **The reference set grows organically** — from K×n_reps REPR medoids at the start to up to `rollout_history_max_refs` (default 2000) distinct samples by mid-training
- **Seeded automatically** — the REPR rollouts from each selection round are always added to the buffer so the buffer is never empty

**Timeline of buffer growth (with fix applied — see bug note below):**

| Training step | Buffer size | Source |
|---|---|---|
| 0 (first selection) | K×n_reps (e.g. 600) | REPR rollouts only |
| 10 | ~730 (600 + ~2 batches×64) | + first training batches |
| 50 | ~1500 | + more batches, old entries start decaying |
| 100+ | ~2000 (capped) | Most-recent 2000 samples, oldest pruned |

> **Note:** If `n_clusters × n_reps ≥ rollout_history_max_refs` (e.g. 300×10=3000 ≥ 2000), the buffer is already at capacity from REPR alone. Training-batch rollouts would be immediately evicted, so the buffer stays locked to the most-recent REPR medoids. In this case the bug below has no impact.

**When to use it:**
- When `n_clusters × n_reps < rollout_history_max_refs` (i.e. the REPR medoids alone don't fill the buffer) and you want broader coverage of the embedding space as a DOTS reference set
- When you want variance estimates to track the *current* policy's capabilities across the training distribution, not just at medoid locations
- Most beneficial after the first 20–50 training steps when the buffer has enough diversity

**Recommended settings for a first run:**
```yaml
use_rollout_history: true
rollout_history_decay_rate: 0.05    # half-life ≈ 14 steps; tune up for faster adaptation
rollout_history_max_age: 500        # discard entries >500 steps old
rollout_history_max_refs: 2000      # cap reference set size (controls DOTS cost)
```

**Important:** `use_rollout_history` requires `strategy: interpolated` — the other strategies (scored, weighted, top_clusters) don't use per-sample DOTS interpolation and won't benefit from the buffer.

---

#### Bug fix: training-batch rollouts not accumulating in history buffer

**Symptom (prior to fix):** With `use_rollout_history=True` and `n_reps` small enough that REPR medoids don't fill the buffer (e.g. k=300, n_reps=2 → 599 medoids < 2000 max_refs), the `DOTS reference: N samples` log line was stuck at exactly the REPR medoid count across all training rounds instead of growing toward `max_refs`. Zero-variance cluster counts were very high (40–130/300) compared to runs where the buffer was full.

**Root cause:** The training loop assigned random `uuid4()` session IDs to each batch (`batch.non_tensor_batch["uid"]`). The `update_rollout_history()` method tried to map these session IDs back to dataset positions via `_uid_to_dataset_idx`, which is keyed on image-path strings like `clevr_math-CLEVR_train_026670.png`. Every lookup returned `None` and was silently dropped — no training-batch rollout ever entered the buffer.

No warning was printed because `_uid_to_dataset_idx` was non-empty (it contained the NPZ UIDs); the early-exit guard only fires when the map is completely absent.

**Fix (applied):** Three files changed:

1. **`verl/utils/dataset/rl_dataset.py`** — `__getitem__` now emits `row_dict["dataset_idx"] = item`, the integer full-dataset index. `torch.utils.data.Subset` maps subset positions to original indices before calling `__getitem__`, so `item` is always the global position even after a reselection rebuild.

2. **`verl/trainer/ppo/ray_trainer.py`** — the `update_rollout_history` call now passes `dataset_indices=batch.non_tensor_batch.get("dataset_idx")`.

3. **`verl/trainer/ppo/data_selector/cluster_selector.py`** — `update_rollout_history` accepts the new `dataset_indices` parameter. When provided, it uses direct integer index lookups (fast path). The legacy string-UID path is kept as fallback. Each call now prints a diagnostic line:
   ```
   [ClusterSelector] update_rollout_history step=N: matched X/Y samples, buffer_unique=Z
   ```
   After the fix, `matched X/Y` should be `128/128` (or whatever the batch size is) every step, and `buffer_unique` should grow each step until it reaches `max_refs`.

**How to verify the fix is working:** Check for these log patterns:
```
# BEFORE fix (broken):
[ClusterSelector] DOTS reference: 599 samples from rollout history (step=139)
#                                 ^^^ frozen at REPR count, never grows

# AFTER fix (working):
[ClusterSelector] update_rollout_history step=1: matched 128/128 samples, buffer_unique=727
[ClusterSelector] update_rollout_history step=2: matched 128/128 samples, buffer_unique=855
...
[ClusterSelector] DOTS reference: 1823 samples from rollout history (step=9)
#                                 ^^^^ growing toward max_refs=2000
```

WandB metrics to watch:
- `data_selection/history_buffer_unique` — should rise from REPR count toward `max_refs`
- `data_selection/history_buffer_total_entries` — counts all temporal entries across all samples

---

---

#### Bug fix: NPZ row order ≠ parquet row order (silent wrong selection)

**Symptom (prior to fix):** Online selection produced results indistinguishable from random or full-dataset training despite the cluster pipeline appearing to "work" (logs showing variance measurements, zero-variance cluster counts, DOTS references). The cluster selector was running but selecting wrong samples because its internal NPZ positions were being used directly as parquet positions.

**Root cause:** `cluster_arrays.npz` is built from a JSON/JSONL file (e.g. `VLAA-Thinking-GRPO-25K_train_90_100.json`) using the offline `00_compute_embeddings.py` + `01_cluster.py` pipeline. The training parquet (`train_90_100.parquet`) is produced by a separate conversion pipeline. These two pipelines produce the **same 22,675 samples but in completely different row orderings** — there is zero positional correspondence between them.

The original code used NPZ row indices everywhere — as REPR parquet indices in `get_reference_indices()`, as selection output in `select()`, and as buffer keys in `update_rollout_history()`. Using an NPZ row 7 as "parquet row 7" picks a completely unrelated sample, making all selection effectively random.

A spot-check confirmed zero positional matches between the two orderings. Matching by `image` path (the field present in both the JSON and in `extra_info['image']` of each parquet row) gives 100% alignment (22,675/22,675 matched).

**Why the logs appeared healthy despite wrong selection:** The variance measurement and DOTS interpolation still ran correctly *relative to the NPZ order* — there were genuine zero-variance clusters, growing reference counts, and score computations. But the final output indices were NPZ positions fed to `torch.utils.data.Subset`, which treated them as parquet positions. The training was effectively random-sampling the parquet.

**Fix (applied):** One new config field and three code changes:

1. **`data_selection.cluster.dataset_json_file`** — path to the JSON/JSONL that was used to build the cluster embeddings. Set this in the launch script (or YAML config).

2. **`cluster_selector.py: _build_alignment_from_json()`** — called during `initialize()`. Loads the JSON, reads `image` fields, then walks the parquet dataset to extract `extra_info['image']`. Builds two maps:
   - `_npz_to_dataset[npz_i] = parquet_i` (shape N_npz, -1 for unmatched)
   - `_dataset_to_npz[parquet_i] = npz_i` (dict, only for matched rows)

3. **`get_reference_indices()`** — now calls `_remap_npz_to_dataset(self._rep_indices)` so REPR rollouts target the correct parquet rows.

4. **`select()`** — all strategy implementations return NPZ indices; the final `_remap_npz_to_dataset()` call converts them to parquet indices before returning.

5. **`update_rewards()`** — incoming `ref_indices` are now parquet positions (from the remapped `get_reference_indices()`). Converts back to NPZ via `_dataset_to_npz` for cluster membership lookup and buffer storage.

6. **`update_rollout_history()` fast path** — converts incoming parquet `dataset_idx` values to NPZ positions via `_dataset_to_npz` before storing in the buffer.

**How to verify alignment is working:** Check the startup logs:
```
# Good — full alignment:
[ClusterSelector] Building NPZ↔parquet alignment from /path/to/dataset.json ...
[ClusterSelector] Alignment: 22675/22675 NPZ rows matched to parquet rows (0 NPZ rows have no parquet counterpart and will be excluded from selection).

# Warning — partial alignment (some JSON samples filtered during parquet creation):
[ClusterSelector] Alignment: 22432/22675 NPZ rows matched to parquet rows (243 NPZ rows have no parquet counterpart and will be excluded from selection).
[ClusterSelector] Unmatched rows are typically samples in the JSON that were filtered out during parquet creation (missing images, preprocessing failures, etc.).

# Bad — dataset_json_file not set:
[ClusterSelector] WARNING: dataset_json_file not set. Assuming NPZ row order == parquet row order. If they differ, REPR rollouts and select() will reference wrong parquet rows. Set data_selection.cluster.dataset_json_file to the JSON source used to build the embeddings.
```

After alignment, REPR rollout logs will show the *same* samples selected each round (medoids are stable), and selection will concentrate on high-variance regions:
```
# With alignment fix:
[ClusterSelector] Selected 2268 samples (strategy=interpolated, budget=2268)
# These 2268 parquet indices now correctly correspond to the NPZ rows predicted
# to have high variance.
```

**Required for all use cases**, not just `use_rollout_history`. Every strategy (`scored`, `weighted`, `top_clusters`, `interpolated`) is affected because they all call `select()` which remaps NPZ→parquet.

---

#### `data_selection.cluster.within_cluster_method` — how samples are chosen within each cluster's allocated budget

| Value | Bash param | Offline equivalent | How | Extra params | Speed |
|---|---|---|---|---|---|
| `centroid_nearest` ★ | `data_selection.cluster.within_cluster_method=centroid_nearest` | `--within_cluster centroid_nearest` | Sort samples in cluster by L2 distance to centroid, take the N closest | none | O(n) |
| `mmd` | `data_selection.cluster.within_cluster_method=mmd` | `--within_cluster mmd` | Greedy MMD coreset: iteratively pick sample that minimises MMD between selected subset and full cluster distribution | `mmd_gamma` | O(n²) |

★ current script uses `centroid_nearest`

- **`centroid_nearest`**: Picks the most "typical" samples — those closest to what the cluster is about. Fast. Good default.
- **`mmd`**: Picks a maximally representative *spread* of samples from the cluster. Better coverage of intra-cluster diversity but O(n²) per cluster (capped at 2000 samples via subsampling). Use when clusters are large and internally varied.

---

#### `data_selection.cluster.representative_method` — how the fixed probe set (rolled out each round) is chosen

| Value | Bash param | How | Extra params | Speed |
|---|---|---|---|---|
| `medoid` ★ | `data_selection.cluster.representative_method=medoid` | Pick `n_reps` samples with highest mean cosine similarity to all other cluster members | none | O(n²) per cluster at init |
| `centroid_nearest` | `data_selection.cluster.representative_method=centroid_nearest` | Pick `n_reps` samples closest in L2 to centroid | none | O(n) per cluster at init |

★ current script uses `medoid`

The representatives are fixed after `initialize()` — they don't change during training. Their rollout rewards change because the policy changes. `medoid` gives true cluster centres; `centroid_nearest` is faster but slightly less precise.

---

#### Offline vs online `interpolated` — same idea, different variance source

The `interpolated` strategy exists in both places and runs the same `dots_interpolate()` logic. The only difference:

| | Offline (`04_select_samples.py`) | Online (`cluster_selector.py`) |
|---|---|---|
| Variance source | Pre-computed from Stage 3 rollout JSONL (static, one checkpoint) | Live rollout on current policy's representatives (updated every N steps/epochs) |
| Adapts during training | No — one-shot | Yes — re-runs each selection round |

**What makes this different from offline cluster selection:**
Offline cluster selection runs the variance measurement once using a fixed checkpoint. Online cluster selection re-runs this measurement on a schedule (`epoch` or `step`) using the *current* policy. As the model improves, cluster variances shift — previously hard clusters become solvable, new clusters become the frontier — and the selection adapts automatically.

## Relationship to `cluster_selection/` (offline pipeline)

The verl module **`verl/trainer/ppo/data_selector/cluster_selector.py` does not import Python code from** `rl_data_selection/.../cluster_selection/`. It is a **self-contained reimplementation** of the same *ideas* and algorithms so training does not depend on repo layout or extra `sys.path` hacks.

What matches the offline pipeline conceptually:

| Offline stage / file | Online `ClusterSelector` equivalent |
|----------------------|-------------------------------------|
| `01_cluster.py` + `cluster_arrays.npz` | Load `cluster_arrays_file` **or** run FAISS spherical KMeans from `embeddings_file` at `initialize()` |
| `02_select_representatives.py` (medoid / centroid_nearest) | `_select_representatives()` with `representative_method` |
| `03_compute_cluster_variance.py` | **Policy-dependent:** `update_rewards()` computes per-cluster variance from **live** reference rollouts (not from a JSONL file) |
| `03b_compute_cluster_scores.py` | `_compute_static_scores()` — transferability + density (geometry only; same formulas) |
| `04_select_samples.py` — `top_clusters`, `weighted`, `scored`, `interpolated`, MMD, `dots_interpolate` | `_select_top_clusters`, `_select_weighted`, `_select_scored`, `_select_interpolated`, `_within_cluster_select` / `_dots_interpolate` |

**Shared artifact:** The recommended path is to produce **`cluster_arrays.npz`** (and optionally embeddings) with the offline `00_`–`02_` scripts, then point `data_selection.cluster.cluster_arrays_file` at that file. The `.npz` is built from a JSON source file that typically has a **different row ordering** from the training parquet (they are produced by independent pipelines). Always set `data_selection.cluster.dataset_json_file` to the source JSON used to build the embeddings so the selector can build the NPZ↔parquet alignment map at startup. See the alignment bug section below.

## Configuration

Add the `data_selection` block to your training YAML config.

**Important:** When using data selection, the dataloader shrinks (fewer batches per epoch). To ensure training runs for the desired number of gradient steps, either:
- Set `trainer.total_training_steps` explicitly in your config, **or**
- Increase `trainer.total_epochs` proportionally (e.g., if selecting 20% of data, multiply epochs by ~5x)

Pass the **full dataset** as `data.train_files` — the selector will choose the subset online. You don't need to pre-filter the parquet.

You can pass `data_selection.*` overrides on the command line just like other config, e.g.:
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=/workspace/data/full_dataset.parquet \
    data_selection.method=cluster \
    data_selection.selection_budget_pct=20.0 \
    data_selection.cluster.cluster_arrays_file=/workspace/data/cluster_arrays.npz \
    trainer.total_epochs=50 \
    ...
```

### Cluster Selection (recommended)
```yaml
data_selection:
  method: cluster
  reselect_interval: 1          # re-select every epoch
  selection_budget_pct: 20.0    # use top 20% of data

  cluster:
    cluster_arrays_file: /path/to/cluster_arrays.npz
    # REQUIRED: path to the JSON/JSONL used to build the cluster embeddings.
    # Enables NPZ↔parquet row alignment. Without this, selection silently picks wrong samples.
    dataset_json_file: /path/to/source_dataset.json
    n_clusters: 50
    n_reps: 5
    strategy: scored             # or: top_clusters, weighted, interpolated
    within_cluster_method: centroid_nearest
    score_temperature: 0.1
    # interpolated-only options:
    dots_temperature: 0.05
    dots_top_k: 64
    dots_diversity: false                    # true → per-cluster allocation with diversity floor
    dots_diversity_use_composite_score: false # true → weight by var×transferability×(1/density)
    dots_diversity_temperature: 0.1          # softmax temperature for cluster allocation
    dots_diversity_anneal: false             # true → decay temperature over training
    dots_diversity_temperature_start: 1.0   # starting temp (hot = diverse)
    dots_diversity_temperature_end: 0.05    # ending temp (cold = focused)
    dots_diversity_temperature_decay: 0.1   # exp decay rate per selection round
    # rollout history buffer (interpolated strategy only):
    use_rollout_history: false              # true → accumulate training rollouts as DOTS refs
    rollout_history_decay_rate: 0.05       # λ: higher = faster decay of old observations
    rollout_history_max_age: 500           # discard entries older than this many steps
    rollout_history_max_refs: 2000         # max reference points passed to DOTS
    # exploration rollouts (breaks feedback loop in history buffer):
    exploration_enabled: false             # true → periodically roll out random un-selected samples
    exploration_pct: 5.0                   # % of dataset to explore each round
    exploration_interval: 2                # explore every N selection rounds
    # image grounding score (multimodal dependency):
    igs_enabled: false                     # true → multiply composite score by IGS
    igs_weight: 1.0                        # exponent on IGS in composite score
```

### DOTS Selection
```yaml
data_selection:
  method: dots
  reselect_interval: 1
  selection_budget_pct: 50.0

  dots:
    ref_size: 256
    alpha: 0.5
    tau: 0.001
```

### Random Baseline
```yaml
data_selection:
  method: random
  reselect_interval: 1
  selection_budget_pct: 20.0
```

### Disabled (default)
```yaml
data_selection:
  method: none
```

## Running Experiments

The ready-to-use launch script is at:

```
verl/examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh
```

### Arguments

```bash
bash run_qwen3_vl-2b_online_selection.sh [ENGINE] [CLUSTER_ARRAYS] [VARIANT] [DATASET_JSON]
```

| Argument | Default | Description |
|---|---|---|
| `ENGINE` | `vllm` | Rollout engine (`vllm` or `sglang`) |
| `CLUSTER_ARRAYS` | `outputs_300_cluster_new/cluster_arrays.npz` | Path to pre-computed cluster arrays from Stage 1 |
| `VARIANT` | `interpolated_weighted` | Which selection configuration to use (see below) |
| `DATASET_JSON` | `VLAA-Thinking-GRPO-25K_train_90_100.json` | **Required.** Path to the JSON/JSONL used to build the cluster embeddings. Used to align NPZ row order with parquet row order at startup. Without this, REPR rollouts and `select()` reference wrong parquet rows. |

### Variants

| VARIANT | Strategy | History buffer | Experiment name suffix |
|---|---|---|---|
| `interpolated_weighted` ★ | `interpolated` + `dots_diversity=true` | `use_rollout_history=true` — training-batch rollouts accumulate with time-decay into the DOTS reference set. Buffer starts at K×n_reps REPR medoids and grows to `rollout_history_max_refs=2000` when `n_clusters × n_reps < 2000` (requires the `dataset_idx` fix — see bug note in rollout history section). | `interpolated_weighted` |
| `interpolated` | `interpolated` + `dots_diversity=true` | Fixed REPR medoids only (K × n_reps refs). No buffer growth. | `interpolated_centroid` |

★ default

### Usage examples

```bash
# Default: interpolated + time-weighted rollout history buffer
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh

# Explicit default:
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh vllm /path/to/cluster_arrays.npz interpolated_weighted

# Without history buffer (compare against default):
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh vllm /path/to/cluster_arrays.npz interpolated

# Override CUDA devices:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh

# Pass extra Hydra overrides (appended after all script params):
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh vllm /path/cluster.npz interpolated_weighted \
    data_selection.cluster.rollout_history_decay_rate=0.1 \
    trainer.total_epochs=20
```

### What the `interpolated_weighted` variant does

```
Step 0  ──► REPR medoids rolled out (600 samples)
             └─► buffer seeded with REPR rollouts
Step 1..N ──► each training batch's rollouts added to buffer
             └─► w(t) = exp(-0.05 × (current_step - t))
                 older observations downweighted automatically
Step 10  ──► selection round
             └─► buffer now has 600 + N×batch REPR+training samples
                 (capped at 2000, most recent kept)
                 DOTS interpolates predicted variance for all 22K samples
                 using this enriched, policy-tracking reference set
```

Rollout history params in effect for this variant:

```yaml
data_selection.cluster.use_rollout_history: true
data_selection.cluster.rollout_history_decay_rate: 0.05   # half-life ~14 steps
data_selection.cluster.rollout_history_max_age: 500        # prune entries >500 steps old
data_selection.cluster.rollout_history_max_refs: 2000      # cap reference set size
```

---

## Preparing Data for Cluster Selection

The cluster selector requires pre-computed embeddings and (optionally) pre-computed clusters. These come from the offline `cluster_selection` pipeline:

### Option A: Provide pre-computed cluster arrays (fastest)

If you've already run the offline pipeline:

```bash
# From the cluster_selection pipeline outputs:
cluster_arrays_file: cluster_selection/outputs_50_cluster/cluster_arrays.npz
```

This `.npz` file contains: `embeddings`, `centroids`, `assignments`, `distances`, `uids`.

### Option B: Provide embeddings only (clusters computed at init)

```bash
# Just provide the raw embeddings:
embeddings_file: cluster_selection/inputs/qwen_embeddings.npz
```

The selector will run FAISS KMeans during `initialize()`. This adds ~30s startup but requires no pre-processing.

### Option C: Run the offline pipeline first

```bash
cd rl_data_selection/cluster_selection/

# Stage 0: Compute embeddings
python 00_compute_embeddings.py --dataset_json /path/to/dataset.json

# Stage 1: Cluster
python 01_cluster.py --n_clusters 50

# Stage 2: Select representatives
python 02_select_representatives.py --method medoid --n_reps 5

# The cluster_arrays.npz from Stage 1 is all the online selector needs.
```

## File Structure

```
verl/trainer/ppo/
├── ray_trainer.py                  ← training loop with data selection hooks
│                                      _init_data_selector(), _maybe_reselect_data(),
│                                      _run_reference_rollouts(), _rebuild_dataloader()
└── data_selector/
    ├── __init__.py                 ← build_selector() factory
    ├── base.py                     ← DataSelector ABC + DataSelectionConfig
    ├── random_selector.py          ← uniform random baseline
    ├── dots_selector.py            ← DOTS difficulty-based selection
    ├── cluster_selector.py         ← cluster-based adaptive selection
    └── README.md                   ← this file
```

## Metrics Logged

When data selection is active, the following metrics are logged at each selection round:

| Metric | Description |
|--------|-------------|
| `data_selection/n_ref_samples` | Number of REPR reference samples rolled out this round |
| `data_selection/ref_reward_mean` | Mean reward across REPR reference rollouts |
| `data_selection/n_selected` | Number of samples selected for training |
| `data_selection/selection_pct` | Percentage of full dataset selected |
| `data_selection/cluster_var_mean` | Mean cluster variance (cluster method) |
| `data_selection/cluster_var_max` | Max cluster variance |
| `data_selection/n_zero_var_clusters` | Number of zero-variance clusters — high values indicate the model has saturated easy/hard clusters; too many reduce signal quality for selection |
| `data_selection/n_active_clusters` | Number of clusters with rollout data |
| `data_selection/ref_solve_none` | Samples model never solves (DOTS) |
| `data_selection/ref_solve_all` | Samples model always solves (DOTS) |
| `data_selection/history_buffer_unique` | **`use_rollout_history` only.** Unique dataset positions in the buffer. Should grow from K×n_reps toward `rollout_history_max_refs` after the dataset_idx fix. If frozen at K×n_reps, training-batch rollouts are not being accumulated. |
| `data_selection/history_buffer_total_entries` | **`use_rollout_history` only.** Total temporal entries summed across all buffer positions (one per step the sample was seen). Useful for gauging decay/pruning behavior. |
| `data_selection/overlap_jaccard` | Jaccard similarity between current and previous selection round. Values >0.9 mean selection is effectively static. |
| `data_selection/cumulative_coverage_pct` | Percentage of all data that has been selected at least once across all rounds. Low values = selection is stuck in a narrow region. |
| `data_selection/cumulative_unique_selected` | Absolute count of unique samples ever selected. |
| `data_selection/selection_round` | Current selection round number. |
| `data_selection/n_clusters_ever_selected` | How many distinct clusters have had samples selected across all rounds. |
| `data_selection/cluster_selection_freq_mean` | Mean per-cluster selection frequency (higher = more concentrated). |
| `data_selection/cluster_selection_gini` | Gini coefficient of per-cluster selection frequency. 0=uniform, 1=all budget in one cluster. |
| `data_selection/n_exploration_samples` | Number of exploration rollout samples this round (exploration feature). |
| `data_selection/exploration_reward_mean` | Mean reward on exploration samples. |
| `data_selection/igs_mean` | Mean Image Grounding Score across clusters (IGS feature). |
| `data_selection/n_multimodal_clusters` | Number of clusters with IGS > 1.5 (strongly multimodal). |

---

### Exploration Rollouts

When `exploration_enabled=true`, the selector periodically rolls out on **random un-selected samples** to break the feedback loop where the history buffer only sees previously-selected data.

**Problem solved:** Without exploration, the DOTS reference set only contains samples that were already selected and trained on. The selector predicts variance by interpolating from this reference → selects similar samples → trains on them → adds to reference. This self-reinforcing loop means the selector never discovers that ignored regions may now be in the model's "zone of proximal development."

**How it works:**
1. Every `exploration_interval` selection rounds, `get_exploration_indices()` picks `exploration_pct`% of the dataset from the **un-selected** pool (samples NOT in the current training set)
2. The trainer rolls out these samples using the same rollout infrastructure as REPR rollouts
3. Results are fed into the history buffer via `update_exploration_rewards()`
4. Next DOTS interpolation now has visibility into previously-ignored regions

**Config:**
```yaml
data_selection.cluster:
  exploration_enabled: true
  exploration_pct: 5.0          # 5% of full dataset = ~1100 random un-selected samples
  exploration_interval: 2       # explore every other selection round
```

**Cost:** Extra rollout time proportional to `exploration_pct`. With 5% and reselect_interval=10, that's ~1100 extra rollouts every 20 training steps — roughly 2x the REPR rollout cost every other round.

---

### Image Grounding Score (IGS)

IGS measures **multimodal dependency**: whether a sample requires the image to answer correctly, or can be solved from text alone.

```
IGS(x) = Var(rewards_with_image) / Var(rewards_without_image)
```

| IGS Value | Interpretation |
|-----------|---------------|
| > 1.5 | Image is essential — genuinely multimodal |
| ≈ 1.0 | Image has no effect — text shortcut exists |
| < 0.5 | Question is trivial regardless of image |

**Integration:** When `igs_enabled=true`, the composite cluster score becomes:
```
score[c] = predicted_var[c] × transferability[c] × (1/density[c]) × IGS[c]^igs_weight
```

This deprioritizes text-shortcuttable clusters and focuses budget on genuinely visual reasoning tasks.

**Two ways to provide IGS:**
1. **Pre-computed** (recommended): Run a one-time blinded rollout, save scores, load with `load_igs_scores(path)`. The file should be a `.npz` with key `igs_scores` of shape `(N,)`.
2. **Online**: Call `update_igs_from_rollouts(indices, rewards_with, rewards_without)` with paired rollout results.

**Config:**
```yaml
data_selection.cluster:
  igs_enabled: false            # Enable IGS in composite scoring
  igs_weight: 1.0               # Exponent on IGS in composite score (higher = stronger preference for multimodal)
```

---

### Selection Overlap Tracking

Automatically tracks how much the selected subset changes between rounds. Helps diagnose whether online selection is truly "dynamic" or effectively static.

**Metrics logged (see table above):**
- `overlap_jaccard` — Jaccard similarity with previous round (1.0 = identical selection)
- `cumulative_coverage_pct` — % of all data ever selected (low = selection is narrow)
- `cluster_selection_gini` — inequality of per-cluster selection frequency

**Console output each round:**
```
[ClusterSelector] Overlap: jaccard=0.723, cumulative_coverage=34.2% (7756/22675)
```

**Offline visualization:** See `cluster_selection/visualize_selection_overlap.py` for detailed plots (Jaccard over time, coverage curves, cluster heatmaps, UMAP projections).

---

## Design Principles

1. **Zero-change default**: When `data_selection.method: none`, the training loop is identical to vanilla verl. No overhead, no risk.

2. **Swappable strategies**: To try a new selection method, implement `DataSelector` and add a case to `build_selector()`. The training loop doesn't change.

3. **Reuse existing infrastructure**: Reference rollouts use the same `generate_sequences()` and reward computation as training. No new model loading or separate inference.

4. **Minimal invasion**: The changes to `ray_trainer.py` are surgical — four new methods, one hook at epoch boundary, one `collate_fn` reference stored. All existing control flow is preserved.

5. **Observable**: Every selection round logs metrics so you can see how the data distribution evolves during training.

## Comparison: DOTS vs Cluster Selection

| Aspect | DOTS | Cluster Selection |
|--------|------|-------------------|
| Reference set | Random ~256 samples each epoch | Fixed cluster medoids (~250) |
| Reference quality | Random → may miss regions | Structurally covers full distribution |
| Prediction method | Teacher model (few-shot regression) | Embedding geometry + cluster variance |
| Granularity | Per-sample difficulty | Per-cluster → within-cluster |
| Extra model needed | Yes (teacher checkpoint) | No (uses pre-computed embeddings) |
| Selection criterion | Difficulty target (alpha) | Variance + transferability + density |
| Computational cost | Teacher inference on full dataset | Cluster variance computation (lightweight) |
| Offline prep needed | Teacher training | Embeddings + clustering |

## Adding a New Selection Method

1. Create `verl/trainer/ppo/data_selector/my_selector.py`
2. Implement the `DataSelector` interface
3. Add the method to `build_selector()` in `__init__.py`
4. Add method-specific config fields to `DataSelectionConfig` in `base.py`
5. Add the YAML config defaults to `ppo_trainer.yaml`

Example skeleton:

```python
from .base import DataSelectionConfig, DataSelector

class MySelector(DataSelector):
    def __init__(self, config: DataSelectionConfig):
        super().__init__(config)

    def initialize(self, dataset, collate_fn=None):
        self._dataset_size = len(dataset)

    def get_reference_indices(self):
        return []  # or return indices to probe

    def update_rewards(self, ref_indices, ref_rewards):
        pass  # process the probe results

    def select(self, budget):
        return list(range(budget))  # return dataset indices
```
