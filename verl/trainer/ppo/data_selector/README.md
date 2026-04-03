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

- **`interpolated`**: Works at per-sample granularity instead of per-cluster. For every sample in the full dataset, predicts its variance by embedding-similarity-weighted average of the representative rollout variances:
  ```
  predicted_var[i] = Σ softmax(cosine_sim(emb[i], rep_embs) / dots_temperature) × rep_variances
  ```
  Then returns the globally top-budget samples. Not constrained to cluster boundaries. The offline `04_select_samples.py --strategy interpolated` does the same thing but with pre-computed static variances; the online version re-runs this with the current policy's variances each round. Expensive: requires a full-dataset embedding pass per selection round.

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

**Shared artifact:** The recommended path is to produce **`cluster_arrays.npz`** (and optionally embeddings) with the offline `00_`–`02_` scripts, then point `data_selection.cluster.cluster_arrays_file` at that file. Row order in the parquet / `RLHFDataset` must align with row order in that `.npz` (same indexing as when embeddings were built).

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
    n_clusters: 50
    n_reps: 5
    strategy: scored
    within_cluster_method: centroid_nearest
    score_temperature: 0.1
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
| `data_selection/n_ref_samples` | Number of reference samples rolled out |
| `data_selection/ref_reward_mean` | Mean reward across reference rollouts |
| `data_selection/n_selected` | Number of samples selected for training |
| `data_selection/selection_pct` | Percentage of full dataset selected |
| `data_selection/cluster_var_mean` | Mean cluster variance (cluster method) |
| `data_selection/cluster_var_max` | Max cluster variance |
| `data_selection/n_zero_var_clusters` | Number of zero-variance clusters |
| `data_selection/n_active_clusters` | Number of clusters with rollout data |
| `data_selection/ref_solve_none` | Samples model never solves (DOTS) |
| `data_selection/ref_solve_all` | Samples model always solves (DOTS) |

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
