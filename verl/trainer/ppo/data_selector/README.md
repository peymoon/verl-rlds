# Online Data Selection for GRPO/PPO Training

## Motivation

Standard verl GRPO training uses a static dataset: every epoch trains on the same data (or a random shuffle). But the model's capability changes dramatically during training — what was challenging at step 100 is trivial at step 1000. The training distribution never adapts to this.

**Online data selection** makes the training distribution adaptive: at regular intervals during training, the system measures what the current policy can and cannot solve, then re-selects the training subset to focus on "informative" samples — not too easy (model already solves them), not too hard (model never solves them), but in the zone where the reward signal has variance and the policy can actually learn.

## How It Works (End-to-End)

```
INITIALIZATION (once at startup)
  ├── Load cluster_arrays.npz (22K × 2048 Qwen3-VL embeddings, K=200 centroids, assignments)
  ├── Build NPZ↔parquet alignment map from dataset_json_file
  ├── Select medoid representatives: K × n_reps = 600 reference probes
  └── Compute static geometry: transferability (inter-cluster cosine sim), density (intra-cluster Gaussian kernel)

ROUND 0 (step 0) — Initial selection
  ├── Reference rollouts: 600 medoids × 8 rollouts each → per-rep reward variance + mean reward
  ├── Seed rollout history buffer with medoid observations
  ├── DOTS interpolate: predict variance for ALL 22K using 600 refs
  │   (cosine-similarity-weighted average, τ=0.05, top_k=64)
  ├── Optional: asymmetric utility reweighting (see below)
  ├── Per-cluster allocation via softmax(composite_score / diversity_temp)
  ├── Select top 10% (2,268 samples) by predicted variance within each cluster
  └── Dataloader rebuilt with these 2,268 samples

STEPS 1–17 — Train + accumulate signal
  ├── Each step: train on 1 batch (128 samples from selected subset)
  ├── GRPO rollouts → rewards → gradients → update policy
  └── update_rollout_history: each batch's per-sample rewards → time-weighted buffer
      Buffer grows: 600 medoids → ~2000 (capped at rollout_history_max_refs)

ROUND 1 (step 18) — Adaptive re-selection
  ├── Reference rollouts: 600 medoids with UPDATED policy → new variance landscape
  ├── DOTS reference set = rollout history buffer (~2000 time-weighted refs)
  │   (recent observations weighted higher: w(t) = exp(-0.05 × Δstep))
  ├── DOTS interpolate: predict variance for all 22K using enriched reference set
  ├── Clusters that WERE hard → now learnable → variance rises → more budget
  ├── Clusters that WERE learnable → now mastered → variance drops → less budget
  └── New selection adapts to policy's evolved capability frontier

EXPLORATION (every other selection round)
  └── 30 random un-selected samples rolled out → results enter buffer
      → DOTS gains visibility into previously ignored regions

...repeats every 18 steps...
```

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
| **`step`** | Every N **completed training steps** | Once **before the first batch** of training (initial subset), then whenever `global_steps % N == 0` after each step's increment. The dataloader is rebuilt and the **iterator is refreshed** so the new subset is used immediately (a plain `for batch in dataloader` would keep the old iterator). |

**Cost:** `step` mode triggers reference rollouts more often (each reselect round). Use a larger `reselect_interval` (e.g. 10–50) if rollouts are expensive.

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

## Current Training Configuration

These are the **active values** in the launch script `run_qwen3_vl-2b_online_selection.sh`:

| Parameter | Value | Rationale |
|---|---|---|
| `n_clusters` | **200** (was 50) | Fine-grained clustering — ~113 samples per cluster |
| `n_reps` | **3** (was 10) | 3 medoids per cluster → 600 reference rollouts per round |
| `reselect_interval` | **18** (was 10) | More training between reselections → richer buffer per round |
| `selection_budget_pct` | 10% | ~2,268 samples from 22,675 pool |
| `strategy` | `interpolated` | Per-sample variance prediction via DOTS |
| `dots_diversity` | `true` | Per-cluster allocation with diversity guarantee |
| `dots_diversity_temperature` | **0.5** (was 0.1) | Warmer allocation → more uniform budget across clusters |
| `dots_temperature` | 0.05 | Sharp DOTS interpolation — nearest refs dominate |
| `dots_top_k` | 64 | Number of nearest references used per prediction |
| `use_rollout_history` | `true` | Buffer grows from 600 medoids to 2000 refs |
| `rollout_history_decay_rate` | 0.05 | Half-life ≈ 14 steps |
| `exploration_enabled` | `true` | Random un-selected samples rolled out periodically |
| `asymmetric_utility_enabled` | **`true`** (new) | Prefer harder samples at equal variance |
| `hard_side_bias` | 0.5 | Mild bias toward hard side |

## Quick Reference: Bash Parameters → Code

Every `data_selection.*` key in the bash script maps directly to a config field. Here is the complete mapping with where each is implemented:

| Bash parameter | Default | Implemented in | What it controls |
|---|---|---|---|
| `data_selection.method` | `none` | `__init__.py: build_selector()` | Which selector class to use |
| `data_selection.reselect_schedule` | `epoch` | `base.py: should_reselect_epoch/step()` | When to trigger re-selection |
| `data_selection.reselect_interval` | `1` | `base.py: should_reselect_epoch/step()` | Every N epochs or steps |
| `data_selection.selection_budget_pct` | `100.0` | `base.py: get_selection_budget()` | % of full dataset to select |
| `data_selection.selection_budget` | `None` | `base.py: get_selection_budget()` | Absolute sample count (overrides pct) |
| `data_selection.global_budget_pct` | `None` | `base.py` + `cluster_selector.py: select()` | Cap on cumulative unique samples over the entire run (% of full dataset). When the union of all ever-selected samples reaches this cap, selection freezes and subsequent reselection rounds are skipped. Set equal to `selection_budget_pct` for a fair comparison with a fixed random baseline. See [Global Budget Cap](#global-budget-cap-global_budget_pct) below. |
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
| `data_selection.cluster.asymmetric_utility_enabled` | `false` | `cluster_selector.py: _select_interpolated()` | When `true`, predicts per-sample mean reward via DOTS alongside variance and reweights selection so that at equal variance, harder (lower predicted mean) samples are preferred. Breaks the symmetry of variance around p=0.5 so "barely passed" 1/8 samples beat "almost mastered" 7/8 samples. Requires `strategy=interpolated`. |
| `data_selection.cluster.hard_side_bias` | `0.5` | `cluster_selector.py: _select_interpolated()` | α in `utility = predicted_var × (1 + α × (0.5 − predicted_mean))`. `0.0` disables the reweight (equivalent to pure variance). `0.5` gives a mild hard-side tilt (1.25× boost at p=0, 0.75× at p=1). `1.0` doubles hard samples vs masters. Values >1 can make `utility` negative and will be clamped to 0. |
| `data_selection.cluster.asymmetric_dead_zone_low` | `0.05` | `cluster_selector.py: _select_interpolated()` | Samples with predicted mean reward below this threshold get `utility=0`. Guards against samples the model is essentially always failing — under GRPO these have no gradient signal (advantage collapses to 0 when all rollouts agree). |
| `data_selection.cluster.asymmetric_dead_zone_high` | `0.95` | `cluster_selector.py: _select_interpolated()` | Symmetric upper dead-zone for samples the model has essentially mastered. Same rationale. |

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

### 4. `cluster` (Cluster-based Selection) ★ Main
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
   - K clusters × n_reps representatives = ~600 probes

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

| Value | Budget allocation formula | Extra params | Speed |
|---|---|---|---|
| `top_clusters` | Sort clusters by variance desc, greedily take all samples from each until budget full | none | fastest |
| `weighted` | `alloc[c] = (var[c] / Σvar) × budget` | none | fast |
| `scored` | `score[c] = var[c] × trans[c] × (1/density[c])`, then `alloc[c] = softmax(score/temp)[c] × budget` | `score_temperature`, `transferability_sim_threshold`, `density_gamma` | fast |
| `interpolated` ★ | Predict per-sample variance via embedding similarity to refs, then per-cluster allocation or global top-k | `dots_temperature`, `dots_top_k`, `dots_diversity` | slower (full-dataset pass) |

★ current script uses `interpolated`

**Detailed explanation of each:**

- **`top_clusters`**: Greedy, sharp. The top 1–5 highest-variance clusters consume the entire budget. All other clusters get zero samples. Use if you want maximum focus on the current learning frontier. Risk: unstable if the top cluster is noisy.

- **`weighted`**: Proportional to variance. All non-zero-variance clusters get *some* samples. Softer and more stable than `top_clusters` but still purely variance-driven — ignores whether learning in one cluster helps others.

- **`scored`**: Composite score combines three signals:
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

  - **`dots_diversity_use_composite_score`** — instead of allocating proportional to mean predicted variance, weight each cluster by `mean_predicted_var × transferability × (1/density)`. This penalises tight redundant clusters and rewards clusters whose learned skills transfer broadly.

  - **`dots_diversity_temperature`** — softmax temperature over cluster scores. High (e.g. 1.0) gives near-uniform allocation (diverse, like random). Low (e.g. 0.05) concentrates budget on top-scoring clusters. Currently set to **0.5** for moderate spread.

  - **`dots_diversity_anneal`** — when enabled, temperature decays exponentially from `dots_diversity_temperature_start` to `dots_diversity_temperature_end` at rate `dots_diversity_temperature_decay` per selection round:
    ```
    temp(t) = temp_end + (temp_start - temp_end) × exp(-decay × t)
    ```
    Intuition: start warm (diverse exploration early in training) and cool down as the model's learning frontier becomes clearer.

---

#### Time-weighted rollout history buffer (`use_rollout_history`)

By default, `interpolated` (and all other strategies) use only the **REPR medoids** as reference points — K×n_reps samples (e.g. 200×3=600). These are structurally spread but their variance signal is limited to a small, fixed probe set.

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

**Timeline of buffer growth:**

| Training step | Buffer size | Source |
|---|---|---|
| 0 (first selection) | K×n_reps (e.g. 600) | REPR rollouts only |
| 10 | ~1880 (600 + ~10 batches×128) | + training batches |
| 18 (first reselect) | ~2000 (capped) | Buffer at capacity, most-recent kept |
| 36+ | ~2000 (stable) | Oldest entries decay, replaced by fresh |

**When to use it:**
- When `n_clusters × n_reps < rollout_history_max_refs` (i.e. the REPR medoids alone don't fill the buffer)
- When you want variance estimates to track the *current* policy's capabilities across the training distribution, not just at medoid locations
- Most beneficial after the first 20–50 training steps when the buffer has enough diversity

**Recommended settings:**
```yaml
use_rollout_history: true
rollout_history_decay_rate: 0.05    # half-life ≈ 14 steps; tune up for faster adaptation
rollout_history_max_age: 500        # discard entries >500 steps old
rollout_history_max_refs: 2000      # cap reference set size (controls DOTS cost)
```

**Important:** `use_rollout_history` requires `strategy: interpolated` — the other strategies (scored, weighted, top_clusters) don't use per-sample DOTS interpolation and won't benefit from the buffer.

---

### Asymmetric utility — breaking variance symmetry (opt-in)

**Problem.** Reward variance is symmetric around `p = 0.5`. A sample where the
policy succeeds on 1/8 rollouts and one where it succeeds on 7/8 rollouts both
have variance `0.109` — the `interpolated` strategy cannot tell them apart.
Empirically the 1/8 case is more informative: it sits at the policy's capability
frontier, and pulling it into the training set *expands* what the model can do,
whereas 7/8 mostly reinforces what it already can.

**Fix.** When `asymmetric_utility_enabled=true`, the selector runs `_dots_interpolate`
**twice** per selection round — once over time-weighted per-sample *variance*
(as before) and once over time-weighted per-sample *mean reward* (a new
`_compute_time_weighted_mean_rewards` helper that reuses the same rollout
buffer). It then forms:

```
utility(i) = predicted_var(i) × (1 + α × (0.5 − predicted_mean(i)))
```

and uses `utility` in place of `predicted_var` for both the global top-k path
and the per-cluster allocation path. At equal variance, harder samples score
higher.

**Dead-zone.** Samples with predicted mean reward outside
`[asymmetric_dead_zone_low, asymmetric_dead_zone_high]` have `utility` zeroed
out. Under GRPO, advantage = `(r − mean) / std` collapses to 0 when all
rollouts agree, so "always wrong" and "always right" samples contribute no
gradient — spending selection budget on them is strictly wasted. The default
`[0.05, 0.95]` is permissive (discards only near-constant samples); tighten to
`[0.1, 0.9]` if you want to push the selection harder into the learnable band.

**Parameters.**

```yaml
data_selection.cluster.asymmetric_utility_enabled: true
data_selection.cluster.hard_side_bias: 0.5          # α in (1 + α*(0.5 − mean))
data_selection.cluster.asymmetric_dead_zone_low: 0.05
data_selection.cluster.asymmetric_dead_zone_high: 0.95
```

**Interaction with other flags.**

- **Works with `dots_diversity` (both modes).** The reweighted `utility` is
  fed into the existing per-cluster softmax allocation, so cluster-level
  diversity, transferability, density, and IGS continue to apply exactly as
  before.
- **Works with `use_rollout_history=true`.** The per-sample mean reward used
  for the reweight comes from the same time-weighted buffer as the variance
  signal, so the prediction tracks the current policy.
- **Requires `strategy=interpolated`.** `top_clusters`, `weighted`, and
  `scored` don't run DOTS interpolation, so there's nothing to reweight there.
- **Backward compatible.** The default is `false`; disabling it reproduces
  the previous behavior exactly.

**Expected log line** (added when the flag is on):

```
[ClusterSelector] asymmetric utility: α=0.50, dead_zone=[0.05,0.95],
                  18432/22500 samples alive, mean p̂ (alive)=0.412
```

`mean p̂ (alive) < 0.5` indicates the selection is sitting in the harder half
of the predicted-difficulty distribution, which is the intended effect.

**When not to enable.** If `ref_reward_mean` in your logs is already hovering
around `0.4–0.5`, the symmetry problem is small and this flag won't help much.
The reweight becomes meaningful once the policy starts mastering a sizeable
fraction of the selected pool (mean reward drifting toward 0.7+).

---

### Global Budget Cap (`global_budget_pct`)

*Available on the `online-data-selection_limit-budget` branch.*

By default, the online selector picks a fresh `selection_budget_pct`% subset each round. Because the model's capabilities change, different samples are selected each round, and the **cumulative** unique data seen over training grows well beyond the per-round budget (often 30–50%+). This makes comparison with a fixed random baseline unfair — the random baseline sees exactly `selection_budget_pct`% unique samples total.

Setting `global_budget_pct` caps the cumulative unique sample count. Once the union of all ever-selected samples reaches the cap, the **pool composition freezes** (no new samples). But periodic reselection rounds continue — they just skip reference/exploration rollouts and instead **reweight within the frozen pool** using training-batch reward variance.

**Typical usage — fair comparison with random 10%:**
```bash
data_selection.selection_budget_pct=10.0 \
data_selection.global_budget_pct=10.0
```

With `global_budget_pct == selection_budget_pct`, the first selection round uses initial-policy rollouts + DOTS interpolation to pick the smartest 10% of the dataset, then freezes the pool. The model trains on exactly 10% unique samples — identical data volume to the random baseline — but the *which* 10% is variance-informed rather than random.

**Adaptive reweighting within the frozen pool:**

After the pool freezes, subsequent reselection rounds are lightweight (no rollouts):

1. Training-batch reward variances continue accumulating in `_rollout_buffer` (requires `use_rollout_history=True`)
2. Every `reselect_interval` steps, `select()` computes time-weighted per-sample variance from the buffer
3. Samples with high variance (still in the learning zone) are sampled more frequently
4. Samples with zero variance (mastered or too hard) are sampled less frequently (but with a 5% floor weight to prevent starvation)
5. The dataloader is rebuilt with variance-weighted sampling (with replacement)

This means: even though the set of unique samples is fixed, the model spends more compute on informative samples — effectively a curriculum within the frozen pool.

**Compute savings:** Reference rollouts (~600 samples × rounds ≈ thousands of inference passes) and exploration rollouts are eliminated. Only the cheap reweight computation runs.

**Config:**
```yaml
data_selection:
  selection_budget_pct: 10.0
  global_budget_pct: 10.0        # freeze pool after first round
  cluster:
    use_rollout_history: true    # required for adaptive reweighting
```

**Console output:**
```
[ClusterSelector] Global budget cap reached: 2268 unique NPZ samples >= 2268 (10.0% of 22675). Pool frozen — subsequent rounds will reweight within this pool using training reward variance.
[ClusterSelector] Frozen reweight round 2: 1847 unique/2268 total, max_reps=4, var=[0.0000, 0.2500], zero_var=312/2268
```

**WandB metrics (frozen pool):**
- `data_selection/frozen_pool_var_mean` — mean per-sample variance in the pool (should decrease as model learns)
- `data_selection/frozen_pool_n_zero_var` — samples with zero variance (mastered/too-hard; downweighted in sampling)
- `data_selection/frozen_pool_n_with_data` — samples with at least one training observation in the buffer

If `use_rollout_history=False`, the reweight falls back to uniform sampling (all pool samples equally likely).

**Online selection vs budget-limited — when to use which:**

| Aspect | Online (default) | Budget-Limited (`global_budget_pct`) |
|---|---|---|
| **Data volume** | ~10% per round but cumulative unique data grows to 30–50%+ | Exactly `global_budget_pct`% unique samples total |
| **Fair baseline comparison** | Unfair vs fixed random (sees more unique data) | Fair — same data volume as random baseline |
| **Compute cost** | Reference rollouts every N steps | Only first-round rollouts; rest are cheap reweights |
| **Adaptation** | Full re-selection from entire dataset | Within-pool reweighting only |
| **Best for** | Maximum adaptation, uncapped experiments | Controlled experiments, ablation studies |
| **Risk** | May over-explore (too many unique samples) | May under-explore (stuck in initial selection) |

---

#### Bug fix: training-batch rollouts not accumulating in history buffer

**Symptom (prior to fix):** With `use_rollout_history=True` and `n_reps` small enough that REPR medoids don't fill the buffer, the `DOTS reference: N samples` log line was stuck at exactly the REPR medoid count across all training rounds instead of growing toward `max_refs`.

**Root cause:** The training loop assigned random `uuid4()` session IDs to each batch. The `update_rollout_history()` method tried to map these back to dataset positions via `_uid_to_dataset_idx`, which is keyed on image-path strings. Every lookup returned `None` — no training-batch rollout ever entered the buffer.

**Fix (applied):** `verl/utils/dataset/rl_dataset.py` now emits `dataset_idx`; `ray_trainer.py` passes it to `update_rollout_history()`; the selector uses direct integer lookups.

**How to verify the fix is working:**
```
# BEFORE fix (broken):
[ClusterSelector] DOTS reference: 599 samples from rollout history (step=139)
#                                 ^^^ frozen at REPR count

# AFTER fix (working):
[ClusterSelector] update_rollout_history step=1: matched 128/128 samples, buffer_unique=727
[ClusterSelector] DOTS reference: 1823 samples from rollout history (step=9)
#                                 ^^^^ growing toward max_refs=2000
```

---

#### Bug fix: NPZ row order ≠ parquet row order (silent wrong selection)

**Symptom (prior to fix):** Online selection produced results indistinguishable from random despite the cluster pipeline appearing to "work" (logs showed variance measurements, DOTS references, etc.).

**Root cause:** `cluster_arrays.npz` and the training parquet are built by separate pipelines with **different row orderings**. The original code used NPZ row indices as parquet positions, picking completely unrelated samples.

**Fix (applied):** New `dataset_json_file` config field. At `initialize()`, the selector loads the JSON, matches by `image` field against the parquet, and builds bidirectional `_npz_to_dataset` / `_dataset_to_npz` alignment maps. All outputs are remapped before returning.

**How to verify:**
```
# Good — full alignment:
[ClusterSelector] Alignment: 22675/22675 NPZ rows matched to parquet rows

# Bad — dataset_json_file not set:
[ClusterSelector] WARNING: dataset_json_file not set. Assuming NPZ row order == parquet row order.
```

**Required for all use cases**, not just `use_rollout_history`. Every strategy is affected.

---

#### `data_selection.cluster.within_cluster_method` — how samples are chosen within each cluster's allocated budget

| Value | How | Speed |
|---|---|---|
| `centroid_nearest` ★ | Sort by L2 distance to centroid, take N closest | O(n) |
| `mmd` | Greedy MMD coreset: iteratively pick sample that minimises MMD between subset and full cluster | O(n²) |

★ current script uses `centroid_nearest`

- **`centroid_nearest`**: Picks the most "typical" samples — those closest to what the cluster is about. Fast. Good default.
- **`mmd`**: Picks a maximally representative *spread* of samples from the cluster. Better coverage of intra-cluster diversity but O(n²) per cluster (capped at 2000 samples via subsampling). Use when clusters are large and internally varied.

---

#### `data_selection.cluster.representative_method` — how the fixed probe set is chosen

| Value | How | Speed |
|---|---|---|
| `medoid` ★ | Pick `n_reps` samples with highest mean cosine similarity to all cluster members | O(n²) per cluster at init |
| `centroid_nearest` | Pick `n_reps` samples closest in L2 to centroid | O(n) per cluster at init |

★ current script uses `medoid`

The representatives are fixed after `initialize()` — they don't change during training. Their rollout rewards change because the policy changes.

---

#### Offline vs online `interpolated` — same idea, different variance source

| | Offline (`04_select_samples.py`) | Online (`cluster_selector.py`) |
|---|---|---|
| Variance source | Pre-computed from Stage 3 rollout JSONL (static, one checkpoint) | Live rollout on current policy's representatives (updated every N steps) |
| Adapts during training | No — one-shot | Yes — re-runs each selection round |

## Relationship to `cluster_selection/` (offline pipeline)

The verl module **`cluster_selector.py` does not import Python code from** `rl_data_selection/.../cluster_selection/`. It is a **self-contained reimplementation** of the same *ideas* and algorithms so training does not depend on repo layout.

| Offline stage / file | Online `ClusterSelector` equivalent |
|---|---|
| `01_cluster.py` + `cluster_arrays.npz` | Load `cluster_arrays_file` **or** run FAISS spherical KMeans at `initialize()` |
| `02_select_representatives.py` | `_select_representatives()` with `representative_method` |
| `03_compute_cluster_variance.py` | `update_rewards()` — live rollouts, not from a JSONL file |
| `03b_compute_cluster_scores.py` | `_compute_static_scores()` — same formulas |
| `04_select_samples.py` | `select()` — same algorithms, adaptive variance |

**Shared artifact:** Produce `cluster_arrays.npz` with the offline `00_`–`01_` scripts, then point `cluster_arrays_file` at it. Always set `dataset_json_file` for correct alignment.

## Configuration

**Important:** When using data selection, the dataloader shrinks (fewer batches per epoch). To ensure training runs for the desired number of gradient steps, either set `trainer.total_training_steps` explicitly or increase `trainer.total_epochs` proportionally.

Pass the **full dataset** as `data.train_files` — the selector will choose the subset online.

### Cluster Selection (recommended)
```yaml
data_selection:
  method: cluster
  reselect_schedule: step
  reselect_interval: 18
  selection_budget_pct: 10.0

  cluster:
    cluster_arrays_file: /path/to/cluster_arrays.npz
    dataset_json_file: /path/to/source_dataset.json     # REQUIRED for alignment
    n_clusters: 200
    n_reps: 3
    strategy: interpolated
    within_cluster_method: centroid_nearest
    representative_method: medoid
    dots_temperature: 0.05
    dots_top_k: 64
    dots_diversity: true
    dots_diversity_use_composite_score: true
    dots_diversity_temperature: 0.5
    use_rollout_history: true
    rollout_history_decay_rate: 0.05
    rollout_history_max_age: 500
    rollout_history_max_refs: 2000
    exploration_enabled: true
    exploration_pct: 5.0
    exploration_pct_base: representatives
    exploration_interval: 2
    asymmetric_utility_enabled: true
    hard_side_bias: 0.5
    asymmetric_dead_zone_low: 0.05
    asymmetric_dead_zone_high: 0.95
    igs_enabled: false
    igs_weight: 1.0
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
| `CLUSTER_ARRAYS` | `outputs_200_cluster_new/cluster_arrays.npz` | Path to pre-computed cluster arrays from Stage 1 |
| `VARIANT` | `interpolated_weighted` | Which selection configuration to use (see below) |
| `DATASET_JSON` | `VLAA-Thinking-GRPO-25K_train_90_100.json` | **Required.** Path to the JSON/JSONL used to build the cluster embeddings. |

### Variants

| VARIANT | Strategy | History buffer | Experiment name suffix |
|---|---|---|---|
| `interpolated_weighted` ★ | `interpolated` + `dots_diversity=true` + composite score | `use_rollout_history=true` — buffer grows from K×n_reps to 2000 | `interpolated_weighted` |
| `interpolated` | `interpolated` + `dots_diversity=true` | Fixed REPR medoids only (K × n_reps refs) | `interpolated_centroid` |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ASYMMETRIC_UTILITY` | `true` | Enable hard-side-biased utility |
| `ASYMMETRIC_BIAS` | `0.5` | α in utility formula |
| `ASYMMETRIC_DEAD_LOW` | `0.05` | Lower dead-zone on predicted mean reward |
| `ASYMMETRIC_DEAD_HIGH` | `0.95` | Upper dead-zone on predicted mean reward |
| `CUDA_VISIBLE_DEVICES` | `1,3,4,5` | GPU selection |
| `EXPLORATION_PCT_BASE` | `representatives` | Base for exploration % calculation |

### Usage examples

```bash
# Default: interpolated + time-weighted rollout history buffer + asymmetric utility
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh

# With asymmetric utility (stronger bias):
ASYMMETRIC_UTILITY=true ASYMMETRIC_BIAS=1.0 bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh

# Without history buffer (ablation):
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh vllm /path/to/cluster_arrays.npz interpolated

# Override CUDA devices:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh

# Pass extra Hydra overrides:
bash examples/grpo_trainer/run_qwen3_vl-2b_online_selection.sh vllm /path/cluster.npz interpolated_weighted /path/dataset.json \
    data_selection.cluster.rollout_history_decay_rate=0.1 \
    trainer.total_epochs=20
```

## Preparing Data for Cluster Selection

### Option A: Provide pre-computed cluster arrays (fastest)

```bash
cluster_arrays_file: cluster_selection/outputs_200_cluster_new/cluster_arrays.npz
```

This `.npz` file contains: `embeddings`, `centroids`, `assignments`, `distances`, `uids`.

### Option B: Provide embeddings only (clusters computed at init)

```bash
embeddings_file: cluster_selection/inputs/qwen_embeddings.npz
```

The selector will run FAISS KMeans during `initialize()`. This adds ~30s startup but requires no pre-processing.

### Option C: Run the offline pipeline first

```bash
cd rl_data_selection/cluster_selection/
python 00_compute_embeddings.py --dataset_json /path/to/dataset.json
python 01_cluster.py --n_clusters 200
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
| `data_selection/n_zero_var_clusters` | Number of zero-variance clusters — high values indicate the model has saturated easy/hard clusters |
| `data_selection/n_active_clusters` | Number of clusters with rollout data |
| `data_selection/history_buffer_unique` | **`use_rollout_history` only.** Unique dataset positions in the buffer. Should grow from K×n_reps toward `max_refs`. If frozen at K×n_reps, training-batch rollouts are not being accumulated. |
| `data_selection/history_buffer_total_entries` | **`use_rollout_history` only.** Total temporal entries across all buffer positions. |
| `data_selection/overlap_jaccard` | Jaccard similarity between current and previous selection round. Values >0.9 mean selection is effectively static. |
| `data_selection/cumulative_coverage_pct` | Percentage of all data that has been selected at least once across all rounds. Low values = selection is stuck in a narrow region. |
| `data_selection/cumulative_unique_selected` | Absolute count of unique samples ever selected. |
| `data_selection/selection_round` | Current selection round number. |
| `data_selection/n_clusters_ever_selected` | How many distinct clusters have had samples selected across all rounds. |
| `data_selection/cluster_selection_freq_mean` | Mean per-cluster selection frequency (higher = more concentrated). |
| `data_selection/cluster_selection_gini` | Gini coefficient of per-cluster selection frequency. 0=uniform, 1=all budget in one cluster. |
| `data_selection/n_exploration_samples` | Number of exploration rollout samples this round. |
| `data_selection/exploration_reward_mean` | Mean reward on exploration samples. |
| `data_selection/igs_mean` | Mean Image Grounding Score across clusters. |
| `data_selection/n_multimodal_clusters` | Number of clusters with IGS > 1.5. |
| `data_selection/frozen_pool_var_mean` | **Global budget cap only.** Mean predicted variance in frozen pool. |
| `data_selection/frozen_pool_n_zero_var` | **Global budget cap only.** Samples with zero predicted variance in pool. |

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

### Exploration Rollouts

When `exploration_enabled=true`, the selector periodically rolls out on **random un-selected samples** to break the feedback loop where the history buffer only sees previously-selected data.

**Problem solved:** Without exploration, the DOTS reference set only contains samples that were already selected and trained on. The selector predicts variance by interpolating from this reference → selects similar samples → trains on them → adds to reference. This self-reinforcing loop means the selector never discovers that ignored regions may now be in the model's "zone of proximal development."

**How it works:**
1. Every `exploration_interval` selection rounds, `get_exploration_indices()` picks `exploration_pct`% of the base from the **un-selected** pool
2. The trainer rolls out these samples using the same rollout infrastructure as REPR rollouts
3. Results are fed into the history buffer via `update_exploration_rewards()`
4. Next DOTS interpolation now has visibility into previously-ignored regions

**Config:**
```yaml
data_selection.cluster:
  exploration_enabled: true
  exploration_pct: 5.0                 # 5% of reference set size (~30 samples)
  exploration_pct_base: representatives  # or "dataset" for % of full dataset
  exploration_interval: 2              # explore every other selection round
```

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
| Reference set | Random ~256 samples each epoch | Fixed cluster medoids (~600) |
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
