# Online Data Selection for GRPO/PPO Training

## Motivation

Standard verl GRPO training uses a static dataset: every epoch trains on the same data (or a random shuffle). But the model's capability changes dramatically during training — what was challenging at step 100 is trivial at step 1000. The training distribution never adapts to this.

**Online data selection** makes the training distribution adaptive: at regular intervals during training, the system measures what the current policy can and cannot solve, then re-selects the training subset to focus on "informative" samples — not too easy (model already solves them), not too hard (model never solves them), but in the zone where the reward signal has variance and the policy can actually learn.

## How It Works (End-to-End)

The pipeline is structured around an **annotation budget**, not a compute
budget: every sample for which the trainer consumes ground-truth (reference
medoid rollouts, training rollouts, and exploration rollouts) is debited
against a single global cap (`global_budget_pct`).  Selection runs in two
phases — **discovery** (growing the annotated pool) and **frozen reweight**
(redistributing compute inside the pool once the cap is hit).

```
INITIALIZATION (once at startup)
  ├── Load cluster_arrays.npz (22K × 2048 Qwen3-VL embeddings, K=200 centroids, assignments)
  ├── Build NPZ↔parquet alignment map from dataset_json_file
  ├── Select medoid representatives: K × n_reps = 600 reference probes
  ├── Compute static geometry: transferability (inter-cluster cosine sim), density (Gaussian kernel)
  └── Compute 2-D PCA projection of embeddings (cached for wandb scatter plots)

ROUND 0 (step 0) — Cold start
  ├── Reference rollouts: 600 medoids × 8 rollouts each → per-rep variance + mean reward
  │   (these 600 samples are debited against the global budget)
  ├── Seed rollout history buffer with medoid observations
  ├── DOTS interpolate: predict variance for ALL 22K using 600 refs
  ├── If asymmetric_utility_enabled: predict mean reward too and reweight
  │   (utility = predicted_var × (1 + α × (0.5 − predicted_mean)), dead-zone clipped)
  ├── Per-cluster softmax allocation over composite scores (var × transferability × 1/density)
  ├── Select top `selection_budget_pct` % (e.g. 0.58% ≈ 128 samples) within each cluster
  │   — the discovery mask `_ever_selected_set` excludes already-selected samples
  └── Dataloader rebuilt with these new samples

STEPS 1..N — Train + accumulate signal (no medoid rollouts)
  ├── Each step: train on 1 batch (128 samples from current selection)
  ├── GRPO rollouts → rewards → gradients → update policy
  ├── update_rollout_history: each batch's per-sample rewards → time-weighted buffer
  │   (buffer grows organically; rollout_history_max_refs=0 → no cap)
  └── Every `reselect_interval` steps → DISCOVERY ROUND (see below)

DISCOVERY ROUND (every `reselect_interval` steps until cap is hit)
  ├── Medoid rollouts SKIPPED (reroll_medoids=false; round-0 medoids only)
  ├── DOTS reference set = entire rollout history buffer (organic, policy-tracking)
  ├── DOTS interpolate variance + (optional) mean over the FULL 22K
  ├── Discovery mask zeros out already-selected samples → top-k yields fresh uniques only
  ├── Per-cluster allocation as before, top-n within each cluster
  └── New batch of `selection_budget` brand-new samples is added to the pool
      → debited against the global budget

GLOBAL CAP REACHED → SWITCH TO FROZEN REWEIGHT
  ├── `_ever_selected_set` ≥ `global_budget_pct × N`  → freeze pool composition
  ├── Subsequent reselection rounds:
  │     • no medoid rollouts (saved compute)
  │     • no exploration rollouts (saved compute)
  │     • DOTS interpolate variance + mean over the frozen pool only
  │     • multinomial-sample with replacement weighted by utility
  └── Effective curriculum within the fixed pool — focus shifts to whichever
      pool members are still in the policy's learning zone

EXPLORATION (optional, off by default)
  └── With exclude_already_selected=true the discovery mask makes random
      exploration largely redundant, so EXPLORATION_ENABLED=false in the
      launch script.  Set to true for ablations.
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
| `n_clusters` | 200 | Fine-grained clustering — ~113 samples per cluster |
| `n_reps` | 3 | 3 medoids per cluster → 600 reference rollouts (round 0 only) |
| `reselect_schedule` | `step` | Reselect on step boundaries, not epoch boundaries |
| `reselect_interval` | **4** | Reselect every 4 training steps — every-step proved too aggressive (burned the cap in ~17 steps and left training stuck in frozen-reweight) |
| `selection_budget_pct` | **0.58%** | ~128 samples per round (one batch). Small per-round budget = many discovery rounds before hitting the cap |
| `global_budget_pct` | **10.0%** | Hard cap on cumulative unique annotated samples — fair comparison vs random 10% baseline |
| `exclude_already_selected` | **`true`** | Discovery mask: each round adds exactly `selection_budget` brand-new uniques (no resampling) |
| `reroll_medoids` | **`false`** | Medoid rollouts run once at round 0; later rounds rely on the rollout-history buffer |
| `strategy` | `interpolated` | Per-sample variance prediction via DOTS |
| `dots_diversity` | `true` | Per-cluster allocation with diversity guarantee |
| `dots_diversity_temperature` | 0.5 | Warmer allocation → more uniform budget across clusters |
| `dots_temperature` | 0.05 | Sharp DOTS interpolation — nearest refs dominate |
| `dots_top_k` | 64 | Number of nearest references used per prediction |
| `use_rollout_history` | `true` | Buffer grows organically from 600 medoids as training progresses |
| `rollout_history_decay_rate` | 0.05 | Half-life ≈ 14 steps |
| `rollout_history_max_refs` | **0** (unlimited) | No artificial cap; rely on `rollout_history_max_age` pruning instead. Older default of 2000 threw away recent observations the trainer had already paid for |
| `exploration_enabled` | **`false`** | Disabled by default — discovery mask already prevents resampling, so the original "diversify the buffer" rationale is moot |
| `asymmetric_utility_enabled` | `true` | Prefer harder samples at equal variance |
| `hard_side_bias` | 0.5 | Mild bias toward hard side |
| `asymmetric_dead_zone_low/high` | 0.05 / 0.95 | Zero out samples the policy already always-fails or always-passes |

## Quick Reference: Bash Parameters → Code

Every `data_selection.*` key in the bash script maps directly to a config field. Here is the complete mapping with where each is implemented:

| Bash parameter | Default | Implemented in | What it controls |
|---|---|---|---|
| `data_selection.method` | `none` | `__init__.py: build_selector()` | Which selector class to use |
| `data_selection.reselect_schedule` | `epoch` | `base.py: should_reselect_epoch/step()` | When to trigger re-selection |
| `data_selection.reselect_interval` | `1` | `base.py: should_reselect_epoch/step()` | Every N epochs or steps |
| `data_selection.selection_budget_pct` | `100.0` | `base.py: get_selection_budget()` | % of full dataset to select |
| `data_selection.selection_budget` | `None` | `base.py: get_selection_budget()` | Absolute sample count (overrides pct) |
| `data_selection.global_budget_pct` | `None` | `base.py` + `cluster_selector.py: select()` | Cap on cumulative unique samples over the entire run (% of full dataset). When the union of all ever-selected samples reaches this cap, selection freezes and subsequent reselection rounds reweight inside the frozen pool. Set equal to `selection_budget_pct` for a fair comparison with a fixed random baseline. |
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
| `data_selection.cluster.rollout_history_max_refs` | `0` | `cluster_selector.py: _compute_time_weighted_variances()` | Maximum number of reference points passed to DOTS (keeps most recent). `0` = unlimited (rely on `rollout_history_max_age` pruning). |
| `data_selection.cluster.exclude_already_selected` | `true` | `cluster_selector.py: _select_interpolated()` | **Discovery mask.** When true, the per-sample interpolated top-k zeros out any sample already in `_ever_selected_set`. Each pre-freeze round therefore adds exactly `selection_budget` brand-new uniques (selection without replacement), so the global cap corresponds to `per_round_budget × n_rounds`. |
| `data_selection.cluster.reroll_medoids` | `false` | `cluster_selector.py: get_reference_indices()` | When false, REPR medoid rollouts only fire on round 0; later rounds rely on the rollout-history buffer instead of re-rolling the same medoids. Saves a substantial amount of inference compute and avoids wasting global budget on medoids that the buffer already supersedes. |
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
- **The reference set grows organically** — from K×n_reps REPR medoids (round 0 only) through training batches that accumulate every step
- **Seeded automatically** — the REPR rollouts from round 0 are always added to the buffer so the buffer is never empty

**Timeline of buffer growth (default v3 settings, reselect_interval=4):**

| Training step | Buffer size | Source |
|---|---|---|
| 0 (round 0) | K×n_reps (e.g. 600) | REPR medoid rollouts only |
| 4 (round 1) | ~1112 (600 + 4×128) | + 4 training batches |
| 8 (round 2) | ~1624 | + 4 more training batches |
| 48 (pool freezes) | ~6700+ | Organic growth, no artificial cap |
| 100+ | Stabilises | `rollout_history_max_age` prunes stale entries |

**When to use it:**
- Always recommended with the `interpolated` strategy — the buffer replaces per-round medoid re-rollouts (which are disabled by default via `reroll_medoids=false`)
- Most beneficial after the first 10–20 training steps when the buffer has enough diversity to give good DOTS predictions without re-rolling the medoids

**Recommended settings:**
```yaml
use_rollout_history: true
rollout_history_decay_rate: 0.05    # half-life ≈ 14 steps; tune up for faster adaptation
rollout_history_max_age: 500        # discard entries >500 steps old
rollout_history_max_refs: 0         # 0 = unlimited; rely on max_age pruning
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

### Annotation budget vs compute budget (v3 design)

The pipeline tracks a single quantity — `_ever_selected_set` — which is the
union of every sample for which the trainer has consumed ground truth.  Three
disjoint sources can add to it, and all three are debited against the same
`global_budget_pct` cap:

| Source | Tracked in | When debited |
|---|---|---|
| `training` | `_ever_selected_train` | Each call to `select()` |
| `medoid` | `_ever_selected_medoid` | Each call to `update_rewards()` (REPR rollouts) |
| `exploration` | `_ever_selected_exploration` | Each call to `update_exploration_rewards()` |

Per-round budget = number of *new uniques* added per `select()` call, because
the discovery mask (`exclude_already_selected=true`) zeros out any sample
already in `_ever_selected_set`.  This gives the trainer a clean
"annotation-units consumed" view that lines up exactly with what a fixed
random baseline at the same `global_budget_pct` would see.

Per-source breakdown is reported as wandb scalars
(`data_selection/budget_breakdown/{training,medoid,exploration}` and the
matching `_pct` variants) and visualised as a stacked area chart over
selection rounds.

### Wandb visualisations

Beyond scalars, the selector logs three diagnostic images per selection round
via `get_wandb_images()`:

| Image | Shows | Notes |
|---|---|---|
| `data_selection/cluster_allocation` | Per-cluster sample count for the most recent round, top-40 clusters, bars coloured by current cluster variance | Always on |
| `data_selection/selected_difficulty` | Histogram of DOTS-predicted mean reward for the most recently selected samples, with frontier line at 0.5 and dead-zone bands shaded | Only when `asymmetric_utility_enabled=true` |
| `data_selection/selection_scatter` | 2-D PCA scatter of all embeddings (background, downsampled) overlaid with the most recently selected samples coloured by predicted mean reward (or variance) | PCA is computed once at `initialize()` and cached |
| `data_selection/budget_breakdown` | Cumulative annotation budget by source (training / medoid / exploration) over selection rounds, with the global cap drawn as a horizontal line | Always on once `_budget_history` has any entries |
| `data_selection/overlap_history` | Jaccard overlap between successive selection rounds, with a 0.5 reference line | Always on after round 2 |

### Global Budget Cap (`global_budget_pct`)

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

#### Bug fix: active probes left `_cluster_variances` empty → silent random fallback

**Symptom (prior to fix):** Ridge runs with `predictor_active_probes=true`
logged `Cluster variances updated: 0 clusters, 0 zero-variance, 0 high-variance`
every round and fell back to `[ClusterSelector] No variance data yet, selecting random`.
Combined with the budget schedule, per-round budget eventually dropped below
`train_batch_size` and the rebuilt dataloader returned 0 batches (stall).

**Root cause:** `update_rewards()` looked up cluster membership via
`idx_to_cluster = dict(zip(self._rep_indices, self._rep_cluster_ids))`,
a dict keyed only on **medoid** indices. Active probes pick samples by UCB
over the full dataset — these are almost never medoids — so every probed
sample failed the `c_id is None` guard, `cluster_rewards` stayed empty, and
`_cluster_variances` was wiped to `{}` at the top of every round.

**Fix (applied):** Look up cluster membership from `self._cluster_ids` (the
per-sample assignments array), not from the rep-only dict. Active probes,
exploration picks, and any other non-medoid ref selection now correctly
contribute to cluster variance.

---

#### Bug fix: `select()` random fallback was too aggressive

**Symptom:** Rounds with empty `_cluster_variances` (e.g. early cold-start,
or when active-probes-bug fired) dropped into a pure random selection even
when `_rollout_buffer` already had per-sample observations the `interpolated`
strategy could have used.

**Fix (applied):** The random-fallback gate in `select()` now checks both
`_cluster_variances` **and** `_rollout_buffer`. For `strategy="interpolated"`,
as long as the rollout buffer or `_rep_variances` has any observations, the
selector dispatches to `_select_interpolated` — which has its own fallback
chain (`_select_weighted` → random) — instead of jumping straight to random.
An empty cluster-variance dict alone is no longer grounds for random.

---

#### Bug fix: per-round discovery budget below `train_batch_size` caused 0 batches

**Symptom:** Under a `budget_schedule` with tapering phases, once cumulative
utilisation crossed a phase boundary (e.g. 50% → 85%), `per_round_pct`
dropped from 1.0 to 0.3. On a 22675-sample dataset that's 68 samples, below
typical `train_batch_size=128+`. The rebuilt dataloader uses `drop_last=True`,
so `StatefulDataLoader` reported `Rebuilt dataloader: 0 batches from 68 samples`
and training stalled.

**Fix (applied):** `DataSelector.set_min_training_pool_size(n)` is called by
the trainer at init with `n = train_batch_size`. At the end of `select()`,
if the strategy returns fewer than `n` indices, the selector pads up to the
floor by drawing from `_ever_selected_set` — samples already annotated and
already counted against the global budget, so padding is free in annotation
terms. If the already-annotated pool is too small (very early rounds), the
padding falls back to random draws from the full parquet dataset; those
extras are **not** added to `_ever_selected_*` so they don't inflate the
annotation count.

**How to verify:** Log line `[ClusterSelector] Padding training pool
{before} -> {after} (floor={n}, source={pool|pool+random|random})` appears
in the round where the phase transition happens, followed by `Rebuilt
dataloader: N batches from >= train_batch_size samples`.

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
| `density_diverse` | Density-aware diverse selection with greedy suppression (see below) | k-NN + greedy loop per cluster |

★ current script uses `medoid`

##### `density_diverse` — density-aware diverse representative selection

Ported from the offline pipeline (`cluster_selection/02_select_representatives.py` on the `benchmark_repr` branch). Combines local density with centroid proximity and enforces spatial diversity across representatives.

**Algorithm:**
1. **k-NN density estimation** — for each point in the cluster, compute density as `1 / (mean cosine distance to k nearest neighbors)`. Points in dense regions score higher.
2. **Composite scoring** — `score_i = density_norm^alpha * centroid_sim^beta`, where `density_norm` is min-max normalised density and `centroid_sim` is the cosine similarity to the cluster centroid (clipped to [0, 1]).
3. **Greedy diverse selection** — pick the highest-scoring point as representative, then suppress (mask out) all points within `diversity_radius` cosine distance of the pick. Repeat until `n_reps` representatives are selected.

**Why it's useful for online selection:**
- **Handles `n_reps > 1` natively** — unlike `medoid` (which uses remove-and-rerun), the suppression radius ensures representatives are spread across different sub-regions of each cluster. This gives better coverage of the cluster's internal structure.
- **Avoids outlier-adjacent probes** — the density term penalises isolated points that might be noise or boundary samples, producing more reliable variance estimates.
- **Downstream benefit** — better-spread representatives → better DOTS interpolation in the `interpolated` strategy, because the reference set covers more of the embedding space.

**Hyperparameters** (set via Hydra overrides or env vars):

| Parameter | Default | Description |
|---|---|---|
| `density_diverse_k` | 10 | k for k-NN density estimation. Larger k → smoother density; smaller k → more local |
| `density_diverse_alpha` | 1.0 | Exponent on the normalised density term. Higher → favour denser regions more |
| `density_diverse_beta` | 1.0 | Exponent on centroid proximity. Higher → favour central samples more |
| `density_diverse_radius` | 0.15 | Cosine-distance suppression radius. After picking a rep, suppress all points with cosine similarity > (1 - radius). Set to 0 to disable diversity enforcement |

**Usage example:**
```bash
REPRESENTATIVE_METHOD=density_diverse bash run_qwen3_vl-2b_online_selection.sh

# With custom hyperparameters:
REPRESENTATIVE_METHOD=density_diverse bash run_qwen3_vl-2b_online_selection.sh \
    vllm /path/cluster_arrays.npz interpolated_weighted /path/dataset.json \
    data_selection.cluster.density_diverse_k=15 \
    data_selection.cluster.density_diverse_alpha=1.5 \
    data_selection.cluster.density_diverse_radius=0.2
```

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
  reselect_interval: 4              # one batch worth of training between discoveries
  selection_budget_pct: 0.58        # ~128 new uniques per round on a 22k dataset
  global_budget_pct: 10.0           # hard cap on cumulative annotated samples

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

    # discovery without replacement + medoid-once-only
    exclude_already_selected: true
    reroll_medoids: false

    use_rollout_history: true
    rollout_history_decay_rate: 0.05
    rollout_history_max_age: 500
    rollout_history_max_refs: 0     # unlimited; rely on age pruning

    # exploration off by default — discovery mask makes it redundant
    exploration_enabled: false
    exploration_pct: 1.0
    exploration_pct_base: representatives
    exploration_interval: 4

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
| `SELECTION_BUDGET_PCT` | `0.58` | Per-round budget as % of dataset (~128 on 22K) |
| `RESELECT_INTERVAL` | `4` | Reselect every N training steps |
| `GLOBAL_BUDGET_PCT` | `10.0` | Hard cap on cumulative annotated samples |
| `EXCLUDE_ALREADY_SELECTED` | `true` | Discovery mask (selection without replacement) |
| `REROLL_MEDOIDS` | `false` | Medoid rollouts on round 0 only |
| `ROLLOUT_HISTORY_MAX_REFS` | `0` | Buffer size cap (0 = unlimited) |
| `EXPLORATION_ENABLED` | `false` | Exploration rollouts (off by default) |
| `EXPLORATION_PCT` | `1.0` | % of base to explore when enabled |
| `EXPLORATION_INTERVAL` | `4` | Explore every N selection rounds |
| `EXPLORATION_PCT_BASE` | `representatives` | Base for exploration % calculation |
| `ASYMMETRIC_UTILITY` | `true` | Enable hard-side-biased utility |
| `ASYMMETRIC_BIAS` | `0.5` | α in utility formula |
| `ASYMMETRIC_DEAD_LOW` | `0.05` | Lower dead-zone on predicted mean reward |
| `ASYMMETRIC_DEAD_HIGH` | `0.95` | Upper dead-zone on predicted mean reward |
| `CUDA_VISIBLE_DEVICES` | `1,3,4,5` | GPU selection |

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
| `data_selection/budget_breakdown/training` | Cumulative unique samples added by `select()` calls. |
| `data_selection/budget_breakdown/medoid` | Cumulative unique samples consumed by REPR medoid rollouts (round 0 only when `reroll_medoids=false`). |
| `data_selection/budget_breakdown/exploration` | Cumulative unique samples consumed by exploration rollouts (zero when exploration is disabled). |
| `data_selection/budget_breakdown/{training,medoid,exploration}_pct` | Same as above, normalized by `cumulative_unique_selected` (sum to 100). |
| `data_selection/global_budget` | Absolute global budget cap (in samples) when `global_budget_pct` is set. |
| `data_selection/global_budget_utilization_pct` | Cumulative unique samples / global cap × 100. Hits 100 at the moment the pool freezes. |
| `data_selection/selection_frozen` | 0 / 1 indicator: 1 once the pool has frozen and frozen-reweight has taken over. |
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

**Disabled by default in v3.**  With `exclude_already_selected=true` the
discovery mask already prevents the selector from re-picking previously-seen
samples, so the original "diversify the buffer" rationale for exploration is
largely moot.  Exploration rollouts also debit the global budget — keeping
them on burns annotation units that would otherwise go to discovery.  Set
`EXPLORATION_ENABLED=true` in the launch script to re-enable it for ablation.

When enabled, the selector periodically rolls out on **random un-selected
samples** to break the feedback loop where the history buffer only sees
previously-selected data.

**How it works:**
1. Every `exploration_interval` selection rounds, `get_exploration_indices()` picks `exploration_pct`% of the base from the **un-selected** pool
2. The trainer rolls out these samples using the same rollout infrastructure as REPR rollouts
3. Results are fed into the history buffer via `update_exploration_rewards()`, which also debits these samples against `_ever_selected_set` (so they count toward the global budget cap, just like medoid and training rollouts)
4. Next DOTS interpolation now has visibility into previously-ignored regions

**Config:**
```yaml
data_selection.cluster:
  exploration_enabled: false           # off by default
  exploration_pct: 1.0                 # 1% of reference set size (~6 samples)
  exploration_pct_base: representatives  # or "dataset" for % of full dataset
  exploration_interval: 4              # explore every 4 selection rounds
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

---

## Variance Predictor Framework

### Motivation

The original DOTS interpolation is a fixed-bandwidth cosine-KNN regressor
(Nadaraya-Watson with softmax temperature). It has three fundamental
limitations:

1. **Pre-aggregation destroys provenance.** `_compute_time_weighted_variances`
   collapses per-sample observations into a single scalar *before* the
   predictor sees them, discarding per-observation step and n_rollouts info.
2. **Time decay in labels, not fit.** The decay factor `exp(-lambda * dt)`
   weights labels, but the predictor itself has no concept of staleness.
3. **No learned parameters, no uncertainty.** The same fixed medoids are probed
   every round regardless of where the predictor is weakest.

The `VariancePredictor` framework (`variance_predictor.py`) replaces the inline
DOTS calls with a pluggable predictor that receives raw observation rows and
decides internally how to weight, fit, and predict.

### Architecture

```
_rollout_buffer
    │
    ▼
_build_observation_rows()          ← raw (npz_idx, step, rewards_array) rows
    │
    ▼
predictor.fit(obs, emb, step)      ← time-decay in fit weights, not labels
    │
    ▼
predictor.predict(emb)             ← PredictionResult(predicted_var, predicted_mean, uncertainty)
    │
    ▼
_select_interpolated / _select_frozen_reweight   ← unchanged allocation logic
```

### Predictor Types

| Type | `predictor_type` | Description |
|------|-----------------|-------------|
| **KNN** | `knn` (default) | Legacy cosine-KNN / Nadaraya-Watson. Zero behavior change from the original code. |
| **Ridge** | `ridge` | Two-head weighted Ridge regression. Head A predicts mean reward `p_hat` (per-rollout supervision); Head B predicts normalized variance `v_hat` directly (per-sample supervision). Returns Head B's output as `predicted_var`, Head A's as `predicted_mean`. Closed-form solution, millisecond refit. Also exposes Head A's Bayesian posterior for active probing. |
| **MLP** | `mlp` | Two-head 2-layer MLP (`d -> hidden -> 2`). Head [0] is the mean-reward logit (BCE loss on per-rollout binary rewards); Head [1] is the direct normalized-variance logit (weighted MSE on per-sample aggregates, sigmoid-squashed to [0,1]). Warm-started across rounds — parameters and optimizer state persist. 5-20 AdamW steps per `fit()` call. |

### Two-head design (Ridge and MLP)

**Why two heads, not one.** The original single-head design regressed on mean
reward `p_hat` and derived variance analytically as `v_hat = p_hat * (1 - p_hat)`.
For binary GRPO rewards this derivation is theoretically correct for a single
Bernoulli trial, but it has a hard consequence at prediction time: the parabola
`p*(1−p)` peaks at **0.25** when `p=0.5`. So every scatter plot of
`predicted_var` vs observed normalized variance ended up with the predicted
axis capped at `[0, 0.25]` while the observed axis spanned `[0, 1]`, making
`R²` meaningless and visually misleading (KNN's free-range interpolator
appeared to "fit" vastly better even though its LOO R² was negative).

Beyond the cosmetic issue, the derivation conflates two distinct quantities
that matter for data selection:

- **Theoretical Bernoulli variance** `p(1−p)` — what head A's derivation
  reports. Informs "where does the policy sit around 50/50?"
- **Normalized empirical variance** `Var(rewards) / (p(1−p))` ∈ `[0, 1]` —
  what the selector actually ranks by. Informs "how much of the theoretical
  maximum uncertainty does this particular prompt realize under this policy?"

A prompt with steady 4/8 splits across rounds has normalized variance ≈ 1
(saturated learning signal), while a prompt that alternates 8/0 and 0/8
across rounds at `p_avg ≈ 0.5` also looks like theoretical variance 0.25 to
Head A but has normalized variance much lower. The ranking target and the
derivation target are not the same thing.

**What the two heads do.**

| | Head A (mean) | Head B (variance) |
|---|---|---|
| Target | per-rollout binary reward | per-sample time-weighted normalized variance |
| Loss | BCE (Ridge: weighted MSE; MLP: BCEwLogits) | weighted MSE on `v_obs ∈ [0, 1]` |
| Rows | 1 per rollout (8× per probed sample) | 1 per probed sample |
| Output at predict | `p_hat ∈ [0, 1]` | `v_hat ∈ [0, 1]` (Ridge: clip; MLP: sigmoid) |

- **`predicted_mean`** still comes from Head A, so asymmetric utility
  (`utility = v̂ · (1 + α·(0.5 − p̂))`), dead-zone filtering, and the Ridge
  Bayesian posterior used by active probes all keep working unchanged.
- **`predicted_var`** now comes from Head B. No 0.25 cap, full dynamic range,
  directly optimized for the quantity the selector will rank on.

**Cold-start behavior.** Before the first fit with enough observed samples,
Head B falls back to the old `p(1−p)` derivation automatically. Subsequent
rounds use the direct head once `obs_var_per_sample` is non-empty. For MLP
specifically, the variance head sigmoid starts at 0.5 and warms up over the
first 3–5 selection rounds as gradients accumulate — this is visible in the
scatter plot as a range that expands from `[0.5, 0.5]` toward `[0, 1]` over
the first few rounds.

**Supervision density.** Head A keeps the 8× supervision benefit of
per-rollout rows (200 probed samples × 8 rollouts = 1,600 training rows).
Head B trains on 200 sample-level rows — much less — but the target is
already aggregated over all rollouts, so the signal-to-noise ratio per row
is much higher. They use the same trunk (MLP) or operate independently
(Ridge) so the total fit cost is within 20% of the single-head version.

**What this affects downstream (automatic).** The following consumers all
pull `result.predicted_var` / `result.predicted_mean` directly, so they
automatically pick up Head B's output:

- `_select_interpolated` — global top-k and per-cluster allocation both
  rank by Head B now.
- `_select_frozen_reweight` — variance-weighted sampling in the frozen pool.
- `predictor_train_r2`, `predictor_train_spearman`, `loo_knn_r2` — diagnostic
  metrics computed at reference points against observed normalized variance.
  These become honest: Ridge/MLP now target the same axis as KNN, so the
  metrics are directly comparable.
- `data_selection/pred_vs_obs_scatter` wandb image — the scatter is
  automatically uncapped, so ridge/MLP plots will span `[0, 1]` on both axes.
- Asymmetric utility and dead-zone filtering — unchanged, still read
  `predicted_mean` from Head A.

**Legacy metric still logged.** `predictor_train_r2_vs_pmean_var` compares
`predicted_var` against `p_empirical · (1 − p_empirical)`. Under the old
single-head design this was the *fair* variance metric; under the two-head
design it now measures how much Head B's direct prediction diverges from
Head A's derived prediction. A large divergence is informative — it means
the dataset's empirical variance structure is not Bernoulli-like and
Head B is capturing structure the derived head cannot.

### Active Probe Selection (Ridge only)

When `predictor_active_probes=true` and `predictor_type=ridge`, the fixed
medoid probe set is replaced with UCB-based acquisition:

```
acq(x) = utility(x) + beta * sigma(x)
```

where `sigma(x)` is the Ridge posterior standard deviation. Points are selected
greedily with diversity suppression (nearby points within cosine distance
`predictor_active_probe_suppress_radius` are masked after each pick). This
closes the loop: the regressor decides where to probe, those rollouts shrink
uncertainty where it matters, and the next round's selection is sharper.

### Validation Metrics

Every selection round logs predictor diagnostics to wandb:

| Metric | Description |
|--------|-------------|
| `data_selection/predictor_train_r2` | R² of predicted vs observed variance at reference points (training error) |
| `data_selection/predictor_train_mae` | Mean absolute error at reference points |
| `data_selection/predictor_train_spearman` | Spearman rank correlation at reference points |
| `data_selection/loo_knn_r2` | **Leave-One-Out KNN R²** — the embedding ceiling. If < 0.15, no regressor can help. |
| `data_selection/loo_knn_mae` | LOO-KNN mean absolute error |
| `data_selection/loo_knn_spearman` | LOO-KNN Spearman rank correlation |
| `data_selection/n_reference_points` | Number of reference points used |
| `data_selection/predictor_type` | 0=knn, 1=ridge, 2=mlp |

A **predicted vs observed scatter plot** (`data_selection/pred_vs_obs_scatter`)
is generated as a wandb image at each selection round.

> **Under the two-head design**, `predictor_train_r2` for Ridge/MLP is now
> directly comparable to KNN — all three predictors are being scored against
> the same target (observed normalized variance on `[0, 1]`). Before the
> two-head fix, Ridge/MLP's `predicted_var` was capped at 0.25 while the
> target spanned `[0, 1]`, so `R²` was dominated by a constant systematic
> offset. A meaningful KNN-vs-Ridge comparison should now use
> `predictor_train_r2` (training error, risks overfitting on KNN) together
> with `loo_knn_r2` (embedding ceiling, independent of which predictor is
> active) — if `loo_knn_r2` is negative, no predictor will help regardless
> of type. KNN's apparent `predictor_train_r2 ≈ 0.9` is a memorisation
> artefact (each ref is nearly its own top neighbor); the honest number is
> `loo_knn_r2`.

### Configuration

All config lives in `ClusterSelectorConfig` (passed via Hydra):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predictor_type` | `knn` | `"knn"`, `"ridge"`, or `"mlp"` |
| `predictor_alpha` | `1.0` | Ridge regularization strength |
| `predictor_mlp_hidden` | `256` | MLP hidden dimension |
| `predictor_mlp_lr` | `1e-3` | MLP learning rate |
| `predictor_mlp_steps` | `10` | Gradient steps per `fit()` call |
| `predictor_mlp_weight_decay` | `1e-3` | MLP weight decay |
| `predictor_active_probes` | `false` | Enable UCB-based active probe selection |
| `predictor_ucb_beta` | `1.0` | UCB exploration coefficient |
| `predictor_active_probe_suppress_radius` | `0.1` | Cosine-distance diversity suppression |

### Shell Script Usage

```bash
# Default: KNN baseline (zero behavior change)
PREDICTOR_TYPE=knn bash run_qwen3_vl-2b_online_selection.sh

# Ridge predictor with joint p-hat head
PREDICTOR_TYPE=ridge bash run_qwen3_vl-2b_online_selection.sh

# Ridge with stronger regularization
PREDICTOR_TYPE=ridge PREDICTOR_ALPHA=10.0 bash run_qwen3_vl-2b_online_selection.sh

# MLP predictor (warm-started across rounds)
PREDICTOR_TYPE=mlp bash run_qwen3_vl-2b_online_selection.sh

# MLP with custom hyperparameters
PREDICTOR_TYPE=mlp PREDICTOR_MLP_HIDDEN=512 PREDICTOR_MLP_STEPS=20 \
    bash run_qwen3_vl-2b_online_selection.sh

# Ridge + active probes (uncertainty-driven probe selection)
PREDICTOR_TYPE=ridge ACTIVE_PROBES=true bash run_qwen3_vl-2b_online_selection.sh

# Ridge + active probes with higher exploration
PREDICTOR_TYPE=ridge ACTIVE_PROBES=true ACTIVE_PROBES_UCB_BETA=2.0 \
    bash run_qwen3_vl-2b_online_selection.sh
```

### Files

| File | Description |
|------|-------------|
| `variance_predictor.py` | `VariancePredictor` ABC, `PredictionResult`, `KNNPredictor`, `RidgePredictor`, `MLPPredictor`, `build_predictor()`, diagnostic utilities |
| `cluster_selector.py` | Integration: `_build_observation_rows()`, predictor wiring in `_select_interpolated` / `_select_frozen_reweight` / `get_reference_indices` / state dict |

### Checkpoint Compatibility

Predictor state is saved in `data_selector.pt` alongside existing selector
state. Old checkpoints without predictor state load cleanly (the predictor
starts from scratch). MLP optimizer and model weights are persisted for
warm-start across resume boundaries.
