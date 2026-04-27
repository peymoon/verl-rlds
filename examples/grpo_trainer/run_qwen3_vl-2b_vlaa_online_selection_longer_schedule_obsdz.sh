#!/usr/bin/env bash
# DISCOVER headline configuration on VLAA-Thinking-GRPO-25K:
#   slow discovery cadence + observed dead-zone (tau_obs=0.80) + asymmetric utility.
#
# This is the VLAA equivalent of run_qwen3_vl-2b_virl39k_online_selection_longer_schedule_obsdz.sh
# and the canonical "DISCOVER + obs. dead-zone (default)" Table 1 row in §4.3.
#
# Two roles depending on the corpus fork (see docs/PAPER_STATUS_AND_PLAN §5.1):
#   - Option A (VLAA headline): this is THE headline DISCOVER run.
#   - Option B (ViRL39K headline): this is the VLAA generalisation row that
#     confirms the cadence finding transfers to a smaller, more-curated corpus.
#
# Cadence design — adapted from the ViRL39K longer_schedule that produced 0.512.
# VLAA differs in three ways that motivate a different schedule:
#   (a) N = 22,675 vs ViRL39K's 34,983 — a smaller candidate pool means each
#       per_round_pct buys absolutely fewer samples.
#   (b) K = 200 (vs 150) — the per-round floor of "1 sample per non-empty cluster"
#       sits at ~200 rather than ~150.
#   (c) The 10% global cap is 2,268 samples — so with the K=200 floor, only
#       ~11 discovery rounds fit at full saturation.
# Schedule below is tuned to exhaust the global cap near step ~270, matching
# the relative cadence of the ViRL39K longer_schedule (which exhausted at ~272):
#   Phase 1: per_round=1.0%,  interval=10  — ~6 rounds × 10 = ~60 steps  (→ 50%)
#   Phase 2: per_round=0.4%,  interval=30  — ~4 rounds × 30 = ~120 steps (→ 85%)
#   Phase 3: per_round=0.15%, interval=50  — ~2 rounds × 50 = ~100 steps (→ 100%)
#   Expected exhaustion: step ~280. Floors at K=200 will dominate per_round_pct
#   when per_round_pct × N < 200.
#
# If Phase 1 logs show exhaustion arriving much earlier than step 60 on the first
# launch, re-launch with intervals scaled up by the observed-vs-expected ratio.
#
# Required prerequisite — verify these exist before launching:
#   /workspace/rl_data_selection/data/vlaa_parquet_splits/train_90_100.parquet
#   /workspace/rl_data_selection/data/vlaa_parquet_splits/test_10_100.parquet
#   /workspace/rl_data_selection/data/VLAA-Thinking/VLAA-Thinking-GRPO-25K_train_90_100.json
#   /workspace/rl_data_selection/benchmark/rl_data_selection/cluster_selection/outputs_200_cluster_new/cluster_arrays.npz
#
# Usage:
#   bash run_qwen3_vl-2b_vlaa_online_selection_longer_schedule_obsdz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----- VLAA paths -----
VLAA_ROOT="${VLAA_ROOT:-/workspace/rl_data_selection/data}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${VLAA_ROOT}/vlaa_parquet_splits/train_90_100.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${VLAA_ROOT}/vlaa_parquet_splits/test_10_100.parquet}"
DATASET_JSON="${DATASET_JSON:-${VLAA_ROOT}/VLAA-Thinking/VLAA-Thinking-GRPO-25K_train_90_100.json}"

# ----- Cluster artefact (paper §3.2 specifies K=200, n_reps=3 for VLAA) -----
K_FINAL="${K_FINAL:-200}"
N_REPS="${N_REPS:-3}"
CLUSTER_ARRAYS="${CLUSTER_ARRAYS:-/workspace/rl_data_selection/benchmark/rl_data_selection/cluster_selection/outputs_200_cluster_new/cluster_arrays.npz}"

export WANDB_API_KEY='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'

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        exit 1
    fi
done

# ----- Slow discovery cadence (VLAA-tuned; see header) -----
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:1.0,interval:10},{until_budget_pct:85,per_round_pct:0.4,interval:30},{until_budget_pct:100,per_round_pct:0.15,interval:50}]'

# Fallback values when no phase matches — used only at the very last fragment
# of the schedule. These are scaled to VLAA (N=22,675).
export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.58}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-10}"

# ----- Predictor -----
export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

# ----- Asymmetric utility (paper §3.5 default) -----
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-true}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.05}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-0.95}"

# ----- Observed dead-zone (paper §3.6, the new contribution) -----
OBS_DEAD_ZONE_HIGH="${OBS_DEAD_ZONE_HIGH:-0.80}"

# ----- Other knobs -----
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
VARIANT="${VARIANT:-interpolated_weighted}"

export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_vlaa_discover"

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$ENGINE" \
    "$CLUSTER_ARRAYS" \
    "$VARIANT" \
    "$DATASET_JSON" \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data_selection.cluster.n_clusters="${K_FINAL}" \
    data_selection.cluster.n_reps="${N_REPS}" \
    data_selection.cluster.observed_deadzone_high="${OBS_DEAD_ZONE_HIGH}" \
    trainer.project_name=verl_grpo_vlaa_discover \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-vlaa_knn_k200_longer_schedule_obsdz080}" \
    +trainer.total_training_steps=300 \
    "$@"
