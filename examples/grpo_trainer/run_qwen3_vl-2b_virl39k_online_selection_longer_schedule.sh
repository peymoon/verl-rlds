#!/usr/bin/env bash
# Stretched budget schedule — exhaust 10% annotation budget closer to step 300
# instead of step 176 (observed on the main KNN run).
#
# Why: on the original schedule the budget cap was hit at step 176/300, leaving
# 41% of training in frozen reweight. Pushing the exhaustion point later keeps
# the predictor learning from on-policy rollouts longer, increases the number
# of discovery rounds, and reduces the window during which new samples cannot
# be added.
#
# Revised schedule (note: per_round_pct is floored at 1-sample-per-non-empty-
# cluster ≈ 150/round, so the effective per-round budget differs from the raw
# pct once per_round_pct × N < 150):
#
#   Phase 1: per_round=0.7%, interval=5  — ~7 rounds × 5 = 35 steps  (→ 50%)
#   Phase 2: per_round=0.25%, interval=15 — ~9 rounds × 15 = 137 steps (→ 85%)
#   Phase 3: per_round=0.10%, interval=25 — ~4 rounds × 25 = 100 steps (→ 100%)
#   Expected exhaustion: step ~272 (vs 176 on the original schedule).
#
# Hypothesis: if this run beats the main KNN run at the same 10% budget,
# the discovery cadence was too front-loaded. If it matches or loses, the
# schedule shape doesn't matter — supports the "discovery is not the driver"
# story from the cluster-only and immediate-freeze ablations.
#
# Required prerequisite:
#   cd /workspace/rl_data_selection/benchmark/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_online_selection_longer_schedule.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
export WANDB_API_KEY='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'
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100_stratified_seed1234}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100_stratified_seed1234}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"
CLUSTER_ARRAYS="${CLUSTER_ARRAYS:-${DS_ROOT}/cluster_arrays/${TRAIN_SPLIT}/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz}"
DATASET_JSON="${DATASET_JSON:-${DS_ROOT}/records/${TRAIN_SPLIT}.jsonl}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd $(cd "$DS_ROOT/.." && pwd) && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

# Stretched schedule. Intervals chosen so total discovery steps ≈ 272.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:0.7,interval:5},{until_budget_pct:85,per_round_pct:0.25,interval:15},{until_budget_pct:100,per_round_pct:0.10,interval:25}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.33}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-5}"

export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
VARIANT="${VARIANT:-interpolated_weighted}"

export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$ENGINE" \
    "$CLUSTER_ARRAYS" \
    "$VARIANT" \
    "$DATASET_JSON" \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data_selection.cluster.n_clusters="${K_FINAL}" \
    data_selection.cluster.n_reps="${N_REPS}" \
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-virl39k_knn_k150_longer_schedule}" \
    +trainer.total_training_steps=300 \
    "$@"
