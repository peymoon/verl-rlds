#!/usr/bin/env bash
# Matched ViRL39K slow-cadence DISCOVER control for the online CAVA run.
#
# This intentionally matches the CAVA/VDR run's dataset, K=150/r=3 clusters,
# slow discovery cadence, global budget, asymmetric utility, and validation
# frequency, while disabling all visual-dependence gates and the observed
# dead-zone. Use it to isolate whether CAVA itself helped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

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
        echo "       Run: cd /workspace/rl_data_selection/benchmark/rl_data_selection && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

if [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="virl39k_slow_no_cava_k${K_FINAL}_r${N_REPS}_g1111_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${WANDB_RUN_ID}}"
export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"

# Same slow discovery cadence as the completed online CAVA/VDR run.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:0.7,interval:5},{until_budget_pct:85,per_round_pct:0.25,interval:15},{until_budget_pct:100,per_round_pct:0.10,interval:25}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.33}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-11.11}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-5}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"

export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-true}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.05}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-0.95}"
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
VARIANT="${VARIANT:-interpolated_weighted}"

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$ENGINE" \
    "$CLUSTER_ARRAYS" \
    "$VARIANT" \
    "$DATASET_JSON" \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data_selection.cluster.n_clusters="${K_FINAL}" \
    data_selection.cluster.n_reps="${N_REPS}" \
    data_selection.cluster.observed_deadzone_high=1.0 \
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    ++trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    ++data_selection.cluster.vdr_enabled=false \
    ++data_selection.cluster.cava_vdr_enabled=false \
    ++data_selection.cluster.cava_use_logp_contrast=false \
    ++data_selection.cluster.cava_weight_static_prior=0.0 \
    ++data_selection.cluster.cava_weight_logp_contrast=0.0 \
    "$@"
