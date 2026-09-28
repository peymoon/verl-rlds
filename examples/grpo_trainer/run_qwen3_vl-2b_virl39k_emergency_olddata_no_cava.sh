#!/usr/bin/env bash
# Emergency one-run recovery target for the paper.
#
# Goal: reproduce the strong old K150/r3 non-stratified KNN run as closely as
# possible while avoiding the CAVA gate, which changed the selected prompt pool
# substantially in the latest runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"

export WANDB_RESUME="${WANDB_RESUME:-never}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"

# Old good-run artifacts:
#   W&B:  run-20260416_091957-onpzqqtp
#   ckpt: v62_k300_r2_density_diverse_knn_0.33_global10.0_interpolated_weighted_asym0.5
TRAIN_PARQUET="${TRAIN_PARQUET:-${DS_ROOT}/parquet/train_90_100.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DS_ROOT}/parquet/test_10_100.parquet}"
CLUSTER_ARRAYS="${CLUSTER_ARRAYS:-${DS_ROOT}/cluster_arrays_90_100/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz}"
DATASET_JSON="${DATASET_JSON:-${DS_ROOT}/records/all.jsonl}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        exit 1
    fi
done

if [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="virl39k_emergency_olddata_no_cava_k${K_FINAL}_r${N_REPS}_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${WANDB_RUN_ID}}"
export WANDB_PROJECT_NAME_OVERRIDE="${WANDB_PROJECT_NAME_OVERRIDE:-verl_grpo_virl39k_baseline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

# Match the strong run's fast progressive discovery schedule.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:1.0,interval:4},{until_budget_pct:85,per_round_pct:0.3,interval:10},{until_budget_pct:100,per_round_pct:0.15,interval:16}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.33}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"

export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"
export REPRESENTATIVE_METHOD="${REPRESENTATIVE_METHOD:-density_diverse}"
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-true}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.05}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-0.95}"
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export SELECTION_REWARD_KEY="${SELECTION_REWARD_KEY:-reward}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

# Keep the observed-score dead zone disabled. The strong run used only the
# asymmetric utility dead zone.
OBS_DEAD_ZONE_HIGH="${OBS_DEAD_ZONE_HIGH:-1.0}"

# TP=1 was faster in the latest successful launches on this machine.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

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
    data_selection.cluster.observed_deadzone_high="${OBS_DEAD_ZONE_HIGH}" \
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    ++trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    ++data_selection.cluster.vdr_enabled=false \
    ++data_selection.cluster.cava_vdr_enabled=false \
    ++data_selection.cluster.cava_use_logp_contrast=false \
    ++data_selection.cluster.cava_apply_to_sample_utility=false \
    ++data_selection.cluster.cava_apply_to_cluster_allocation=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    "$@"
