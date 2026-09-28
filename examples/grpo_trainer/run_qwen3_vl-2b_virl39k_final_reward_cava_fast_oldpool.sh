#!/usr/bin/env bash
# Final ViRL39K DISCOVER run: old shaped-reward selector signal + restored
# frozen-pool semantics + fast cadence + asymmetric utility + CAVA drop-vision
# log-prob gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
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
    WANDB_RUN_ID="virl39k_final_reward_cava_fast_oldpool_k${K_FINAL}_r${N_REPS}_seed1234_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${WANDB_RUN_ID}}"
export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

# Old fast discovery cadence from the high v62/knn_v2 scaffold.
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

# Keep observed dead-zone disabled; use the asymmetric dead-zone instead.
OBS_DEAD_ZONE_HIGH="${OBS_DEAD_ZONE_HIGH:-1.0}"

# CAVA/drop-vision online log-prob gate.
VDR_LOGP_MAX_SAMPLES="${VDR_LOGP_MAX_SAMPLES:-32}"
VDR_LOGP_MIN_POINTS="${VDR_LOGP_MIN_POINTS:-32}"
VDR_LOGP_SAMPLE_POLICY="${VDR_LOGP_SAMPLE_POLICY:-random}"
VDR_LOGP_DECAY_RATE="${VDR_LOGP_DECAY_RATE:-0.05}"
VDR_LOGP_MAX_AGE="${VDR_LOGP_MAX_AGE:-500}"
VDR_GATE_FLOOR="${VDR_GATE_FLOOR:-0.70}"
VDR_GATE_THRESHOLD="${VDR_GATE_THRESHOLD:-0.0}"
VDR_GATE_TEMPERATURE="${VDR_GATE_TEMPERATURE:-1.0}"

# The previous reward-key run used TP=1 and was much faster. Keep that default,
# but allow TENSOR_MODEL_PARALLEL_SIZE=4 if startup memory becomes an issue.
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
    ++data_selection.cluster.cava_vdr_enabled=true \
    ++data_selection.cluster.cava_weight_static_prior=0.0 \
    ++data_selection.cluster.cava_use_logp_contrast=true \
    ++data_selection.cluster.cava_weight_logp_contrast=1.0 \
    ++data_selection.cluster.cava_logp_decay_rate="${VDR_LOGP_DECAY_RATE}" \
    ++data_selection.cluster.cava_logp_max_age="${VDR_LOGP_MAX_AGE}" \
    ++data_selection.cluster.cava_logp_min_points_for_interp="${VDR_LOGP_MIN_POINTS}" \
    ++data_selection.cluster.cava_gate_floor="${VDR_GATE_FLOOR}" \
    ++data_selection.cluster.cava_gate_threshold="${VDR_GATE_THRESHOLD}" \
    ++data_selection.cluster.cava_gate_temperature="${VDR_GATE_TEMPERATURE}" \
    ++data_selection.cluster.cava_robust_normalize=true \
    ++data_selection.cluster.cava_apply_to_sample_utility=true \
    ++data_selection.cluster.cava_apply_to_cluster_allocation=false \
    ++data_selection.cluster.cava_cluster_gate_blend=0.0 \
    ++data_selection.cluster.cava_logp_every_n_steps=1 \
    ++data_selection.cluster.cava_logp_max_samples_per_step="${VDR_LOGP_MAX_SAMPLES}" \
    ++data_selection.cluster.cava_null_image_mode=drop_vision \
    ++data_selection.cluster.cava_logp_sample_policy="${VDR_LOGP_SAMPLE_POLICY}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    "$@"
