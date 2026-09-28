#!/usr/bin/env bash
# Canonical clean_v1 seed-sweep preset for the ViRL39K online KNN selector.
#
# Across seeds this holds the dataset split, cluster artifacts, selector
# hyperparameters, unique-data budget, and number of optimizer steps fixed.
# RUN_SEED is propagated through Python hashing, the dataset loader, selector,
# FSDP actor, actor data loader, and rollout sampler by the shared launcher.
#
# Deliberately disabled until separately validated: CAVA/VDR gates, asymmetric
# utility/dead zones, exploration, and active probes. Medoid probe samples count
# toward the exact 10% cumulative unique-data cap.
#
# Usage:
#   RUN_SEED=42 bash run_qwen3_vl-2b_virl39k_seeded_knn_clean.sh
#   RUN_SEED=123 bash run_qwen3_vl-2b_virl39k_seeded_knn_clean.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
RUN_SEED="${RUN_SEED:-42}"
RESUME_MODE="${RESUME_MODE:-disable}"
FULL_DETERMINISM="${FULL_DETERMINISM:-true}"

if ! [[ "$RUN_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RUN_SEED must be a non-negative integer, got '$RUN_SEED'" >&2
    exit 1
fi
case "$RESUME_MODE" in
    disable) WANDB_RESUME_DEFAULT=never ;;
    auto|resume_path) WANDB_RESUME_DEFAULT=allow ;;
    *)
        echo "ERROR: RESUME_MODE must be disable, auto, or resume_path" >&2
        exit 1
        ;;
esac
export RUN_SEED RESUME_MODE FULL_DETERMINISM
export WANDB_RESUME="${WANDB_RESUME:-$WANDB_RESUME_DEFAULT}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100_stratified_seed1234}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100_stratified_seed1234}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"
CLUSTER_ARRAYS="${CLUSTER_ARRAYS:-${DS_ROOT}/cluster_arrays/${TRAIN_SPLIT}/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz}"
DATASET_JSON="${DATASET_JSON:-${DS_ROOT}/records/${TRAIN_SPLIT}.jsonl}"
FULL_SOURCE_JSON="${FULL_SOURCE_JSON:-${DS_ROOT}/records/all.jsonl}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON" "$FULL_SOURCE_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd /workspace/rl_data_selection/benchmark/rl_data_selection && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

FULL_SOURCE_SIZE=$(wc -l < "$FULL_SOURCE_JSON")
CANONICAL_BUDGET_COUNT=$((FULL_SOURCE_SIZE * 10 / 100))
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "[clean_v1] Source-level 10% cap: ${CANONICAL_BUDGET_COUNT}/${FULL_SOURCE_SIZE} prompts"
echo "[clean_v1] Input artifact SHA-256:"
sha256sum "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON" "$FULL_SOURCE_JSON"
echo "[clean_v1] Critical source SHA-256:"
sha256sum \
    "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_seeded_knn_clean.sh" \
    "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$VERL_ROOT/verl/trainer/ppo/data_selector/cluster_selector.py" \
    "$VERL_ROOT/verl/trainer/ppo/data_selector/variance_predictor.py" \
    "$VERL_ROOT/verl/trainer/ppo/ray_trainer.py"
if git -C "$VERL_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[clean_v1] verl commit: $(git -C "$VERL_ROOT" rev-parse HEAD)"
    if ! git -C "$VERL_ROOT" diff --quiet --ignore-submodules -- ||
       ! git -C "$VERL_ROOT" diff --cached --quiet --ignore-submodules --; then
        echo "[clean_v1] WARNING: tracked verl sources are dirty; commit before a paper run." >&2
    fi
fi

if [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="virl39k_clean_v1_seed${RUN_SEED}_knn_k${K_FINAL}_r${N_REPS}_budget10_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${WANDB_RUN_ID}}"
export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_seeded_clean"

# Phased discovery spans the 300-step run instead of exhausting the pool early.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:1.0,interval:4},{until_budget_pct:85,per_round_pct:0.3,interval:10},{until_budget_pct:100,per_round_pct:0.15,interval:16}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.58}"
# The exact source-level 10% cap includes medoid probes and matches
# make_random_subset.py's random-10% baseline.
export GLOBAL_BUDGET_COUNT="${GLOBAL_BUDGET_COUNT:-$CANONICAL_BUDGET_COUNT}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"

# Locked clean_v1 selector choices.
export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-false}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.0}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-1.0}"
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export SELECTION_REWARD_KEY="${SELECTION_REWARD_KEY:-acc}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-true}"
export REPRESENTATIVE_METHOD="${REPRESENTATIVE_METHOD:-density_diverse}"
export EXCLUDE_ALREADY_SELECTED="${EXCLUDE_ALREADY_SELECTED:-true}"
export REROLL_MEDOIDS="${REROLL_MEDOIDS:-false}"
export EXPLORATION_ENABLED="${EXPLORATION_ENABLED:-false}"
export ACTIVE_PROBES="${ACTIVE_PROBES:-false}"

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
    trainer.project_name=verl_grpo_virl39k_seeded_clean \
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
