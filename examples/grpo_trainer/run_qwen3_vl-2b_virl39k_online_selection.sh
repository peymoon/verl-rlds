#!/usr/bin/env bash
# ViRL39K online data-selection run for Qwen3-VL-2B.
#
# Delegates to run_qwen3_vl-2b_online_selection.sh after pointing the train
# parquet, val parquet, cluster_arrays, and dataset_json at the ViRL39K
# artifacts produced by `bash dataset_prep/prepare.sh virl39k`.
#
# Required prerequisite:
#   cd /workspace/rl_data_selection/benchmark/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# That pipeline writes:
#   data/virl39k/parquet/train_90_100.parquet
#   data/virl39k/parquet/test_10_100.parquet
#   data/virl39k/records/all.jsonl
#   data/virl39k/cluster_arrays/outputs_K${K}_r${N_REPS}/cluster_arrays.npz
#
# Env vars (overridable):
#   DATA_ROOT        root under which data/virl39k/ lives
#                    (default: /workspace/rl_data_selection/benchmark/rl_data_selection/data)
#   K_FINAL          K used when cluster_arrays was built (default: 300)
#   N_REPS           n_reps used when cluster_arrays was built (default: 3)
#   TRAIN_SPLIT      parquet stem to train on (default: train_90_100)
#   TEST_SPLIT       parquet stem for validation (default: test_10_100)
#   PREDICTOR_TYPE   knn | ridge | mlp (default: knn)
#   SELECTION_BUDGET_PCT, GLOBAL_BUDGET_PCT, BUDGET_SCHEDULE  — forwarded through
#   CUDA_VISIBLE_DEVICES — as usual
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_online_selection.sh
#   PREDICTOR_TYPE=ridge ACTIVE_PROBES=true \
#       bash run_qwen3_vl-2b_virl39k_online_selection.sh
#   K_FINAL=200 N_REPS=3 \
#       bash run_qwen3_vl-2b_virl39k_online_selection.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
export WANDB_API_KEY='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'
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"
CLUSTER_ARRAYS="${DS_ROOT}/cluster_arrays_90_100/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz"
DATASET_JSON="${DS_ROOT}/records/all.jsonl"

# Sanity checks — fail fast with an actionable message instead of letting
# verl complain about a missing parquet 300 lines into the launch.
for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd $(cd "$DS_ROOT/.." && pwd) && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

# ViRL39K is ~39k samples. With n_reps=3 × K=150 = 450 medoid probes
# (3,600 sequences per round at n=8), each cluster gets 24 binary rollout
# samples for variance estimation — cleaner early-round cluster scores than
# n_reps=2. The phased schedule (50% of budget at per_round=1.0, interval=4)
# runs ~5 discovery rounds before tapering.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:1.0,interval:4},{until_budget_pct:85,per_round_pct:0.3,interval:10},{until_budget_pct:100,per_round_pct:0.15,interval:16}]'

# Slightly smaller per-round increment than VLAA since 1% of 39k (390
# samples) is already > 2 batches (256) and would overshoot the
# discovery budget by end of round. 0.33% ≈ 128 samples = one batch.
export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.33}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-4}"

export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

# ViRL39K categories are genuine topic tags (math/spatial/tables/...) so
# the IGS multimodal signal and composite scoring are both informative.
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

# verl positional args: ENGINE CLUSTER_ARRAYS VARIANT DATASET_JSON
if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
VARIANT="${VARIANT:-interpolated_weighted}"

# Override project name so wandb groups these runs separately from VLAA.
export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"

# The main launcher takes positional args for cluster_arrays / dataset_json;
# we pass our ViRL39K paths plus a project-name override through Hydra.
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
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-virl39k_knn_k150}" \
    +trainer.total_training_steps=300 \
    "$@"
