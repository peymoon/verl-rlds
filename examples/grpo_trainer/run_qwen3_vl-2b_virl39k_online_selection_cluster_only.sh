#!/usr/bin/env bash
# Cluster-only ablation (no DOTS per-sample predictor).
#
# Isolates the contribution of the variance predictor. Keeps cluster-level
# allocation (weighted by cluster variance) but removes:
#   - DOTS per-sample interpolation (strategy=weighted instead of interpolated)
#   - composite scoring (no transferability × 1/density reweighting)
#   - asymmetric utility (no predicted_mean dependence)
#   - rollout-history buffer (no policy-tracking reference set)
#
# Hypothesis: if this run matches or beats the full online-KNN run, the
# per-sample predictor contributes nothing on ViRL39K (consistent with
# loo_knn_r2 ≈ −0.14 observed on the KNN run). Paper should then reframe
# the contribution as "cluster-structured online reweighting".
#
# Reuses the same phased budget schedule so the comparison is apples-to-apples
# on the discovery / frozen-reweight split.
#
# Required prerequisite:
#   cd /workspace/rl_data_selection/benchmark/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_online_selection_cluster_only.sh
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

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd $(cd "$DS_ROOT/.." && pwd) && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

# Single-shot schedule: select full 10% in round 0 via cluster variance, then
# frozen reweight for the remaining ~299 steps using cluster-level variance only
# (no per-sample KNN predictor). interval:1000 > total steps so no round 2 fires.
# This mirrors immediate_freeze but replaces KNN reweighting with cluster variance,
# making it a clean ablation of the predictor's contribution.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:100,per_round_pct:9.9,interval:1000}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-10.0}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-16}"

# Predictor type is irrelevant here — we override the strategy to bypass
# per-sample interpolation. Keep knn for log consistency.
export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-true}"

# Disable asymmetric utility — it depends on predicted_mean from Head A,
# which we are ablating out.
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-false}"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
# VARIANT must stay interpolated_weighted so the launcher wires the rollout
# history buffer on for frozen reweight, but we override the strategy below.
VARIANT="${VARIANT:-interpolated_weighted}"

export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"

# The key ablation: strategy=top_clusters picks top-k clusters by cluster-level
# variance and samples uniformly at random within each selected cluster.
# No DOTS interpolation, no per-sample variance predictor, no composite score.
#
# BUG FIX (previous run stuck at 8.2% budget): strategy=weighted under-selects
# because per-cluster softmax allocation with a small per_round_pct gets floored
# at ceiling(target/K) per cluster, which can become infeasible once clusters
# are exhausted of unseen samples. top_clusters allocates budget to the k
# highest-variance clusters without a softmax floor, so it exhausts the full
# budget over the schedule.
#
# We also force-disable dots_diversity + composite + asymmetric so the only
# signal is cluster-level variance.
exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$ENGINE" \
    "$CLUSTER_ARRAYS" \
    "$VARIANT" \
    "$DATASET_JSON" \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data_selection.cluster.n_clusters="${K_FINAL}" \
    data_selection.cluster.n_reps="${N_REPS}" \
    data_selection.cluster.strategy=top_clusters \
    data_selection.cluster.dots_diversity=false \
    data_selection.cluster.dots_diversity_use_composite_score=false \
    data_selection.cluster.asymmetric_utility_enabled=false \
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-virl39k_cluster_only_k150}" \
    +trainer.total_training_steps=300 \
    "$@"
