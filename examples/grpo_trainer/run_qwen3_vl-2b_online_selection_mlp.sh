#!/usr/bin/env bash
# Thin wrapper: online selection with the 2-layer MLP variance predictor.
#
# Delegates to run_qwen3_vl-2b_online_selection.sh with PREDICTOR_TYPE=mlp.
# The MLP is warm-started across selection rounds — weights and optimizer
# state persist, so each round takes a small number of AdamW steps on the
# growing observation buffer. Loss is per-rollout BCE with effective-sample
# -size normalization so gradient magnitudes stay stable as the buffer grows.
# Predictions are v_hat = σ(logit) * (1 - σ(logit)).
#
# What this run tests:
#   1. Does a non-linear predictor recover signal the Ridge head misses?
#      Primary diagnostic: data_selection/predictor_train_r2_vs_pmean_var
#      and data_selection/loo_knn_r2 over training rounds.
#   2. Warm-start stability — predictor_train_r2 should trend upward round
#      over round, NOT oscillate. If it oscillates, PREDICTOR_MLP_LR is too
#      high or PREDICTOR_MLP_STEPS too large.
#   3. Does the MLP suffer the same embedding-ceiling (LOO_R² ≈ 0) as KNN
#      and Ridge? If yes, the ceiling is the frozen Qwen3-VL embedding,
#      not the regressor — next move would be fine-tuning a small embedding
#      adapter instead of swapping predictor families again.
#
# Active probing requires posterior uncertainty, which the MLP does not
# expose. ACTIVE_PROBES is force-disabled in this wrapper.
#
# Env knobs specific to this run:
#   PREDICTOR_MLP_HIDDEN        hidden dim (default 256)
#   PREDICTOR_MLP_LR            AdamW lr (default 1e-3)
#   PREDICTOR_MLP_STEPS         inner steps per selection round (default 10)
#   PREDICTOR_MLP_WEIGHT_DECAY  AdamW wd (default 1e-3)
#
# Usage:
#   bash run_qwen3_vl-2b_online_selection_mlp.sh
#   PREDICTOR_MLP_HIDDEN=512 PREDICTOR_MLP_STEPS=20 \
#       bash run_qwen3_vl-2b_online_selection_mlp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PREDICTOR_TYPE=mlp
export PREDICTOR_MLP_HIDDEN=${PREDICTOR_MLP_HIDDEN:-256}
export PREDICTOR_MLP_LR=${PREDICTOR_MLP_LR:-1e-3}
export PREDICTOR_MLP_STEPS=${PREDICTOR_MLP_STEPS:-10}
export PREDICTOR_MLP_WEIGHT_DECAY=${PREDICTOR_MLP_WEIGHT_DECAY:-1e-3}

# MLP does not expose posterior uncertainty → active probing is a no-op and
# would silently fall back to fixed medoids. Disable explicitly.
export ACTIVE_PROBES=false

export NORMALIZE_VARIANCE=${NORMALIZE_VARIANCE:-true}
export COUNT_MEDOIDS_IN_BUDGET=${COUNT_MEDOIDS_IN_BUDGET:-false}

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" "$@"
