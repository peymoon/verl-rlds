#!/usr/bin/env bash
# Thin wrapper: online selection with the Ridge variance predictor.
#
# Delegates to run_qwen3_vl-2b_online_selection.sh with PREDICTOR_TYPE=ridge.
# Ridge fits closed-form weighted least squares on cached Qwen3-VL embeddings
# with per-rollout BCE-style supervision (each binary rollout outcome is one
# row), predicts p_hat and v_hat = p_hat*(1-p_hat), and exposes posterior
# uncertainty σ(x) = sqrt(α · xᵀ(XᵀWX + αI)⁻¹ x) that is used for UCB active
# probing in discovery rounds.
#
# What this run tests:
#   1. Does a learned predictor beat the legacy cosine-KNN interpolation
#      (v6 fixed) at the same global budget? Compare val_accuracy and
#      predictor_train_r2_vs_pmean_var over training.
#   2. Does UCB-driven active probing (ACTIVE_PROBES=true) improve
#      sample-efficiency vs fixed-medoid probing?
#   3. Does predictor uncertainty correlate with observed error?
#      (read `data_selection/predictor_uncertainty_*` from wandb, compare
#       against residuals on the diagnostic LOO set.)
#
# Env knobs specific to this run (overridable):
#   PREDICTOR_ALPHA          Ridge L2 penalty, default 1.0
#   ACTIVE_PROBES            true (default here) — use UCB acquisition
#   ACTIVE_PROBES_UCB_BETA   β in acq = utility + β·σ, default 1.0
#   ACTIVE_PROBES_SUPPRESS_RADIUS  cosine suppression radius, default 0.1
#
# Usage:
#   bash run_qwen3_vl-2b_online_selection_ridge.sh
#   PREDICTOR_ALPHA=0.3 ACTIVE_PROBES_UCB_BETA=1.5 \
#       bash run_qwen3_vl-2b_online_selection_ridge.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PREDICTOR_TYPE=ridge
export PREDICTOR_ALPHA=${PREDICTOR_ALPHA:-1.0}
export ACTIVE_PROBES=${ACTIVE_PROBES:-true}
export ACTIVE_PROBES_UCB_BETA=${ACTIVE_PROBES_UCB_BETA:-1.0}
export ACTIVE_PROBES_SUPPRESS_RADIUS=${ACTIVE_PROBES_SUPPRESS_RADIUS:-0.1}

# Ridge's joint p-hat head makes normalize_variance redundant (v_hat is
# already on [0, 0.25] via p*(1-p)). Keep it on for parity with KNN so the
# side-by-side comparison is apples-to-apples.
export NORMALIZE_VARIANCE=${NORMALIZE_VARIANCE:-true}
export COUNT_MEDOIDS_IN_BUDGET=${COUNT_MEDOIDS_IN_BUDGET:-false}

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" "$@"
