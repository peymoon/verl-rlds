#!/usr/bin/env bash
# Paper-faithful final preset: clean_v1 pipeline + the selector regime of the
# two historical ~13 runs (nonstrat_v62 13.12, virl39k_knn_150 12.84).
#
# Layers exactly three deltas on top of run_qwen3_vl-2b_virl39k_seeded_knn_clean.sh:
#
#   1. ASYMMETRIC_UTILITY=true, dead zone [0.05, 0.95], bias 0.5
#      - matches the 13.12 run config (asym0.5) and the paper's Eq. 6 utility
#        u_i = v_hat * (1 + 0.5*(0.5 - mu_hat)) * 1{0.05 <= mu_hat <= 0.95}.
#   2. SELECTION_REWARD_KEY=reward
#      - the selector consumes the full training reward (0.9*acc + 0.1*format),
#        the graded 4-level signal both ~13 runs used. The clean preset's
#        'acc' (binary) is what collapsed the variance signal under the old
#        Bernoulli-envelope normalization; with fixed-max scaling either key
#        is graded, but 'reward' reproduces the successful regime and matches
#        the paper, whose v_i(t) is the variance of the verifier reward r.
#   3. WANDB naming: final_v1 instead of clean_v1.
#
# Everything else inherits the audited clean pipeline: exact qid bijection
# (fail-closed), min(4*var,1) scaling, exact 3,887-prompt source-level cap
# INCLUDING medoid probes, full RUN_SEED wiring, CAVA/VDR/exploration/active
# probes off, no resume from legacy checkpoints.
#
# NOTE the one deliberate difference from the 13.12 run: that run did NOT
# count its 449 probes in the cap (3,947 uniques, 11.31% exposure). This
# preset keeps COUNT_MEDOIDS_IN_BUDGET=true so the run matches the paper's
# claimed 10.00% total verified exposure. Do not flip it without also
# changing the paper's exposure table.
#
# Evaluate BOTH step 240 and step 300 (pre-registered endpoints).
# Regime signature to watch on val-core/TIGER-Lab/ViRL39K/acc/mean@1:
#   - still rising at step 300  -> the 13.12-like regime (expect 300 >= 240)
#   - peaking ~step 270-280 then declining -> overfit regime (expect 240 > 300)
#
# Usage:
#   RUN_SEED=42 bash run_qwen3_vl-2b_virl39k_seeded_knn_final.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-true}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.05}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-0.95}"

# v2 fix (defect 2): 'acc' is binary, so min(4*var,1) yields the intended
# symmetric 5-level signal. With the composite 'reward' key the same scaling
# sharpens peak/tail selection from 1.12x to 2.77x versus the historical run,
# over-concentrating the pool on 4/8-difficulty prompts.
export SELECTION_REWARD_KEY="${SELECTION_REWARD_KEY:-acc}"

# v2 fix (defect 1): the inherited schedule could not exhaust the cap inside
# 300 steps once medoid probes counted toward it, so the pool never froze and
# every round trained on a padded 128-row single-batch shard. Each phase here
# stays above train_batch_size=128 (0.4% of 34,983 = 140) and the cap is
# reached near step 150-170, restoring the frozen-replay phase.
# clean.sh does `unset BUDGET_SCHEDULE` before hardcoding its own, so this must
# be a trailing Hydra override (++ forces over its earlier + assignment).
# Simulated: 50% @step20, 85% @step90, cap/freeze @step154 -> 146 replay steps,
# closely matching the historical run's 176/124 discovery-to-replay balance.
#
# NOTE: this must NOT be written as BUDGET_SCHEDULE_V2="${BUDGET_SCHEDULE_V2:-literal}".
# bash mis-parses literal unescaped {..},{..} inside a ${VAR:-default} word —
# it silently drops the first '}' and appends a stray one at the end, which
# broke Hydra's override grammar on the first v2 launch attempt. Verified by
# reproducing byte-for-byte in isolation before landing this fix.
if [ -z "${BUDGET_SCHEDULE_V2:-}" ]; then
    BUDGET_SCHEDULE_V2='[{until_budget_pct:50,per_round_pct:1.0,interval:4},{until_budget_pct:85,per_round_pct:0.5,interval:10},{until_budget_pct:100,per_round_pct:0.4,interval:16}]'
fi

RUN_SEED="${RUN_SEED:-42}"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
if [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="virl39k_final_v2_seed${RUN_SEED}_knn_k${K_FINAL}_r${N_REPS}_asym_acc_budget10_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID RUN_SEED K_FINAL N_REPS

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_seeded_knn_clean.sh" \
    "$@" \
    ++data_selection.cluster.budget_schedule="${BUDGET_SCHEDULE_V2}"
