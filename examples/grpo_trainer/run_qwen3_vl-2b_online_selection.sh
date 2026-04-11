set -x
# Online data selection: train on the FULL dataset, let the selector choose
# the training subset adaptively each epoch based on current model capability.
#
# Key differences from the standard run script:
#   1. data.train_files points to the FULL dataset (not a pre-selected subset)
#   2. data_selection.method=cluster enables online cluster-based selection
#   3. data_selection.selection_budget_pct=10.0 selects top 10% each step
#   4. data_selection.global_budget_pct caps cumulative unique samples over the
#      entire run. Set equal to selection_budget_pct (e.g. both 10%) for a fair
#      comparison with a fixed random baseline. When hit, pool composition
#      freezes and reference/exploration rollouts are skipped (saves compute).
#      Periodic reselection rounds continue to reweight within the frozen pool
#      using training-batch reward variance (requires use_rollout_history=true).
#      Set via GLOBAL_BUDGET_PCT env var; default=null (no cap).
#   5. data_selection.cluster.cluster_arrays_file points to pre-computed clusters
#      (outputs_300_cluster_new/cluster_arrays.npz from Stage 1)
#   6. data_selection.cluster.dataset_json_file points to the JSON used to build
#      the embeddings — required for correct NPZ↔parquet index alignment
#   7. trainer.total_epochs is increased since each epoch trains on 10% of data
#
# Arguments:
#   ENGINE          vllm or sglang (default: vllm)
#   CLUSTER_ARRAYS  path to cluster_arrays.npz
#                   (default: outputs_300_cluster_new/cluster_arrays.npz)
#   VARIANT         interpolated_weighted (default) or interpolated
#   DATASET_JSON    path to the JSON/JSONL used to build the cluster embeddings
#                   REQUIRED for correct NPZ↔parquet row alignment.
#                   Without this, REPR rollouts and select() reference wrong parquet rows.
#                   (default: VLAA-Thinking-GRPO-25K_train_90_100.json)
#
# Variants:
#   interpolated_weighted  strategy=interpolated + dots_diversity=true
#                          + dots_diversity_use_composite_score=true
#                          + use_rollout_history=true
#                          Cluster allocation weighted by var×transferability×(1/density).
#                          Training-batch rollouts accumulate with time-decay into the
#                          DOTS reference set, growing from ~250 REPR rollouts to up to
#                          2000 diverse samples mid-training.
#   interpolated           strategy=interpolated + dots_diversity=true
#                          Fixed REPR medoids only (K=50 x n_reps=5 = 250 refs),
#                          no composite score. Use as ablation against
#                          interpolated_weighted to isolate the benefit of the
#                          rollout history buffer and composite scoring.
#
# Reselection cadence:
#   data_selection.reselect_schedule=step   → triggers every N training steps
#   data_selection.reselect_interval=10     → re-select every 10 steps
#   Cost: reference rollouts on 250 medoids at each round. Increase interval
#   if rollouts are expensive.
#
# Pre-requisite: run the offline cluster pipeline (stages 0-1) to produce
#   cluster_arrays.npz. See cluster_selection/README.md for instructions.
#
# Environment variables (optional):
#   ASYMMETRIC_UTILITY   false (default) | true — enable hard-side-biased
#                        utility = predicted_var * (1 + α*(0.5 - predicted_mean)).
#                        Requires VARIANT=interpolated_weighted or interpolated.
#                        See README §"Asymmetric utility" for details.
#   ASYMMETRIC_BIAS      α in the utility expression (default 0.5).
#   ASYMMETRIC_DEAD_LOW  lower dead-zone on predicted mean reward (default 0.05).
#   ASYMMETRIC_DEAD_HIGH upper dead-zone on predicted mean reward (default 0.95).
#
# Usage examples:
#   # Default: interpolated + time-weighted rollout history buffer
#   bash run_qwen3_vl-2b_online_selection.sh
#
#   # Same, but with asymmetric utility enabled (α=0.5, default dead-zone):
#   ASYMMETRIC_UTILITY=true bash run_qwen3_vl-2b_online_selection.sh
#
#   # Stronger hard-side bias and tighter dead-zone:
#   ASYMMETRIC_UTILITY=true ASYMMETRIC_BIAS=1.0 \
#       ASYMMETRIC_DEAD_LOW=0.1 ASYMMETRIC_DEAD_HIGH=0.9 \
#       bash run_qwen3_vl-2b_online_selection.sh
#
#   # Explicit defaults:
#   bash run_qwen3_vl-2b_online_selection.sh vllm /path/to/cluster_arrays.npz interpolated_weighted /path/to/dataset.json
#
#   # Without history buffer (baseline comparison):
#   bash run_qwen3_vl-2b_online_selection.sh vllm /path/to/cluster_arrays.npz interpolated /path/to/dataset.json
#
#   # Override CUDA devices:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_qwen3_vl-2b_online_selection.sh
#
#   # Fair comparison with random 10% (freeze after first smart selection):
#   GLOBAL_BUDGET_PCT=10.0 bash run_qwen3_vl-2b_online_selection.sh
#
#   # Pass extra Hydra overrides (appended after all positional params):
#   bash run_qwen3_vl-2b_online_selection.sh vllm /path/cluster.npz interpolated_weighted /path/dataset.json \
#       data_selection.cluster.rollout_history_decay_rate=0.1 \
#       trainer.total_epochs=20

print_help() {
        cat <<'EOF'
Usage:
    run_qwen3_vl-2b_online_selection.sh [ENGINE] [CLUSTER_ARRAYS] [VARIANT] [DATASET_JSON] [HYDRA_OVERRIDES...]

Positional args:
    ENGINE          vllm | sglang (default: vllm)
    CLUSTER_ARRAYS  path to cluster_arrays.npz
                                    (default: /workspace/rl_data_selection/benchmark/rl_data_selection/cluster_selection/outputs_300_cluster_new/cluster_arrays.npz)
    VARIANT         interpolated_weighted | interpolated (default: interpolated_weighted)
    DATASET_JSON    JSON/JSONL used to build cluster embeddings (required for NPZ↔parquet remap)
                                    (default: /workspace/rl_data_selection/data/VLAA-Thinking/VLAA-Thinking-GRPO-25K_train_90_100.json)

Important selector method options (from cluster_selector.py):
    data_selection.cluster.strategy:
        top_clusters | weighted | scored | interpolated

    data_selection.cluster.within_cluster_method:
        centroid_nearest | mmd

    data_selection.cluster.representative_method:
        medoid | centroid_nearest

Variant behavior:
    interpolated_weighted -> strategy=interpolated, dots_diversity=true,
                                                     dots_diversity_use_composite_score=true,
                                                     use_rollout_history=true
    interpolated          -> strategy=interpolated, dots_diversity=true,
                                                     dots_diversity_use_composite_score=false,
                                                     use_rollout_history=false

Common env vars:
    GLOBAL_BUDGET_PCT, SELECTION_BUDGET_PCT, RESELECT_INTERVAL,
    EXCLUDE_ALREADY_SELECTED, REROLL_MEDOIDS,
    EXPLORATION_ENABLED, EXPLORATION_PCT, EXPLORATION_INTERVAL, EXPLORATION_PCT_BASE,
    ASYMMETRIC_UTILITY, ASYMMETRIC_BIAS, ASYMMETRIC_DEAD_LOW, ASYMMETRIC_DEAD_HIGH,
    ROLLOUT_HISTORY_MAX_REFS, CUDA_VISIBLE_DEVICES

Examples:
    bash run_qwen3_vl-2b_online_selection.sh
    bash run_qwen3_vl-2b_online_selection.sh --help
    GLOBAL_BUDGET_PCT=10.0 bash run_qwen3_vl-2b_online_selection.sh
    bash run_qwen3_vl-2b_online_selection.sh vllm /path/cluster_arrays.npz interpolated /path/dataset.json \
            data_selection.cluster.within_cluster_method=mmd \
            data_selection.cluster.n_reps=4
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        print_help
        exit 0
fi

ENGINE=${1:-vllm}
CLUSTER_ARRAYS=${2:-/workspace/rl_data_selection/benchmark/rl_data_selection/cluster_selection/outputs_50_cluster_new/cluster_arrays.npz}
VARIANT=${3:-interpolated_weighted}
# Path to the JSON/JSONL that was used to build the cluster embeddings.
# Required to correctly align NPZ row order with parquet row order — these
# two orderings are typically different.  The selector uses this to remap
# NPZ indices → parquet indices for REPR rollouts, select(), and history buffer.
DATASET_JSON=${4:-/workspace/rl_data_selection/data/VLAA-Thinking/VLAA-Thinking-GRPO-25K_train_90_100.json}
shift 4 || true

# --- Variant-specific flags ---
if [ "$VARIANT" = "interpolated_weighted" ]; then
    # Interpolated strategy + diversity floor + composite cluster scoring
    # + time-weighted rollout history buffer.
    # Cluster allocation weighted by mean_predicted_var × transferability × (1/density).
    # The buffer seeds from REPR medoid rollouts and grows organically with each
    # training batch, giving DOTS an increasingly rich, policy-tracking reference set.
    USE_ROLLOUT_HISTORY=true
    DOTS_DIVERSITY=true
    DOTS_COMPOSITE=true
    EXP_SUFFIX="interpolated_weighted"
elif [ "$VARIANT" = "interpolated" ]; then
    # Interpolated strategy + diversity floor, fixed REPR medoids only, no composite score.
    # Use as an ablation against interpolated_weighted to isolate the benefit
    # of the rollout history buffer and composite scoring.
    USE_ROLLOUT_HISTORY=false
    DOTS_DIVERSITY=true
    DOTS_COMPOSITE=false
    EXP_SUFFIX="interpolated_centroid"
else
    echo "ERROR: Unknown VARIANT='$VARIANT'. Valid values: interpolated_weighted, interpolated"
    exit 1
fi

# --- Asymmetric utility (hard-side bias) ---
# Variance is symmetric around p=0.5: a sample the policy solves 1/8 times
# looks identical to one it solves 7/8 times to the pure `interpolated`
# strategy.  When ASYMMETRIC_UTILITY=true, the selector predicts per-sample
# mean reward via DOTS in parallel with variance and reweights:
#   utility = predicted_var * (1 + ASYMMETRIC_BIAS * (0.5 - predicted_mean))
# so at equal variance, harder samples (lower predicted mean) score higher.
# Samples near p=0 or p=1 are zeroed out — under GRPO they produce no
# gradient signal, so spending budget on them is wasted.
# See README.md §"Asymmetric utility — breaking variance symmetry" for details.
ASYMMETRIC_UTILITY=${ASYMMETRIC_UTILITY:-true}
ASYMMETRIC_BIAS=${ASYMMETRIC_BIAS:-0.5}
ASYMMETRIC_DEAD_LOW=${ASYMMETRIC_DEAD_LOW:-0.05}
ASYMMETRIC_DEAD_HIGH=${ASYMMETRIC_DEAD_HIGH:-0.95}

if [ "$ASYMMETRIC_UTILITY" = "true" ]; then
    EXP_SUFFIX="${EXP_SUFFIX}_asym${ASYMMETRIC_BIAS}"
fi

# --- Global budget cap ---
# Hard cap on cumulative unique samples ever selected across the whole run
# (% of full dataset).  Once hit, the pool freezes and reselection rounds
# reweight inside the frozen pool using DOTS-predicted variance (and mean,
# when asymmetric utility is on).  Set equal to (per-round budget × n_rounds)
# for a fair comparison against a fixed random baseline at the same %.
GLOBAL_BUDGET_PCT=${GLOBAL_BUDGET_PCT:-10.0}

# --- Reroll medoids each round? ---
# When false, the medoid reference rollouts run only on round 0 (cold start),
# and subsequent rounds rely on the rollout-history buffer accumulated from
# training-step rewards.  This was the supervisor's recommendation: re-rolling
# fixed medoids every round wastes compute because the buffer already tracks
# fresher policy-state observations from neighbouring samples.
REROLL_MEDOIDS=${REROLL_MEDOIDS:-false}

# --- Discovery cadence ---
# Per-round budget = how many *new* samples to add per reselection (with the
# exclude_already_selected mask in v3, this is the literal new-unique increment).
# RESELECT_INTERVAL = how many training steps between reselections.
# At batch_size=128 the natural unit is one batch.  Defaults below: pick one
# fresh batch every step until the global cap is hit (~17 rounds for 10% of
# 22k = 2.2k unique samples), then frozen reweight takes over for the rest of
# training.
SELECTION_BUDGET_PCT=${SELECTION_BUDGET_PCT:-0.58}   # ~128 samples on 22k dataset
# Reselect every 4 training steps — every-step proved too aggressive: it
# burns the global cap in ~17 steps (with budget 0.58% × 17 ≈ 10%) which
# leaves the bulk of training in the frozen-reweight phase. Stretching the
# discovery cadence to every 4 steps gives the policy time to actually
# learn from the new samples before adding more, while the rollout-history
# buffer continues to absorb training-batch rewards on every step.
RESELECT_INTERVAL=${RESELECT_INTERVAL:-4}
EXCLUDE_ALREADY_SELECTED=${EXCLUDE_ALREADY_SELECTED:-true}

# --- Exploration ---
# With exclude_already_selected=true the discovery mask already prevents
# re-picking samples, so the original "diversify the buffer" rationale for
# exploration is largely subsumed.  Disabled by default; set
# EXPLORATION_ENABLED=true to re-enable for ablations.
EXPLORATION_ENABLED=${EXPLORATION_ENABLED:-false}
EXPLORATION_PCT=${EXPLORATION_PCT:-1.0}
EXPLORATION_INTERVAL=${EXPLORATION_INTERVAL:-4}

# Buffer cap: 0 = unlimited (rely on rollout_history_max_age pruning).  Older
# values like 2000 throw away recent observations the trainer already paid for.
ROLLOUT_HISTORY_MAX_REFS=${ROLLOUT_HISTORY_MAX_REFS:-0}

EXP_NAME="v3_k50_r10_perRound${SELECTION_BUDGET_PCT}_global${GLOBAL_BUDGET_PCT}_${EXP_SUFFIX}"
EXPLORATION_PCT_BASE=${EXPLORATION_PCT_BASE:-representatives}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,3,4,5"} \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/rl_data_selection/data/vlaa_parquet_splits/train_90_100.parquet \
    data.val_files=/workspace/rl_data_selection/data/vlaa_parquet_splits/test_10_100.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data_selection.method=cluster \
    data_selection.reselect_schedule=step \
    data_selection.reselect_interval=$RESELECT_INTERVAL \
    data_selection.selection_budget_pct=$SELECTION_BUDGET_PCT \
    data_selection.global_budget_pct=$GLOBAL_BUDGET_PCT \
    data_selection.cluster.cluster_arrays_file=$CLUSTER_ARRAYS \
    data_selection.cluster.dataset_json_file=$DATASET_JSON \
    data_selection.cluster.n_clusters=50 \
    data_selection.cluster.n_reps=10 \
    data_selection.cluster.strategy=interpolated \
    data_selection.cluster.within_cluster_method=centroid_nearest \
    data_selection.cluster.representative_method=medoid \
    data_selection.cluster.dots_temperature=0.05 \
    data_selection.cluster.dots_diversity_temperature=0.5 \
    data_selection.cluster.dots_top_k=64 \
    data_selection.cluster.dots_diversity=$DOTS_DIVERSITY \
    data_selection.cluster.dots_diversity_use_composite_score=$DOTS_COMPOSITE \
    data_selection.cluster.use_rollout_history=$USE_ROLLOUT_HISTORY \
    data_selection.cluster.rollout_history_decay_rate=0.05 \
    data_selection.cluster.rollout_history_max_age=500 \
    data_selection.cluster.rollout_history_max_refs=$ROLLOUT_HISTORY_MAX_REFS \
    data_selection.cluster.exclude_already_selected=$EXCLUDE_ALREADY_SELECTED \
    data_selection.cluster.reroll_medoids=$REROLL_MEDOIDS \
    data_selection.cluster.exploration_enabled=$EXPLORATION_ENABLED \
    data_selection.cluster.exploration_pct=$EXPLORATION_PCT \
    data_selection.cluster.exploration_pct_base=$EXPLORATION_PCT_BASE \
    data_selection.cluster.exploration_interval=$EXPLORATION_INTERVAL \
    data_selection.cluster.igs_enabled=false \
    data_selection.cluster.igs_weight=1.0 \
    data_selection.cluster.asymmetric_utility_enabled=$ASYMMETRIC_UTILITY \
    data_selection.cluster.hard_side_bias=$ASYMMETRIC_BIAS \
    data_selection.cluster.asymmetric_dead_zone_low=$ASYMMETRIC_DEAD_LOW \
    data_selection.cluster.asymmetric_dead_zone_high=$ASYMMETRIC_DEAD_HIGH \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_vlaa_grpo_full' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.total_epochs=170 \
    trainer.default_local_dir=/workspace/rl_data_selection/peyman/outputs/checkpoints/online_selection/${EXP_NAME} \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.rollout_data_dir=/workspace/rl_data_selection/peyman/outputs/rollouts/online_selection/${EXP_NAME} \
    trainer.val_before_train=False \
    trainer.validation_data_dir=/workspace/rl_data_selection/peyman/outputs/rollouts/online_selection/${EXP_NAME}_val "$@"
