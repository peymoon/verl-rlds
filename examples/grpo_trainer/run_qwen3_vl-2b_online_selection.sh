set -x
# Online data selection: train on the FULL dataset, let the selector choose
# the training subset adaptively each epoch based on current model capability.
#
# Key differences from the standard run script:
#   1. data.train_files points to the FULL dataset (not a pre-selected subset)
#   2. data_selection.method=cluster enables online cluster-based selection
#   3. data_selection.selection_budget_pct=20.0 selects top 20% each epoch
#   4. data_selection.cluster.cluster_arrays_file points to pre-computed clusters
#   5. trainer.total_epochs is increased since each epoch trains on 20% of data
#
# For reselection every N *steps* instead of every epoch:
#   data_selection.reselect_schedule=step data_selection.reselect_interval=10
#
# Pre-requisite: run the offline cluster pipeline (stages 0-1) to get cluster_arrays.npz
# See cluster_selection/README.md for instructions.

ENGINE=${1:-vllm}
CLUSTER_ARRAYS=${2:-/workspace/rl_data_selection//benchmark/rl_data_selection/cluster_selection/outputs_200_cluster/cluster_arrays.npz}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"3,4,5,7"} \
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
    data_selection.reselect_interval=10 \
    data_selection.selection_budget_pct=10.0 \
    data_selection.cluster.cluster_arrays_file=$CLUSTER_ARRAYS \
    data_selection.cluster.n_clusters=200 \
    data_selection.cluster.n_reps=3 \
    data_selection.cluster.strategy=interpolated \
    data_selection.cluster.within_cluster_method=centroid_nearest \
    data_selection.cluster.representative_method=medoid \
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
    trainer.experiment_name='selected_k200_r3_interpolated_centroid_top10pct_cluster_online_10pct_qwen3_vl_2b' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=/workspace/peyman/outputs/checkpoints/online_selection/selected_k200_r3_interpolated_centroid_top10pct_cluster_online_10pct \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.rollout_data_dir=/workspace/peyman/outputs/rollouts/online_selection/selected_k200_r3_interpolated_centroid_top10pct_cluster_online_10pct \
    trainer.val_before_train=False \
    trainer.validation_data_dir=/workspace/peyman/outputs/rollouts/online_selection/selected_k200_r3_interpolated_centroid_top10pct_cluster_online_10pct_val "$@"
