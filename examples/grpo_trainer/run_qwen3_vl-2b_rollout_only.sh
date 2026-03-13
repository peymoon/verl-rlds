set -x
ENGINE=${1:-vllm}
# Point val_files at the full ~25K-row parquet created with train_ratio=0.0:
#   cd /workspace/rl_data_selection
#   python prepare_vlaa_dataset.py --output_dir data/vlaa_grpo --train_ratio 0.0
# That writes data/vlaa_grpo_full/test.parquet with ALL rows.
#
# total_epochs=0 + val_before_train=True → runs _validate() once then exits.
# No FSDP actor update, no optimizer state, no rollout buffer accumulation.
# val_batch_size can be larger than train_batch_size (no gradient memory needed).
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"} \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/rl_data_selection/data/vlaa_grpo_20/test.parquet \
    data.val_files=/workspace/rl_data_selection/data/vlaa_grpo_full_test/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
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
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_vlaa_rollout_only' \
    trainer.experiment_name='qwen3_vl_2b_rollout_full' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=0 \
    trainer.default_local_dir=/workspace/peyman/outputs/checkpoints/vlaa_rollout_only \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=/workspace/peyman/outputs/rollouts/vlaa_full_rollout $@
