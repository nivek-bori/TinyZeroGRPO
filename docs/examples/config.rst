   data:
     tokenizer: null
     train_files: data/cryptarithm/train.parquet
     val_files: /data/cryptarithm/test.parquet
     prompt_key: prompt
     max_prompt_length: 512
     max_response_length: 2048
     train_batch_size: 256
     val_batch_size: 256
     return_raw_input_ids: False  # This should be set to true when the tokenizer between policy and rm differs
     return_raw_chat: False
   actor_rollout_ref:
     hybrid_engine: True
     model:
       path: Qwen/Qwen2.5-3B-Instruct
       external_lib: null
       override_config: {}
       enable_gradient_checkpointing: False
     actor:
       strategy: fsdp  # This is for backward-compatibility
       ppo_mini_batch_size: 128
       ppo_micro_batch_size: 64
       grad_clip: 1.0
       clip_ratio: 0.2
       entropy_coeff: 0.001
       ppo_epochs: 1
       shuffle: True
       optim:
         lr: 1e-6
         lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
         min_lr_ratio: null   # only useful for warmup with cosine
         warmup_style: constant  # select from constant/cosine
         total_training_steps: -1  # must be override by program
       fsdp_config:
         wrap_policy:
           # transformer_layer_cls_to_wrap: None
           min_num_params: 0
         param_offload: False
         grad_offload: False
         optimizer_offload: False
     ref:
       fsdp_config:
         param_offload: False
         wrap_policy:
           # transformer_layer_cls_to_wrap: None
           min_num_params: 0
       log_prob_micro_batch_size: 128
     rollout:
       name: vllm
       temperature: 1.0
       top_k: -1 # 0 for hf rollout, -1 for vllm rollout
       top_p: 1
       response_length: ${data.max_response_length}
       # for vllm rollout
       dtype: float16 # should align with FSDP
       gpu_memory_utilization: 0.3
       ignore_eos: False
       enforce_eager: True
       free_cache_engine: True
       load_format: dummy_dtensor # or dummy_hf or dummy_megatron
       tensor_model_parallel_size: 2
       max_num_batched_tokens: 8192
       max_num_seqs: 1024
       log_prob_micro_batch_size: 128
       # for vllm and hf rollout
       do_sample: True
   algorithm:
     gamma: 1.0
     lam: 1.0
     adv_estimator: gae
     kl_penalty: kl  # how to estimate kl divergence
     kl_ctrl:
       type: fixed
       kl_coef: 0.002
   trainer:
     total_epochs: 1
     project_name: rented_test
     experiment_name: test
     logger: ['console',]
     nnodes: 1
     n_gpus_per_node: 2
     save_freq: -1
     test_freq: -1
     critic_warmup: 0
     default_hdfs_dir: ./checkpionts/${trainer.experiment_name}
     default_local_dir: ./checkpoints/${trainer.project_name}/${trainer.experiment_name}