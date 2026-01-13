

export LLM_NAME="Qwen3-8B"
export LLM_PATH="../../llm_path/Qwen/Qwen3-8B"



## Step 1: Cold-start SFT (We have provided the ckpts of these two SFT-initialized ones)
#accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml   CoEvolve_SFT.py \
#                      --SFT_mode "thought_completion" \
#                      --num_train_epochs 3  \
#                      --output_dir  ckpt_SFT
#
#accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml   CoEvolve_SFT.py \
#                      --SFT_mode "model_pxy" \
#                      --num_train_epochs 3 \
#                      --output_dir  ckpt_SFT



# Step 2: CoEvolve RL

accelerate launch --config_file accelerate_configs/multi_gpu_42splitgpu.yaml   CoEvolve_RL.py \
                      --co_evolve_iter 1 \
                      --optimized_policy_type thought_policy 


accelerate launch --config_file accelerate_configs/multi_gpu_42splitgpu.yaml   CoEvolve_RL.py \
                      --co_evolve_iter 1 \
                      --optimized_policy_type pxy_policy --max_completion_length 256
