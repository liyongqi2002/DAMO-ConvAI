export CUDA_VISIBLE_DEVICES=0,1,2,3



#### Step 1: pre-collect data to calculate R-EMID
python collect_evaluated_policy_generations.py



export LLM_NAME="Qwen3-8B"
export LLM_PATH="../../llm_path/Qwen/Qwen3-8B" # the path you put your VLM


#### Step 2: calculate R-EMID
export THOUGHT_LLM_MODE="ckpt_GRPO_CoEvolve/PolicyType[thought_policy]-ITER[2]-BASED[Qwen3-8B]#ProbLLM"
export PXY_LLM_MODE="ckpt_GRPO_CoEvolve/PolicyType[pxy_policy]-ITER[2]-BASED[Qwen3-8B]#ProbLLM"
python REMID_thought_collection.py
python REMID_estimation.py
