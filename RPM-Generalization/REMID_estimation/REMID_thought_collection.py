

import sys
sys.path.append('..')
sys.path.append('../test_generation')



import random
import numpy as np
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel

from templates import import_template
from policy_test import format_dialogue_simple,prepare_vllm,get_vllm_input
from vllm.lora.request import LoRARequest

LLM_path = os.environ.get('LLM_PATH')
LLM_name = os.environ.get('LLM_NAME')

Thought_LLM_Mode = os.environ.get('THOUGHT_LLM_MODE')
str_Thought_LLM_Mode=Thought_LLM_Mode.replace("/","-")



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    bench_version="v15"
    set_seeds(seed=42)

    if "GRPO" in Thought_LLM_Mode:
        lora_initialize = f"../estimator_training/ckpt_SFT/METHOD[thought_completion]-BASED[{LLM_name}]#ProbLLM"

        model_path_merged_initial_lora = "temp_lora_initialize"

        base_model = AutoModelForCausalLM.from_pretrained(LLM_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)

        peft_model = PeftModel.from_pretrained(base_model, lora_initialize)
        merged_model = peft_model.merge_and_unload()

        tokenizer.save_pretrained(model_path_merged_initial_lora)
        merged_model.save_pretrained(model_path_merged_initial_lora)

        llm, sampling_params, processor = prepare_vllm(model_path=model_path_merged_initial_lora, use_lora=True)

    else:
        llm, sampling_params, processor = prepare_vllm(model_path=LLM_path, use_lora=True)

    llm_lora_path_module = f"../estimator_training/{Thought_LLM_Mode}"

    collected_policy_generations_filepath = f"CPG-BVersion[{bench_version}].json"
    with open(collected_policy_generations_filepath, encoding="utf-8", mode="r") as f:
        raw_policy_generations = json.load(f)


    map_index2sampleID=[]
    vllm_inputs = []
    for index, (sample_id, sample_ins) in enumerate(raw_policy_generations.items()):

        sample_ID = sample_ins["sample_ID"]
        user_persona = sample_ins["user_persona"]
        agent_character = str(sample_ins["agent_character"])
        dialogue_context = sample_ins["dialogue_context"]
        str_dialogue_context = format_dialogue_simple(dialogue_context)

        system_prompt, base_prompt = import_template(mode="thought_completion")
        SFT_input = base_prompt.format(
            user_persona=user_persona,
            agent_character=agent_character,
            str_dialogue_context=str_dialogue_context,
            # agent_golden_response=instance["agent_golden_response"],
            agent_golden_response="Please give your analysis without any given reference response",
        )


        vllm_input = get_vllm_input(
            SFT_input,
            processor=processor,
            specified_sys_prompt=system_prompt)

        vllm_inputs.append(vllm_input)
        map_index2sampleID.append(sample_ID)

    vllm_inputs=vllm_inputs

    outputs = llm.generate(vllm_inputs,
                           sampling_params=sampling_params,
                           lora_request=LoRARequest("adapter", 1, llm_lora_path_module))

    SThought_dict={}
    for idx in range(len(outputs)):
        generated_text = outputs[idx].outputs[0].text
        sample_ID=map_index2sampleID[idx]
        SThought_dict[sample_ID]=generated_text

    output_filepath = f"[{str_Thought_LLM_Mode}]-[FullSize]-SThought_dict.json.tmp"
    with open(output_filepath, 'w', encoding="utf-8") as f:
        json.dump(SThought_dict, f, indent=2)






