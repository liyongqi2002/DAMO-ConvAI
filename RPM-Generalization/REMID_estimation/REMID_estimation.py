import copy
import os
import os.path as osp
import argparse
import math
import json
import random
import pickle
import pdb
import warnings

from tqdm import trange, tqdm


import sys
sys.path.append('..')
sys.path.append('../test_generation')


import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from datasets import load_dataset



from peft import PeftModel
import torch.multiprocessing as mp



try:
    mp.set_start_method('spawn', force=False)  # force=False 表示如果已经设置就跳过
except RuntimeError:
    print("Start method already set")


from emi_utils import Structural_EMI



from scipy.stats import pearsonr, spearmanr, kendalltau
import math

from policy_test import format_dialogue_simple
from templates import import_template



Thought_LLM_Mode = os.environ.get('THOUGHT_LLM_MODE')
str_Thought_LLM_Mode=Thought_LLM_Mode.replace("/","-")


Pxy_LLM_Mode = os.environ.get('PXY_LLM_MODE')
str_Pxy_LLM_Mode=Pxy_LLM_Mode.replace("/","-")

llm_lora_path_ConditionalProb=f"../estimator_training/{Pxy_LLM_Mode}"
LLM_path = os.environ.get('LLM_PATH')
LLM_name = os.environ.get('LLM_NAME')



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":

    if "GRPO" in Pxy_LLM_Mode:
        lora_initialize=f"../estimator_training/ckpt_SFT/METHOD[RLMid_SFT_model_pxy]-BASED[{LLM_name}]#ProbLLM"

        model_path_merged_initial_lora="temp_lora_initialize_PXYPART"
        if not os.path.exists(model_path_merged_initial_lora):
            base_model = AutoModelForCausalLM.from_pretrained(LLM_path,trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)

            peft_model = PeftModel.from_pretrained(base_model, lora_initialize)
            merged_model = peft_model.merge_and_unload()

            tokenizer.save_pretrained(model_path_merged_initial_lora)
            merged_model.save_pretrained(model_path_merged_initial_lora)
        base_LLM_path=model_path_merged_initial_lora
    else:
        base_LLM_path=LLM_path
    emi_estimator = Structural_EMI(base_LLM_path,llm_lora_path_ConditionalProb, num_gpus = torch.cuda.device_count(), gpu_id_list=None)


    bench_version="v15"

    set_seeds(seed=42)

    filepath_SThought_dict = f"[{str_Thought_LLM_Mode}]-[FullSize]-SThought_dict.json.tmp"
    with open(filepath_SThought_dict, encoding="utf-8", mode="r") as f:
        SThought_dict = json.load(f)


    collected_policy_generations_filepath = f"CPG-BVersion[{bench_version}].json"
    with open(collected_policy_generations_filepath, encoding="utf-8", mode="r") as f:
        raw_policy_generations = json.load(f)


    policy_dict={}
    # categories = ["IDTest", "OOD1Test", "OOD2Test", "OOD3Test"]
    categories = ['IDTest', 'german',  'spanish', 'chinese', 'japanese', 'korean', 'Literature', 'Film & Television', 'Theater', 'Gaming', 'TurnLevelComposition', 'WordLevelComposition']

    # 遍历每个样本
    for index, (sample_id, sample_ins) in enumerate(raw_policy_generations.items()):
        EMI_Inference_SThought = SThought_dict[sample_ins["sample_ID"]]
        try:
            EMI_Inference_SThought = EMI_Inference_SThought.split("[Core Features of the Golden Response]")[-1].replace("```", "")
        except:
            try:
                EMI_Inference_SThought = EMI_Inference_SThought.split("Core Features of the Golden Response")[-1]
            except:
                try:
                    EMI_Inference_SThought = EMI_Inference_SThought.split("Trial 3")[-1]
                except:
                    EMI_Inference_SThought = EMI_Inference_SThought


        category_match = sample_ins["subset_tag"]

        golden_response = sample_ins["agent_golden_response"]

        model_responses = sample_ins["model_response"]
        for policy, model_response in model_responses.items():
            if policy not in policy_dict:
                policy_dict[policy]={}
            if category_match not in policy_dict[policy]:
                policy_dict[policy][category_match]=[]

            policy_dict[policy][category_match].append({
                "user_persona": sample_ins["user_persona"],
                "str_agent_character": str(sample_ins["agent_character"]),
                "str_dialogue_context": format_dialogue_simple(sample_ins["dialogue_context"]),
                "theta_response": model_response,
                "golden_response": golden_response,
                "EMI_Inference_SThought": EMI_Inference_SThought,
            })

    output_filepath=f"[{str_Thought_LLM_Mode}]-[{str_Pxy_LLM_Mode}]-all_TEMID_dict.json.tmp"
    if os.path.exists(output_filepath):
        with open(output_filepath, encoding="utf-8", mode="r") as f:
            all_EMI_dict = json.load(f)
    else:
        all_EMI_dict = {}

    dict_ref_mi_cache={}
    with torch.inference_mode():

        for policy in list(policy_dict.keys()):
            for category in categories:
                print(f"processing {policy} {category}")

                EMI_instances=policy_dict[policy][category]

                converted_batch = {
                    "x_message": [],
                    "y_theta_message": [],
                    "y_golden_message": [],
                }
                for idx,EMI_instance in enumerate(EMI_instances):
                    user_persona = EMI_instance["user_persona"]
                    str_agent_character = EMI_instance["str_agent_character"]
                    str_dialogue_context = EMI_instance["str_dialogue_context"]

                    theta_response = EMI_instance["theta_response"]
                    golden_response = EMI_instance["golden_response"]
                    EMI_Inference_SThought = EMI_instance["EMI_Inference_SThought"]

                    system_prompt, base_prompt = import_template(mode="model_pxy")

                    SFT_input = base_prompt.format(
                        user_persona=user_persona,
                        agent_character=str_agent_character,
                        str_dialogue_context=str_dialogue_context,
                    )
                    SFT_input = f"{SFT_input}\n\n## Core Features of the Golden Response\n```{EMI_Inference_SThought}```\n\n"

                    x_message = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": SFT_input},
                            {"role": "assistant", "content": f""},
                        ]
                    }

                    y_theta_message = {
                        "messages": [
                            {"role": "assistant", "content": f"## Agent Response\n{theta_response}"},
                        ]
                    }
                    y_golden_message = {
                        "messages": [
                            {"role": "assistant", "content": f"## Agent Response\n{golden_response}"},
                        ]
                    }

                    converted_batch["x_message"].append(x_message)
                    converted_batch["y_theta_message"].append(y_theta_message)
                    converted_batch["y_golden_message"].append(y_golden_message)


                ##########################################################################################################
                # src_emi, model_mi, ref_mi = emi_estimator.forward(x_message=converted_batch["x_message"],
                #                                                   y_theta_message=converted_batch["y_theta_message"],
                #                                                   y_golden_message=converted_batch["y_golden_message"])
                ##########################################################################################################

                model_mi = emi_estimator.club_mi(converted_batch["x_message"], converted_batch["y_theta_message"]).item()
                if category not in dict_ref_mi_cache:
                    ref_mi = emi_estimator.club_mi(converted_batch["x_message"], converted_batch["y_golden_message"]).item()
                    dict_ref_mi_cache[category] = ref_mi
                else:
                    print(f"Loading from cache for {category} from {str(dict_ref_mi_cache)}")
                    ref_mi = dict_ref_mi_cache[category]
                src_emi = model_mi - ref_mi


                processed_name=f"{policy} ### {category}"
                EMI_dict={
                    "processed_name": processed_name,
                    "emi_score":{
                        "src_emi": src_emi,
                        "model_mi": model_mi,
                        "ref_mi": ref_mi,
                    }
                }
                print(f"the src_emi is {src_emi}")
                print(f"the model_mi is {model_mi}")
                print(f"the ref_mi is {ref_mi}")
                print(EMI_dict)

                all_EMI_dict[processed_name]=EMI_dict
            output_filepath = f"[{str_Thought_LLM_Mode}]-[{str_Pxy_LLM_Mode}]-all_TEMID_dict.json.tmp"
            with open(output_filepath, 'w', encoding="utf-8") as f:
                json.dump(all_EMI_dict, f, indent=2)

    print(all_EMI_dict)
    with open(output_filepath, 'w', encoding="utf-8") as f:
        json.dump(all_EMI_dict, f, indent=2)
