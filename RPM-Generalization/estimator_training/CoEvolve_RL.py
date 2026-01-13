import copy
import json
import os
import sys
from collections import defaultdict

from grpo_trainer import GRPOTrainer
from grpo_config import GRPOConfig

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,

    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from datasets import load_dataset
import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import re
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset

import argparse
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scipy.stats import spearmanr


# Define and parse arguments.
@dataclass
class ScriptArguments:
    # training parameters

    output_dir: Optional[str] = field(default="ckpt_GRPO_CoEvolve", metadata={"help": "the ckpt"}, )

    max_prompt_length: Optional[int] = field(default=3072, metadata={"help": "the max_prompt_length"}, )
    max_completion_length: Optional[int] = field(default=2048, metadata={"help": "the max_completion_length"}, )

    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the num_train_epochs"}, )

    co_evolve_iter: Optional[int] = field(default=1, metadata={"help": "the co_evolve_iter"}, )

    optimized_policy_type: Optional[str] = field(default=None, metadata={
        "help": "the optimized_policy_type, thought_policy or pxy_policy"}, )


sys.path.append('..')
from templates import import_template

sys.path.append('../test_generation')
from policy_test import format_dialogue_simple

#########################################################################################################################
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

warnings.filterwarnings("ignore")

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


def forward_single_gpu(gpu_id, model_path, lora_path, text_batch, batch_size_per_gpu, context_lengths,
                       return_dict=None):
    device = f"cuda:{gpu_id}"
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_8bit=True,
                                                 torch_dtype=torch.float16)
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path).to(device)
    else:
        print(f"In this stage, we adopt the original base LLM for judge {model_path}")

    all_log_probs = []

    for i in tqdm(range(0, len(text_batch), batch_size_per_gpu),
                  total=len(text_batch) // batch_size_per_gpu,
                  desc=f"Performing on GPU: {gpu_id}"):

        sub_batch = text_batch[i:i + batch_size_per_gpu]
        sub_context_lengths = context_lengths[i:i + batch_size_per_gpu]
        # print(sub_context_lengths)
        # assert 1==0

        # Tokenize
        templated_texts = [
            tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            for item in sub_batch
        ]

        inputs = tokenizer(templated_texts, padding=True, truncation=True,
                           return_tensors="pt", max_length=4096).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.shape)  # shape: [B, T-1]

            # 对于每个样本，取 context_length 后面的部分
            batch_log_probs = []
            for idx in range(loss.shape[0]):
                start_idx = sub_context_lengths[idx] - 1  # 因为已经移位了
                if start_idx < 0:
                    start_idx = 0
                if start_idx >= loss.shape[1]:
                    print(loss.shape)
                    print(sub_context_lengths)
                    print("OUT OF CONTEXT LENGTH")
                    print(sub_batch[idx])
                    batch_log_probs.append(torch.tensor(0.0))
                else:
                    masked_loss = loss[idx, start_idx:] * shift_attention_mask[idx, start_idx:]
                    num_tokens = shift_attention_mask[idx, start_idx:].sum().item()  # 实际 token 数量
                    if num_tokens == 0:
                        log_prob_norm = torch.tensor(0.0)  # 防止除零
                    else:
                        log_prob_norm = (-masked_loss.sum() / num_tokens).cpu()
                    batch_log_probs.append(log_prob_norm)

            batch_log_probs = torch.stack(batch_log_probs)
            all_log_probs.append(batch_log_probs)

    final_log_probs = torch.cat(all_log_probs, dim=0)
    return_dict[gpu_id] = final_log_probs


# ! Perplexity_Estimator Estimator
class Perplexity_Estimator(nn.Module):
    def __init__(self, model_path, lora_path, num_gpus, gpu_id_list=None):
        super(Perplexity_Estimator, self).__init__()
        self.model_path = model_path
        self.lora_path = lora_path
        self.num_gpus = num_gpus
        self.gpu_id_list = []
        if gpu_id_list is None:
            for id in range(self.num_gpus):
                self.gpu_id_list.append(id)
        else:
            self.gpu_id_list = gpu_id_list

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", load_in_8bit=True,
        #                                              torch_dtype=torch.float16)
        # self.model = PeftModel.from_pretrained(self.model, self.lora_path)

    def get_context_lengths(self, x_list):
        """
        对 x_list 中的每个样本进行 tokenization，返回其 token 长度。
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        context_lengths = []
        for item in x_list:
            # 假设每个 item 是 {"messages": [...]}
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = tokenizer(text, padding=True, truncation=True,
                               return_tensors="pt", max_length=4096)
            length = tokens["input_ids"].shape[1]
            context_lengths.append(length)
        return context_lengths

    def batch_log_conditional_prob(self, x_list, xy_list, batch_size_per_gpu=2):
        context_lengths = self.get_context_lengths(x_list)

        total_samples = len(xy_list)
        samples_per_gpu = math.ceil(total_samples / self.num_gpus)
        # print(total_samples)
        # print(self.num_gpus)
        # print(samples_per_gpu)

        batches = []
        batch_context_lengths = []

        for i in range(0, total_samples, samples_per_gpu):
            batches.append(xy_list[i:i + samples_per_gpu])
            batch_context_lengths.append(context_lengths[i:i + samples_per_gpu])

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for gpu_index, gpu_id in enumerate(self.gpu_id_list):
            p = mp.Process(
                target=forward_single_gpu,
                args=(gpu_id, self.model_path, self.lora_path, batches[gpu_index], batch_size_per_gpu,
                      batch_context_lengths[gpu_index], return_dict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = [return_dict[gpu_id] for gpu_id in self.gpu_id_list]
        final_probs = torch.cat(results, dim=0)
        return final_probs


#########################################################################################################################

system_prompt_for_SThought, base_prompt_for_SThought = import_template(mode="thought_completion")
system_prompt_for_model_pxy, base_prompt_for_model_pxy = import_template(mode="model_pxy")

LLM_path = os.environ.get('LLM_PATH')
LLM_name = os.environ.get('LLM_NAME')


def read_RL_DS(script_args, tokenizer, version):
    # 第一步，修改源头数据来源。
    # cold start data -> full-size training + test data
    if script_args.optimized_policy_type == "thought_policy":
        # for all the optimization of thought_policy, we only need the original bench data
        raw_filepath = f"../../Benchmark/{version}/PDGBench.json"
        with open(raw_filepath, encoding="utf-8", mode="r") as f:
            original_data = json.load(f)
            original_instances = []
            for split in original_data:
                split_instances = original_data[split]
                original_instances.extend(split_instances)
    else:
        # else, we need read the last-iter output data from the thought policy to co-evolve
        raw_filepath = f"CoEvolveData_Iter[{script_args.co_evolve_iter}].json.tmp"
        with open(raw_filepath, encoding="utf-8", mode="r") as f:
            original_instances = json.load(f)

    original_instances = original_instances[:]

    processed_instances = []
    for instance in original_instances:
        sample_ID = instance["sample_ID"]
        if "IDTrain" in sample_ID:
            # 不考虑 IDTrain samples作为RL的样本
            continue

        user_persona = instance["user_persona"]
        agent_character = str(instance["agent_character"])
        dialogue_context = instance["dialogue_context"]
        str_dialogue_context = format_dialogue_simple(dialogue_context)

        if script_args.optimized_policy_type == "thought_policy":
            user_prompt = base_prompt_for_SThought.format(
                user_persona=user_persona,
                agent_character=agent_character,
                str_dialogue_context=str_dialogue_context,
                agent_golden_response="Please give your analysis without any given reference response",
            )
            prompt_for_model_pxy_part1 = base_prompt_for_model_pxy.format(
                user_persona=user_persona,
                agent_character=agent_character,
                str_dialogue_context=str_dialogue_context,
            )
            agent_golden_response = instance["agent_golden_response"]
            expected_output_for_model_pxy = f"## Agent Response\n{agent_golden_response}"

            processed_instance = {
                "prompt": [
                    {"role": "system", "content": system_prompt_for_SThought},
                    {"role": "user", "content": user_prompt},
                ],
                "reward_aux_info": {
                    "prompt_for_model_pxy_part1": prompt_for_model_pxy_part1,
                    "expected_output_for_model_pxy": expected_output_for_model_pxy,
                },
            }

        elif script_args.optimized_policy_type == "pxy_policy":
            system_prompt = system_prompt_for_model_pxy

            prompt_for_model_pxy_part1 = base_prompt_for_model_pxy.format(
                user_persona=user_persona,
                agent_character=agent_character,
                str_dialogue_context=str_dialogue_context,
            )
            agent_golden_response = instance["agent_golden_response"]
            expected_output_for_model_pxy = f"## Agent Response\n{agent_golden_response}"

            thought_policy_generation = instance["thought_policy_generation"]
            prompt_for_model_pxy = f"{prompt_for_model_pxy_part1}\n\n## Core Features of the Golden Response\n```{thought_policy_generation}```\n\n"

            prompt_message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_for_model_pxy},
                # {"role": "assistant", "content": ""},  # 这里是为了把pxy policy的think符号给去掉
            ]
            processed_prompt = tokenizer.apply_chat_template(
                prompt_message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            # print(processed_prompt)
            # assert 1==0

            processed_instance = {
                "prompt": processed_prompt,
                "reward_aux_info": {
                    "prompt_for_model_pxy": prompt_for_model_pxy,
                    "expected_output_for_model_pxy": expected_output_for_model_pxy,
                },
            }


        else:
            raise NotImplementedError("optimized_policy_type is not given.")

        processed_instances.append(processed_instance)

    random.shuffle(processed_instances)
    processed_instances = random.sample(processed_instances, k=1000)

    train_dataset = Dataset.from_list(processed_instances)
    return train_dataset


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, ModelConfig))
    script_args, model_args = parser.parse_args_and_config()

    version = "v15"

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_path,
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2",
        # quantization_config=bnb_config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True, use_fast=True)
    train_dataset = read_RL_DS(script_args=script_args, tokenizer=tokenizer, version=version)

    if script_args.optimized_policy_type == "thought_policy":
        if script_args.co_evolve_iter == 1:
            optimized_policy_lora_path = f"ckpt_SFT/METHOD[thought_completion]-BASED[{LLM_name}]#ProbLLM"
            reward_model_lora_path = f"ckpt_SFT/METHOD[model_pxy]-BASED[{LLM_name}]#ProbLLM"
        else:
            optimized_policy_lora_path = f"{script_args.output_dir}/PolicyType[thought_policy]-ITER[{script_args.co_evolve_iter - 1}]-BASED[{LLM_name}]#ProbLLM"
            reward_model_lora_path = f"{script_args.output_dir}/PolicyType[pxy_policy]-ITER[{script_args.co_evolve_iter - 1}]-BASED[{LLM_name}]#ProbLLM"

        specific_output_dir = f"{script_args.output_dir}/PolicyType[thought_policy]-ITER[{script_args.co_evolve_iter}]-BASED[{LLM_name}]#ProbLLM"

    elif script_args.optimized_policy_type == "pxy_policy":
        if script_args.co_evolve_iter == 1:
            optimized_policy_lora_path = f"ckpt_SFT/METHOD[RLMid_SFT_model_pxy]-BASED[{LLM_name}]#ProbLLM"
            reward_model_lora_path = None
        else:
            optimized_policy_lora_path = f"{script_args.output_dir}/PolicyType[pxy_policy]-ITER[{script_args.co_evolve_iter - 1}]-BASED[{LLM_name}]#ProbLLM"
            reward_model_lora_path = None

        specific_output_dir = f"{script_args.output_dir}/PolicyType[pxy_policy]-ITER[{script_args.co_evolve_iter}]-BASED[{LLM_name}]#ProbLLM"
    else:
        raise NotImplementedError("optimized_policy_type is not given.")

    print(f"LOADING THE optimized_policy FROM: {optimized_policy_lora_path}")
    print(f"LOADING THE reward_model FROM: {reward_model_lora_path}")
    print(f"Trained Policy Saved in: {specific_output_dir}")

    optimized_policy = PeftModel.from_pretrained(base_model, optimized_policy_lora_path)
    optimized_policy = optimized_policy.merge_and_unload()

    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        output_dir=specific_output_dir,
        learning_rate=1e-5,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward

        per_device_train_batch_size=4,  # batch size per device during training
        gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},

        num_train_epochs=script_args.num_train_epochs,
        bf16=True,
        tf32=True,  # use tf32 precision

        # Parameters that control de data preprocessing
        num_generations=8,  # default: 8
        max_prompt_length=script_args.max_prompt_length,  # default: 512
        max_completion_length=script_args.max_completion_length,  # default: 256

        # Parameters related to reporting and saving
        report_to="none",
        logging_steps=1,
        # push_to_hub=True,
        save_strategy="steps",
        save_steps=50,

        # vllm usage (如果采样的模型是lora后的模型，不太方便用vllm开关？)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_tensor_parallel_size=1,
    )


    def perplexity_reward_ThoughtPolicy(completions, reward_aux_info, **kwargs):
        # specify the gpu for the EMI class
        RM_num_gpus = int(torch.cuda.device_count())
        if RM_num_gpus == 4:
            # RM_gpu_id_list = [2, 3]
            RM_gpu_id_list = [3]

        else:
            raise NotImplementedError("RM_num_gpus is not given.")

        perplexity_estimator = Perplexity_Estimator(LLM_path,
                                                    lora_path=reward_model_lora_path,
                                                    num_gpus=len(RM_gpu_id_list),
                                                    gpu_id_list=RM_gpu_id_list)

        all_x_list = []
        all_xy_list = []

        all_thoughts = []
        for completion, p_dict in zip(completions, reward_aux_info):
            content = completion[0]["content"]

            prompt_for_model_pxy_part1 = p_dict["prompt_for_model_pxy_part1"]
            expected_output_for_model_pxy = p_dict["expected_output_for_model_pxy"]

            prompt_for_model_pxy_part2 = content.split("[Core Features of the Golden Response]")[-1].replace("```", "")
            prompt_for_model_pxy = f"{prompt_for_model_pxy_part1}\n\n## Core Features of the Golden Response\n```{prompt_for_model_pxy_part2}```\n\n"

            xy = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_model_pxy},
                    {"role": "user", "content": prompt_for_model_pxy},
                    {"role": "assistant", "content": expected_output_for_model_pxy}
                ]
            }
            all_xy_list.append(xy)

            x = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_model_pxy},
                    {"role": "user", "content": prompt_for_model_pxy},
                    {"role": "assistant", "content": ""}
                ]
            }
            all_x_list.append(x)

            all_thoughts.append(prompt_for_model_pxy_part2)

        all_positive_log_probs = perplexity_estimator.batch_log_conditional_prob(all_x_list, all_xy_list)

        rewards = torch.exp(all_positive_log_probs).tolist()

        for idx in range(len(rewards)):
            print("========================================")
            print("reward", rewards[idx])
            # print("full thought path", completions[idx][0]["content"])
            print("thought", all_thoughts[idx])

            print("#####")
            print("expected_output_for_model_pxy", reward_aux_info[idx]["expected_output_for_model_pxy"])

        return rewards


    def format_reward_ThoughtPolicy(completions, reward_aux_info, **kwargs):
        rewards = []

        # 正则表达式匹配章节标题
        section_pattern = {
            "part1": r"###\s*Part 1:\s*Restatement of Key Information",
            "user_persona": r"####\s*Key Information in User Persona",
            "agent_character": r"####\s*Key Information in Agent Character",
            "dialogue_context": r"####\s*Key Information in Dialogue Context",
            "summary": r"###\s*Summary of Key Information",
            "part2": r"###\s*Part 2:\s*Iterative Revision",
            "part3": r"###\s*Part 3:\s*Final Feature Set of the Golden Response"
        }

        for completion in completions:
            content = completion[0]["content"]
            # print(content)
            score = 0.0

            try:
                # === 第一部分：检查 [Core Features of the Golden Response] 是否在代码块中 ===
                core_features_in_code_block = False
                code_blocks = re.findall(r"```[\s\S]*?\[Core Features of the Golden Response\][\s\S]*?```", content)
                if code_blocks:
                    core_features_in_code_block = True
                score += 0.5 if core_features_in_code_block else 0.0
            except Exception as e:
                pass

            try:
                # === 第二部分：检查章节结构完整性 ===
                structure_complete = all([
                    re.search(section_pattern["part1"], content),
                    re.search(section_pattern["user_persona"], content),
                    re.search(section_pattern["agent_character"], content),
                    re.search(section_pattern["dialogue_context"], content),
                    re.search(section_pattern["summary"], content),
                    re.search(section_pattern["part2"], content),
                    re.search(section_pattern["part3"], content)
                ])
                score += 0.5 if structure_complete else 0.0
            except Exception as e:
                pass
            rewards.append(score)

        return rewards


    def length_reward_ThoughtPolicy(completions, reward_aux_info, **kwargs):
        rewards = []

        for completion in completions:
            content = completion[0]["content"]
            # print(content)
            score = 0


            try:
                # === 第一部分：检查 [Core Features of the Golden Response] 是否在代码块中 ===
                core_features_match = re.findall(r"```[\s\S]*?\[Core Features of the Golden Response\][\s\S]*?```", content)
                if core_features_match:
                    core_text = content.split("[Core Features of the Golden Response]")[-1].replace("```", "")

                    # 去除 Markdown 和代码块标记
                    # clean_text = re.sub(r"```[\s\S]*?```", "", core_text)  # 删除嵌套的代码块
                    # clean_text = re.sub(r"[^\w\s]", "", clean_text)  # 可选：删除标点
                    word_count = len(core_text.split(" "))
                    # print(f"Word core_text: {core_text}")
                    # print(f"Word Cou0nt: {word_count}")

                    target_length = 600

                    if word_count == 0:
                        score = 0.0
                    elif word_count <= target_length:
                        score = word_count / target_length
                    else:
                        score = 1.0
            except Exception as e:
                pass


            rewards.append(score)

        return rewards

    def perplexity_reward_PxyPolicy(completions, reward_aux_info, **kwargs):
        # specify the gpu for the EMI class
        RM_num_gpus = int(torch.cuda.device_count())
        if RM_num_gpus == 4:
            # RM_gpu_id_list = [2, 3]
            RM_gpu_id_list = [3]

        else:
            raise NotImplementedError("RM_num_gpus is not given.")

        system_prompt_for_PxyReward = """If the given Reference Golden Response matches 100% with the "Core Features of the Golden Response" as described, then please determine what percentage of the Core Features is matched by the given Candidate Response. Please output the percentage value directly, without analysis."""

        perplexity_estimator = Perplexity_Estimator(LLM_path,
                                                    lora_path=reward_model_lora_path,
                                                    num_gpus=len(RM_gpu_id_list),
                                                    gpu_id_list=RM_gpu_id_list)

        order1_all_x_list = []
        order1_all_xy_golden_list = []

        all_roll_outs = []
        for completion, p_dict in zip(completions, reward_aux_info):
            # content = completion[0]["content"]
            content = completion

            expected_output_for_model_pxy = p_dict["expected_output_for_model_pxy"]

            prompt_for_model_pxy = p_dict["prompt_for_model_pxy"]

            instruction = """If the given Reference Golden Response matches 100% with the "Core Features of the Golden Response" as described, then please determine what percentage of the Core Features is matched by the given Candidate Response. Please output the percentage value directly, without analysis."""
            prompt_for_model_PxyReward = f"""{prompt_for_model_pxy}\n\n```Reference Golden Response: {expected_output_for_model_pxy}```\n\n```Candidate Response: {content}```\n\n {instruction}\n\n Percentage: """

            x = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_PxyReward},
                    {"role": "user", "content": prompt_for_model_PxyReward},
                    {"role": "assistant", "content": ""}
                ]
            }

            xy_golden = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_PxyReward},
                    {"role": "user", "content": prompt_for_model_PxyReward},
                    {"role": "assistant", "content": """100%"""}
                ]
            }

            order1_all_x_list.append(x)
            order1_all_xy_golden_list.append(xy_golden)

            all_roll_outs.append(content)

        # 这个顺序中，把golden的放进来，作为某种参考
        order2_all_x_list = []
        order2_all_xy_golden_list = []
        for completion, p_dict in zip(completions, reward_aux_info):
            # content = completion[0]["content"]
            expected_output_for_model_pxy = p_dict["expected_output_for_model_pxy"]

            prompt_for_model_pxy = p_dict["prompt_for_model_pxy"]

            instruction = """If the given Reference Golden Response matches 100% with the "Core Features of the Golden Response" as described, then please determine what percentage of the Core Features is matched by the given Candidate Response. Please output the percentage value directly, without analysis."""
            prompt_for_model_PxyReward = f"""{prompt_for_model_pxy}\n\n```Reference Golden Response: {expected_output_for_model_pxy}```\n\n```Candidate Response: {expected_output_for_model_pxy}```\n\n {instruction}\n\n Percentage: """

            x = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_PxyReward},
                    {"role": "user", "content": prompt_for_model_PxyReward},
                    {"role": "assistant", "content": ""}
                ]
            }

            xy_golden = {
                "messages": [
                    {"role": "system", "content": system_prompt_for_PxyReward},
                    {"role": "user", "content": prompt_for_model_PxyReward},
                    {"role": "assistant", "content": """100%"""}
                ]
            }

            order2_all_x_list.append(x)
            order2_all_xy_golden_list.append(xy_golden)

        order1_golden_all_conditional_prob = perplexity_estimator.batch_log_conditional_prob(order1_all_x_list,
                                                                                             order1_all_xy_golden_list)
        order2_golden_all_conditional_prob = perplexity_estimator.batch_log_conditional_prob(order2_all_x_list,
                                                                                             order2_all_xy_golden_list)

        order1_prob = torch.exp(order1_golden_all_conditional_prob)
        order2_prob = torch.exp(order2_golden_all_conditional_prob)
        # diff = order1_prob - order2_prob
        # rewards = 1 + torch.clamp(diff, max=0)  # clamp 将大于 0 的值设为 0
        # rewards = rewards.tolist()  # 转为 Python list

        diff = order1_golden_all_conditional_prob - order2_golden_all_conditional_prob
        rewards = torch.clamp(torch.exp(diff), max=1)  # clamp 将大于 1 的值设为 1

        for idx in range(len(rewards)):
            print("========================================")
            print("perplexity reward", rewards[idx])
            print("diff", diff[idx])
            print("order1_prob", order1_prob[idx])
            print("order2_prob", order2_prob[idx])

            # print("full thought path", completions[idx][0]["content"])
            print("roll_out", all_roll_outs[idx])
            print("#####")
            print("expected_output_for_model_pxy", reward_aux_info[idx]["expected_output_for_model_pxy"])
        return rewards


    def format_reward_PxyPolicy(completions, reward_aux_info, **kwargs):
        rewards = []

        # 正则表达式匹配章节标题
        section_pattern = {
            "part1": r"##\s*Agent Response\s*",
        }

        for completion in completions:
            # content = completion[0]["content"]
            content = completion
            # print(content)
            score = 0.0

            try:
                # === 第二部分：检查章节结构完整性 ===
                structure_complete = all([
                    re.search(section_pattern["part1"], content)
                ])
                score += 1 if structure_complete else 0.0
            except Exception as e:
                pass
            rewards.append(score)

        return rewards


    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    if script_args.optimized_policy_type == "thought_policy":
        reward_funcs = [format_reward_ThoughtPolicy,length_reward_ThoughtPolicy, perplexity_reward_ThoughtPolicy]
    elif script_args.optimized_policy_type == "pxy_policy":
        reward_funcs = [format_reward_PxyPolicy,perplexity_reward_PxyPolicy]
    else:
        raise NotImplementedError("optimized_policy_type is not given.")

    trainer = GRPOTrainer(
        model=optimized_policy,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

