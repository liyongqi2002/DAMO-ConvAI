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
    """
    Args:
        gpu_id: GPU ID
        model_path: 基础模型路径
        lora_path: LoRA 权重路径
        text_batch: list of dict，包含 messages 字段
        batch_size_per_gpu: 每个 GPU 上的小 batch 大小
        context_lengths: 每个样本中 x 的 token 长度（用于切片）
        return_dict: 共享字典，用于返回结果
    """

    device = f"cuda:{gpu_id}"
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_8bit=True,
                                                 torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, lora_path).to(device)

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
            # for idx in range(loss.shape[0]):
            #     start_idx = sub_context_lengths[idx] - 1  # 因为已经移位了
            #     if start_idx < 0:
            #         start_idx = 0
            #     if start_idx >= loss.shape[1]:
            #         print(loss.shape)
            #         print(sub_context_lengths)
            #         print("OUT OF CONTEXT LENGTH")
            #         print(sub_batch[idx])
            #         batch_log_probs.append(torch.tensor(0.0))
            #     else:
            #         masked_loss = loss[idx, start_idx:] * shift_attention_mask[idx, start_idx:]
            #         log_prob = -masked_loss.sum().cpu()
            #         batch_log_probs.append(log_prob)

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


# ! Structural_EMI Estimator
class Structural_EMI(nn.Module):
    def __init__(self, model_path, lora_path, num_gpus, gpu_id_list=None):
        super(Structural_EMI, self).__init__()
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

    def batch_log_conditional_prob(self, x_list, xy_list, batch_size_per_gpu=4):
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

    def club_mi(self, x_list, y_list):
        B = len(x_list)

        xy_list = []
        for idx, y in enumerate(y_list):
            xy = copy.deepcopy(x_list[idx])
            xy["messages"][-1] = y["messages"][0]
            xy_list.append(xy)

        # Step 1: 计算正样本对 log p(y_i | x_i)
        positive_log_probs = self.batch_log_conditional_prob(x_list, xy_list)  # shape: [B]

        # Step 2: 构造负样本对：固定 x_i，让 y_j ≠ y_i，形成 (x_i, y_j)
        # 可以使用 batch 内所有其他 y_j 作为负样本（in-batch negatives）
        ####################################################################################
        ############ 原始版本 ################################################################
        # # 复制 y_list 形成矩阵 [B, B]
        # all_x = []
        # all_xy = []
        # for i in range(B):
        #     for j in range(B):
        #         all_x.append(x_list[i])
        #
        #         xy = copy.deepcopy(x_list[i])
        #         xy["messages"][-1] = y_list[j]["messages"][0]
        #         all_xy.append(xy)
        #
        # # 得到所有组合的 log probs: shape [B, B]
        # all_log_probs = self.batch_log_conditional_prob(all_x, all_xy).view(B, B)
        #
        # # 屏蔽掉对角线上的正样本（避免自己跟自己比）
        # mask = (~torch.eye(B, dtype=bool))
        # negative_log_probs = all_log_probs.masked_select(mask).view(B, B - 1)
        # # 对负样本求均值
        # negative_log_probs = negative_log_probs.mean(dim=1)  # shape [B]
        #
        # # CLUB mi
        # mi_estimate = (positive_log_probs - negative_log_probs).mean()
        ####################################################################################
        all_x = []
        all_xy = []
        for i in range(B):
            for j in range(B):
                if i==j:
                    continue
                all_x.append(x_list[i])

                xy = copy.deepcopy(x_list[i])
                xy["messages"][-1] = y_list[j]["messages"][0]
                all_xy.append(xy)

        sample_num=min(len(all_x), 1000) # in our test, the final mi_estimate will be stable when the number of negative paris >1000
        indexes = range(len(all_x))
        sampled_indexes = random.sample(indexes, sample_num)
        sampled_all_x = [all_x[index] for index in sampled_indexes]
        sampled_all_xy = [all_xy[index] for index in sampled_indexes]
        negative_log_probs = self.batch_log_conditional_prob(sampled_all_x, sampled_all_xy)

        mi_estimate = positive_log_probs.mean() - negative_log_probs.mean()

        # print("==================================================================================")
        # print(f"{sample_num} negative pairs negative_log_probs.mean(): {negative_log_probs.mean()}")
        # print(f"{sample_num} negative pairs mi_estimate: {mi_estimate}")




        return mi_estimate

    def forward(self, x_message, y_theta_message, y_golden_message):

        mi_theta = self.club_mi(x_message, y_theta_message).item()
        mi_golden = self.club_mi(x_message, y_golden_message).item()
        emi = mi_theta - mi_golden

        return emi, mi_theta, mi_golden


