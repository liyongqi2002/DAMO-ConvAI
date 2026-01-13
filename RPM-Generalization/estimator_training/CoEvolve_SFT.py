import json

import json
import os

import numpy as np
import torch
from datasets import load_dataset, Dataset
from numpy import mean
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

from dataclasses import dataclass, field
from typing import Dict, Optional

import argparse

import sys

sys.path.append('..')
from templates import import_template
sys.path.append('../test_generation')
from policy_test import format_dialogue_simple



# Define and parse arguments.
@dataclass
class ScriptArguments:
    # training parameters

    output_dir: Optional[str] = field(default="ckpt_SFT", metadata={"help": "the ckpt"}, )

    max_seq_length: Optional[int] = field(default=2048, metadata={"help": "the max_seq_length"}, )

    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the num_train_epochs"}, )

    SFT_mode: Optional[str] = field(default="", metadata={"help": "the SFT_mode"}, )
    co_evolve_iter: Optional[int] = field(default=1, metadata={"help": "the co_evolve_iter"}, )


def import_LLM_name():
    LLM_path = os.environ.get('LLM_PATH')
    LLM_name = os.environ.get('LLM_NAME')

    return LLM_path, LLM_name


def read_ColdStartSFT_DS(raw_filepath, mode):
    with open(raw_filepath, encoding="utf-8", mode="r") as f:
        original_data = json.load(f)

    processed_instances = []
    for instance in original_data:
        try:
            user_persona = instance["user_persona"]
            agent_character = str(instance["agent_character"])
            dialogue_context = instance["dialogue_context"]
            str_dialogue_context =format_dialogue_simple(dialogue_context)

            system_prompt,base_prompt=import_template(mode=mode)

            thought_result = str(instance["SThought"])

            if mode == "thought_completion":
                SFT_input = base_prompt.format(
                    user_persona=user_persona,
                    agent_character=agent_character,
                    str_dialogue_context=str_dialogue_context,
                    # agent_golden_response=instance["agent_golden_response"],
                    agent_golden_response="Please give your analysis without any given reference response",
                )
                SFT_output = f"{thought_result}"

            elif mode == "model_pxy":
                SFT_input = base_prompt.format(
                    user_persona=user_persona,
                    agent_character=agent_character,
                    str_dialogue_context=str_dialogue_context,
                )
                core_features=thought_result.split("[Core Features of the Golden Response]")[-1].replace("```","")
                SFT_input = f"{SFT_input}\n\n## Core Features of the Golden Response\n```{core_features}```\n\n"

                agent_golden_response = instance["agent_golden_response"]
                SFT_output = f"## Agent Response\n{agent_golden_response}"
            else:
                print("mode must be 'thought_completion' or 'model_pxy'")

            processed_instance = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": SFT_input},
                    {"role": "assistant", "content": SFT_output}
                ]
            }
            # print(processed_instance)
            # assert 1==0
            processed_instances.append(processed_instance)
        except Exception as e:
            print(e)
            continue

    # processed_instances=processed_instances[:100]
    print(f"num of seqs: {len(processed_instances)}")

    ds = Dataset.from_list(processed_instances)

    return ds


LLM_path, LLM_name = import_LLM_name()

if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, ModelConfig))
    script_args, model_args = parser.parse_args_and_config()

    if script_args.SFT_mode=="thought_completion":
        script_args.max_seq_length = 2048 * 3
    elif script_args.SFT_mode=="model_pxy":
        script_args.max_seq_length = 2048 * 2
    elif script_args.SFT_mode=="RLMid_SFT_model_pxy":
        script_args.max_seq_length = 2048 * 2
    else:
        raise NotImplementedError
    
    version = "v15"

    if script_args.SFT_mode in ["thought_completion","model_pxy"]:
        ColdStart_size = 1000
        raw_filepath = f"./rl_data/{version}-ColdStart_instances-{ColdStart_size}.json"
        train_dataset = read_ColdStartSFT_DS(raw_filepath=raw_filepath,
                         mode=script_args.SFT_mode)
    elif script_args.SFT_mode in ["RLMid_SFT_model_pxy"]:
        raw_filepath = f"CoEvolveData_Iter[{script_args.co_evolve_iter}].json.tmp"
        with open(raw_filepath, encoding="utf-8", mode="r") as f:
            original_instances = json.load(f)

        processed_instances = []
        for instance in original_instances:
            sample_ID = instance["sample_ID"]
            user_persona = instance["user_persona"]
            agent_character = str(instance["agent_character"])
            dialogue_context = instance["dialogue_context"]
            str_dialogue_context = format_dialogue_simple(dialogue_context)

            core_features = instance["thought_policy_generation"]

            system_prompt,base_prompt=import_template(mode="model_pxy")

            SFT_input = base_prompt.format(
                    user_persona=user_persona,
                    agent_character=agent_character,
                    str_dialogue_context=str_dialogue_context,
            )
            SFT_input = f"{SFT_input}\n\n## Core Features of the Golden Response\n```{core_features}```\n\n"

            agent_golden_response = instance["agent_golden_response"]
            SFT_output = f"## Agent Response\n{agent_golden_response}"

            processed_instance = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": SFT_input},
                    {"role": "assistant", "content": SFT_output}
                ]
            }

            processed_instances.append(processed_instance)

        train_dataset = Dataset.from_list(processed_instances)

    else:
        raise NotImplementedError


    training_args = SFTConfig(
        output_dir=f"{script_args.output_dir}/METHOD[{script_args.SFT_mode}]-BASED[{LLM_name}]#ProbLLM",
        # directory to save and repository id

        num_train_epochs=script_args.num_train_epochs,  # number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        # optim="paged_adamw_32bit",  # use fused adamw optimizer
        logging_steps=5,  # log every n steps
        # save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",  # use cosine learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="none",  # report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # use reentrant checkpointing
        # dataset_text_field="",  # need a dummy field for collator
        # dataset_kwargs={"skip_prepare_dataset": False},  # important for collator
        max_seq_length=script_args.max_seq_length,
        dataset_num_proc=64,
        save_strategy="steps",
        save_steps=100,
    )
    training_args.remove_unused_columns = False

    ################
    # Model, Tokenizer
    ################
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = LLM_path
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    ################
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules='all-linear',
        task_type="CAUSAL_LM",
    )


    ################
    # Training
    ################

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

