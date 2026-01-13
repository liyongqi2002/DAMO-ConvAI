import random
import warnings


warnings.filterwarnings("ignore")

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


import copy
import json
import os





def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)


def prepare_vllm(model_path,
                 use_lora=False,
                 gpu_memory_utilization=0.85,
                 max_tokens=2048,
                 temperature=0,
                 top_p=0.001,
                 repetition_penalty=1.05,
                 ):
    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    processor.chat_template = chat_template
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=use_lora,
        trust_remote_code=True,
        # max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        # stop_token_ids=[],
        stop=["</Agent Response>"],
    )

    return llm, sampling_params, processor


def get_vllm_input(prompt, processor, specified_sys_prompt=None):
    if specified_sys_prompt is None:
        sys_prompt = "You are a helpful assistant."
    else:
        sys_prompt = specified_sys_prompt

    conversation = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    # print(conversation)

    processed_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # print(processed_prompt)
    # assert 1==0

    return processed_prompt


def format_dialogue_simple(dialogue_list):
    result = []

    for i in range(0, len(dialogue_list), 2):
        user_item = dialogue_list[i]
        user_text = user_item["user_query"]
        turn_number = (i // 2) + 1  

        if i + 1 < len(dialogue_list):
            agent_item = dialogue_list[i + 1]
            agent_text = agent_item["agent_response"]
            block = f"User Query {turn_number}: {user_text}\nAgent Response {turn_number}: {agent_text}\n"
        else:
            block = f"User Query {turn_number}: {user_text}\n"
            last_turn_number = turn_number

        result.append(block)

    result.append(f"Agent Response {last_turn_number}: ")

    return "\n".join(result)


def naive_LLMs_test(test_data, llm, sampling_params, processor, lora_path, policy_name):
    vllm_inputs = []
    index2sample_ID = []
    sample_ID2instance = {}
    for key in ["IDTest", "OOD1Test", "OOD2Test", "OOD3Test"]:

        sub_test_data = test_data[key][:]

        for test_instance in sub_test_data:
            test_sample_ID = test_instance["sample_ID"]

            user_persona = test_instance['user_persona']
            agent_character = test_instance['agent_character']
            dialogue_context = test_instance['dialogue_context']

            str_dialogue_context = format_dialogue_simple(dialogue_context)

            if lora_path == "Naive":
                prompt = f"User Persona: {user_persona}\nAgent Character: {agent_character}\nDialogue Context: {str_dialogue_context}\n"
            else:
                # use the sft or RL evaluated policy
                prompt = f"<User Persona>{user_persona}</User Persona>\n<Agent Character>{agent_character}</Agent Character>\n<Dialogue Context>{str_dialogue_context}</Dialogue Context>\n"

            # print(prompt)
            # assert 1 == 0

            specified_sys_prompt = f"""You are serving for the user with the given persona. Please act as the given agent character"""

            vllm_input = get_vllm_input(
                prompt,
                processor=processor,
                specified_sys_prompt=specified_sys_prompt)
            vllm_inputs.append(vllm_input)

            index2sample_ID.append(test_sample_ID)
            sample_ID2instance[test_sample_ID] = copy.deepcopy(test_instance)

    # vllm_inputs = vllm_inputs[:10]

    if lora_path == "Naive":
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    else:
        print(lora_path)

        from vllm.lora.request import LoRARequest
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params,
                               lora_request=LoRARequest("adapter", 1, lora_path))

    test_results = []
    for idx in range(len(outputs)):
        generated_text = outputs[idx].outputs[0].text
        response = generated_text
        # print(response)

        if "</Agent Response>" in response:
            response = response.split("</Agent Response>")[0]
        if "<Agent Response>" in response:
            response = response.split("<Agent Response>")[-1]
        # print(response)

        original_test_instance = sample_ID2instance[index2sample_ID[idx]]
        original_test_instance["model_response"] = {
            "model_response": response,
            "policy_name": policy_name,
            "aux_info": {
                "generated_text": generated_text
            },

        }
        test_results.append(original_test_instance)

    return test_results


LLM_path = os.environ.get('LLM_PATH')  # the path of the base LLM
LLM_name = os.environ.get('LLM_NAME')  # the name of the base LLM

bench_version = os.environ.get('BENCH_VERSION')

lora_suffix = os.environ.get('LORA_SUFFIX')

if __name__ == "__main__":


    set_seeds(seed=42)
    print(lora_suffix)

    test_data_filepath = f"../../Benchmark/{bench_version}/PDGBench.json"
    print(test_data_filepath)
    with open(test_data_filepath, encoding="utf-8", mode="r") as f:
        test_data = json.load(f)

    max_tokens = 256

    TrainedSFTPolicy_lora_path = f"../evaluated_policy_training/ckpt_SFT/BVersion[{bench_version}]-BASED[{LLM_name}]#TrainedSFTPolicy"
    TrainedThinkingSFTPolicy_lora_path = f"../evaluated_policy_training/ckpt_ThinkingSFT/BVersion[{bench_version}]-BASED[{LLM_name}]#TrainedThinkingSFTPolicy"

    if "TrainedSFTPolicy" in lora_suffix:  # train from BASE
        lora_path = TrainedSFTPolicy_lora_path
    elif "TrainedDASFTPolicy" in lora_suffix:  # train from BASE
        lora_path = f"../evaluated_policy_training/ckpt_DASFT/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
    elif "TrainedDPOPolicy" in lora_suffix:  # train from SFT
        lora_path = f"../evaluated_policy_training/ckpt_DPO/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
    elif "TrainedRejectSFTPolicy" in lora_suffix:  # train from SFT
        lora_path = f"../evaluated_policy_training/ckpt_RejectSFT/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
    elif "TrainedGRPOPolicy" in lora_suffix:  # train from SFT
        lora_path = f"../evaluated_policy_training/ckpt_GRPO/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
    elif "TrainedMIGRPOPolicy" in lora_suffix:  # train from SFT
        lora_path = f"../evaluated_policy_training/ckpt_MIGRPO/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
    elif "TrainedNaiveGRPOPolicy" in lora_suffix:  # train from SFT
        lora_path = f"../evaluated_policy_training/ckpt_NaiveGRPO/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"

    elif "TrainedThinkingSFTPolicy" in lora_suffix:  # train from BASE
        lora_path = f"../evaluated_policy_training/ckpt_ThinkingSFT/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
        max_tokens = 512
    elif "TrainedThinkingFullSFTPolicy" in lora_suffix:  # train from BASE
        lora_path = f"../evaluated_policy_training/ckpt_ThinkingFullSFT/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
        max_tokens = 512
    elif "TrainedThinkingGRPOPolicy" in lora_suffix:  # train from ThinkingSFT
        lora_path = f"../evaluated_policy_training/ckpt_ThinkingGRPO/BVersion[{bench_version}]-BASED[{LLM_name}]#{lora_suffix}"
        max_tokens = 512

    elif "ThinkingGRPOCoRLStage2CKPT" in lora_suffix:  # train from a few
        lora_path = f"../evaluated_policy_training/ckpt_ThinkingGRPOCoRL/PolicyType[MidSFTOne]-ITER[1]-BVersion[{bench_version}]-BASED[{LLM_name}]"
        max_tokens = 512

    else:
        lora_path = "Naive"

    print(lora_path)

    if lora_suffix in ["Naive"]:
        llm, sampling_params, processor = prepare_vllm(model_path=LLM_path, max_tokens=max_tokens)
    elif lora_suffix in ["TrainedDPOPolicy", "TrainedRejectSFTPolicy", "TrainedGRPOPolicy", "TrainedMIGRPOPolicy",
                         "TrainedNaiveGRPOPolicy", "TrainedThinkingGRPOPolicy",
                         "ThinkingGRPOCoRLStage2CKPT" 
                         ]:
        if lora_suffix in ["TrainedThinkingGRPOPolicy"]:
            lora_initialize_list = [TrainedThinkingSFTPolicy_lora_path]
        elif lora_suffix in ["ThinkingGRPOCoRLStage2CKPT"]:
            lora_initialize_list = [
                TrainedThinkingSFTPolicy_lora_path,
                f"../evaluated_policy_training/ckpt_ThinkingGRPOCoRL/PolicyType[thought_policy]-ITER[1]-BVersion[{bench_version}]-BASED[{LLM_name}]",
            ]
        else:
            lora_initialize_list = [TrainedSFTPolicy_lora_path]

        model_path_merged_initial_lora = "temp_lora_initialize"

        base_model = AutoModelForCausalLM.from_pretrained(LLM_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(LLM_path, trust_remote_code=True)

        from peft import PeftModel

        merged_model = base_model
        for lora_initialize in lora_initialize_list:
            merged_model = PeftModel.from_pretrained(merged_model, lora_initialize)
            merged_model = merged_model.merge_and_unload()

        tokenizer.save_pretrained(model_path_merged_initial_lora)
        merged_model.save_pretrained(model_path_merged_initial_lora)

        llm, sampling_params, processor = prepare_vllm(model_path=model_path_merged_initial_lora, use_lora=True,
                                                       max_tokens=max_tokens)

    else:  # for the TrainedSFTPolicy, TrainedDASFTPolicy, TrainedThinkingSFTPolicy, TrainedThinkingFullSFTPolicy, start from the initial llm
        print(LLM_path)
        llm, sampling_params, processor = prepare_vllm(model_path=LLM_path, use_lora=True, max_tokens=max_tokens)

    benchmarked_policy = f"Lora[{lora_suffix}]-BLLM[{LLM_name}]"
    test_results = naive_LLMs_test(test_data, llm, sampling_params, processor, lora_path=lora_path,
                                   policy_name=benchmarked_policy)

    if not os.path.exists(f"test_results/{bench_version}"):
        os.mkdir(f"test_results/{bench_version}")

    test_results_filepath = f"test_results/{bench_version}/TR-BVersion[{bench_version}]-{benchmarked_policy}.json"
    with open(test_results_filepath, 'w', encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
