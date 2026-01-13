import copy
import json

if __name__ == '__main__':
    bench_version="v15"
    collected_policies=[
        "Lora[TrainedSFTPolicy]-BLLM[Qwen3-4B-Base]",
    ]


    test_results_file_path = f"../test_generation/test_results/{bench_version}/TR-BVersion[{bench_version}]-{collected_policies[0]}.json"
    with open(test_results_file_path, encoding="utf-8", mode="r") as f:
        test_instances = json.load(f)

    dict_test_instances={}
    for test_instance in test_instances:
        sample_ID = test_instance["sample_ID"]
        dict_test_instances[sample_ID] = copy.deepcopy(test_instance)
        dict_test_instances[sample_ID]["model_response"]={}

    for collected_policy in collected_policies:
        test_results_file_path=f"../test_generation/test_results/{bench_version}/TR-BVersion[{bench_version}]-{collected_policy}.json"
        try:
            with open(test_results_file_path, encoding="utf-8", mode="r") as f:
                test_results = json.load(f)
        except:
            print("NOT FOUND test FILE")
            continue

        for test_result in test_results:
            sample_ID=test_result["sample_ID"]
            model_response=test_result["model_response"]["model_response"]

            dict_test_instances[sample_ID]["model_response"][collected_policy]=model_response


    collected_policy_generations_filepath = f"CPG-BVersion[{bench_version}].json"
    with open(collected_policy_generations_filepath, 'w', encoding="utf-8") as f:
        json.dump(dict_test_instances, f, indent=2)