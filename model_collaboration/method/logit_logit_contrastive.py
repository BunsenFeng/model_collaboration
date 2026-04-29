import json
import torch
from model_collaboration.data import eval
from model_collaboration.utils import logit_arithmetic
from transformers import AutoTokenizer
from model_collaboration.method import distributed_generation

# return a function that runs logit contrastive with top-k and bottom-k
def logit_operation_generator(k, lambda_=0.2):
    def operation(logits_list):
        assert len(logits_list) == 2*k, "logits_list length must be 2*k"
        final_logits = torch.zeros_like(logits_list[0])
        for i in range(k):
            final_logits += logits_list[i]  # top-k
            final_logits -= logits_list[i + k]  # bottom-k
        final_logits_output = lambda_ * final_logits + logits_list[0]  # apply that offset to the top-1 logits
        return final_logits_output
    return operation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    batch_size = hyperparameters.get("batch_size")
    max_new_tokens = hyperparameters.get("max_response_length")
    temperature = hyperparameters.get("temperature")

    # method-specific hyperparameters
    k = hyperparameters.get("k", 1) # top-k and bottom-k
    lambda_ = hyperparameters.get("lambda_", 0.2) # scaler for summed logit contrastion

    # evaluate models on the dev set to know the top-k and bottom-k models
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    list_of_input_list = [dev_input_list for _ in model_names]

    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(avg_dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task, avg_dev_score))

    # get the indices of top-k and bottom-k models
    sorted_indices = sorted(range(len(list_of_dev_scores)), key=lambda i: list_of_dev_scores[i], reverse=True)
    top_k_indices = sorted_indices[:k]
    bottom_k_indices = sorted_indices[-k:]
    selected_model_indices = top_k_indices + bottom_k_indices

    # contrastive decoding on the test set
    tokenizer = AutoTokenizer.from_pretrained(model_names[0], use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logit_calc_object = logit_arithmetic.LogitArithmetic(
        model_names=[model_names[i] for i in selected_model_indices],
        model_devices=["cuda:{}".format(gpu_id) for gpu_id in gpu_ids],
        tokenizer=tokenizer
    )

    test_input_list = eval.prepare_inputs(task, task_type, "test")
    outputs = logit_calc_object.batch_generate(
        prompts=test_input_list,
        tokenizer=logit_calc_object.tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temprature=temperature,
        arithmetic_func=logit_operation_generator(k, lambda_)
    )

    test_scores = eval.get_scores(task, task_type, "test", outputs)
    avg_test_scores = sum(test_scores) / len(test_scores)
    print("Final test {} score after logit contrastive: {}".format(task, avg_test_scores))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_scores,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_logit_contrastive.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()