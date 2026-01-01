import json
from model_collaboration.data import eval
from model_collaboration.utils import logit_arithmetic
from transformers import AutoTokenizer
from model_collaboration.method import distributed_generation

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
    mode = hyperparameters.get("mode", "average") # average or optimized

    if mode == "optimized":
        raise NotImplementedError("Optimized logit fusion is not implemented yet.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_names[0], use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logit_calc_object = logit_arithmetic.LogitArithmetic(
        model_names=model_names,
        model_devices=["cuda:{}".format(gpu_id) for gpu_id in gpu_ids],
        tokenizer=tokenizer
    )

    # preparing inputs
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    outputs = logit_calc_object.batch_generate(
        prompts=test_input_list,
        tokenizer=logit_calc_object.tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temprature=temperature
    )

    test_scores = eval.get_scores(task, task_type, "test", outputs)
    avg_test_scores = sum(test_scores) / len(test_scores)
    print("Final test {} score after logit fusion: {}".format(task, avg_test_scores))

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
    log_filename = "model_collaboration/logs/{}_{}_{}_logit_fusion.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()