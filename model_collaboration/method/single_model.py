import os
import json
import torch
from tqdm import tqdm
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    max_new_tokens = hyperparameters.get("max_response_length", 100)
    temperature = hyperparameters.get("temperature", 0.7)
    top_p = hyperparameters.get("top_p", 0.9)
    batch_size = hyperparameters.get("batch_size", 8)

    assert len(model_names) == 1, "This method only supports a single model."

    # evaluate on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")

    # list_of_input_list = [test_input_list]
    # list_of_output_list = distributed_generation.distributed_generation(
    #     model_names,
    #     list_of_input_list,
    #     gpu_ids
    # )

    # set to multiple devices in the list of gpu_ids
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])

    model_name = model_names[0]

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    output_list = []
    for i in tqdm(range(0, len(test_input_list), batch_size)):
        batch_inputs = test_input_list[i:i+batch_size]
        # try to apply chat template
        try:
            chat_inputs = []
            for input in batch_inputs:
                chat = [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input}
                ]
                chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                chat_inputs.append(chat_input)
        except:
            chat_inputs = batch_inputs
        
        inputs = tokenizer(chat_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        output_list.extend(decoded_outputs)

    test_scores = eval.get_scores(task, task_type, "test", output_list)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Model: {}, test {} score: {}".format(model_names[0], task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "output": output_list[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)

    # file name with task, model name, and avg_test_score with 4 decimal places
    simple_model_name = model_names[0].split("/")[-1]
    log_filename = "model_collaboration/logs/{}_{}_{}_single_model.json".format(task, simple_model_name, round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

if __name__ == "__main__":
    run_method()