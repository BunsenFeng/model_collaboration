import os
import json
import torch
from tqdm import tqdm
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Models to exclude from 8-bit quantization (fall back to bf16 for these)
NO_8BIT_MODELS = {
    "openai/gpt-oss-20b",       # trust_remote_code conflicts with bitsandbytes
    "google/gemma-3-12b-it",    # CUDA device-side assert with 8-bit
}

# Qwen3 models support enable_thinking=False in apply_chat_template
QWEN3_MODELS = {
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
}

# DeepSeek-R1 models always emit <think>...</think>; strip it from output before scoring
STRIP_THINK_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}


def strip_think_tags(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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
    load_in_8bit = hyperparameters.get("load_in_8bit", False)

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

    if load_in_8bit and model_name not in NO_8BIT_MODELS:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
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
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if model_name in QWEN3_MODELS:
                    kwargs["enable_thinking"] = False
                chat_input = tokenizer.apply_chat_template(chat, **kwargs)
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
        if model_name in STRIP_THINK_MODELS:
            decoded_outputs = [strip_think_tags(o) for o in decoded_outputs]
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