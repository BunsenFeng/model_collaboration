import os
import glob
import json
import torch
from tqdm import tqdm
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Models to exclude from 8-bit quantization (fall back to bf16 for these)
NO_8BIT_MODELS = {
    "openai/gpt-oss-20b",                          # trust_remote_code conflicts with bitsandbytes
    "google/gemma-3-12b-it",                       # CUDA device-side assert with 8-bit
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",       # 8-bit overhead too slow for SSM layers
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",  # same
}

# Models that need a reduced batch size to avoid OOM in bf16
SMALL_BATCH_MODELS = {
    "openai/gpt-oss-20b": 16,
    # NOTE: 30B was batch_size=8 when run in bf16; with 8-bit it uses default 32 (fits on H100)
}

# Models that use per-instance resume by default (skip already-generated outputs on restart)
RESUME_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
}

# Models that need more tokens to finish chain-of-thought before answering
LONG_RESPONSE_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 2048,
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

# gpt-oss-20b emits a spurious "assistantfinal" prefix before the answer due to chat template
STRIP_ASSISTANT_FINAL_MODELS = {
    "openai/gpt-oss-20b",
}


def strip_think_tags(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def strip_assistant_final(text: str) -> str:
    # gpt-oss-20b appends "assistantfinal<ANSWER>" at the end; extract just the answer
    idx = text.lower().find("assistantfinal")
    if idx >= 0:
        return text[idx + len("assistantfinal"):].strip()
    return text

def find_existing_log(task, simple_model_name):
    pattern = "model_collaboration/logs/{}_{}*_single_model.json".format(task, simple_model_name)
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def checkpoint_path(task, simple_model_name):
    return "model_collaboration/logs/{}_{}_ckpt.json".format(task, simple_model_name)

def load_checkpoint(task, simple_model_name):
    path = checkpoint_path(task, simple_model_name)
    if os.path.exists(path):
        return eval._retry_read_json(path)
    return None

def save_checkpoint(task, simple_model_name, input_list, output_list):
    data = {inp: out for inp, out in zip(input_list, output_list) if out is not None}
    with open(checkpoint_path(task, simple_model_name), "w") as f:
        json.dump(data, f)

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
    ratio = hyperparameters.get("ratio", 1.0)

    assert len(model_names) == 1, "This method only supports a single model."

    model_name = model_names[0]
    simple_model_name = model_name.split("/")[-1]

    if model_name in SMALL_BATCH_MODELS:
        batch_size = min(batch_size, SMALL_BATCH_MODELS[model_name])
    if model_name in LONG_RESPONSE_MODELS:
        max_new_tokens = max(max_new_tokens, LONG_RESPONSE_MODELS[model_name])

    resume = hyperparameters.get("resume", False) or model_name in RESUME_MODELS

    # evaluate on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test", ratio=ratio)

    # Pre-fill output list from checkpoint (mid-run save) or existing result log
    output_list = [None] * len(test_input_list)
    if resume:
        existing_map = {}
        ckpt = load_checkpoint(task, simple_model_name)
        if ckpt:
            existing_map = ckpt
            print("Resume: loaded checkpoint for {}/{}".format(task, simple_model_name))
        else:
            existing_log = find_existing_log(task, simple_model_name)
            if existing_log:
                existing_data = eval._retry_read_json(existing_log)
                existing_map = {e["input"]: e["output"] for e in existing_data.get("logs", [])}
        if existing_map:
            for i, inp in enumerate(test_input_list):
                if inp in existing_map:
                    output_list[i] = existing_map[inp]
            n_existing = sum(o is not None for o in output_list)
            print("Resume: loaded {}/{} existing outputs".format(n_existing, len(test_input_list)))

    missing_indices = [i for i, o in enumerate(output_list) if o is None]

    if missing_indices:
        if load_in_8bit and model_name not in NO_8BIT_MODELS:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        for batch_start in tqdm(range(0, len(missing_indices), batch_size)):
            batch_idx = missing_indices[batch_start:batch_start+batch_size]
            batch_inputs = [test_input_list[i] for i in batch_idx]
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
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if model_name in STRIP_THINK_MODELS:
                decoded_outputs = [strip_think_tags(o) for o in decoded_outputs]
            if model_name in STRIP_ASSISTANT_FINAL_MODELS:
                decoded_outputs = [strip_assistant_final(o) for o in decoded_outputs]
            for k, idx in enumerate(batch_idx):
                output_list[idx] = decoded_outputs[k]
            if resume:
                save_checkpoint(task, simple_model_name, test_input_list, output_list)

    test_scores = eval.get_scores(task, task_type, "test", output_list, ratio=ratio)
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

    log_filename = "model_collaboration/logs/{}_{}_{}_single_model.json".format(task, simple_model_name, round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    ckpt = checkpoint_path(task, simple_model_name)
    if os.path.exists(ckpt):
        os.remove(ckpt)

if __name__ == "__main__":
    run_method()
