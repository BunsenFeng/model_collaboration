import json
import torch
import copy
import random
import numpy as np
import nevergrad as ng
from typing import List, Optional
from functools import partial
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict, set_peft_model_state_dict

from model_collaboration.data import eval
from model_collaboration.utils import lora_check

# ==========================================
# 1. Helper Functions
# ==========================================

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: str, device: str):
    """Load base model and adapters into memory."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        peft_model = PeftModel.from_pretrained(base_model, lora_module_list[0])
    except Exception as e:
        raise Exception(f'{lora_module_list[0]} is unable to load into the model {model_name_or_path}. Error: {e}')
        
    peft_model.eval()

    print("> Begin to load LoRA modules into memory cache...")
    cache = {}
    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        
        try:
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Module {peft_model_id} cannot be merged due to architecture mismatch.')
                
    return peft_model, tokenizer, cache

def default_l1_regularization(weights, coef):
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return coef * sum_of_squares

def get_score_by_generation(weights, model, tokenizer, cache, input_texts, task, task_type, device, max_new_tokens, regular_coef):
    """Optimization objective: Merge weights -> Generate -> Score."""
    # Synthesize Weights
    final_state_dict = {}
    lora_module_list = list(cache.keys())
    keys = cache[lora_module_list[0]].keys()
    
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = final_state_dict[key] + weights[i] * lora_state_dict[key]
    
    set_peft_model_state_dict(model, final_state_dict)
    
    # Generate
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Evaluate
    try:
        scores = eval.get_scores(task, task_type, "dev", decoded_outputs)
        avg_score = sum(scores) / len(scores)
    except Exception as e:
        print(f"Error during scoring: {e}")
        avg_score = 0.0
    
    print("avg_score: ", avg_score)
    print("weights: ", weights)
    
    reg = default_l1_regularization(weights, regular_coef)
    objective = (1.0 - avg_score) + reg
    return objective

def inference_on_test_set(model, tokenizer, input_texts, batch_size, device, max_new_tokens):
    """Run batch inference on test set."""
    all_outputs = []
    total = len(input_texts)
    
    for i in tqdm(range(0, total, batch_size), desc="Test Set Inference"):
        batch_inputs = input_texts[i : i + batch_size]
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_outputs.extend(decoded_batch)
        
    return all_outputs

# ==========================================
# 2. Main Logic
# ==========================================

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    print("Running weight_lorahub method (Direct Inference Mode)")
    
    # 1. Setup Parameters
    seed = hyperparameters.get("seed", 42)
    lorahub_dev_samples = hyperparameters.get("lorahub_dev_samples", 5) 
    max_inference_step = hyperparameters.get("max_inference_step", 20)
    lora_weight_bound = hyperparameters.get("lora_weight_bound", 1.5)
    regular_coef = hyperparameters.get("regular_coef", 0.05)
    max_response_length = hyperparameters.get("max_response_length", 256)
    test_batch_size = hyperparameters.get("batch_size", 4) 
    if test_batch_size is None: test_batch_size = 4

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Checks & Data Loading
    for model_name in model_names:
        if not lora_check.is_lora_adapter_peft(model_name):
            raise ValueError("Model {} is not a LoRA adapter".format(model_name))
            
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")[:lorahub_dev_samples]
    base_model_path = PeftConfig.from_pretrained(model_names[0]).base_model_name_or_path
    device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"

    # 3. Load Model & Cache
    model, tokenizer, cache = load_base_model_and_lora_modules(model_names, base_model_path, device)

    # 4. Optimization Loop (Nevergrad)
    number_of_loras = len(model_names)
    get_score_partial = partial(
        get_score_by_generation, 
        model=model, tokenizer=tokenizer, cache=cache, 
        input_texts=dev_input_list, task=task, task_type=task_type, 
        device=device, max_new_tokens=max_response_length, regular_coef=regular_coef
    )

    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[lora_weight_bound] * number_of_loras,
        lower=[-lora_weight_bound] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    
    print(f"> Begin LoRAHub optimization...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_weights = recommendation.value
    print("Final Weights:", final_weights)

    # =========================================================================
    # 5. Apply Optimal Weights
    # =========================================================================
    # We apply the found weights directly to the in-memory model.
    # This avoids saving the model to disk and reloading it.
    print("> Applying optimal weights to the in-memory model...")
    
    final_state_dict = {}
    lora_module_list = list(cache.keys())
    keys = cache[lora_module_list[0]].keys()
    
    # Calculate weighted sum of adapters
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = final_weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = final_state_dict[key] + final_weights[i] * lora_state_dict[key]
    
    # Update model weights
    set_peft_model_state_dict(model, final_state_dict)
    
    del cache # Clear adapter cache to free memory
    torch.cuda.empty_cache()

    # =========================================================================
    # 6. Final Test Set Evaluation
    # =========================================================================
    # Run inference on the full test set using the optimized model.
    print("> Evaluating optimized model on Test Set ...")
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    
    test_outputs = inference_on_test_set(
        model=model,
        tokenizer=tokenizer,
        input_texts=test_input_list,
        batch_size=test_batch_size,
        device=device,
        max_new_tokens=max_response_length
    )

    # Score the results
    test_score = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_score) / len(test_score)
    print("LoRAHub test {} score: {}".format(task, avg_test_score))

    # =========================================================================
    # 7. Logging
    # =========================================================================
    # Save the experiment results, weights, and predictions to a JSON file.
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "learned_weights": final_weights.tolist(),
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_score[i]
        }
        experiment_logs["logs"].append(log)

    log_filename = "model_collaboration/logs/{}_{}_{}_lorahub.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    pass