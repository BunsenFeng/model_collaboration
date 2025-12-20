"""
Weight-level: ExPO (Model Extrapolation)

Implementation of "Model Extrapolation Expedites Alignment" (ACL 2025)
Paper: https://arxiv.org/abs/2404.16792

ExPO extrapolates model weights beyond the DPO/RLHF checkpoint in the direction
away from the SFT checkpoint. This amplifies the alignment effect.

Formula: extrapolated_weight = dpo_weight + alpha * (dpo_weight - sft_weight)
       = (1 + alpha) * dpo_weight - alpha * sft_weight

Supports multiple model pairs: evaluates on dev set and selects the best pair.
Models can be specified as:
  - Alternating list: [sft1, dpo1, sft2, dpo2, ...] (default)
  - Or via hyperparameters: sft_models=[...], dpo_models=[...]
"""
import os
import json
import random
import shutil
import torch
from tqdm import tqdm
from data import eval
from utils import lora_check
from method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer


def extrapolate_models(sft_model_path, dpo_model_path, alpha, output_path, gpu_id):
    """
    Perform model extrapolation: new_weight = dpo_weight + alpha * (dpo_weight - sft_weight)
    
    Args:
        sft_model_path: Path to the SFT model (base/reference model)
        dpo_model_path: Path to the DPO/RLHF model (aligned model)
        alpha: Extrapolation coefficient (typically 0.3 or 0.5)
        output_path: Path to save the extrapolated model
        gpu_id: GPU to use for loading models (not used - loads on CPU for efficiency)
    """
    # Load models on CPU to avoid GPU memory issues during weight manipulation
    # This is more memory efficient since we only need to manipulate weights, not run inference
    
    print(f"Loading SFT model from: {sft_model_path} (on CPU)")
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading DPO model from: {dpo_model_path} (on CPU)")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Verify models have the same architecture
    sft_state_dict = sft_model.state_dict()
    dpo_state_dict = dpo_model.state_dict()
    
    if len(sft_state_dict) != len(dpo_state_dict):
        raise ValueError(
            f"Model architecture mismatch: SFT has {len(sft_state_dict)} parameters, "
            f"DPO has {len(dpo_state_dict)} parameters"
        )
    
    print(f"Extrapolating with alpha={alpha}...")
    # Perform extrapolation: new = dpo + alpha * (dpo - sft)
    total = len(dpo_state_dict)
    for name, dpo_param in tqdm(dpo_model.named_parameters(), total=total, desc="Extrapolating"):
        sft_param = sft_state_dict[name]
        # new_weight = dpo_weight + alpha * (dpo_weight - sft_weight)
        dpo_param.data = dpo_param.data + alpha * (dpo_param.data - sft_param.data)
    
    # Save the extrapolated model
    print(f"Saving extrapolated model to: {output_path}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    dpo_model.save_pretrained(output_path)
    
    # Save tokenizer from the DPO model
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    # Clean up to free memory
    del sft_model
    del dpo_model
    del sft_state_dict
    del dpo_state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path


def parse_model_pairs(model_names, hyperparameters):
    """
    Parse model names into SFT-DPO pairs.
    
    Supports two formats:
    1. Alternating list in model_names: [sft1, dpo1, sft2, dpo2, ...]
    2. Explicit lists in hyperparameters: sft_models=[...], dpo_models=[...]
    
    Returns:
        List of tuples: [(sft1, dpo1), (sft2, dpo2), ...]
    """
    sft_models = hyperparameters.get("sft_models", None)
    dpo_models = hyperparameters.get("dpo_models", None)
    
    if sft_models is not None and dpo_models is not None:
        # Explicit specification via hyperparameters
        if len(sft_models) != len(dpo_models):
            raise ValueError(
                f"sft_models and dpo_models must have the same length. "
                f"Got {len(sft_models)} SFT models and {len(dpo_models)} DPO models."
            )
        pairs = list(zip(sft_models, dpo_models))
    else:
        # Alternating list: [sft1, dpo1, sft2, dpo2, ...]
        if len(model_names) % 2 != 0:
            raise ValueError(
                f"model_names must have even length for alternating SFT-DPO pairs. "
                f"Got {len(model_names)} models. "
                f"Alternatively, specify sft_models and dpo_models in hyperparameters."
            )
        pairs = [(model_names[i], model_names[i + 1]) for i in range(0, len(model_names), 2)]
    
    return pairs


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Run ExPO (Model Extrapolation) method with support for multiple model pairs.
    
    Args:
        task: str, the name of the task
        task_type: str, the type of the task (e.g., "multiple_choice", "exact_match", etc.)
        gpu_ids: list of int, the GPU ids to use
        model_names: list of str, models in alternating order [sft1, dpo1, sft2, dpo2, ...]
                     OR any list if sft_models/dpo_models specified in hyperparameters
        hyperparameters: dict, method-specific hyperparameters:
            - alpha: float, extrapolation coefficient (default: 0.3)
            - mode: str, "fixed" or "optimized" (default: "fixed")
            - alpha_candidates: list of float, candidates for optimization (default: [0.1, 0.2, 0.3, 0.4, 0.5])
            - sft_models: list of str, explicit list of SFT models (optional)
            - dpo_models: list of str, explicit list of DPO models (optional)
            - pair_selection: str, criterion for selecting best pair: "dpo_score", "improvement", "sft_score" (default: "dpo_score")
    """
    
    print("=" * 60)
    print("ExPO: Model Extrapolation Expedites Alignment")
    print("=" * 60)
    
    # Parse model pairs
    model_pairs = parse_model_pairs(model_names, hyperparameters)
    num_pairs = len(model_pairs)
    
    print(f"Number of model pairs: {num_pairs}")
    for i, (sft, dpo) in enumerate(model_pairs):
        print(f"  Pair {i + 1}: SFT={sft}, DPO={dpo}")
    print("Make sure each pair shares the same model architecture, or expect errors.")
    print("=" * 60)
    
    # Convert LoRA adapters to full models if necessary
    all_models = []
    for sft, dpo in model_pairs:
        all_models.extend([sft, dpo])
    all_models = lora_check.lora_to_full(all_models)
    
    # Reconstruct pairs after LoRA conversion
    model_pairs = [(all_models[i], all_models[i + 1]) for i in range(0, len(all_models), 2)]
    
    # Extract hyperparameters
    mode = hyperparameters.get("mode", "fixed")
    alpha = hyperparameters.get("alpha", 0.3)
    alpha_candidates = hyperparameters.get("alpha_candidates", [0.1, 0.2, 0.3, 0.4, 0.5])
    pair_selection = hyperparameters.get("pair_selection", "dpo_score")
    
    # Create output directory
    expo_base_path = hyperparameters.get("expo_base_path", "logs/expo/")
    expo_base_path = expo_base_path.rstrip("/") + "_" + task + "/"
    if os.path.exists(expo_base_path):
        expo_base_path = expo_base_path.rstrip("/") + "_" + str(random.randint(0, 10000000)) + "/"
        print(f"Warning: expo base path already exists. Using new path: {expo_base_path}")
    os.makedirs(expo_base_path, exist_ok=True)
    
    gpu_id = gpu_ids[0]
    
    # Step 1: Evaluate all models on the dev set to select the best pair
    if num_pairs > 1:
        print("\n" + "=" * 60)
        print("Step 1: Evaluating all models on dev set to select best pair")
        print("=" * 60)
        
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        
        # Flatten all models for evaluation
        all_model_paths = []
        for sft, dpo in model_pairs:
            all_model_paths.extend([sft, dpo])
        
        # Evaluate all models
        list_of_input_list = [dev_input_list for _ in all_model_paths]
        list_of_output_list = distributed_generation.distributed_generation(
            all_model_paths,
            list_of_input_list,
            gpu_ids
        )
        
        # Calculate scores for each model
        all_dev_scores = []
        for i, model_path in enumerate(all_model_paths):
            dev_outputs = list_of_output_list[i]
            dev_scores = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_scores) / len(dev_scores)
            all_dev_scores.append(avg_dev_score)
            model_type = "SFT" if i % 2 == 0 else "DPO"
            pair_idx = i // 2 + 1
            print(f"Pair {pair_idx} {model_type} ({model_path}): dev {task} score = {avg_dev_score:.4f}")
        
        # Calculate pair scores based on selection criterion
        pair_scores = []
        for i in range(num_pairs):
            sft_score = all_dev_scores[i * 2]
            dpo_score = all_dev_scores[i * 2 + 1]
            
            if pair_selection == "dpo_score":
                # Select pair with best DPO model
                score = dpo_score
            elif pair_selection == "improvement":
                # Select pair with largest improvement from SFT to DPO
                score = dpo_score - sft_score
            elif pair_selection == "sft_score":
                # Select pair with best SFT model
                score = sft_score
            else:
                raise ValueError(f"Unknown pair_selection criterion: {pair_selection}")
            
            pair_scores.append(score)
            print(f"Pair {i + 1} selection score ({pair_selection}): {score:.4f}")
        
        # Select the best pair
        best_pair_idx = pair_scores.index(max(pair_scores))
        best_sft, best_dpo = model_pairs[best_pair_idx]
        
        print(f"\nBest pair selected (by {pair_selection}): Pair {best_pair_idx + 1}")
        print(f"  SFT: {best_sft}")
        print(f"  DPO: {best_dpo}")
    else:
        # Only one pair, use it directly
        best_sft, best_dpo = model_pairs[0]
        print(f"\nUsing single model pair:")
        print(f"  SFT: {best_sft}")
        print(f"  DPO: {best_dpo}")
    
    # Step 2: Optimize or fix alpha
    if mode == "optimized":
        print("\n" + "=" * 60)
        print(f"Step 2: Optimizing alpha from candidates: {alpha_candidates}")
        print("=" * 60)
        
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        
        best_alpha = alpha_candidates[0]
        best_dev_score = -float("inf")
        
        for candidate_alpha in alpha_candidates:
            print(f"\n--- Testing alpha = {candidate_alpha} ---")
            
            # Create extrapolated model for this alpha
            extrapolated_model_path = os.path.join(expo_base_path, f"expo_alpha_{candidate_alpha}")
            extrapolate_models(
                sft_model_path=best_sft,
                dpo_model_path=best_dpo,
                alpha=candidate_alpha,
                output_path=extrapolated_model_path,
                gpu_id=gpu_id
            )
            
            # Evaluate on dev set
            list_of_output_list = distributed_generation.distributed_generation(
                [extrapolated_model_path],
                [dev_input_list],
                [gpu_id]
            )
            
            dev_outputs = list_of_output_list[0]
            dev_scores = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_scores) / len(dev_scores)
            
            print(f"Alpha {candidate_alpha}: dev {task} score = {avg_dev_score:.4f}")
            
            if avg_dev_score > best_dev_score:
                best_dev_score = avg_dev_score
                best_alpha = candidate_alpha
        
        alpha = best_alpha
        print(f"\nBest alpha found: {alpha} (dev score: {best_dev_score:.4f})")
    else:
        print(f"\nUsing fixed alpha = {alpha}")
    
    # Step 3: Create final extrapolated model with the chosen alpha
    print("\n" + "=" * 60)
    print("Step 3: Creating final extrapolated model")
    print("=" * 60)
    
    final_model_path = os.path.join(expo_base_path, "final_model")
    extrapolate_models(
        sft_model_path=best_sft,
        dpo_model_path=best_dpo,
        alpha=alpha,
        output_path=final_model_path,
        gpu_id=gpu_id
    )
    
    # Step 4: Evaluate on test set
    print("\n" + "=" * 60)
    print("Step 4: Evaluating on test set")
    print("=" * 60)
    
    # Force garbage collection and clear CUDA cache before evaluation
    import gc
    gc.collect()
    # Clear cache on all specified GPUs
    for gid in gpu_ids:
        try:
            with torch.cuda.device(gid):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    
    # Use only the first GPU to avoid multiprocessing issues with single model
    list_of_output_list = distributed_generation.distributed_generation(
        [final_model_path],
        [test_input_list],
        [gpu_ids[0]]  # Use single GPU for single model evaluation
    )
    
    test_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    
    print(f"\nExPO test {task} score: {avg_test_score:.4f}")
    print(f"  Selected pair: SFT={best_sft}, DPO={best_dpo}")
    print(f"  Alpha: {alpha}")
    
    # Save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "expo",
        "model_names": model_names,
        "model_pairs": [[sft, dpo] for sft, dpo in model_pairs],
        "selected_sft_model": best_sft,
        "selected_dpo_model": best_dpo,
        "hyperparameters": hyperparameters,
        "alpha": alpha,
        "mode": mode,
        "pair_selection": pair_selection,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)
    
    # Save to a json file
    log_filename = f"logs/{task}_{len(model_pairs) * 2}_{round(avg_test_score, 4)}_expo.json"
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    
    print(f"\nLogs saved to: {log_filename}")
    print(f"Extrapolated model saved to: {final_model_path}")
    
    return 0


if __name__ == "__main__":
    run_method()
