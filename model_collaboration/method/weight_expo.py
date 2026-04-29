"""
Weight-level: ExPO (Model Extrapolation)

Inspired by "Model Extrapolation Expedites Alignment" (ACL 2025)
Paper: https://arxiv.org/abs/2404.16792

This method extrapolates model weights as a collaboration strategy.
Given n models with the same architecture, it evaluates them on the dev set
and extrapolates from lower-performing to higher-performing models.

Formula: extrapolated_weight = target_weight + alpha * (target_weight - source_weight)
       = (1 + alpha) * target_weight - alpha * source_weight

Modes:
- worst_to_best: Extrapolate from worst model to best model
- topk_bottomk: Merge top-k models, merge bottom-k models, extrapolate from bottom-k to top-k
- pairs (legacy): Specify explicit SFT-DPO pairs for extrapolation
"""
import os
import json
import random
import shutil
import torch
from tqdm import tqdm
from model_collaboration.data import eval
from model_collaboration.utils import lora_check
from model_collaboration.utils.swarm import lora_merge
from model_collaboration.method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer


def extrapolate_models(source_model_path, target_model_path, alpha, output_path, gpu_id):
    """
    Perform model extrapolation: new_weight = target_weight + alpha * (target_weight - source_weight)
    
    This pushes the weights further in the direction from source to target.
    
    Args:
        source_model_path: Path to the source model (lower performance / base)
        target_model_path: Path to the target model (higher performance / aligned)
        alpha: Extrapolation coefficient (typically 0.3 or 0.5)
        output_path: Path to save the extrapolated model
        gpu_id: GPU to use (not used - loads on CPU for efficiency)
    """
    # Load models on CPU to avoid GPU memory issues during weight manipulation
    print(f"Loading source model from: {source_model_path} (on CPU)")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading target model from: {target_model_path} (on CPU)")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Verify models have the same architecture
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    if len(source_state_dict) != len(target_state_dict):
        raise ValueError(
            f"Model architecture mismatch: source has {len(source_state_dict)} parameters, "
            f"target has {len(target_state_dict)} parameters"
        )
    
    print(f"Extrapolating with alpha={alpha}...")
    # Perform extrapolation: new = target + alpha * (target - source)
    total = len(target_state_dict)
    for name, target_param in tqdm(target_model.named_parameters(), total=total, desc="Extrapolating"):
        source_param = source_state_dict[name]
        # new_weight = target_weight + alpha * (target_weight - source_weight)
        target_param.data = target_param.data + alpha * (target_param.data - source_param.data)
    
    # Save the extrapolated model
    print(f"Saving extrapolated model to: {output_path}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    target_model.save_pretrained(output_path)
    
    # Save tokenizer from the target model
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    # Clean up to free memory
    del source_model
    del target_model
    del source_state_dict
    del target_state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Run ExPO (Model Extrapolation) method.
    
    Args:
        task: str, the name of the task
        task_type: str, the type of the task (e.g., "multiple_choice", "exact_match", etc.)
        gpu_ids: list of int, the GPU ids to use
        model_names: list of str, n models with the same architecture
        hyperparameters: dict, method-specific hyperparameters:
            - alpha: float, extrapolation coefficient (default: 0.3)
            - mode: str, extrapolation strategy (default: "worst_to_best")
                - "worst_to_best": extrapolate from worst to best model
                - "topk_bottomk": extrapolate from merged bottom-k to merged top-k
                - "pairs": legacy mode with explicit SFT-DPO pairs
            - k: int, number of models for top-k/bottom-k mode (default: 1)
            - alpha_mode: str, "fixed" or "optimized" (default: "fixed")
            - alpha_candidates: list of float, for optimization (default: [0.1, 0.2, 0.3, 0.4, 0.5])
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("ExPO: Model Extrapolation for Collaboration")
    print("=" * 60)
    print(f"Number of models: {len(model_names)}")
    for i, name in enumerate(model_names):
        print(f"  Model {i + 1}: {name}")
    print("Make sure all models share the same architecture, or expect errors.")
    print("=" * 60)
    
    # Convert LoRA adapters to full models if necessary
    model_names = lora_check.lora_to_full(model_names)
    
    # Extract hyperparameters
    mode = hyperparameters.get("mode", "worst_to_best")
    alpha = hyperparameters.get("alpha", 0.3)
    alpha_mode = hyperparameters.get("alpha_mode", "fixed")
    alpha_candidates = hyperparameters.get("alpha_candidates", [0.1, 0.2, 0.3, 0.4, 0.5])
    k = hyperparameters.get("k", 1)
    
    # Create output directory
    expo_base_path = hyperparameters.get("expo_base_path", "model_collaboration/logs/expo/")
    expo_base_path = expo_base_path.rstrip("/") + "_" + task + "/"
    if os.path.exists(expo_base_path):
        expo_base_path = expo_base_path.rstrip("/") + "_" + str(random.randint(0, 10000000)) + "/"
        print(f"Warning: expo base path already exists. Using new path: {expo_base_path}")
    os.makedirs(expo_base_path, exist_ok=True)
    
    gpu_id = gpu_ids[0]
    
    # Handle legacy pairs mode
    if mode == "pairs":
        return run_pairs_mode(task, task_type, gpu_ids, model_names, hyperparameters, expo_base_path)
    
    # Step 1: Evaluate all models on the dev set
    print("\n" + "=" * 60)
    print("Step 1: Evaluating all models on dev set")
    print("=" * 60)
    
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    
    list_of_input_list = [dev_input_list for _ in model_names]
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )
    
    # Calculate scores for each model
    model_scores = []
    for i, model_name in enumerate(model_names):
        dev_outputs = list_of_output_list[i]
        dev_scores = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_scores) / len(dev_scores)
        model_scores.append(avg_dev_score)
        print(f"Model {i + 1} ({model_name}): dev {task} score = {avg_dev_score:.4f}")
    
    # Rank models by score
    ranked_indices = sorted(range(len(model_names)), key=lambda i: model_scores[i], reverse=True)
    print(f"\nModel ranking (best to worst):")
    for rank, idx in enumerate(ranked_indices):
        print(f"  Rank {rank + 1}: {model_names[idx]} (score: {model_scores[idx]:.4f})")
    
    # Step 2: Determine source and target models based on mode
    print("\n" + "=" * 60)
    print(f"Step 2: Preparing models for extrapolation (mode: {mode})")
    print("=" * 60)
    
    if mode == "worst_to_best":
        # Extrapolate from worst to best
        source_model_path = model_names[ranked_indices[-1]]  # worst
        target_model_path = model_names[ranked_indices[0]]   # best
        print(f"Source (worst): {source_model_path} (score: {model_scores[ranked_indices[-1]]:.4f})")
        print(f"Target (best): {target_model_path} (score: {model_scores[ranked_indices[0]]:.4f})")
        
    elif mode == "topk_bottomk":
        # Merge top-k and bottom-k, then extrapolate
        k = min(k, len(model_names) // 2)  # Ensure k is valid
        if k < 1:
            k = 1
        
        top_k_indices = ranked_indices[:k]
        bottom_k_indices = ranked_indices[-k:]
        
        top_k_models = [model_names[i] for i in top_k_indices]
        bottom_k_models = [model_names[i] for i in bottom_k_indices]
        
        print(f"Top-{k} models (to merge as target):")
        for idx in top_k_indices:
            print(f"  - {model_names[idx]} (score: {model_scores[idx]:.4f})")
        print(f"Bottom-{k} models (to merge as source):")
        for idx in bottom_k_indices:
            print(f"  - {model_names[idx]} (score: {model_scores[idx]:.4f})")
        
        # Merge top-k models (uniform weights)
        if k == 1:
            target_model_path = top_k_models[0]
            source_model_path = bottom_k_models[0]
        else:
            print(f"\nMerging top-{k} models...")
            target_model_path = os.path.join(expo_base_path, "merged_topk")
            top_k_weights = [1.0 / k] * k
            lora_merge(
                weights=top_k_weights,
                lora_name_list=top_k_models,
                output_path=target_model_path,
                gpu_id=gpu_id
            )
            
            print(f"Merging bottom-{k} models...")
            source_model_path = os.path.join(expo_base_path, "merged_bottomk")
            bottom_k_weights = [1.0 / k] * k
            lora_merge(
                weights=bottom_k_weights,
                lora_name_list=bottom_k_models,
                output_path=source_model_path,
                gpu_id=gpu_id
            )
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'worst_to_best', 'topk_bottomk', 'pairs'")
    
    # Step 3: Optimize or fix alpha
    if alpha_mode == "optimized":
        print("\n" + "=" * 60)
        print(f"Step 3: Optimizing alpha from candidates: {alpha_candidates}")
        print("=" * 60)
        
        best_alpha = alpha_candidates[0]
        best_dev_score = -float("inf")
        
        for candidate_alpha in alpha_candidates:
            print(f"\n--- Testing alpha = {candidate_alpha} ---")
            
            extrapolated_model_path = os.path.join(expo_base_path, f"expo_alpha_{candidate_alpha}")
            extrapolate_models(
                source_model_path=source_model_path,
                target_model_path=target_model_path,
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
    
    # Step 4: Create final extrapolated model
    print("\n" + "=" * 60)
    print("Step 4: Creating final extrapolated model")
    print("=" * 60)
    
    final_model_path = os.path.join(expo_base_path, "final_model")
    extrapolate_models(
        source_model_path=source_model_path,
        target_model_path=target_model_path,
        alpha=alpha,
        output_path=final_model_path,
        gpu_id=gpu_id
    )
    
    # Step 5: Evaluate on test set
    print("\n" + "=" * 60)
    print("Step 5: Evaluating on test set")
    print("=" * 60)
    
    # Force garbage collection before evaluation
    import gc
    gc.collect()
    for gid in gpu_ids:
        try:
            with torch.cuda.device(gid):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except:
            pass
    
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    
    list_of_output_list = distributed_generation.distributed_generation(
        [final_model_path],
        [test_input_list],
        [gpu_ids[0]]
    )
    
    test_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    
    print(f"\n{'=' * 60}")
    print(f"ExPO test {task} score: {avg_test_score:.4f}")
    print(f"  Mode: {mode}")
    print(f"  Alpha: {alpha}")
    print(f"  Source: {source_model_path}")
    print(f"  Target: {target_model_path}")
    print(f"{'=' * 60}")
    
    # Save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "expo",
        "mode": mode,
        "model_names": model_names,
        "model_scores": {model_names[i]: model_scores[i] for i in range(len(model_names))},
        "source_model": source_model_path,
        "target_model": target_model_path,
        "hyperparameters": hyperparameters,
        "alpha": alpha,
        "alpha_mode": alpha_mode,
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
    
    log_filename = f"model_collaboration/logs/{task}_{len(model_names)}_{round(avg_test_score, 4)}_expo.json"
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    
    print(f"\nLogs saved to: {log_filename}")
    print(f"Extrapolated model saved to: {final_model_path}")
    
    return 0


def run_pairs_mode(task, task_type, gpu_ids, model_names, hyperparameters, expo_base_path):
    """
    Legacy mode: Handle explicit SFT-DPO pairs.
    
    Models should be in alternating order: [sft1, dpo1, sft2, dpo2, ...]
    Or specified via sft_models and dpo_models in hyperparameters.
    """
    print(f"\nRunning in legacy 'pairs' mode")
    
    # Parse model pairs
    sft_models = hyperparameters.get("sft_models", None)
    dpo_models = hyperparameters.get("dpo_models", None)
    
    if sft_models is not None and dpo_models is not None:
        if len(sft_models) != len(dpo_models):
            raise ValueError("sft_models and dpo_models must have the same length.")
        pairs = list(zip(sft_models, dpo_models))
    else:
        if len(model_names) % 2 != 0:
            raise ValueError("For pairs mode, model_names must have even length: [sft1, dpo1, sft2, dpo2, ...]")
        pairs = [(model_names[i], model_names[i + 1]) for i in range(0, len(model_names), 2)]
    
    print(f"Number of pairs: {len(pairs)}")
    for i, (sft, dpo) in enumerate(pairs):
        print(f"  Pair {i + 1}: source={sft}, target={dpo}")
    
    alpha = hyperparameters.get("alpha", 0.3)
    alpha_mode = hyperparameters.get("alpha_mode", "fixed")
    alpha_candidates = hyperparameters.get("alpha_candidates", [0.1, 0.2, 0.3, 0.4, 0.5])
    pair_selection = hyperparameters.get("pair_selection", "dpo_score")
    gpu_id = gpu_ids[0]
    
    # Evaluate all models if multiple pairs
    if len(pairs) > 1:
        print("\n" + "=" * 60)
        print("Evaluating all models on dev set to select best pair")
        print("=" * 60)
        
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        
        all_model_paths = []
        for sft, dpo in pairs:
            all_model_paths.extend([sft, dpo])
        
        list_of_input_list = [dev_input_list for _ in all_model_paths]
        list_of_output_list = distributed_generation.distributed_generation(
            all_model_paths,
            list_of_input_list,
            gpu_ids
        )
        
        all_dev_scores = []
        for i, model_path in enumerate(all_model_paths):
            dev_outputs = list_of_output_list[i]
            dev_scores = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_scores) / len(dev_scores)
            all_dev_scores.append(avg_dev_score)
            model_type = "source" if i % 2 == 0 else "target"
            pair_idx = i // 2 + 1
            print(f"Pair {pair_idx} {model_type} ({model_path}): dev score = {avg_dev_score:.4f}")
        
        # Select best pair
        pair_scores = []
        for i in range(len(pairs)):
            sft_score = all_dev_scores[i * 2]
            dpo_score = all_dev_scores[i * 2 + 1]
            
            if pair_selection == "dpo_score":
                score = dpo_score
            elif pair_selection == "improvement":
                score = dpo_score - sft_score
            elif pair_selection == "sft_score":
                score = sft_score
            else:
                score = dpo_score
            
            pair_scores.append(score)
        
        best_pair_idx = pair_scores.index(max(pair_scores))
        source_model_path, target_model_path = pairs[best_pair_idx]
        print(f"\nBest pair selected: Pair {best_pair_idx + 1}")
    else:
        source_model_path, target_model_path = pairs[0]
    
    print(f"  Source: {source_model_path}")
    print(f"  Target: {target_model_path}")
    
    # Optimize alpha if needed
    if alpha_mode == "optimized":
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        best_alpha = alpha_candidates[0]
        best_dev_score = -float("inf")
        
        for candidate_alpha in alpha_candidates:
            extrapolated_model_path = os.path.join(expo_base_path, f"expo_alpha_{candidate_alpha}")
            extrapolate_models(source_model_path, target_model_path, candidate_alpha, extrapolated_model_path, gpu_id)
            
            list_of_output_list = distributed_generation.distributed_generation(
                [extrapolated_model_path], [dev_input_list], [gpu_id]
            )
            dev_outputs = list_of_output_list[0]
            dev_scores = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_scores) / len(dev_scores)
            
            print(f"Alpha {candidate_alpha}: dev score = {avg_dev_score:.4f}")
            if avg_dev_score > best_dev_score:
                best_dev_score = avg_dev_score
                best_alpha = candidate_alpha
        
        alpha = best_alpha
        print(f"Best alpha: {alpha}")
    
    # Create final model
    final_model_path = os.path.join(expo_base_path, "final_model")
    extrapolate_models(source_model_path, target_model_path, alpha, final_model_path, gpu_id)
    
    # Evaluate on test set
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    list_of_output_list = distributed_generation.distributed_generation(
        [final_model_path], [test_input_list], [gpu_ids[0]]
    )
    
    test_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    
    print(f"\nExPO test {task} score: {avg_test_score:.4f} (alpha={alpha})")
    
    # Save logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "expo",
        "mode": "pairs",
        "model_names": model_names,
        "source_model": source_model_path,
        "target_model": target_model_path,
        "hyperparameters": hyperparameters,
        "alpha": alpha,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    
    for i in range(len(test_input_list)):
        experiment_logs["logs"].append({
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_scores[i]
        })
    
    log_filename = f"model_collaboration/logs/{task}_{len(model_names)}_{round(avg_test_score, 4)}_expo.json"
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    
    print(f"Logs saved to: {log_filename}")
    print(f"Extrapolated model saved to: {final_model_path}")
    
    return 0


if __name__ == "__main__":
    run_method()
