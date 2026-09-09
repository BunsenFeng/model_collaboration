import os
import json
import random
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Iterable, Sequence
from collections import defaultdict, deque
from pathlib import Path
import csv
import numpy as np
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from model_collaboration.utils import distributed_dpo, distributed_grpo
import logging
import inspect

logger = logging.getLogger(__name__)

def _safe_name(model_path: str) -> str:
    """
    Generate unique directory names for adapters based on model paths
    Use a hash or the last part of the path to create a safe directory name   
    """
    # Use the last part of the path, replacing / with _
    safe_name = model_path.replace("/", "_").replace("\\", "_")
    # Limit length to avoid filesystem issues
    if len(safe_name) > 100:
        import hashlib
        safe_name = hashlib.md5(model_path.encode()).hexdigest()[:16]
    return safe_name

def _judge_batch_with_model(
    judge_name: str,
    judge_model: str,
    pairs: List[Dict[str, Any]],
    gpu_id: int,
    batch_size: int,
    base_dir: Optional[str] = None,
    num_rounds: int = 1,
    max_response_length: int = 256,
    temperature: float = 1e-5,
    top_p: float = 1.0,
) -> None:
    """
    Single-judge scoring for a batch of pairs.

    Mirrors the structure of Judge._process_pairs_batch + run_judges:
      - Scores both responses in each pair and writes into pair['judges'][judge_name]['rounds'].
      - Saves intermediate chunk results under base_dir/intermediate_results/<judge_name>/chunk_x.json.
    """
    if not pairs:
        return

    # Output directory for intermediate judge results
    output_dir = None
    if base_dir is not None:
        output_dir = os.path.join(base_dir, "intermediate_results", _safe_name(judge_name))
        os.makedirs(output_dir, exist_ok=True)

    # Split pairs into chunks to avoid OOM
    chunk_size = 50
    pair_chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    # Configure generation hyperparameters for the judge model
    # These can be controlled via config.json (hyperparameters),
    # and are passed in from run_method -> run_judges_sparta.
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    for chunk_idx, chunk in enumerate(pair_chunks):
        # Initialize 'judges' structure on each pair
        for pair in chunk:
            pair.setdefault("judges", {})
            pair["judges"].setdefault(judge_name, {"rounds": []})

        # Can support multiple judge rounds; default is 1
        all_rounds = []
        for _ in range(num_rounds):
            # Build all judge prompts for this round
            instructions: List[str] = []
            index_map: Dict[int, Tuple[int, int]] = {}
            for p_idx, pair in enumerate(chunk):
                instr_text = pair.get("instruction", "")
                responses = pair.get("responses", [])
                for r_idx, resp in enumerate(responses):
                    prompt = f"""
Please judge the following response based on the question and the response to be evaluated.
Question: {instr_text}
Response to be evaluated: {resp}

Operation: Output ONLY a JSON object with one score in this exact format. Score must be in the range of 1 to 10.
Your output should be like this:
{{"score": score}}
"""
                    instructions.append(prompt)
                    index_map[len(instructions) - 1] = (p_idx, r_idx)

            if not instructions:
                continue

            # Call distributed_generation as the judge backend
            judge_outputs_lists = distributed_generation.distributed_generation(
                [judge_model],
                [instructions],
                [gpu_id],
                max_response_length=max_response_length,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                max_parallel_models=1,
            )
            judge_outputs = judge_outputs_lists[0]

            # Parse a single scalar score from judge output
            def _extract_single_score(text: Optional[str]) -> Optional[int]:
                if text is None:
                    return None
                try:
                    s = text.strip()
                    try:
                        data = json.loads(s)
                        if isinstance(data, dict) and "score" in data:
                            val = data["score"]
                            if isinstance(val, (int, float)) and 1 <= val <= 10:
                                return int(val)
                    except json.JSONDecodeError:
                        pass
                    patterns = [
                        r'{\s*"score"\s*:\s*(\d+)\s*}',
                        r'"score"\s*:\s*(\d+)',
                        r'score\s*[:=]\s*(\d+)',
                        r'Score:\s*(\d+)',
                        r'(\d+)\s*/\s*10',
                    ]
                    for pat in patterns:
                        matches = re.findall(pat, s, flags=re.IGNORECASE)
                        for m in matches:
                            try:
                                v = int(m)
                                if 1 <= v <= 10:
                                    return v
                            except Exception:
                                continue
                except Exception:
                    return None
                return None

            # Build round_results for this round
            round_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for flat_idx, resp in enumerate(judge_outputs):
                if flat_idx not in index_map:
                    continue
                p_idx, r_idx = index_map[flat_idx]
                round_results.setdefault(p_idx, {})
                round_results[p_idx].setdefault(
                    r_idx,
                    {"score": None, "response": resp, "error": None},
                )
                if resp is not None:
                    sc = _extract_single_score(resp)
                    if sc is not None:
                        round_results[p_idx][r_idx]["score"] = sc
                    else:
                        round_results[p_idx][r_idx]["error"] = "Failed to extract score"

            all_rounds.append(round_results)

        # Write all_rounds back into the pairs in this chunk
        for p_idx, pair in enumerate(chunk):
            judge_entry = pair["judges"][judge_name]
            for round_results in all_rounds:
                res = round_results.get(p_idx, {})
                has_error = (
                    res.get(0, {}).get("error") is not None
                    or res.get(1, {}).get("error") is not None
                )
                if has_error:
                    scores = [5.0, 5.0]
                    default_scores_used = True
                else:
                    scores = []
                    for i in range(2):
                        if i in res and res[i].get("score") is not None:
                            scores.append(float(res[i]["score"]))
                        else:
                            scores.append(5.0)
                    default_scores_used = len(scores) != 2

                round_data = {
                    "scores": scores,
                    "responses": {
                        "response_0": res.get(0, {}).get("response"),
                        "response_1": res.get(1, {}).get("response"),
                        "error_0": res.get(0, {}).get("error"),
                        "error_1": res.get(1, {}).get("error"),
                        "default_scores_used": default_scores_used,
                    },
                }
                judge_entry["rounds"].append(round_data)

        # Save intermediate chunk results
        if output_dir is not None:
            save_path = os.path.join(output_dir, f"chunk_{chunk_idx}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)

def calculate_judge_averages_sparta(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute average scores per judge for each pair, mirroring calculate_judge_averages.

    For each pair['judges'][judge_name], add 'ave_scores': [ave0, ave1].
    """
    for item in pairs:
        judges = item.get("judges", {})
        for judge_name, judge_data in judges.items():
            rounds = judge_data.get("rounds", [])
            scores0, scores1 = [], []
            all_default = True
            for rd in rounds:
                if not rd.get("responses", {}).get("default_scores_used", False):
                    all_default = False
                sc = rd.get("scores", [])
                if len(sc) >= 2:
                    scores0.append(sc[0])
                    scores1.append(sc[1])
            if all_default:
                judge_data["ave_scores"] = [5.0, 5.0]
            else:
                ave0 = float(np.mean(scores0)) if scores0 else 0.0
                ave1 = float(np.mean(scores1)) if scores1 else 0.0
                judge_data["ave_scores"] = [round(ave0, 2), round(ave1, 2)]
    return pairs

def _update_exp3_weights(
    instr_select_configs: Dict[str, Any],
    judged_pairs: List[Dict[str, Any]],
    instructions: List[str],
    iterations: int,
) -> np.ndarray:
    """
    EXP3 prompt-weight update for Stackelberg instruction selection.
    """
    
    # # Theoretically optimal learning rate
    T = iterations
    K = len(instructions)
    # eta = np.sqrt((2 * np.log(K)) / (K * T))
    
    # Get old weights + hyperparameter
    weights = np.asarray(instr_select_configs["weights"], dtype=float)
    gamma = instr_select_configs["gamma"]
    probs = _exp3_probs(weights, gamma, K)

    # Adversarial instruction selector reward
    rewards_active = np.zeros_like(weights, dtype=float)
    rewards_opponent = np.zeros_like(weights, dtype=float)
    for pair in judged_pairs:
        scores = pair.get("scores")
        instr_idx = pair.get("instruction_idx")

        # Accumulate rewards in the case of prompts selected multiple times
        # Reward is bounded from 0 to 1 (inclusive)
        r_active = _exp3_reward(
            model_score=scores[0],
            pair_scores=scores,
            instr_select_configs=instr_select_configs,
            iterations=iterations,
        )
        r_opponent = _exp3_reward(
            model_score=scores[1],
            pair_scores=scores,
            instr_select_configs=instr_select_configs,
            iterations=iterations,
        )
        rewards_active[instr_idx] += r_active / probs[instr_idx]
        rewards_opponent[instr_idx] += r_opponent / probs[instr_idx]

    # Weight update
    reward = (rewards_active + rewards_opponent) / 2
    log_weights = np.log(np.maximum(weights, 1e-12)) + (1 / K) * reward
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)

    return weights

def _update_exp3_per_model_weights(
    instr_select_configs: Dict[str, Any],
    judged_pairs: List[Dict[str, Any]],
    instructions: List[str],
    iterations: int,
):
    """
    EXP3 prompt-weight update for Stackelberg instruction selection, per model.
    """
    
    # # Theoretically optimal learning rate
    T = iterations
    K = len(instructions)
    # eta = np.sqrt((2 * np.log(K)) / (K * T))
    
    # Get old weights + hyperparameter
    weights = instr_select_configs["weights"]
    gamma = instr_select_configs["gamma"]
    model_key_by_path = instr_select_configs["model_key_by_path"]

    # Adversarial instruction selector reward
    rewards_by_model: Dict[str, np.ndarray] = {
        model_key: np.zeros(K, dtype=float)
        for model_key in weights
    }

    for pair in judged_pairs:
        scores = pair.get("scores")
        instr_idx = pair.get("instruction_idx")
        models = pair.get("models")

        # Reward calculation, which is bounded from 0 to 1 (inclusive)
        active_model_path = models[0]
        active_key = model_key_by_path.get(active_model_path, active_model_path)
        active_probs = _exp3_probs(weights[active_key], gamma, K)
        r_active = _exp3_reward(
            model_score=scores[0],
            pair_scores=scores,
            instr_select_configs=instr_select_configs,
            iterations=iterations,
        )
        rewards_by_model[active_key][instr_idx] += r_active / active_probs[instr_idx]

        # opp_model_path = models[1]
        # opp_key = model_key_by_path.get(opp_model_path, opp_model_path)
        # opp_probs = _exp3_probs(weights[opp_key], gamma, K)
        # r_opponent = _exp3_reward(
        #     model_score=scores[1],
        #     pair_scores=scores,
        #     instr_select_configs=instr_select_configs,
        #     iterations=iterations,
        # )
        # rewards_by_model[opp_key][instr_idx] += r_opponent / opp_probs[instr_idx]

    # Weight update
    new_weights: Dict[str, np.ndarray] = {}
    for model_key, old_weights in weights.items():
        model_rewards = rewards_by_model.get(
            model_key,
            np.zeros(K, dtype=float),
        )

        log_weights = np.log(np.maximum(old_weights, 1e-12)) + (1 / K) * model_rewards
        log_weights -= np.max(log_weights)

        new_weights[model_key] = np.exp(log_weights)

    return new_weights

def _exp3_probs(weights: np.ndarray, gamma: float, num_actions: int) -> np.ndarray:
    """
    Convert raw EXP3 weights into a valid probability distribution.
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    probs = (1.0 - gamma) * (weights / weights.sum()) + gamma / num_actions
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum()
    return probs

def _exp3_reward(
    model_score: float,
    pair_scores: List[float],
    instr_select_configs: Dict[str, Any],
    iterations: int,
) -> float:
    """
    Weighted sum reward for adversarial instruction selector.
    """

    def _difficulty_reward(
        score: float,
        instr_select_configs: Dict[str, Any],
    ) -> float:
        """
        Reward low-scoring model outputs with a min threshold for nonzero reward. 
        In other words, prompts that are too hard should receive a reward of 0.
        This component is normalized to [0, 1] assuming judge scores are in [1, 10].
        """
        score_threshold = instr_select_configs["score_threshold"]
        if score < score_threshold:
            return 0.0

        reward = (10.0 - score) / (10.0 - score_threshold)
        return reward

    difficulty_weight = instr_select_configs["difficulty_reward_weight"]
    r_diff = _difficulty_reward(model_score, instr_select_configs)

    def _preference_quality_reward(
        scores: List[float],
        instr_select_configs: Dict[str, Any],
        iterations: int,
    ) -> float:
        """
        Reward preference pairs whose normalized score gap is close to the target gap for this iteration.
        """
        # Find actual score gap
        score_gap = abs(scores[0] - scores[1]) / 9.0

        # Find scheduled ideal score gap
        iteration = instr_select_configs["iteration"]
        tau = iteration / max(iterations - 1, 1)

        gap_start = instr_select_configs["ideal_start_gap"]
        gap_end = instr_select_configs["ideal_end_gap"]
        target_gap = gap_end + (gap_start - gap_end) * (1.0 - tau)

        sigma = instr_select_configs["preference_gap_sigma"]

        reward = math.exp(-((score_gap - target_gap) ** 2) / (2.0 * max(sigma, 1e-12) ** 2))
        return reward
    
    preference_quality_weight = instr_select_configs["preference_quality_reward_weight"]
    r_pref = _preference_quality_reward(pair_scores, instr_select_configs, iterations)

    reward_method = instr_select_configs["reward_method"]
    if reward_method == "difficulty_only":
        reward = r_diff
    elif reward_method == "preference_quality_only":
        reward = r_pref
    else:
        total_w = difficulty_weight + preference_quality_weight
        difficulty_weight /= total_w
        preference_quality_weight /= total_w

        reward = difficulty_weight * r_diff + preference_quality_weight * r_pref
    return reward

def save_exp3_weights(
    base_dir: str,
    iteration: int,
    instr_select_method: str,
    instr_select_configs: Dict[str, Any],
    instructions: List[str],
    top_k: int = 20,
) -> None:
    
    if instr_select_method not in ["exp3", "exp3_per_model"]:
        return
    if "weights" not in instr_select_configs:
        return

    save_dir = os.path.join(base_dir, f"iteration_{iteration}", "analysis")
    os.makedirs(save_dir, exist_ok=True)

    gamma = instr_select_configs["gamma"]
    num_instructions = len(instructions)

    payload: Dict[str, Any] = {
        "iteration": iteration,
        "instruction_selection": instr_select_method,
        "instructions": [
            {
                "instruction_idx": int(i),
                "instruction": instructions[i],
            }
            for i in range(num_instructions)
        ],
    }

    weights = instr_select_configs["weights"]

    if instr_select_method == "exp3":
        weight_vec = np.asarray(weights, dtype=float)
        prob_vec = _exp3_probs(weight_vec, gamma, num_instructions)

        top_indices = np.argsort(prob_vec)[::-1][:top_k]

        payload["weights"] = [float(x) for x in weight_vec]
        payload["probs"] = [float(x) for x in prob_vec]
        payload["top_instructions"] = [
            {
                "rank": int(rank + 1),
                "instruction_idx": int(idx),
                "prob": float(prob_vec[idx]),
                "weight": float(weight_vec[idx]),
                "instruction": instructions[idx],
            }
            for rank, idx in enumerate(top_indices)
        ]

    elif instr_select_method == "exp3_per_model":
        payload["weights"] = {}
        payload["probs"] = {}
        payload["top_instructions_by_model"] = {}

        for model_key, model_weights in weights.items():
            weight_vec = np.asarray(model_weights, dtype=float)
            prob_vec = _exp3_probs(weight_vec, gamma, num_instructions)

            top_indices = np.argsort(prob_vec)[::-1][:top_k]

            payload["weights"][model_key] = [float(x) for x in weight_vec]
            payload["probs"][model_key] = [float(x) for x in prob_vec]
            payload["top_instructions_by_model"][model_key] = [
                {
                    "rank": int(rank + 1),
                    "instruction_idx": int(idx),
                    "prob": float(prob_vec[idx]),
                    "weight": float(weight_vec[idx]),
                    "instruction": instructions[idx],
                }
                for rank, idx in enumerate(top_indices)
            ]

    save_path = os.path.join(save_dir, "exp3_weights.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[Sparta] Iter {iteration}: Saved EXP3 weights to {save_path}")

def load_exp3_weights(
    base_dir: str,
    checkpoint_iteration: int,
    instr_select_method: str,
    instructions: List[str],
    model_names: List[str],
) -> Any:
    """
    Load EXP3 weights saved by save_exp3_weights().

    Expected path:
      base_dir/iteration_{checkpoint_iteration}/analysis/exp3_weights.json

    Supports instr_select_method == "exp3" and "exp3_per_model"
    """
    weights_path = os.path.join(
        base_dir,
        f"iteration_{checkpoint_iteration}",
        "analysis",
        "exp3_weights.json",
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Could not resume EXP3: missing saved weights at {weights_path}. "
            "Expected to load weights from the previous completed iteration."
        )
    
    with open(weights_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    saved_method = payload.get("instruction_selection")
    if saved_method != instr_select_method:
        raise ValueError(
            f"EXP3 checkpoint method mismatch: "
            f"saved method={saved_method!r}, current method={instr_select_method!r}."
        )

    if instr_select_method == "exp3":
        if "weights" not in payload:
            raise ValueError(f"EXP3 checkpoint missing 'weights': {weights_path}")

        weights = np.asarray(payload["weights"], dtype=float)
        print(f"[Sparta] Loaded EXP3 weights from {weights_path}")
        return weights
    
    if instr_select_method == "exp3_per_model":
        saved_weights_by_model = payload.get("weights")
        if not isinstance(saved_weights_by_model, dict):
            raise ValueError(
                f"Per-model EXP3 checkpoint missing dict 'weights': {weights_path}"
            )

        weights_by_model: Dict[str, np.ndarray] = {}

        for model_name in model_names:
            if model_name not in saved_weights_by_model:
                print(
                    f"[Sparta] Warning: no saved EXP3 weights for model {model_name!r}; "
                    "initializing this model's instruction weights to uniform."
                )
                weights_by_model[model_name] = np.ones(len(instructions), dtype=float)
                continue

            weights_by_model[model_name] = np.asarray(
                saved_weights_by_model[model_name],
                dtype=float,
            )

        extra_models = sorted(set(saved_weights_by_model.keys()) - set(model_names))
        if extra_models:
            print(
                f"[Sparta] Warning: ignoring EXP3 weights for models not in current "
                f"model_names: {extra_models}"
            )

        print(f"[Sparta] Loaded per-model EXP3 weights from {weights_path}")
        return weights_by_model
    
    raise ValueError(f"Unsupported instruction selection method: {instr_select_method}")

def save_adapter_dev_scores(
    base_dir: str,
    adapter_dev_records: List[Dict[str, Any]],
    task: str
) -> None:
    """
    Save dev-set performance for each evaluated adapter/model.
    """

    analysis_dir = os.path.join(base_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    csv_path = os.path.join(analysis_dir, f"{task}_adapter_dev_scores.csv")

    fieldnames = [
        "adapter_key",
        "model",
        "iteration",
        "adapter_path",
        "dev_score",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in adapter_dev_records:
            writer.writerow({
                "adapter_key": record.get("adapter_key"),
                "model": record.get("model"),
                "iteration": record.get("iteration"),
                "adapter_path": record.get("adapter_path"),
                "dev_score": record.get("dev_score"),
            })

    print(f"[Sparta] Saved adapter dev scores to {csv_path}")

def run_judges_sparta(
    judge_models: List[str],
    pairs: List[Dict[str, Any]],
    gpu_ids: List[int],
    batch_size: int = 8,
    num_rounds: int = 1,
    base_dir: Optional[str] = None,
    max_response_length: int = 256,
    temperature: float = 1e-5,
    top_p: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Multi-judge wrapper similar to the original run_judges.

    - Supports multiple judge models; each judge is bound to a GPU (cycled if fewer GPUs).
    - Each judge scores pairs in chunks and saves intermediate_results/<judge_name>/chunk_x.json.
    - Returns pairs with a populated 'judges' structure; call calculate_judge_averages_sparta afterwards.
    """
    if not judge_models:
        return pairs

    if not gpu_ids:
        gpu_ids = [0]

    for idx, judge_model in enumerate(judge_models):
        gpu_id = gpu_ids[idx % len(gpu_ids)]

        # judge_models are already HuggingFace identifiers, use them directly
        judge_model_path = judge_model
        # Use the full path as judge_name to match with model_ratings keys
        judge_name = judge_model

        print(f"[Sparta] Running judge {judge_model_path} on GPU {gpu_id}")
        _judge_batch_with_model(
            judge_name=judge_name,
            judge_model=judge_model_path,
            pairs=pairs,
            gpu_id=gpu_id,
            batch_size=batch_size,
            base_dir=base_dir,
            num_rounds=num_rounds,
            max_response_length=64,
            temperature=1e-5,
            top_p=1.0,
        )

    pairs = calculate_judge_averages_sparta(pairs)
    return pairs

"""
Rating logic is implemented via RatingSystem / RatingSystemDynamicWeighted /
RatingSystemStaticWeighted below. The older _update_reputation helper is no longer used.
"""

def save_judged_pairs_sparta(judged_pairs: List[Dict[str, Any]], base_dir: str, iteration: int) -> None:
    """
    Save judged_pairs under model_collaboration/logs/text_sparta_stackelberg/iteration_k/judged_results/judged_pairs.json,
    mirroring the original save_judged_pairs behavior.
    """
    try:
        save_dir = os.path.join(base_dir, f"iteration_{iteration}", "judged_results")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "judged_pairs.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)
        print(f"[Sparta] Judged pairs saved to: {file_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(save_dir, f"judged_pairs_{timestamp}.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)
        print(f"[Sparta] Backup saved to: {backup_path}")
    except Exception as e:
        print(f"[Sparta] Error saving judged pairs: {e}")


def save_rating_history_sparta(
    rating_history: List[Dict[str, Any]],
    base_dir: str,
    iteration: int,
) -> None:
    """
    Save rating_history to model_collaboration/logs/text_sparta_stackelberg/ as a JSON snapshot, matching save_rating_history.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"iteration_{iteration}_rating_history.json")
        history_data = {
            "iteration": iteration,
            "total_pairs": len(rating_history),
            "history": rating_history,
            "final_ratings": rating_history[-1]["ratings"] if rating_history else None,
            "timestamp": datetime.now().isoformat(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"[Sparta] Detailed rating history saved to {file_path}")
    except Exception as e:
        print(f"[Sparta] Error saving rating history to JSON: {e}")


def filter_tie_sparta(preference_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out preference pairs where score_diff == 0 (ties).
    """
    return [pair for pair in preference_pairs if pair.get("score_diff", 0.0) != 0.0]


def save_preference_pairs_to_json_sparta(
    preference_pairs: List[Dict[str, Any]],
    base_dir: str,
    filename: str = "preference_pairs.json",
) -> str:
    """
    Save preference_pairs to the given directory and return the file path.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preference_pairs, f, indent=2, ensure_ascii=False)
        print(f"[Sparta] Preference pairs saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"[Sparta] Error saving preference pairs to JSON: {e}")
        return ""


class RatingSystem:
    """
    Simplified version of the original RatingSystem (no plotting), used for the "normal" mode.
    """

    def __init__(
        self,
        model_scores: Dict[str, Dict[str, float]],
        initial_K: float,
        min_K: float,
        delta_history: Optional[Dict[str, List[float]]] = None,
        window_size: int = 10,
        min_deviation: float = 0.1,
        epsilon: float = 0.01,
        decay_rate: float = 0.9,
        decay_steps: int = 10,
        scaling_factor: float = 20.0,
        freeze_ratings: bool = False,
        debug: bool = False,
    ):
        self.initial_K = initial_K
        self.min_K = min_K
        self.K = initial_K
        self.model_ratings = {m: info.copy() for m, info in model_scores.items()}
        self.window_size = window_size
        self.min_deviation = min_deviation
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.scaling_factor = scaling_factor
        self.freeze_ratings = freeze_ratings
        self.debug = debug

        if delta_history is None:
            self.delta_history = {model: [] for model in model_scores}
        else:
            self.delta_history = delta_history
            for model in model_scores:
                if model in delta_history and len(delta_history[model]) >= 2:
                    new_deviation = float(np.std(delta_history[model]))
                    self.model_ratings[model]["deviation"] = max(
                        new_deviation, self.min_deviation
                    )

        self.update_count = 0

    def select_preference_response(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Follow the original logic: use judge ratings as weights and ave_scores to get weighted scores.
        Returns a dict with chosen_model / rejected_model / score_diff / weighted_scores.
        """
        models = pair.get("models", [])
        responses = pair.get("responses", [])
        judges = pair.get("judges", {})
        if len(models) != 2 or len(responses) != 2 or not judges:
            return None

        model_a, model_b = models
        response_a, response_b = responses

        total_weight = 0.0
        weighted_score_a = 0.0
        weighted_score_b = 0.0

        for judge_name, judge_info in judges.items():
            
            if judge_name in [model_a, model_b]:
                continue
            if judge_name not in self.model_ratings:
                continue

            judge_rating = self.model_ratings[judge_name]["score"]
            ave = judge_info.get("ave_scores")
            if not ave or len(ave) < 2:
                continue
            score_a, score_b = float(ave[0]), float(ave[1])

            weighted_score_a += judge_rating * score_a
            weighted_score_b += judge_rating * score_b
            total_weight += judge_rating

        if total_weight <= 0.0:
            return None

        weighted_score_a /= total_weight
        weighted_score_b /= total_weight
        score_diff = weighted_score_a - weighted_score_b

        if weighted_score_a > weighted_score_b:
            return {
                "instruction": pair.get("instruction", ""),
                "chosen": response_a,
                "rejected": response_b,
                "chosen_model": model_a,
                "rejected_model": model_b,
                "score_diff": float(score_diff),
                "weighted_scores": [float(weighted_score_a), float(weighted_score_b)],
            }
        else:
            return {
                "instruction": pair.get("instruction", ""),
                "chosen": response_b,
                "rejected": response_a,
                "chosen_model": model_b,
                "rejected_model": model_a,
                "score_diff": float(-score_diff),
                "weighted_scores": [float(weighted_score_b), float(weighted_score_a)],
            }

    def update_ratings_from_judges(self, pairs: Any) -> None:
        """
        Update ratings and deviations (normal version, no static/dynamic weight).
        Input can be a single dict or list[dict].
        """
        if self.freeze_ratings:
            return

        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be a dict or list of dicts.")

        self.update_count += 1
        self.K = max(
            self.min_K,
            self.initial_K * (self.decay_rate ** (self.update_count / self.decay_steps)),
        )

        model_deltas = {model: [] for model in self.model_ratings}
        old_deviations = {
            model: self.model_ratings[model]["deviation"] for model in self.model_ratings
        }

        for pair in pairs:
            if not isinstance(pair, dict) or "models" not in pair:
                continue
            model_a, model_b = pair["models"]
            judges = pair.get("judges", {})

            numerator = 0.0
            denominator = 0.0

            for judge_name, judge_info in judges.items():
                if judge_name in [model_a, model_b]:
                    continue
                if judge_name not in self.model_ratings:
                    continue
                judge_rating = self.model_ratings[judge_name]["score"]
                ave = judge_info.get("ave_scores")
                if not ave or len(ave) < 2:
                    continue
                score_a, score_b = float(ave[0]), float(ave[1])
                numerator += judge_rating * (score_a - score_b)
                denominator += judge_rating

            if denominator == 0.0:
                continue

            score_diff = numerator / denominator

            for i, model_i in enumerate([model_a, model_b]):
                model_j = model_b if i == 0 else model_a
                R_i = self.model_ratings[model_i]["score"]
                R_j = self.model_ratings[model_j]["score"]
                sigma_i = self.model_ratings[model_i]["deviation"]
                sigma_j = self.model_ratings[model_j]["deviation"]

                combined_deviation = math.sqrt(sigma_i**2 + sigma_j**2)
                if combined_deviation == 0.0:
                    combined_deviation = 1e-6

                phi_forward = 0.5 * (
                    1.0
                    + math.erf((R_i - R_j) / (math.sqrt(2.0) * combined_deviation))
                )
                phi_backward = 0.5 * (
                    1.0
                    + math.erf((R_j - R_i) / (math.sqrt(2.0) * combined_deviation))
                )

                delta = (
                    self.K
                    * (score_diff if i == 0 else -score_diff)
                    * math.tanh(sigma_i)
                    * max(abs(phi_forward - phi_backward), self.epsilon)
                )
                delta /= self.scaling_factor

                old_score = self.model_ratings[model_i]["score"]
                new_score = max(10.0, old_score + delta)
                actual_delta = new_score - old_score

                self.model_ratings[model_i]["score"] = new_score
                model_deltas[model_i].append(actual_delta)

        for model, deltas in model_deltas.items():
            if not deltas:
                continue
            self.delta_history.setdefault(model, [])
            self.delta_history[model].extend(deltas)
            self.delta_history[model] = self.delta_history[model][-self.window_size :]
            if len(self.delta_history[model]) >= 2:
                new_dev = float(np.std(self.delta_history[model]))
                self.model_ratings[model]["deviation"] = max(
                    new_dev, self.min_deviation
                )

        if self.debug:
            print(f"\nUpdate count: {self.update_count}")
            print(f"Current K value: {self.K:.2f}")
            print("\nDeviation changes:")
            for model in self.model_ratings:
                print(
                    f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}"
                )

    def get_all_ratings(self) -> Dict[str, Dict[str, float]]:
        return self.model_ratings


class RatingSystemDynamicWeighted(RatingSystem):
    """
    Dynamic-weighted variant: extends RatingSystem with dynamic weights computed from previous
    iterations' model_info, following the original script.
    """

    def __init__(
        self,
        model_scores: Dict[str, Dict[str, float]],
        initial_K: float,
        min_K: float,
        delta_history: Optional[Dict[str, List[float]]] = None,
        base_dir: Optional[str] = None,
        current_iteration: Optional[int] = None,
        window_size: int = 10,
        min_deviation: float = 0.1,
        epsilon: float = 0.01,
        decay_rate: float = 0.9,
        decay_steps: int = 10,
        scaling_factor: float = 10.0,
        freeze_ratings: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            model_scores=model_scores,
            initial_K=initial_K,
            min_K=min_K,
            delta_history=delta_history,
            window_size=window_size,
            min_deviation=min_deviation,
            epsilon=epsilon,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            scaling_factor=scaling_factor,
            freeze_ratings=freeze_ratings,
            debug=debug,
        )
        self.base_dir = base_dir
        self.current_iteration = current_iteration
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> Dict[str, float]:
        weights = {model: 1.0 for model in self.model_ratings.keys()}
        if not self.base_dir or self.current_iteration is None:
            return weights
        try:
            if self.current_iteration >= 8:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                if os.path.exists(weights_path):
                    return eval._retry_read_json(weights_path)
                return weights

            if self.current_iteration >= 2:
                prev_iter = self.current_iteration - 1
                prev_path = os.path.join(
                    self.base_dir, f"iteration_{prev_iter}", "model_info.json"
                )
                if not os.path.exists(prev_path):
                    return weights
                prev_info = eval._retry_read_json(prev_path)
                sorted_models = sorted(
                    prev_info.keys(),
                    key=lambda x: prev_info[x]["score"],
                )
                num_weighted = self.current_iteration - 1
                for i in range(min(num_weighted, len(sorted_models))):
                    model = sorted_models[i]
                    if i == 0:
                        weights[model] = 0.0
                    else:
                        weights[model] = 0.1 * i

                if self.current_iteration == 7:
                    weights_path = os.path.join(
                        self.base_dir, "iteration_7", "weights.json"
                    )
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    with open(weights_path, "w") as f:
                        json.dump(weights, f, indent=2)
            return weights
        except Exception as e:
            print(f"[Sparta] Error calculating dynamic weights: {e}")
            return weights

    def update_ratings_from_judges(self, pairs: Any) -> None:
        if self.freeze_ratings:
            return
        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be dict or list of dicts.")

        self.update_count += 1
        self.K = max(
            self.min_K,
            self.initial_K * (self.decay_rate ** (self.update_count / self.decay_steps)),
        )

        model_deltas = {model: [] for model in self.model_ratings}
        old_deviations = {
            model: self.model_ratings[model]["deviation"] for model in self.model_ratings
        }

        for pair in pairs:
            if not isinstance(pair, dict) or "models" not in pair:
                continue
            model_a, model_b = pair["models"]
            judges = pair.get("judges", {})

            numerator = 0.0
            denominator = 0.0

            for judge_name, judge_info in judges.items():
                if judge_name in [model_a, model_b]:
                    continue
                if judge_name not in self.model_ratings:
                    continue
                judge_rating = self.model_ratings[judge_name]["score"]
                ave = judge_info.get("ave_scores")
                if not ave or len(ave) < 2:
                    continue
                score_a, score_b = float(ave[0]), float(ave[1])
                score_a *= self.weights.get(model_a, 1.0)
                score_b *= self.weights.get(model_b, 1.0)
                numerator += judge_rating * (score_a - score_b)
                denominator += judge_rating

            if denominator == 0.0:
                continue

            score_diff = numerator / denominator

            for i, model_i in enumerate([model_a, model_b]):
                model_j = model_b if i == 0 else model_a
                R_i = self.model_ratings[model_i]["score"]
                R_j = self.model_ratings[model_j]["score"]
                sigma_i = self.model_ratings[model_i]["deviation"]
                sigma_j = self.model_ratings[model_j]["deviation"]

                combined_deviation = math.sqrt(sigma_i**2 + sigma_j**2)
                if combined_deviation == 0.0:
                    combined_deviation = 1e-6

                phi_forward = 0.5 * (
                    1.0
                    + math.erf((R_i - R_j) / (math.sqrt(2.0) * combined_deviation))
                )
                phi_backward = 0.5 * (
                    1.0
                    + math.erf((R_j - R_i) / (math.sqrt(2.0) * combined_deviation))
                )

                delta = (
                    self.K
                    * (score_diff if i == 0 else -score_diff)
                    * math.tanh(sigma_i)
                    * max(abs(phi_forward - phi_backward), self.epsilon)
                )
                delta /= self.scaling_factor  # 10.0/scale - dynamic weighted

                old_score = self.model_ratings[model_i]["score"]
                new_score = max(10.0, old_score + delta)
                actual_delta = new_score - old_score

                self.model_ratings[model_i]["score"] = new_score
                model_deltas[model_i].append(actual_delta)

        for model, deltas in model_deltas.items():
            if not deltas:
                continue
            self.delta_history.setdefault(model, [])
            self.delta_history[model].extend(deltas)
            self.delta_history[model] = self.delta_history[model][-self.window_size :]
            if len(self.delta_history[model]) >= 2:
                new_dev = float(np.std(self.delta_history[model]))
                self.model_ratings[model]["deviation"] = max(
                    new_dev, self.min_deviation
                )

        if self.debug:
            print(f"\nUpdate count: {self.update_count}")
            print(f"Current K value: {self.K:.2f}")
            print("\nDeviation changes:")
            for model in self.model_ratings:
                print(
                    f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}"
                )

    def get_weights(self) -> Dict[str, float]:
        return self.weights


class RatingSystemStaticWeighted(RatingSystem):
    """
    Static-weighted variant: uses iteration history to gradually assign fixed weights to more models.
    """

    def __init__(
        self,
        model_scores: Dict[str, Dict[str, float]],
        initial_K: float,
        min_K: float,
        delta_history: Optional[Dict[str, List[float]]] = None,
        base_dir: Optional[str] = None,
        current_iteration: Optional[int] = None,
        window_size: int = 10,
        min_deviation: float = 0.1,
        epsilon: float = 0.01,
        decay_rate: float = 0.9,
        decay_steps: int = 10,
        scaling_factor: float = 20.0,
        freeze_ratings: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            model_scores=model_scores,
            initial_K=initial_K,
            min_K=min_K,
            delta_history=delta_history,
            window_size=window_size,
            min_deviation=min_deviation,
            epsilon=epsilon,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            scaling_factor=scaling_factor,
            freeze_ratings=freeze_ratings,
            debug=debug,
        )
        self.base_dir = base_dir
        self.current_iteration = current_iteration
        self.weights = self._calculate_static_weights()

    def _calculate_static_weights(self) -> Dict[str, float]:
        weights = {model: 1.0 for model in self.model_ratings.keys()}
        if not self.base_dir or self.current_iteration is None:
            return weights
        try:
            if self.current_iteration >= 8:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                if os.path.exists(weights_path):
                    return eval._retry_read_json(weights_path)
                return weights

            weighted_models: List[str] = []
            for iter_num in range(2, self.current_iteration + 1):
                prev_iter = iter_num - 1
                prev_path = os.path.join(
                    self.base_dir, f"iteration_{prev_iter}", "model_info.json"
                )
                if not os.path.exists(prev_path):
                    continue
                prev_info = eval._retry_read_json(prev_path)
                remaining_models = [
                    model
                    for model in prev_info.keys()
                    if model not in weighted_models
                ]
                if not remaining_models:
                    continue
                sorted_models = sorted(
                    remaining_models, key=lambda x: prev_info[x]["score"]
                )
                model = sorted_models[0]
                weighted_models.append(model)
                idx = len(weighted_models) - 1
                if idx == 0:
                    weights[model] = 0.0
                else:
                    weights[model] = 0.1 * idx

            if self.current_iteration == 7:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                with open(weights_path, "w") as f:
                    json.dump(weights, f, indent=2)
            return weights
        except Exception as e:
            print(f"[Sparta] Error calculating static weights: {e}")
            return weights

    def update_ratings_from_judges(self, pairs: Any) -> None:
        if self.freeze_ratings:
            return
        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be dict or list of dicts.")

        self.update_count += 1
        self.K = max(
            self.min_K,
            self.initial_K * (self.decay_rate ** (self.update_count / self.decay_steps)),
        )

        model_deltas = {model: [] for model in self.model_ratings}
        old_deviations = {
            model: self.model_ratings[model]["deviation"] for model in self.model_ratings
        }

        for pair in pairs:
            if not isinstance(pair, dict) or "models" not in pair:
                continue
            model_a, model_b = pair["models"]
            judges = pair.get("judges", {})

            numerator = 0.0
            denominator = 0.0

            for judge_name, judge_info in judges.items():
                if judge_name in [model_a, model_b]:
                    continue
                if judge_name not in self.model_ratings:
                    continue
                judge_rating = self.model_ratings[judge_name]["score"]
                ave = judge_info.get("ave_scores")
                if not ave or len(ave) < 2:
                    continue
                score_a, score_b = float(ave[0]), float(ave[1])
                score_a *= self.weights.get(model_a, 1.0)
                score_b *= self.weights.get(model_b, 1.0)
                numerator += judge_rating * (score_a - score_b)
                denominator += judge_rating

            if denominator == 0.0:
                continue

            score_diff = numerator / denominator

            for i, model_i in enumerate([model_a, model_b]):
                model_j = model_b if i == 0 else model_a
                R_i = self.model_ratings[model_i]["score"]
                R_j = self.model_ratings[model_j]["score"]
                sigma_i = self.model_ratings[model_i]["deviation"]
                sigma_j = self.model_ratings[model_j]["deviation"]

                combined_deviation = math.sqrt(sigma_i**2 + sigma_j**2)
                if combined_deviation == 0.0:
                    combined_deviation = 1e-6

                phi_forward = 0.5 * (
                    1.0
                    + math.erf((R_i - R_j) / (math.sqrt(2.0) * combined_deviation))
                )
                phi_backward = 0.5 * (
                    1.0
                    + math.erf((R_j - R_i) / (math.sqrt(2.0) * combined_deviation))
                )

                delta = (
                    self.K
                    * (score_diff if i == 0 else -score_diff)
                    * math.tanh(sigma_i)
                    * max(abs(phi_forward - phi_backward), self.epsilon)
                )
                delta /= self.scaling_factor  # static - 20.0/scale

                old_score = self.model_ratings[model_i]["score"]
                new_score = max(10.0, old_score + delta)
                actual_delta = new_score - old_score

                self.model_ratings[model_i]["score"] = new_score
                model_deltas[model_i].append(actual_delta)

        for model, deltas in model_deltas.items():
            if not deltas:
                continue
            self.delta_history.setdefault(model, [])
            self.delta_history[model].extend(deltas)
            self.delta_history[model] = self.delta_history[model][-self.window_size :]
            if len(self.delta_history[model]) >= 2:
                new_dev = float(np.std(self.delta_history[model]))
                self.model_ratings[model]["deviation"] = max(
                    new_dev, self.min_deviation
                )

        if self.debug:
            print(f"\nUpdate count: {self.update_count}")
            print(f"Current K value: {self.K:.2f}")
            print("\nDeviation changes:")
            for model in self.model_ratings:
                print(
                    f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}"
                )

    def get_weights(self) -> Dict[str, float]:
        return self.weights


# ======================================================================================
# Stackelberg method
# ======================================================================================

_GLOBAL_LEADER_KEY = "__global__"
_VALID_TRAINING_ALGORITHMS = {"dpo", "grpo"}
_VALID_LEADER_TYPES = {"probabilistic", "uniform"}
_VALID_LEADER_SCOPES = {"global", "per_model"}


def _save_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_training_algorithm(value: Any) -> str:
    value = str(value or "dpo").strip().lower().replace("-", "_")
    aliases = {
        "preference": "dpo",
        "preference_optimization": "dpo",
        "direct_preference_optimization": "dpo",
        "online_grpo": "grpo",
    }
    value = aliases.get(value, value)
    if value not in _VALID_TRAINING_ALGORITHMS:
        raise ValueError(
            f"training_algorithm must be one of {sorted(_VALID_TRAINING_ALGORITHMS)}, got {value!r}."
        )
    return value


def _normalize_leader_type(value: Any) -> str:
    value = str(value or "probabilistic").strip().lower().replace("-", "_")
    aliases = {
        "exp3": "probabilistic",
        "bandit": "probabilistic",
        "random": "uniform",
    }
    value = aliases.get(value, value)
    if value not in _VALID_LEADER_TYPES:
        raise ValueError(f"leader_type must be one of {sorted(_VALID_LEADER_TYPES)}, got {value!r}.")
    return value


def _resolve_modes(hyperparameters: Dict[str, Any]) -> Tuple[str, str, str]:
    training_algorithm = _normalize_training_algorithm(
        hyperparameters.get("training_algorithm", "dpo")
    )
    leader_type = _normalize_leader_type(
        hyperparameters.get("leader_type", "probabilistic")
    )

    legacy_instruction_selection = hyperparameters.get("instruction_selection", "").lower()
    default_scope = "per_model" if legacy_instruction_selection == "exp3_per_model" else "global"
    leader_scope = hyperparameters.get("leader_scope", default_scope).strip().lower()
    if leader_scope not in _VALID_LEADER_SCOPES:
        raise ValueError(f"leader_scope must be one of {sorted(_VALID_LEADER_SCOPES)}, got {leader_scope!r}.")
    if leader_type == "uniform":
        leader_scope = "global"
    return training_algorithm, leader_type, leader_scope


def _weights_to_probs(weights: np.ndarray, uniform_mix: float) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("Leader weights must be a non-empty one-dimensional vector.")
    probs = weights / weights.sum()
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    probs = (1.0 - uniform_mix) * probs + uniform_mix / len(weights)
    probs = np.maximum(probs, 0.0)
    return probs / probs.sum()


def _entropy_normalized(probs: np.ndarray) -> float:
    probs = np.maximum(np.asarray(probs, dtype=float), 1e-12)
    if len(probs) <= 1:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / math.log(len(probs)))


def _apply_entropy_floor(weights: np.ndarray, entropy_floor: float) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    entropy_floor = float(np.clip(entropy_floor, 0.0, 1.0))
    if entropy_floor <= 0.0:
        return weights
    probs = weights / weights.sum()
    if _entropy_normalized(probs) >= entropy_floor:
        return weights
    uniform = np.ones_like(probs) / len(probs)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        mixed = (1.0 - mid) * probs + mid * uniform
        if _entropy_normalized(mixed) >= entropy_floor:
            hi = mid
        else:
            lo = mid
    mixed = (1.0 - hi) * probs + hi * uniform
    return np.maximum(mixed / max(float(mixed.mean()), 1e-12), 1e-12)


def _leader_keys(scope: str, model_names: Sequence[str]) -> List[str]:
    return [_GLOBAL_LEADER_KEY] if scope == "global" else list(model_names)


def _initialize_leader_state(
    base_dir: str,
    instructions: List[str],
    model_names: List[str],
    task: str,
    task_type: str,
    leader_type: str,
    leader_scope: str,
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    k = len(instructions)
    keys = _leader_keys(leader_scope, model_names)
    initial_vector = np.ones(k, dtype=float)

    return {
        "leader_type": leader_type,
        "leader_scope": leader_scope,
        "weights": {key: initial_vector.copy() for key in keys},
    }


def _serialize_leader_state(
    state: Dict[str, Any],
    instructions: List[str],
    model_names: List[str],
    uniform_mix: float,
    iteration: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "iteration": int(iteration),
        "leader_type": state["leader_type"],
        "leader_scope": state["leader_scope"],
        "weights": {},
        "probs": {},
        "entropy_normalized": {},
        "top_instructions": {},
    }
    for key, vector in state["weights"].items():
        weights = np.asarray(vector, dtype=float)
        probs = _weights_to_probs(weights, uniform_mix=uniform_mix)
        top_indices = np.argsort(probs)[::-1][: min(20, len(probs))]
        payload["weights"][key] = [float(x) for x in weights]
        payload["probs"][key] = [float(x) for x in probs]
        payload["entropy_normalized"][key] = _entropy_normalized(probs)
        payload["top_instructions"][key] = [
            {
                "rank": rank + 1,
                "instruction_idx": int(idx),
                "weight": float(weights[idx]),
                "prob": float(probs[idx]),
                "instruction": instructions[idx],
            }
            for rank, idx in enumerate(top_indices)
        ]
    return payload


def _save_leader_state(
    base_dir: str,
    state: Dict[str, Any],
    instructions: List[str],
    model_names: List[str],
    uniform_mix: float,
    iteration: int,
    initial: bool = False,
) -> str:
    if initial:
        path = os.path.join(base_dir, "analysis", "leader_initial_state.json")
    else:
        path = os.path.join(base_dir, f"iteration_{iteration}", "analysis", "leader_state.json")
    _save_json(
        path,
        _serialize_leader_state(
            state=state,
            instructions=instructions,
            model_names=model_names,
            uniform_mix=uniform_mix,
            iteration=iteration,
        ),
    )
    return path


def _load_leader_state(
    base_dir: str,
    checkpoint_iteration: int,
    instructions: List[str],
    model_names: List[str],
    expected_type: str,
    expected_scope: str,
) -> Dict[str, Any]:
    path = os.path.join(base_dir, f"iteration_{checkpoint_iteration}", "analysis", "leader_state.json")
    if os.path.exists(path):
        payload = _read_json(path)
        saved_type = payload.get("leader_type")
        saved_scope = payload.get("leader_scope")
        if saved_type != expected_type or saved_scope != expected_scope:
            raise ValueError(
                "Leader checkpoint/config mismatch: "
                f"saved=({saved_type}, {saved_scope}), requested=({expected_type}, {expected_scope})."
            )
        weights = {
            key: np.asarray(vector, dtype=float)
            for key, vector in payload.get("weights", {}).items()
        }
        expected_keys = set(_leader_keys(expected_scope, model_names))
        if set(weights) != expected_keys:
            raise ValueError(f"Leader checkpoint keys mismatch at {path}.")
        if any(len(vector) != len(instructions) for vector in weights.values()):
            raise ValueError(f"Leader checkpoint prompt count mismatch at {path}.")
        return {"leader_type": expected_type, "leader_scope": expected_scope, "weights": weights}

    # Backward-compatible fallback for the old exp3 script.
    if expected_type == "probabilistic":
        legacy = os.path.join(
            base_dir,
            f"iteration_{checkpoint_iteration}",
            "analysis",
            "exp3_weights.json",
        )
        if os.path.exists(legacy):
            payload = _read_json(legacy)
            raw = payload.get("weights")
            if expected_scope == "global" and isinstance(raw, list):
                return {
                    "leader_type": expected_type,
                    "leader_scope": expected_scope,
                    "weights": {_GLOBAL_LEADER_KEY: np.asarray(raw, dtype=float)},
                }
            if expected_scope == "per_model" and isinstance(raw, dict):
                return {
                    "leader_type": expected_type,
                    "leader_scope": expected_scope,
                    "weights": {
                        model: np.asarray(raw.get(model, np.ones(len(instructions))), dtype=float)
                        for model in model_names
                    },
                }

    raise FileNotFoundError(f"No leader checkpoint found for iteration {checkpoint_iteration}: {path}")


def _leader_vector_for_model(state: Dict[str, Any], model_name: str) -> np.ndarray:
    key = _GLOBAL_LEADER_KEY if state["leader_scope"] == "global" else model_name
    if key not in state["weights"]:
        raise KeyError(f"Missing leader weights for {key!r}.")
    return np.asarray(state["weights"][key], dtype=float)


def _select_opponent(
    current_model: str,
    model_names: List[str],
    model_ratings: Dict[str, Dict[str, float]],
    random_match_prob: float,
    num_opponents: int,
    opponent_selection: str,
    iteration: int,
    total_iterations: int,
    reputation_gap_sigma: float,
) -> str:
    opponents = [m for m in model_names if m != current_model]
    if not opponents:
        raise ValueError("At least two models are required for a duel.")
    if random.random() < random_match_prob:
        return random.choice(opponents)

    current_score = float(model_ratings.get(current_model, {}).get("score", 100.0))
    candidates = [
        (other, abs(current_score - float(model_ratings.get(other, {}).get("score", 100.0))))
        for other in opponents
    ]
    top_k = max(1, min(int(num_opponents), len(candidates)))
    if opponent_selection == "lowest_diff":
        candidates.sort(key=lambda x: x[1])
        return random.choice([name for name, _ in candidates[:top_k]])
    if opponent_selection == "highest_diff":
        candidates.sort(key=lambda x: x[1], reverse=True)
        return random.choice([name for name, _ in candidates[:top_k]])
    if opponent_selection not in {"schedule_decreasing", "schedule_increasing"}:
        raise ValueError(f"Invalid opponent_selection={opponent_selection!r}.")

    tau = iteration / max(total_iterations - 1, 1)
    target_gap = 1.0 - tau if opponent_selection == "schedule_decreasing" else tau
    max_diff = max(diff for _, diff in candidates) if candidates else 0.0
    names: List[str] = []
    scores: List[float] = []
    sigma = max(float(reputation_gap_sigma), 1e-12)
    for name, diff in candidates:
        normalized_gap = 0.0 if max_diff <= 1e-12 else float(np.clip(diff / max_diff, 0.0, 1.0))
        names.append(name)
        scores.append(math.exp(-((normalized_gap - target_gap) ** 2) / (2.0 * sigma**2)))
    probs = np.asarray(scores, dtype=float)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return random.choice(opponents)
    probs /= probs.sum()
    return str(np.random.choice(names, p=probs))


def _sample_duels_unified(
    instructions: List[str],
    model_names: List[str],
    model_ratings: Dict[str, Dict[str, float]],
    leader_state: Dict[str, Any],
    num_duels: int,
    leader_uniform_mix: float,
    random_match_prob: float,
    num_opponents: int,
    opponent_selection: str,
    iteration: int,
    total_iterations: int,
    reputation_gap_sigma: float,
) -> List[Dict[str, Any]]:
    if len(model_names) < 3:
        raise ValueError("At least three models are required: two duelers and one peer judge.")
    if not instructions:
        return []
    duels: List[Dict[str, Any]] = []
    for duel_id in range(int(num_duels)):
        active_model = model_names[duel_id % len(model_names)]
        vector = _leader_vector_for_model(leader_state, active_model)
        if leader_state["leader_type"] == "uniform":
            probs = np.ones(len(instructions), dtype=float) / len(instructions)
        else:
            probs = _weights_to_probs(vector, uniform_mix=leader_uniform_mix)
        instruction_idx = int(np.random.choice(len(instructions), p=probs))
        opponent_model = _select_opponent(
            current_model=active_model,
            model_names=model_names,
            model_ratings=model_ratings,
            random_match_prob=random_match_prob,
            num_opponents=num_opponents,
            opponent_selection=opponent_selection,
            iteration=iteration,
            total_iterations=total_iterations,
            reputation_gap_sigma=reputation_gap_sigma,
        )
        judges = [m for m in model_names if m not in {active_model, opponent_model}]
        duels.append(
            {
                "duel_id": int(duel_id),
                "pair_id": int(duel_id),
                "instruction_idx": instruction_idx,
                "instruction": instructions[instruction_idx],
                "models": [active_model, opponent_model],
                "judge_names": judges,
                "sample_prob": float(probs[instruction_idx]),
                "old_weight": float(vector[instruction_idx]),
            }
        )
    return duels


def _save_prompt_sampling_manifest(base_dir: str, iteration: int, duels: List[Dict[str, Any]]) -> None:
    _save_json(
        os.path.join(base_dir, f"iteration_{iteration}", "analysis", "prompt_sampling_manifest.json"),
        {"iteration": int(iteration), "duels": duels},
    )


def _generate_offline_duel_responses(
    duels: List[Dict[str, Any]],
    current_model_paths: Dict[str, str],
    gpu_ids: List[int],
    max_response_length: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    max_parallel_models: Optional[int],
) -> List[Dict[str, Any]]:
    tasks_by_model: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for duel in duels:
        for model_name in duel["models"]:
            tasks_by_model[model_name].append((int(duel["duel_id"]), duel["instruction"]))

    active_names = [m for m in current_model_paths if tasks_by_model.get(m)]
    model_paths = [current_model_paths[m] for m in active_names]
    input_lists = [[prompt for _, prompt in tasks_by_model[m]] for m in active_names]
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        big_model_mode=False,
    )
    output_lists = distributed_generation.distributed_generation(
        model_paths,
        input_lists,
        gpu_ids,
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        max_parallel_models=max_parallel_models,
    )

    response_by_duel_model: Dict[Tuple[int, str], str] = {}
    for model_name, assignments, outputs in zip(active_names, [tasks_by_model[m] for m in active_names], output_lists):
        if len(assignments) != len(outputs):
            raise RuntimeError(f"Generation length mismatch for {model_name}.")
        for (duel_id, _), output in zip(assignments, outputs):
            response_by_duel_model[(duel_id, model_name)] = output

    raw_pairs: List[Dict[str, Any]] = []
    for duel in duels:
        duel_id = int(duel["duel_id"])
        model_a, model_b = duel["models"]
        if (duel_id, model_a) not in response_by_duel_model or (duel_id, model_b) not in response_by_duel_model:
            continue
        pair = dict(duel)
        pair["model_paths"] = [current_model_paths[model_a], current_model_paths[model_b]]
        pair["responses"] = [
            response_by_duel_model[(duel_id, model_a)],
            response_by_duel_model[(duel_id, model_b)],
        ]
        pair["judges"] = {}
        raw_pairs.append(pair)
    return raw_pairs


def _judge_offline_duels(
    raw_pairs: List[Dict[str, Any]],
    model_names: List[str],
    current_model_paths: Dict[str, str],
    model_ratings: Dict[str, Dict[str, float]],
    gpu_ids: List[int],
    base_dir: str,
    judge_batch_size: int,
    judge_rounds: int,
    judge_max_response_length: int,
    judge_temperature: float,
    judge_top_p: float,
) -> List[Dict[str, Any]]:
    judge_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in raw_pairs:
        for judge_name in pair.get("judge_names", []):
            judge_groups[judge_name].append(pair)

    for idx, (judge_name, pairs) in enumerate(judge_groups.items()):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        _judge_batch_with_model(
            judge_name=judge_name,
            judge_model=current_model_paths[judge_name],
            pairs=pairs,
            gpu_id=gpu_id,
            batch_size=judge_batch_size,
            base_dir=base_dir,
            num_rounds=judge_rounds,
            max_response_length=judge_max_response_length,
            temperature=judge_temperature,
            top_p=judge_top_p,
        )

    judged_pairs = calculate_judge_averages_sparta(raw_pairs)
    complete: List[Dict[str, Any]] = []
    for pair in judged_pairs:
        weighted = np.zeros(2, dtype=float)
        total_weight = 0.0
        for judge_name, judge_info in pair.get("judges", {}).items():
            if judge_name in pair.get("models", []):
                continue
            ave = judge_info.get("ave_scores")
            if not ave or len(ave) < 2:
                continue
            judge_weight = float(model_ratings.get(judge_name, {}).get("score", 1.0))
            weighted += judge_weight * np.asarray(ave[:2], dtype=float)
            total_weight += judge_weight
        if total_weight <= 0.0:
            pair["scores"] = [5.0, 5.0]
        else:
            pair["scores"] = [float(x) for x in weighted / total_weight]
        model_a, model_b = pair["models"]
        score_a, score_b = pair["scores"]
        pair["model_level_scores"] = {model_a: score_a, model_b: score_b}
        pair["model_level_rewards"] = {
            model_a: float(np.clip((score_a - 1.0) / 9.0, 0.0, 1.0)),
            model_b: float(np.clip((score_b - 1.0) / 9.0, 0.0, 1.0)),
        }
        pair["score_diff"] = float(score_a - score_b)
        pair["winner"] = model_a if score_a > score_b else model_b if score_b > score_a else None
        pair["winner_index"] = 0 if score_a > score_b else 1 if score_b > score_a else None
        pair["response_summaries"] = {
            model_a: [
                {
                    "reward": pair["model_level_rewards"][model_a],
                    "raw_score_1_to_10": score_a,
                    "completion_excerpt": str(pair["responses"][0])[:800],
                    "judge_scores": {
                        j: info.get("ave_scores", [None, None])[0]
                        for j, info in pair.get("judges", {}).items()
                    },
                }
            ],
            model_b: [
                {
                    "reward": pair["model_level_rewards"][model_b],
                    "raw_score_1_to_10": score_b,
                    "completion_excerpt": str(pair["responses"][1])[:800],
                    "judge_scores": {
                        j: info.get("ave_scores", [None, None])[1]
                        for j, info in pair.get("judges", {}).items()
                    },
                }
            ],
        }
        complete.append(pair)
    return complete


def _build_rating_system(
    score_type: str,
    model_ratings: Dict[str, Dict[str, float]],
    delta_history: Dict[str, List[float]],
    base_dir: str,
    iteration: int,
    hyperparameters: Dict[str, Any],
) -> RatingSystem:
    common = dict(
        model_scores=model_ratings,
        initial_K=float(hyperparameters.get("initial_k", 10.0)),
        min_K=float(hyperparameters.get("min_k", 5.0)),
        delta_history=delta_history,
        window_size=int(hyperparameters.get("window_size", 10)),
        min_deviation=float(hyperparameters.get("min_deviation", 0.1)),
        epsilon=float(hyperparameters.get("epsilon", 0.01)),
        decay_rate=float(hyperparameters.get("decay_rate", 0.9)),
        decay_steps=int(hyperparameters.get("decay_steps", 10)),
        scaling_factor=float(hyperparameters.get("scaling_factor", 20.0)),
        freeze_ratings=bool(hyperparameters.get("freeze_ratings", False)),
        debug=bool(hyperparameters.get("debug", False)),
    )
    if score_type == "dynamic":
        return RatingSystemDynamicWeighted(base_dir=base_dir, current_iteration=iteration, **common)
    if score_type == "static":
        return RatingSystemStaticWeighted(base_dir=base_dir, current_iteration=iteration, **common)
    if score_type != "normal":
        raise ValueError("score_type must be 'normal', 'dynamic', or 'static'.")
    return RatingSystem(**common)


def _update_leader_state(
    state: Dict[str, Any],
    judged_duels: List[Dict[str, Any]],
    instructions: List[str],
    model_names: List[str],
    base_dir: str,
    iteration: int,
    total_iterations: int,
    task: str,
    task_type: str,
    training_algorithm: str,
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    leader_type = state["leader_type"]
    if leader_type == "uniform":
        return state

    valid_duels = [
        duel
        for duel in judged_duels
        if isinstance(duel.get("scores"), list)
        and len(duel["scores"]) >= 2
        and duel.get("instruction_idx") is not None
    ]
    if not valid_duels:
        print("[Stackelberg] No complete scored duels; preserving leader weights.")
        return state

    # Probabilistic EXP3 leader.
    instr_configs = {
        "reward_method": hyperparameters.get("reward_method", "weighted"),
        "gamma": float(hyperparameters.get("instr_sample_gamma", hyperparameters.get("leader_uniform_mix", 0.2))),
        "difficulty_reward_weight": float(hyperparameters.get("difficulty_reward_weight", 0.3)),
        "score_threshold": float(hyperparameters.get("score_threshold", 3.0)),
        "preference_quality_reward_weight": float(hyperparameters.get("preference_quality_reward_weight", 0.7)),
        "ideal_start_gap": float(hyperparameters.get("ideal_start_gap", 0.6)),
        "ideal_end_gap": float(hyperparameters.get("ideal_end_gap", 0.15)),
        "preference_gap_sigma": float(hyperparameters.get("preference_gap_sigma", 0.15)),
        "iteration": int(iteration),
    }
    if instr_configs["reward_method"] not in {"weighted", "difficulty_only", "preference_quality_only"}:
        raise ValueError("Invalid reward_method for the probabilistic leader.")

    if state["leader_scope"] == "global":
        instr_configs["weights"] = np.asarray(state["weights"][_GLOBAL_LEADER_KEY], dtype=float)
        state["weights"][_GLOBAL_LEADER_KEY] = _update_exp3_weights(
            instr_select_configs=instr_configs,
            judged_pairs=valid_duels,
            instructions=instructions,
            iterations=total_iterations,
        )
    else:
        instr_configs["weights"] = {
            model: np.asarray(state["weights"][model], dtype=float)
            for model in model_names
        }
        instr_configs["model_key_by_path"] = {model: model for model in model_names}
        state["weights"] = _update_exp3_per_model_weights(
            instr_select_configs=instr_configs,
            judged_pairs=valid_duels,
            instructions=instructions,
            iterations=total_iterations,
        )
    return state


def _load_ratings_and_history(
    base_dir: str,
    iteration: int,
    model_names: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]], int]:
    previous_info_path = os.path.join(base_dir, f"iteration_{iteration - 1}", "model_info.json")
    if iteration > 0 and os.path.exists(previous_info_path):
        previous = _read_json(previous_info_path)
        ratings = {
            model: {
                "score": float(previous.get(model, {}).get("score", 100.0)),
                "deviation": float(previous.get(model, {}).get("deviation", 0.5)),
            }
            for model in model_names
        }
    else:
        ratings = {model: {"score": 100.0, "deviation": 0.5} for model in model_names}

    delta_path = os.path.join(base_dir, "rating_deltas.json")
    history = {model: [] for model in model_names}
    update_count = 0
    if os.path.exists(delta_path):
        try:
            payload = _read_json(delta_path)
            raw = payload.get("delta_history", {})
            history = {model: list(raw.get(model, [])) for model in model_names}
            update_count = int(payload.get("update_count", 0))
        except Exception:
            pass
    return ratings, history, update_count


def _save_ratings_and_history(
    base_dir: str,
    iteration: int,
    ratings: Dict[str, Dict[str, float]],
    delta_history: Dict[str, List[float]],
    update_count: int,
) -> None:
    _save_json(
        os.path.join(base_dir, f"iteration_{iteration}", "model_info.json"),
        {
            model: {
                "score": float(info["score"]),
                "deviation": float(info["deviation"]),
            }
            for model, info in ratings.items()
        },
    )
    _save_json(
        os.path.join(base_dir, "rating_deltas.json"),
        {"delta_history": delta_history, "update_count": int(update_count)},
    )


def _restore_latest_adapters(
    base_dir: str,
    model_names: List[str],
    start_iteration: int,
    training_algorithm: str,
) -> Tuple[Dict[str, str], List[Tuple[int, str, str]]]:
    current_paths = {model: model for model in model_names}
    entries: List[Tuple[int, str, str]] = []
    if start_iteration <= 0:
        return current_paths, entries
    prefix = f"{training_algorithm}_"
    for model in model_names:
        for previous_iteration in range(start_iteration - 1, -1, -1):
            candidate = os.path.join(
                base_dir,
                f"iteration_{previous_iteration}",
                f"{prefix}{_safe_name(model)}",
            )
            if os.path.isdir(candidate):
                current_paths[model] = candidate
                entries.append((previous_iteration, model, candidate))
                break
    return current_paths, entries


def _write_grpo_train_files_by_model(
    duels: List[Dict[str, Any]],
    model_names: List[str],
    dataset_dir: str,
) -> Dict[str, str]:
    os.makedirs(dataset_dir, exist_ok=True)
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {model: [] for model in model_names}
    for duel in duels:
        model_a, model_b = duel["models"]
        judges_json = json.dumps(duel["judge_names"], ensure_ascii=False)
        for active_model, opponent_model in ((model_a, model_b), (model_b, model_a)):
            rows_by_model[active_model].append(
                {
                    "prompt": duel["instruction"],
                    "instruction_idx": int(duel["instruction_idx"]),
                    "duel_id": int(duel["duel_id"]),
                    "active_model": active_model,
                    "opponent_model": opponent_model,
                    "judge_names_json": judges_json,
                }
            )
    paths: Dict[str, str] = {}
    for model, rows in rows_by_model.items():
        if not rows:
            continue
        path = os.path.join(dataset_dir, f"grpo_prompts_{_safe_name(model)}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        _append_jsonl(path, rows)
        paths[model] = path
    return paths


def _load_reward_log(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _aggregate_grpo_reward_logs(
    duels: List[Dict[str, Any]],
    trained_output_paths_by_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    reward_rows: List[Dict[str, Any]] = []
    for output_path in trained_output_paths_by_model.values():
        reward_rows.extend(_load_reward_log(os.path.join(output_path, "reward_logs", "online_rewards.jsonl")))

    by_duel_model: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in reward_rows:
        try:
            duel_id = int(row.get("duel_id"))
            model_name = str(row.get("model_name"))
        except Exception:
            continue
        by_duel_model[duel_id][model_name].append(row)

    judged_duels: List[Dict[str, Any]] = []
    for duel in duels:
        duel_id = int(duel["duel_id"])
        model_scores: Dict[str, float] = {}
        model_rewards: Dict[str, float] = {}
        response_summaries: Dict[str, List[Dict[str, Any]]] = {}
        for model_name in duel["models"]:
            rows = by_duel_model.get(duel_id, {}).get(model_name, [])
            if not rows:
                continue
            raw_scores = [float(row.get("raw_score_1_to_10", 5.0)) for row in rows]
            rewards = [float(row.get("reward", 0.0)) for row in rows]
            model_scores[model_name] = float(np.mean(raw_scores))
            model_rewards[model_name] = float(np.mean(rewards))
            sorted_rows = sorted(rows, key=lambda row: float(row.get("reward", 0.0)), reverse=True)
            candidates = sorted_rows[:2] + sorted_rows[-2:] if len(sorted_rows) > 2 else sorted_rows
            summaries: List[Dict[str, Any]] = []
            seen = set()
            for row in candidates:
                key = (str(row.get("completion", "")), float(row.get("reward", 0.0)))
                if key in seen:
                    continue
                seen.add(key)
                summaries.append(
                    {
                        "reward": float(row.get("reward", 0.0)),
                        "raw_score_1_to_10": float(row.get("raw_score_1_to_10", 5.0)),
                        "completion_excerpt": str(row.get("completion", ""))[:800],
                        "judge_scores": row.get("judge_scores", {}),
                    }
                )
            response_summaries[model_name] = summaries

        scored = dict(duel)
        scored["model_level_scores"] = model_scores
        scored["model_level_rewards"] = model_rewards
        scored["response_summaries"] = response_summaries
        model_a, model_b = duel["models"]
        if model_a in model_scores and model_b in model_scores:
            score_a = model_scores[model_a]
            score_b = model_scores[model_b]
            scored["scores"] = [score_a, score_b]
            scored["score_diff"] = float(score_a - score_b)
            scored["winner"] = model_a if score_a > score_b else model_b if score_b > score_a else None
            scored["winner_index"] = 0 if score_a > score_b else 1 if score_b > score_a else None
        else:
            scored["scores"] = []
            scored["score_diff"] = None
            scored["winner"] = None
            scored["winner_index"] = None
        judged_duels.append(scored)
    return judged_duels


def _update_reputations_from_grpo_duels(
    model_ratings: Dict[str, Dict[str, float]],
    judged_duels: List[Dict[str, Any]],
    delta_history: Dict[str, List[float]],
    update_count: int,
    hyperparameters: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]], int, List[Dict[str, Any]]]:
    if bool(hyperparameters.get("freeze_ratings", False)):
        return model_ratings, delta_history, update_count, []

    initial_k = float(hyperparameters.get("initial_k", 10.0))
    min_k = float(hyperparameters.get("min_k", 5.0))
    window_size = int(hyperparameters.get("window_size", 10))
    min_deviation = float(hyperparameters.get("min_deviation", 0.1))
    epsilon = float(hyperparameters.get("epsilon", 0.01))
    decay_rate = float(hyperparameters.get("decay_rate", 0.9))
    decay_steps = float(hyperparameters.get("decay_steps", 10))
    scaling_factor = float(hyperparameters.get("scaling_factor", 20.0))
    debug = bool(hyperparameters.get("debug", False))
    rating_history: List[Dict[str, Any]] = []

    for duel_index, duel in enumerate(judged_duels):
        scores = duel.get("scores")
        if not isinstance(scores, list) or len(scores) < 2:
            continue
        model_a, model_b = duel["models"]
        score_diff = float(scores[0] - scores[1])
        update_count += 1
        k_value = max(min_k, initial_k * (decay_rate ** (update_count / max(decay_steps, 1e-12))))
        model_deltas: Dict[str, List[float]] = {model: [] for model in model_ratings}
        for index, model_i in enumerate((model_a, model_b)):
            model_j = model_b if index == 0 else model_a
            rating_i = float(model_ratings[model_i]["score"])
            rating_j = float(model_ratings[model_j]["score"])
            sigma_i = float(model_ratings[model_i]["deviation"])
            sigma_j = float(model_ratings[model_j]["deviation"])
            combined = math.sqrt(sigma_i**2 + sigma_j**2) or 1e-6
            phi_forward = 0.5 * (1.0 + math.erf((rating_i - rating_j) / (math.sqrt(2.0) * combined)))
            phi_backward = 0.5 * (1.0 + math.erf((rating_j - rating_i) / (math.sqrt(2.0) * combined)))
            signed_diff = score_diff if index == 0 else -score_diff
            delta = (
                k_value
                * signed_diff
                * math.tanh(sigma_i)
                * max(abs(phi_forward - phi_backward), epsilon)
                / max(scaling_factor, 1e-12)
            )
            new_rating = max(10.0, rating_i + delta)
            actual_delta = new_rating - rating_i
            model_ratings[model_i]["score"] = new_rating
            model_deltas[model_i].append(actual_delta)

        for model, deltas in model_deltas.items():
            if not deltas:
                continue
            delta_history.setdefault(model, []).extend(deltas)
            delta_history[model] = delta_history[model][-window_size:]
            if len(delta_history[model]) >= 2:
                model_ratings[model]["deviation"] = max(
                    float(np.std(delta_history[model])),
                    min_deviation,
                )
        rating_history.append(
            {
                "duel_index": duel_index,
                "duel_id": duel.get("duel_id"),
                "duel": duel,
                "ratings": {
                    model: {
                        "score": float(info["score"]),
                        "deviation": float(info["deviation"]),
                    }
                    for model, info in model_ratings.items()
                },
            }
        )
        if debug:
            print(f"[Stackelberg] GRPO rating update {duel_index}: score_diff={score_diff:.3f}")
    return model_ratings, delta_history, update_count, rating_history


def _run_dpo_iteration(
    task: str,
    task_type: str,
    gpu_ids: List[int],
    model_names: List[str],
    hyperparameters: Dict[str, Any],
    base_dir: str,
    iteration: int,
    total_iterations: int,
    instructions: List[str],
    current_model_paths: Dict[str, str],
    leader_state: Dict[str, Any],
    model_ratings: Dict[str, Dict[str, float]],
    delta_history: Dict[str, List[float]],
    update_count: int,
) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, List[float]], int, List[Tuple[int, str, str]]]:
    iter_dir = os.path.join(base_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    seed = int(hyperparameters.get("seed", 42)) + iteration
    random.seed(seed)
    np.random.seed(seed)

    num_duels = int(hyperparameters.get("num_duels_per_iteration", len(instructions)))
    duels = _sample_duels_unified(
        instructions=instructions,
        model_names=model_names,
        model_ratings=model_ratings,
        leader_state=leader_state,
        num_duels=num_duels,
        leader_uniform_mix=float(
            hyperparameters.get("leader_uniform_mix", hyperparameters.get("instr_sample_gamma", 0.2))
        ),
        random_match_prob=float(hyperparameters.get("random_match_prob", 0.2)),
        num_opponents=int(hyperparameters.get("num_opponents", 3)),
        opponent_selection=hyperparameters.get("opponent_selection", "schedule_decreasing"),
        iteration=iteration,
        total_iterations=total_iterations,
        reputation_gap_sigma=float(hyperparameters.get("reputation_gap_sigma", 0.15)),
    )
    _save_prompt_sampling_manifest(base_dir, iteration, duels)

    raw_pairs = _generate_offline_duel_responses(
        duels=duels,
        current_model_paths=current_model_paths,
        gpu_ids=gpu_ids,
        max_response_length=int(hyperparameters.get("max_response_length", 256)),
        temperature=float(hyperparameters.get("temperature", 0.7)),
        top_p=float(hyperparameters.get("top_p", 0.9)),
        batch_size=int(hyperparameters.get("batch_size", 1)),
        max_parallel_models=int(hyperparameters.get("max_parallel_generation_models", len(gpu_ids))),
    )
    print(f"[Stackelberg] Iter {iteration}: generated {len(raw_pairs)} offline duel pairs.")
    if not raw_pairs:
        return current_model_paths, leader_state, model_ratings, delta_history, update_count, []

    judged_pairs = _judge_offline_duels(
        raw_pairs=raw_pairs,
        model_names=model_names,
        current_model_paths=current_model_paths,
        model_ratings=model_ratings,
        gpu_ids=gpu_ids,
        base_dir=base_dir,
        judge_batch_size=int(hyperparameters.get("judge_batch_size", 8)),
        judge_rounds=int(hyperparameters.get("judge_rounds", 1)),
        judge_max_response_length=int(hyperparameters.get("judge_max_response_length", 64)),
        judge_temperature=float(hyperparameters.get("judge_temperature", 1e-5)),
        judge_top_p=float(hyperparameters.get("judge_top_p", 1.0)),
    )
    save_judged_pairs_sparta(judged_pairs, base_dir, iteration)

    score_type = hyperparameters.get("score_type", "normal")
    rating_system = _build_rating_system(
        score_type=score_type,
        model_ratings=model_ratings,
        delta_history=delta_history,
        base_dir=base_dir,
        iteration=iteration,
        hyperparameters=hyperparameters,
    )
    rating_system.update_count = update_count
    rating_history: List[Dict[str, Any]] = []
    for pair_index, pair in enumerate(judged_pairs):
        rating_system.update_ratings_from_judges(pair)
        rating_history.append(
            {
                "pair_index": pair_index,
                "pair": pair,
                "ratings": {
                    model: {
                        "score": float(info["score"]),
                        "deviation": float(info["deviation"]),
                    }
                    for model, info in rating_system.get_all_ratings().items()
                },
            }
        )
    model_ratings = rating_system.get_all_ratings()
    delta_history = rating_system.delta_history
    update_count = int(rating_system.update_count)
    _save_ratings_and_history(base_dir, iteration, model_ratings, delta_history, update_count)
    save_rating_history_sparta(rating_history, base_dir, iteration)

    leader_state = _update_leader_state(
        state=leader_state,
        judged_duels=judged_pairs,
        instructions=instructions,
        model_names=model_names,
        base_dir=base_dir,
        iteration=iteration,
        total_iterations=total_iterations,
        task=task,
        task_type=task_type,
        training_algorithm="dpo",
        hyperparameters=hyperparameters,
    )

    preference_pairs: List[Dict[str, Any]] = []
    for pair in judged_pairs:
        preference = rating_system.select_preference_response(pair)
        if preference is not None:
            preference_pairs.append(preference)
    preference_pairs = filter_tie_sparta(preference_pairs)
    dataset_dir = os.path.join(iter_dir, "dataset")
    preference_path = save_preference_pairs_to_json_sparta(preference_pairs, dataset_dir)

    new_entries: List[Tuple[int, str, str]] = []
    if preference_pairs and preference_path:
        output_paths = [os.path.join(iter_dir, f"dpo_{_safe_name(model)}") for model in model_names]
        _call_with_supported_kwargs(
            distributed_dpo.distributed_dpo,
            {
                "list_of_model_names": [current_model_paths[model] for model in model_names],
                "list_of_dpo_data_paths": [preference_path for _ in model_names],
                "list_of_gpu_ids": gpu_ids[: len(model_names)] or [gpu_ids[0]],
                "list_of_output_model_paths": output_paths,
                "parallel_training": bool(hyperparameters.get("parallel_dpo_training", True)),
                "batch_size": int(hyperparameters.get("dpo_batch_size", 1)),
                "gradient_accumulation_steps": int(hyperparameters.get("dpo_gradient_accumulation_steps", 16)),
                "learning_rate": float(hyperparameters.get("dpo_learning_rate", 1e-6)),
                "epoch": float(hyperparameters.get("dpo_epoch", 1.0)),
            },
        )
        for model, output_path in zip(model_names, output_paths):
            current_model_paths[model] = output_path
            new_entries.append((iteration, model, output_path))
    return current_model_paths, leader_state, model_ratings, delta_history, update_count, new_entries


def _run_grpo_iteration(
    task: str,
    task_type: str,
    gpu_ids: List[int],
    model_names: List[str],
    hyperparameters: Dict[str, Any],
    base_dir: str,
    iteration: int,
    total_iterations: int,
    instructions: List[str],
    current_model_paths: Dict[str, str],
    leader_state: Dict[str, Any],
    model_ratings: Dict[str, Dict[str, float]],
    delta_history: Dict[str, List[float]],
    update_count: int,
) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, List[float]], int, List[Tuple[int, str, str]]]:
    if hyperparameters.get("score_type", "normal") != "normal":
        raise ValueError("GRPO currently supports score_type='normal' only, matching the original GRPO method.")
    iter_dir = os.path.join(base_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    seed = int(hyperparameters.get("seed", 42)) + iteration
    random.seed(seed)
    np.random.seed(seed)

    duels = _sample_duels_unified(
        instructions=instructions,
        model_names=model_names,
        model_ratings=model_ratings,
        leader_state=leader_state,
        num_duels=int(hyperparameters.get("num_duels_per_iteration", len(instructions))),
        leader_uniform_mix=float(
            hyperparameters.get("leader_uniform_mix", hyperparameters.get("instr_sample_gamma", 0.2))
        ),
        random_match_prob=float(hyperparameters.get("random_match_prob", 0.2)),
        num_opponents=int(hyperparameters.get("num_opponents", 3)),
        opponent_selection=hyperparameters.get("opponent_selection", "schedule_decreasing"),
        iteration=iteration,
        total_iterations=total_iterations,
        reputation_gap_sigma=float(hyperparameters.get("reputation_gap_sigma", 0.15)),
    )
    _save_prompt_sampling_manifest(base_dir, iteration, duels)
    dataset_dir = os.path.join(iter_dir, "dataset")
    train_paths_by_model = _write_grpo_train_files_by_model(duels, model_names, dataset_dir)
    if not train_paths_by_model:
        return current_model_paths, leader_state, model_ratings, delta_history, update_count, []

    iteration_model_paths = current_model_paths.copy()
    train_models = [model for model in model_names if model in train_paths_by_model]
    output_paths = [os.path.join(iter_dir, f"grpo_{_safe_name(model)}") for model in train_models]
    trained_paths = _call_with_supported_kwargs(
        distributed_grpo.distributed_grpo_with_judges,
        {
            "list_of_model_names": [iteration_model_paths[model] for model in train_models],
            "list_of_train_data_paths": [train_paths_by_model[model] for model in train_models],
            "list_of_gpu_ids": gpu_ids[: len(train_models)] or [gpu_ids[0]],
            "list_of_output_model_paths": output_paths,
            "list_of_original_model_names": train_models,
            "judge_model_paths_by_name": iteration_model_paths,
            "judge_weights_by_name": {model: float(model_ratings[model]["score"]) for model in model_names},
            "judge_gpu_ids": hyperparameters.get("online_judge_gpu_ids", None),
            "parallel_training": bool(hyperparameters.get("parallel_grpo_training", False)),
            "gpus_per_grpo_job": int(hyperparameters.get("gpus_per_grpo_job", 1)),
            "max_parallel_grpo_jobs": int(hyperparameters.get("max_parallel_grpo_jobs", len(gpu_ids))),
            "batch_size": int(hyperparameters.get("grpo_batch_size", 1)),
            "gradient_accumulation_steps": int(hyperparameters.get("grpo_gradient_accumulation_steps", 4)),
            "learning_rate": float(hyperparameters.get("grpo_learning_rate", 1e-6)),
            "epoch": float(hyperparameters.get("grpo_epoch", 1.0)),
            "num_generations": int(hyperparameters.get("grpo_num_generations", 4)),
            "max_completion_length": int(hyperparameters.get("grpo_max_completion_length", 256)),
            "beta": float(hyperparameters.get("grpo_beta", 0.0)),
            "epsilon": float(hyperparameters.get("grpo_epsilon", 0.2)),
            "scale_rewards": hyperparameters.get("grpo_scale_rewards", "group"),
            "loss_type": hyperparameters.get("grpo_loss_type", "grpo"),
            "reward_scale": hyperparameters.get("reward_scale", "zero_one"),
            "judge_batch_size": int(hyperparameters.get("judge_batch_size", 4)),
            "judge_max_response_length": int(hyperparameters.get("judge_max_response_length", 64)),
            "judge_temperature": float(hyperparameters.get("judge_temperature", 1e-5)),
            "judge_top_p": float(hyperparameters.get("judge_top_p", 1.0)),
            "seed": seed,
        },
    )
    trained_by_model = {model: path for model, path in zip(train_models, trained_paths)}
    judged_duels = _aggregate_grpo_reward_logs(duels, trained_by_model)
    _save_json(os.path.join(iter_dir, "judged_results", "judged_duels.json"), judged_duels)

    model_ratings, delta_history, update_count, rating_history = _update_reputations_from_grpo_duels(
        model_ratings=model_ratings,
        judged_duels=judged_duels,
        delta_history=delta_history,
        update_count=update_count,
        hyperparameters=hyperparameters,
    )
    _save_ratings_and_history(base_dir, iteration, model_ratings, delta_history, update_count)
    save_rating_history_sparta(rating_history, base_dir, iteration)

    leader_state = _update_leader_state(
        state=leader_state,
        judged_duels=judged_duels,
        instructions=instructions,
        model_names=model_names,
        base_dir=base_dir,
        iteration=iteration,
        total_iterations=total_iterations,
        task=task,
        task_type=task_type,
        training_algorithm="grpo",
        hyperparameters=hyperparameters,
    )

    new_entries: List[Tuple[int, str, str]] = []
    for model, path in trained_by_model.items():
        current_model_paths[model] = path
        new_entries.append((iteration, model, path))
    return current_model_paths, leader_state, model_ratings, delta_history, update_count, new_entries


def _call_with_supported_kwargs(function: Any, kwargs: Dict[str, Any]) -> Any:
    """Call a project utility while tolerating version-specific optional parameters."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        # Some wrapped/c-extension callables do not expose a usable signature.
        return function(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(**filtered)


def _safe_average(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("Evaluation returned no scores.")
    return float(sum(float(score) for score in scores) / len(scores))


def _evaluate_adapters_and_test(
    task: str,
    task_type: str,
    gpu_ids: List[int],
    model_names: List[str],
    hyperparameters: Dict[str, Any],
    base_dir: str,
    run_id: str,
    training_algorithm: str,
    leader_type: str,
    leader_scope: str,
    current_model_paths: Dict[str, str],
    all_adapter_entries: List[Tuple[int, str, str]],
    total_iterations: int,
) -> float:
    print("[Sparta] All iterations complete. Beginning evaluation...")
    prefix = f"{training_algorithm}_"
    discovered: List[Tuple[int, str, str]] = []
    for iteration in range(total_iterations):
        for model in model_names:
            path = os.path.join(base_dir, f"iteration_{iteration}", f"{prefix}{_safe_name(model)}")
            if os.path.isdir(path):
                discovered.append((iteration, model, path))
    seen = set()
    entries: List[Tuple[int, str, str]] = []
    for entry in all_adapter_entries + discovered:
        if entry[2] in seen:
            continue
        seen.add(entry[2])
        entries.append(entry)

    eval_max_response_length = int(
        hyperparameters.get("eval_max_response_length", hyperparameters.get("max_response_length", 256))
    )
    eval_temperature = float(hyperparameters.get("eval_temperature", hyperparameters.get("temperature", 0.7)))
    eval_top_p = float(hyperparameters.get("eval_top_p", hyperparameters.get("top_p", 0.9)))
    eval_batch_size = int(hyperparameters.get("eval_batch_size", hyperparameters.get("batch_size", 1)))
    distributed_generation.update_generation_hyperparameters(
        max_response_length=eval_max_response_length,
        temperature=eval_temperature,
        top_p=eval_top_p,
        batch_size=eval_batch_size,
        big_model_mode=False,
    )

    dev_inputs = eval.prepare_inputs(task, task_type, "dev")
    dev_records: List[Dict[str, Any]] = []
    if entries:
        print("[Sparta] Adapters found. Evaluating...")
        paths = [entry[2] for entry in entries]
        outputs_by_path = distributed_generation.distributed_generation(
            paths,
            [dev_inputs for _ in paths],
            gpu_ids,
            max_response_length=eval_max_response_length,
            temperature=eval_temperature,
            top_p=eval_top_p,
            batch_size=eval_batch_size,
            max_parallel_models=int(hyperparameters.get("max_parallel_generation_models", len(gpu_ids))),
        )
        for (iteration, model, path), outputs in zip(entries, outputs_by_path):
            score = _safe_average(eval.get_scores(task, task_type, "dev", outputs))
            dev_records.append(
                {
                    "adapter_key": f"{model}_iter{iteration}",
                    "model": model,
                    "iteration": int(iteration),
                    "adapter_path": path,
                    "dev_score": score,
                }
            )
        save_adapter_dev_scores(base_dir, dev_records, task)
        best_record = max(dev_records, key=lambda record: record["dev_score"])
        best_model_name = str(best_record["model"])
        best_model_iteration = int(best_record["iteration"])
        best_model_path = str(best_record["adapter_path"])
        dev_scores = {record["adapter_key"]: record["dev_score"] for record in dev_records}
    else:
        print("[Sparta] No adapters found. Evaluating...")
        paths = [current_model_paths[model] for model in model_names]
        outputs_by_path = distributed_generation.distributed_generation(
            paths,
            [dev_inputs for _ in paths],
            gpu_ids,
            max_response_length=eval_max_response_length,
            temperature=eval_temperature,
            top_p=eval_top_p,
            batch_size=eval_batch_size,
            max_parallel_models=int(hyperparameters.get("max_parallel_generation_models", len(gpu_ids))),
        )
        scores = [
            _safe_average(eval.get_scores(task, task_type, "dev", outputs))
            for outputs in outputs_by_path
        ]
        best_index = int(np.argmax(scores))
        best_model_name = model_names[best_index]
        best_model_iteration = total_iterations - 1
        best_model_path = paths[best_index]
        dev_scores = {model: score for model, score in zip(model_names, scores)}

    test_inputs = eval.prepare_inputs(task, task_type, "test")
    test_outputs = distributed_generation.distributed_generation(
        [best_model_path],
        [test_inputs],
        gpu_ids,
        max_response_length=eval_max_response_length,
        temperature=eval_temperature,
        top_p=eval_top_p,
        batch_size=eval_batch_size,
        max_parallel_models=1,
    )[0]
    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    average_test_score = _safe_average(test_scores)
    logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_sparta_stackelberg",
        "training_algorithm": training_algorithm,
        "leader_type": leader_type,
        "leader_scope": leader_scope,
        "run_id": run_id,
        "model_names": model_names,
        "best_model": best_model_name,
        "best_model_iteration": best_model_iteration,
        "best_model_path": best_model_path,
        "hyperparameters": hyperparameters,
        "avg_test_score": average_test_score,
        "dev_scores": dev_scores,
        "logs": [
            {"input": prompt, "output": output, "score": score}
            for prompt, output, score in zip(test_inputs, test_outputs, test_scores)
        ],
    }
    log_path = os.path.join(
        base_dir,
        f"{task}_{len(model_names)}_{round(average_test_score, 4)}_"
        f"{training_algorithm}_{leader_type}_stackelberg.json",
    )
    _save_json(log_path, logs)
    print(
        f"[Stackelberg] Best model={best_model_name}, iteration={best_model_iteration}, "
        f"test_score={average_test_score:.6f}"
    )
    return average_test_score


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    training_algorithm, leader_type, leader_scope = _resolve_modes(hyperparameters)
    if not gpu_ids:
        gpu_ids = [0]
    if len(model_names) < 3:
        raise ValueError("At least three models are required for peer-judged Stackelberg training.")

    num_iterations = int(hyperparameters.get("num_iterations", 1))
    start_iteration = int(hyperparameters.get("current_iteration", 0))
    root_base_dir = str(
        hyperparameters.get("base_dir", os.path.join("model_collaboration", "logs", "text_sparta_stackelberg"))
    )
    requested_run_id = hyperparameters.get("run_id", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if start_iteration > 0:
        if not requested_run_id:
            raise ValueError("Resuming requires hyperparameters['run_id'].")
        run_id = requested_run_id
        base_dir = os.path.join(root_base_dir, run_id)
        if not os.path.isdir(base_dir):
            raise ValueError(f"Cannot resume: run directory does not exist at {base_dir}")
    else:
        run_id = timestamp if not requested_run_id else f"{requested_run_id}_{timestamp}"
        base_dir = os.path.join(root_base_dir, run_id)
    os.makedirs(base_dir, exist_ok=True)

    total_iterations = start_iteration + num_iterations
    num_instructions = int(hyperparameters.get("num_instructions", 500 if training_algorithm == "dpo" else 128))
    all_instructions = eval.prepare_inputs(task, task_type, "dev")
    instructions = all_instructions[: min(num_instructions, len(all_instructions))]
    if not instructions:
        raise ValueError(f"No dev instructions found for task={task!r}, task_type={task_type!r}.")

    resolved = {
        "task": task,
        "task_type": task_type,
        "training_algorithm": training_algorithm,
        "leader_type": leader_type,
        "leader_scope": leader_scope,
        "run_id": run_id,
        "start_iteration": start_iteration,
        "num_iterations": num_iterations,
        "num_instructions": len(instructions),
        "model_names": model_names,
        "gpu_ids": gpu_ids,
        "hyperparameters": hyperparameters,
    }
    config_path = os.path.join(base_dir, "resolved_run_config.json")
    if start_iteration > 0 and os.path.exists(config_path):
        previous = _read_json(config_path)
        for key in ("training_algorithm", "leader_type", "leader_scope"):
            if previous.get(key) != resolved.get(key):
                raise ValueError(
                    f"Cannot change {key} while resuming: saved={previous.get(key)!r}, requested={resolved.get(key)!r}."
                )
    else:
        _save_json(config_path, resolved)

    print(
        f"[Stackelberg] Run directory: {base_dir}\n"
        f"[Stackelberg] training_algorithm={training_algorithm}, "
        f"leader_type={leader_type}, leader_scope={leader_scope}"
    )

    current_model_paths, all_adapter_entries = _restore_latest_adapters(
        base_dir=base_dir,
        model_names=model_names,
        start_iteration=start_iteration,
        training_algorithm=training_algorithm,
    )
    if start_iteration > 0:
        leader_state = _load_leader_state(
            base_dir=base_dir,
            checkpoint_iteration=start_iteration - 1,
            instructions=instructions,
            model_names=model_names,
            expected_type=leader_type,
            expected_scope=leader_scope,
        )
    else:
        leader_state = _initialize_leader_state(
            base_dir=base_dir,
            instructions=instructions,
            model_names=model_names,
            task=task,
            task_type=task_type,
            leader_type=leader_type,
            leader_scope=leader_scope,
            hyperparameters=hyperparameters,
        )
        _save_leader_state(
            base_dir=base_dir,
            state=leader_state,
            instructions=instructions,
            model_names=model_names,
            uniform_mix=float(
                hyperparameters.get("leader_uniform_mix", hyperparameters.get("instr_sample_gamma", 0.2))
            ),
            iteration=-1,
            initial=True,
        )

    for local_iteration in range(num_iterations):
        iteration = start_iteration + local_iteration
        print(f"[Stackelberg] Iteration {iteration} starting.")
        model_ratings, delta_history, update_count = _load_ratings_and_history(
            base_dir=base_dir,
            iteration=iteration,
            model_names=model_names,
        )
        if training_algorithm == "dpo":
            result = _run_dpo_iteration(
                task=task,
                task_type=task_type,
                gpu_ids=gpu_ids,
                model_names=model_names,
                hyperparameters=hyperparameters,
                base_dir=base_dir,
                iteration=iteration,
                total_iterations=total_iterations,
                instructions=instructions,
                current_model_paths=current_model_paths,
                leader_state=leader_state,
                model_ratings=model_ratings,
                delta_history=delta_history,
                update_count=update_count,
            )
        else:
            result = _run_grpo_iteration(
                task=task,
                task_type=task_type,
                gpu_ids=gpu_ids,
                model_names=model_names,
                hyperparameters=hyperparameters,
                base_dir=base_dir,
                iteration=iteration,
                total_iterations=total_iterations,
                instructions=instructions,
                current_model_paths=current_model_paths,
                leader_state=leader_state,
                model_ratings=model_ratings,
                delta_history=delta_history,
                update_count=update_count,
            )
        (
            current_model_paths,
            leader_state,
            model_ratings,
            delta_history,
            update_count,
            new_entries,
        ) = result
        all_adapter_entries.extend(new_entries)
        _save_leader_state(
            base_dir=base_dir,
            state=leader_state,
            instructions=instructions,
            model_names=model_names,
            uniform_mix=float(
                hyperparameters.get("leader_uniform_mix", hyperparameters.get("instr_sample_gamma", 0.2))
            ),
            iteration=iteration,
        )

    _evaluate_adapters_and_test(
        task=task,
        task_type=task_type,
        gpu_ids=gpu_ids,
        model_names=model_names,
        hyperparameters=hyperparameters,
        base_dir=base_dir,
        run_id=run_id,
        training_algorithm=training_algorithm,
        leader_type=leader_type,
        leader_scope=leader_scope,
        current_model_paths=current_model_paths,
        all_adapter_entries=all_adapter_entries,
        total_iterations=total_iterations,
    )
    return 0

