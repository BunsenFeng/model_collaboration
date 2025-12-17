import os
import json
import random
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from data import eval
from method import distributed_generation
from utils import distributed_dpo
import logging

logger = logging.getLogger(__name__)


def _pairwise_competition(
    gpu_ids: List[int],
    model_names: List[str],
    model_name_mapping: Optional[Dict[str, str]],
    instructions: List[str],
    random_match_prob: float = 0.2,
    num_opponents: int = 3,
    model_reputation: Optional[Dict[str, float]] = None,
    max_response_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Build pairwise competitions between models on a list of instructions.

    For each instruction we select a first model and an opponent model.
    Opponents are chosen either at random (with probability random_match_prob)
    or by nearest reputation score (model_reputation), similar to Competition.get_opponent.
    Returns a list of raw_pairs ready for judging.
    """

    if len(model_names) < 2 or not instructions:
        return []

    # Opponent selection: if model_reputation is provided, follow Competition.get_opponent
    # and prefer opponents with similar scores; otherwise fall back to pure random sampling.
    def get_opponent(current_model: str) -> str:
        opponents = [m for m in model_names if m != current_model]
        if not opponents:
            return current_model

        # If reputation is not provided, fall back to pure random
        if not model_reputation:
            return random.choice(opponents)

        # Reputation score of current model
        current_score = model_reputation.get(current_model)
        if current_score is None:
            return random.choice(opponents)

        # With probability random_match_prob, pick a fully random opponent
        if random.random() < random_match_prob:
            return random.choice(opponents)

        # Otherwise sort by score difference and sample from the closest num_opponents
        potential_opponents: List[Tuple[str, float]] = []
        for other in opponents:
            other_score = model_reputation.get(other)
            if other_score is None:
                # If there is no reputation yet, treat as worst match (infinite diff)
                diff = float("inf")
            else:
                diff = abs(current_score - other_score)
            potential_opponents.append((other, diff))

        potential_opponents.sort(key=lambda x: x[1])
        top_k = potential_opponents[: max(1, min(num_opponents, len(potential_opponents)))]
        return random.choice([name for name, _ in top_k])

    model_tasks: Dict[str, List[str]] = {m: [] for m in model_names}
    instruction_pairs: List[Tuple[str, str, str]] = []

    # 1) Full loops: each loop has len(model_names) instructions, mirroring Competition.run
    num_loops = len(instructions) // len(model_names)
    for loop_idx in range(num_loops):
        start_idx = loop_idx * len(model_names)
        end_idx = start_idx + len(model_names)
        loop_instructions = instructions[start_idx:end_idx]

        # Randomly shuffle model order in each loop
        loop_models = random.sample(model_names, len(model_names))

        for model_idx, first_model in enumerate(loop_models):
            instr = loop_instructions[model_idx]
            opponent_model = get_opponent(first_model)

            instruction_pairs.append((instr, first_model, opponent_model))
            model_tasks[first_model].append(instr)
            model_tasks[opponent_model].append(instr)

    # 2) Remaining instructions (fewer than one full loop)
    remaining_start = num_loops * len(model_names)
    for instr in instructions[remaining_start:]:
        first_model = random.choice(model_names)
        opponent_model = get_opponent(first_model)

        instruction_pairs.append((instr, first_model, opponent_model))
        model_tasks[first_model].append(instr)
        model_tasks[opponent_model].append(instr)

    # 3) Use distributed_generation to generate all (model, instruction) responses
    logical_model_names: List[str] = []
    list_of_input_list: List[List[str]] = []
    for m in model_names:
        if model_tasks[m]:
            logical_model_names.append(m)
            list_of_input_list.append(model_tasks[m])

    if not logical_model_names:
        return []

    # Map logical model names to actual HF/local paths if a mapping is provided.
    # If no mapping is given, fall back to using the logical name directly.
    inference_model_names: List[str] = []
    if model_name_mapping:
        for m in logical_model_names:
            inference_model_names.append(model_name_mapping.get(m, m))
    else:
        inference_model_names = logical_model_names

    # Configure generation hyperparameters via the shared helper.
    # These values are passed in from the method's config (hyperparameters)
    # so that generation here is consistent with config.json.
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    list_of_output_list = distributed_generation.distributed_generation(
        inference_model_names,
        list_of_input_list,
        gpu_ids,
    )

    # Build an index: {model_name: {instruction: response}}
    model_responses: Dict[str, Dict[str, str]] = {}
    for m, ins_list, out_list in zip(
        logical_model_names, list_of_input_list, list_of_output_list
    ):
        model_responses[m] = {ins: resp for ins, resp in zip(ins_list, out_list)}

    # 4) Build raw_pairs, aligned with Competition.pair format
    raw_pairs: List[Dict[str, Any]] = []
    for instr, model_a, model_b in instruction_pairs:
        if (
            model_a in model_responses
            and model_b in model_responses
            and instr in model_responses[model_a]
            and instr in model_responses[model_b]
        ):
            new_pair = {
                "instruction": instr,
                "models": [model_a, model_b],
                "responses": [
                    model_responses[model_a][instr],
                    model_responses[model_b][instr],
                ],
                "judges": {},
            }
            raw_pairs.append(new_pair)

    return raw_pairs

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
        output_dir = os.path.join(base_dir, "intermediate_results", judge_name)
        os.makedirs(output_dir, exist_ok=True)

    # Split pairs into chunks to avoid OOM
    chunk_size = 50
    pair_chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    # Configure generation hyperparameters for the judge model
    # These can be controlled via config.json (hyperparameters),
    # and are passed in from run_method -> run_judges_sparta.
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_response_length,
        # temperature must be > 0 for transformers; use a very small value to approximate greedy
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

def _aggregate_scores(
    scored_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize pair['scores'] to a length-2 float list and derive score_diff and winner.

    - Ensure scores is length-2 (pad with 5.0 if necessary).
    - score_diff = scores[0] - scores[1].
    - winner: 0 if first response wins, 1 if second wins, None for tie.
    """
    if not scored_pairs:
        return scored_pairs

    aggregated_pairs = scored_pairs
    for pair in aggregated_pairs:
        scores = pair.get("scores")
        # 规范化为长度 2 的列表
        if not isinstance(scores, list):
            scores = []
        scores = [float(s) for s in scores[:2]]
        while len(scores) < 2:
            scores.append(5.0)

        score_diff = scores[0] - scores[1]
        pair["scores"] = scores
        pair["score_diff"] = float(score_diff)

        if score_diff > 0:
            pair["winner"] = 0
        elif score_diff < 0:
            pair["winner"] = 1
        else:
            pair["winner"] = None

    return aggregated_pairs

def run_judges_sparta(
    judge_models: List[str],
    pairs: List[Dict[str, Any]],
    gpu_ids: List[int],
    model_name_mapping: Optional[Dict[str, str]] = None,
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

        # judge_name is the logical name used in rating and logging;
        # judge_model_path is the actual HF/local identifier used for loading.
        judge_name = judge_model
        judge_model_path = (
            model_name_mapping.get(judge_model, judge_model)
            if model_name_mapping is not None
            else judge_model
        )

        print(f"[Sparta] Running judge {judge_name} (model: {judge_model_path}) on GPU {gpu_id}")
        _judge_batch_with_model(
            judge_name=judge_name,
            judge_model=judge_model_path,
            pairs=pairs,
            gpu_id=gpu_id,
            batch_size=batch_size,
            base_dir=base_dir,
            num_rounds=num_rounds,
            max_response_length=max_response_length,
            temperature=temperature,
            top_p=top_p,
        )

    # 计算每个 judge 的 ave_scores
    pairs = calculate_judge_averages_sparta(pairs)
    return pairs

"""
Rating logic is implemented via RatingSystem / RatingSystemDynamicWeighted /
RatingSystemStaticWeighted below. The older _update_reputation helper is no longer used.
"""

def save_judged_pairs_sparta(judged_pairs: List[Dict[str, Any]], base_dir: str, iteration: int) -> None:
    """
    Save judged_pairs under logs/text_sparta/iteration_k/judged_results/judged_pairs.json,
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
    Save rating_history to logs/text_sparta/ as a JSON snapshot, matching save_rating_history.
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
            # 若 judge 也是参赛模型，则可按原脚本跳过
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
                    with open(weights_path, "r") as f:
                        return json.load(f)
                return weights

            if self.current_iteration >= 2:
                prev_iter = self.current_iteration - 1
                prev_path = os.path.join(
                    self.base_dir, f"iteration_{prev_iter}", "model_info.json"
                )
                if not os.path.exists(prev_path):
                    return weights
                with open(prev_path, "r") as f:
                    prev_info = json.load(f)
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
                    with open(weights_path, "r") as f:
                        return json.load(f)
                return weights

            weighted_models: List[str] = []
            for iter_num in range(2, self.current_iteration + 1):
                prev_iter = iter_num - 1
                prev_path = os.path.join(
                    self.base_dir, f"iteration_{prev_iter}", "model_info.json"
                )
                if not os.path.exists(prev_path):
                    continue
                with open(prev_path, "r") as f:
                    prev_info = json.load(f)
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

        print(f"\nUpdate count: {self.update_count}")
        print(f"Current K value: {self.K:.2f}")
        print("\nDeviation changes:")
        for model in self.model_ratings:
            print(
                f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}"
            )

    def get_weights(self) -> Dict[str, float]:
        return self.weights

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Implement your approach here.
    Args:
        task: str, the name of the task
        task_type: str, the type of the task (e.g., "multiple_choice", "exact_match", etc.)
        gpu_ids: list of int, the GPU ids to use for distributed generation
        model_names: list of str, the names of the models to use
        hyperparameters: dict, method-specific hyperparameters
        You get these arguments from the config file that users pass in.
    """

    # 1. optionally, extract the general hyperparameters from the hyperparameters dict
    # these four are included in any config file by default
    # this is useful if you are handling generation/finetuning without the (amazing) helper functions provided
    # these generation args will be auto-configured for the helper functions

    # max_response_length = hyperparameters.get("max_response_length")
    # temperature = hyperparameters.get("temperature")
    # top_p = hyperparameters.get("top_p")
    # batch_size = hyperparameters.get("batch_size")

    # 2. optionally, extract the method-specific hyperparameters from the hyperparameters dict
    # users would pass this in via the config file, through the "hyperparameters" dict
    # you need to tell users to set them in the readme
    # you should also provide a default value (in most cases)

    # method_specific_hyperparameter_a = hyperparameters.get("method_specific_hyperparameter_a", default_value_a)
    # method_specific_hyperparameter_b = hyperparameters.get("method_specific_hyperparameter_b", default_value_b)
    # no default value, will throw an error if the user doesn't provide it in the config file
    # method_specific_hyperparameter_c = hyperparameters.get("method_specific_hyperparameter_c")

    # 3. optionally, do something based on the dev set of the dataset
    # could be: selecting a model as the summarizer/evaluator/... based on dev performance
    # could be: finetuning the models somehow on the dev set
    # could be: setting some sort of hyperparameter/threshold based on the dev set
    # it's ok that your approach doesn't have this step, e.g. multiagent debate
    # if you ever saves anything during this step, make sure to save it in `logs/<your_method_name>/`!

    # a most simple example, select the best model based on dev set performance
    dev_input_list = eval.prepare_inputs(task, task_type, "dev") # grab the inputs for the dev set

    # evaluate every model on it through distributed generation
    list_of_input_list = [dev_input_list for _ in model_names] # replicate the dev inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(dev_input_list)

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs) # send the outputs to the eval module to get a list of per-input scores
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(avg_dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task, avg_dev_score))

    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    print("Best model selected for final generation: {}".format(best_model_name))
    # you can then use best_model_name somehow in the next step

    # 4. evaluate the approach on the test set
    # based on the stuff you did in the dev set, you arrived at some final approach
    # generate responses with it, evaluate it, do logging

    test_input_list = eval.prepare_inputs(task, task_type, "test") # grab the inputs for the test set

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Sparta competition + multi-judge + reputation + DPO preference generation.
    """

    # ------------------------- 0. The number of iterations（Sparta's own iterations） -------------------------
    sparta_iterations = int(hyperparameters.get("sparta_iterations", 1))
    base_start_iter = int(hyperparameters.get("iteration", 0))
    base_dir = hyperparameters.get("base_dir", os.path.join("logs", "text_sparta"))

    # Track current model paths (for adapter handling across iterations)
    # Initially, use model_name_mapping if provided, otherwise use model_names directly
    initial_model_name_mapping: Optional[Dict[str, str]] = hyperparameters.get("model_name_mapping")
    current_model_paths: Dict[str, str] = {}
    for m in model_names:
        if initial_model_name_mapping and m in initial_model_name_mapping:
            current_model_paths[m] = initial_model_name_mapping[m]
        else:
            current_model_paths[m] = m

    for it in range(sparta_iterations):
        iteration = base_start_iter + it  # the global iteration number of the current iteration

        # ------------------------- 1. Read hyperparameters -------------------------
        # General generation hyperparameters (kept in sync with config.json)
        max_response_length = int(hyperparameters.get("max_response_length", 256))
        temperature = float(hyperparameters.get("temperature", 0.7))
        top_p = float(hyperparameters.get("top_p", 0.9))
        batch_size = int(hyperparameters.get("batch_size", 1))

        # Judge generation hyperparameters (also configurable via config.json)
        judge_max_response_length = int(
            hyperparameters.get("judge_max_response_length", max_response_length)
        )
        judge_temperature = float(
            hyperparameters.get("judge_temperature", 1e-5)
        )
        judge_top_p = float(
            hyperparameters.get("judge_top_p", 1.0)
        )

        num_instructions = int(hyperparameters.get("num_instructions", 500))
        random_match_prob = float(hyperparameters.get("random_match_prob", 0.2))
        num_opponents = int(hyperparameters.get("num_opponents", 3))

        initial_K = float(hyperparameters.get("initial_k", 10.0))
        min_K = float(hyperparameters.get("min_k", 5.0))
        window_size = int(hyperparameters.get("window_size", 10))
        min_deviation = float(hyperparameters.get("min_deviation", 0.1))
        epsilon = float(hyperparameters.get("epsilon", 0.01))
        decay_rate = float(hyperparameters.get("decay_rate", 0.9))
        decay_steps = int(hyperparameters.get("decay_steps", 10))
        scaling_factor = float(hyperparameters.get("scaling_factor", 20.0))

        judge_models: List[str] = hyperparameters.get("judge_models", model_names)
        # Use current_model_paths as the model_name_mapping for this iteration
        # This allows us to use DPO-trained adapters from previous iterations
        model_name_mapping: Optional[Dict[str, str]] = current_model_paths.copy() if current_model_paths else None
        judge_batch_size = int(hyperparameters.get("judge_batch_size", 8))
        judge_rounds = int(hyperparameters.get("judge_rounds", 1))

        # ------------------------- 2. Initialize / Read the previous iteration's model_ratings + delta_history -------------------------
        iter_dir_prev = os.path.join(base_dir, f"iteration_{iteration-1}")
        model_info_path_prev = os.path.join(iter_dir_prev, "model_info.json")
        if iteration > 0 and os.path.exists(model_info_path_prev):
            with open(model_info_path_prev, "r", encoding="utf-8") as f:
                prev_info = json.load(f)
            model_ratings: Dict[str, Dict[str, float]] = {
                m: {
                    "score": float(prev_info[m].get("score", 100.0)),
                    "deviation": float(prev_info[m].get("deviation", 0.5)),
                }
                for m in prev_info
            }
        else:
            model_ratings = {m: {"score": 100.0, "deviation": 0.5} for m in model_names}
        for m in model_names:
            model_ratings.setdefault(m, {"score": 100.0, "deviation": 0.5})

        delta_history_path = os.path.join(base_dir, "rating_deltas.json")
        delta_history: Dict[str, List[float]] = {m: [] for m in model_ratings}
        update_count = 0
        if os.path.exists(delta_history_path):
            try:
                with open(delta_history_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                raw_hist = payload.get("delta_history", {})
                for m in model_ratings:
                    delta_history[m] = raw_hist.get(m, [])
                update_count = int(payload.get("update_count", 0))
            except Exception:
                pass

        # ------------------------- 3. Prepare dev instructions -------------------------
        all_instructions = eval.prepare_inputs(task, task_type, "dev")
        if num_instructions < len(all_instructions):
            instructions = all_instructions[:num_instructions]
        else:
            instructions = all_instructions
        print(f"[Sparta] Iter {iteration}: Using {len(instructions)} dev instructions.")

        # ------------------------- 4. pairwise competition -------------------------
        model_reputation = {m: model_ratings[m]["score"] for m in model_ratings}
        raw_pairs = _pairwise_competition(
            gpu_ids=gpu_ids,
            model_names=model_names,
            model_name_mapping=model_name_mapping,
            instructions=instructions,
            random_match_prob=random_match_prob,
            num_opponents=num_opponents,
            model_reputation=model_reputation,
            max_response_length=max_response_length,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        print(f"[Sparta] Iter {iteration}: Generated {len(raw_pairs)} raw pairs.")
        if not raw_pairs:
            continue

        # ------------------------- 5. Multiple judges scoring + ave_scores -------------------------
        judged_pairs = run_judges_sparta(
            judge_models=judge_models,
            pairs=raw_pairs,
            gpu_ids=gpu_ids,
            model_name_mapping=model_name_mapping,
            batch_size=judge_batch_size,
            num_rounds=judge_rounds,
            base_dir=base_dir,
            max_response_length=judge_max_response_length,
            temperature=judge_temperature,
            top_p=judge_top_p,
        )

        for pair in judged_pairs:
            judges = pair.get("judges", {})
            if not judges:
                continue
            w0_sum = w1_sum = total_weight = 0.0
            for jname, jinfo in judges.items():
                ave = jinfo.get("ave_scores")
                if not ave or len(ave) < 2:
                    continue
                score_a, score_b = float(ave[0]), float(ave[1])
                judge_weight = float(model_ratings.get(jname, {}).get("score", 1.0))
                w0_sum += judge_weight * score_a
                w1_sum += judge_weight * score_b
                total_weight += judge_weight
            if total_weight <= 0:
                continue
            pair["scores"] = [w0_sum / total_weight, w1_sum / total_weight]

        aggregated_pairs = _aggregate_scores(judged_pairs)

        # ------------------------- 6. RatingSystem update (normal / dynamic / static) -------------------------
        score_type = hyperparameters.get("score_type", "normal")  # normal / dynamic / static

        if score_type == "dynamic":
            rating_system = RatingSystemDynamicWeighted(
                model_scores=model_ratings,
                initial_K=initial_K,
                min_K=min_K,
                delta_history=delta_history,
                base_dir=base_dir,
                current_iteration=iteration,
                window_size=window_size,
                min_deviation=min_deviation,
                epsilon=epsilon,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                scaling_factor=scaling_factor,
                freeze_ratings=bool(hyperparameters.get("freeze_ratings", False)),
            )
        elif score_type == "static":
            rating_system = RatingSystemStaticWeighted(
                model_scores=model_ratings,
                initial_K=initial_K,
                min_K=min_K,
                delta_history=delta_history,
                base_dir=base_dir,
                current_iteration=iteration,
                window_size=window_size,
                min_deviation=min_deviation,
                epsilon=epsilon,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                scaling_factor=scaling_factor,
                freeze_ratings=bool(hyperparameters.get("freeze_ratings", False)),
            )
        else:
            rating_system = RatingSystem(
                model_scores=model_ratings,
                initial_K=initial_K,
                min_K=min_K,
                delta_history=delta_history,
                window_size=window_size,
                min_deviation=min_deviation,
                epsilon=epsilon,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                scaling_factor=scaling_factor,
                freeze_ratings=bool(hyperparameters.get("freeze_ratings", False)),
            )

        rating_history: List[Dict[str, Any]] = []
        for idx_pair, pair in enumerate(judged_pairs):
            rating_system.update_ratings_from_judges(pair)
            current_ratings = rating_system.get_all_ratings()
            rating_history.append(
                {
                    "pair_index": idx_pair,
                    "pair": pair,
                    "ratings": {
                        model: {
                            "score": info["score"],
                            "deviation": info["deviation"],
                        }
                        for model, info in current_ratings.items()
                    },
                }
            )

        model_ratings = rating_system.get_all_ratings()

        # Write the latest delta_history back (RatingSystem has already updated self.delta_history)
        delta_history = rating_system.delta_history if hasattr(
            rating_system, "delta_history"
        ) else delta_history

        # ------------------------- 7. Save model_info.json and rating_deltas.json + rating_history -------------------------
        iter_dir = os.path.join(base_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        model_info_path = os.path.join(iter_dir, "model_info.json")
        serializable_info = {
            m: {
                "score": float(model_ratings[m]["score"]),
                "deviation": float(model_ratings[m]["deviation"]),
            }
            for m in model_ratings
        }
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(serializable_info, f, ensure_ascii=False, indent=2)
        print(f"[Sparta] Iter {iteration}: Saved model_info to {model_info_path}")

        payload = {"delta_history": delta_history, "update_count": getattr(rating_system, "update_count", 0)}
        with open(delta_history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        save_rating_history_sparta(rating_history, base_dir, iteration)
        save_judged_pairs_sparta(judged_pairs, base_dir, iteration)

        # ------------------------- 8. Generate preference_pairs (using select_preference_response + filter_tie) -------------------------
        preference_pairs: List[Dict[str, Any]] = []
        for pair in judged_pairs:
            pref = rating_system.select_preference_response(pair)
            if pref is not None:
                preference_pairs.append(pref)

        old_len = len(preference_pairs)
        preference_pairs = filter_tie_sparta(preference_pairs)
        logger.info(
            f"[Sparta] Iter {iteration}: preference_pairs {old_len} -> {len(preference_pairs)} after tie filter"
        )

        dataset_dir = os.path.join(iter_dir, "dataset")
        pref_path = save_preference_pairs_to_json_sparta(preference_pairs, dataset_dir)

        # ------------------------- 9. Optional DPO -------------------------
        enable_dpo = bool(hyperparameters.get("enable_dpo", False))
        if enable_dpo and preference_pairs and pref_path:
            dpo_model_names = hyperparameters.get("dpo_model_names", model_names)
            dpo_data_paths = [pref_path for _ in dpo_model_names]
            dpo_gpu_ids = gpu_ids[: len(dpo_model_names)] or [0]
            dpo_output_model_paths = [
                os.path.join(iter_dir, f"dpo_{m}") for m in dpo_model_names
            ]

            # Use current_model_paths to resolve logical names to actual paths for DPO training.
            # This allows DPO to be applied on top of previously trained adapters.
            dpo_hf_model_names = [
                current_model_paths.get(m, m) for m in dpo_model_names
            ]

            dpo_batch_size = int(hyperparameters.get("dpo_batch_size", 1))
            dpo_grad_acc = int(
                hyperparameters.get("dpo_gradient_accumulation_steps", 16)
            )
            dpo_lr = float(hyperparameters.get("dpo_learning_rate", 1e-6))
            dpo_epoch = int(hyperparameters.get("dpo_epoch", 1))

            print(f"[Sparta] Iter {iteration}: Starting DPO for {dpo_model_names}")
            distributed_dpo.distributed_dpo(
                list_of_model_names=dpo_hf_model_names,
                list_of_dpo_data_paths=dpo_data_paths,
                list_of_gpu_ids=dpo_gpu_ids,
                list_of_output_model_paths=dpo_output_model_paths,
                batch_size=dpo_batch_size,
                gradient_accumulation_steps=dpo_grad_acc,
                learning_rate=dpo_lr,
                epoch=dpo_epoch,
            )
            print(f"[Sparta] Iter {iteration}: DPO finished.")
            
            # Update current_model_paths to use DPO-trained adapter paths for next iteration
            # This mirrors the approach in text_multiagent_finetuning.py where
            # finetuned model paths replace the base model paths.
            for idx, m in enumerate(dpo_model_names):
                if m in current_model_paths:
                    current_model_paths[m] = dpo_output_model_paths[idx]
                    print(f"[Sparta] Iter {iteration}: Updated model path for {m} -> {dpo_output_model_paths[idx]}")

    return 0

if __name__ == "__main__":
    run_method()