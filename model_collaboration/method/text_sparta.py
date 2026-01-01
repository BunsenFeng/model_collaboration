import os
import json
import random
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from model_collaboration.utils import distributed_dpo
import logging

logger = logging.getLogger(__name__)


def _pairwise_competition(
    gpu_ids: List[int],
    model_names: List[str],
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
    # model_names are already HuggingFace identifiers, use them directly
    active_model_names: List[str] = []
    list_of_input_list: List[List[str]] = []
    for m in model_names:
        if model_tasks[m]:
            active_model_names.append(m)
            list_of_input_list.append(model_tasks[m])

    if not active_model_names:
        return []

    # Configure generation hyperparameters via the shared helper.
    # These values are passed in from the method's config (hyperparameters)
    # so that generation here is consistent with config.json.
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    # Use model_names directly as HuggingFace identifiers
    list_of_output_list = distributed_generation.distributed_generation(
        active_model_names,
        list_of_input_list,
        gpu_ids,
    )

    # Build an index: {model_name: {instruction: response}}
    model_responses: Dict[str, Dict[str, str]] = {}
    for m, ins_list, out_list in zip(
        active_model_names, list_of_input_list, list_of_output_list
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
            max_response_length=max_response_length,
            temperature=temperature,
            top_p=top_p,
        )

    pairs = calculate_judge_averages_sparta(pairs)
    return pairs

"""
Rating logic is implemented via RatingSystem / RatingSystemDynamicWeighted /
RatingSystemStaticWeighted below. The older _update_reputation helper is no longer used.
"""

def save_judged_pairs_sparta(judged_pairs: List[Dict[str, Any]], base_dir: str, iteration: int) -> None:
    """
    Save judged_pairs under model_collaboration/logs/text_sparta/iteration_k/judged_results/judged_pairs.json,
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
    Save rating_history to model_collaboration/logs/text_sparta/ as a JSON snapshot, matching save_rating_history.
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

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Sparta competition + multi-judge + reputation + DPO preference generation.
    
    This method implements an iterative competition-based training approach where:
    1. Models compete pairwise on instructions
    2. Other models (judges) score the competition responses
    3. Model ratings are updated based on judge scores
    4. Preference pairs are generated from competitions
    5. DPO training is applied to improve models
    6. The best adapter across all iterations is selected via dev set evaluation
    
    Args:
        task: Task name (e.g., "gsm8k", "mmlu")
        task_type: Task type (e.g., "exact_match", "multiple_choice")
        gpu_ids: List of GPU IDs to use for distributed processing
        model_names: List of model identifiers. Can be:
            - HuggingFace Hub identifiers (e.g., "allenai/Llama-3.1-Tulu-3-8B-SFT")
            - Local paths (absolute or relative, e.g., "/path/to/model" or "./models/my_model")
            - Any mix of the above
            The code supports any number of any models without requiring pre-defined mappings.
        hyperparameters: Dictionary containing hyperparameters. Supported keys:
            
            **Iteration Control:**
            - num_iterations (int, default=1): Number of Sparta iterations to run
            - current_iteration (int, default=0): Starting iteration number (for resuming)
            - base_dir (str, default="model_collaboration/logs/text_sparta"): Base directory for saving logs and models
            
            **Generation Hyperparameters (used for both competition and judging):**
            - max_response_length (int, default=256): Maximum number of tokens to generate
            - temperature (float, default=0.7): Sampling temperature for generation
            - top_p (float, default=0.9): Nucleus sampling parameter
            - batch_size (int, default=1): Batch size for generation
            
            **Judge Parameters:**
            - judge_batch_size (int, default=8): Batch size for judge model generation
            - judge_rounds (int, default=1): Number of judging rounds per pair
            
            **Competition Parameters:**
            - num_instructions (int, default=500): Number of instructions to use for competition
            - random_match_prob (float, default=0.2): Probability of random opponent selection
            - num_opponents (int, default=3): Number of top-K opponents to consider for matching
            
            **Rating System Parameters:**
            - initial_k (float, default=10.0): Initial K value for rating updates
            - min_k (float, default=5.0): Minimum K value (after decay)
            - window_size (int, default=10): Window size for deviation calculation
            - min_deviation (float, default=0.1): Minimum deviation value
            - epsilon (float, default=0.01): Small epsilon for numerical stability
            - decay_rate (float, default=0.9): Decay rate for K value
            - decay_steps (int, default=10): Steps for K decay
            - scaling_factor (float, default=20.0): Scaling factor for rating updates
            - score_type (str, default="normal"): Rating system type: "normal", "dynamic", or "static"
            - freeze_ratings (bool, default=False): If True, ratings are not updated
            
            **Debug:**
            - debug (bool, default=False): If True, prints detailed rating update information
            
    Returns:
        int: Always returns 0 on successful completion
        
    Note:
        - Judges are dynamically selected from the model pool for each pair (models that didn't compete)
        - DPO training uses default hyperparameters (batch_size=1, gradient_accumulation_steps=16, 
          learning_rate=1e-6, epoch=1)
        - All adapters from all iterations are evaluated on dev set, and the best one is selected
        - LoRA adapters are automatically detected and merged before training new adapters
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # ------------------------- 0. Load all hyperparameters in one place -------------------------
    # Iteration control
    num_iterations = int(hyperparameters.get("num_iterations", 3))  # Number of iterations to run
    start_iteration = int(hyperparameters.get("current_iteration", 0))  # Starting iteration number (for resuming)
    base_dir = hyperparameters.get("base_dir", os.path.join("model_collaboration/logs", "text_sparta"))
    
    # General generation hyperparameters (used for both competition and judging)
    max_response_length = int(hyperparameters.get("max_response_length", 256))
    temperature = float(hyperparameters.get("temperature", 0.7))
    top_p = float(hyperparameters.get("top_p", 0.9))
    batch_size = int(hyperparameters.get("batch_size", 1))
    
    # Judge operational parameters (judges are dynamically selected from model_names pool for each pair)
    judge_batch_size = int(hyperparameters.get("judge_batch_size", 8))
    judge_rounds = int(hyperparameters.get("judge_rounds", 1))
    
    # Competition and instruction parameters
    num_instructions = int(hyperparameters.get("num_instructions", 500))
    random_match_prob = float(hyperparameters.get("random_match_prob", 0.2))
    num_opponents = int(hyperparameters.get("num_opponents", 3))
    
    # Rating system parameters
    initial_K = float(hyperparameters.get("initial_k", 10.0))
    min_K = float(hyperparameters.get("min_k", 5.0))
    window_size = int(hyperparameters.get("window_size", 10))
    min_deviation = float(hyperparameters.get("min_deviation", 0.1))
    epsilon = float(hyperparameters.get("epsilon", 0.01))
    decay_rate = float(hyperparameters.get("decay_rate", 0.9))
    decay_steps = int(hyperparameters.get("decay_steps", 10))
    scaling_factor = float(hyperparameters.get("scaling_factor", 20.0))
    score_type = hyperparameters.get("score_type", "normal")  # normal / dynamic / static
    freeze_ratings = bool(hyperparameters.get("freeze_ratings", False))
    debug = bool(hyperparameters.get("debug", False))

    # Track current model paths (for adapter handling across iterations)
    # model_names are already HuggingFace identifiers, use them directly
    current_model_paths: Dict[str, str] = {m: m for m in model_names}

    for it in range(num_iterations):
        iteration = start_iteration + it  # the global iteration number of the current iteration

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
            model_names=[current_model_paths[m] for m in model_names],
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
        # Optimize: Group pairs by judge models to avoid loading models multiple times
        # For each pair, judges are models from the pool that didn't participate in that specific competition
        # pair["models"] contains model paths (from current_model_paths), need to map back to original names
        # Build reverse mapping: model_path -> original_model_name
        path_to_name = {current_model_paths.get(m, m): m for m in model_names}
        
        # Step 1: Collect all pairs that need judging and group them by judge model
        # judge_model_path -> list of pairs that need this judge
        judge_groups: Dict[str, List[Dict[str, Any]]] = {}
        pairs_without_judges: List[Dict[str, Any]] = []
        
        for pair in raw_pairs:
            # Get the two model paths that competed in this pair
            competing_model_paths = pair.get("models", [])
            # Map back to original model names
            competing_model_names = {path_to_name.get(path, path) for path in competing_model_paths}
            # Judges are all models in the pool except the two that competed
            pair_judge_models = [m for m in model_names if m not in competing_model_names]
            
            if not pair_judge_models:
                # If all models competed, skip judging (shouldn't happen with >2 models)
                pairs_without_judges.append(pair)
                continue
            
            # Group pairs by judge model path
            for judge_model_name in pair_judge_models:
                judge_model_path = current_model_paths.get(judge_model_name, judge_model_name)
                if judge_model_path not in judge_groups:
                    judge_groups[judge_model_path] = []
                # Use the same pair object reference so modifications are in-place
                judge_groups[judge_model_path].append(pair)
        
        # Step 2: Judge all pairs for each judge model in one batch (avoids reloading models)
        judge_paths_list = list(judge_groups.keys())
        
        for idx, judge_model_path in enumerate(judge_paths_list):
            # Get the original judge model name for judge_name
            judge_model_name = None
            for m in model_names:
                if current_model_paths.get(m, m) == judge_model_path:
                    judge_model_name = m
                    break
            if judge_model_name is None:
                judge_model_name = judge_model_path
            
            # Assign GPU for this judge model (round-robin)
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            
            pairs_to_judge = judge_groups[judge_model_path]
            print(f"[Sparta] Iter {iteration}: Judging {len(pairs_to_judge)} pairs with judge {judge_model_name} on GPU {gpu_id}")
            
            # Judge all pairs for this judge model in one batch
            # Note: _judge_batch_with_model modifies pairs in-place, so raw_pairs will be updated
            _judge_batch_with_model(
                judge_name=judge_model_name,
                judge_model=judge_model_path,
                pairs=pairs_to_judge,
                gpu_id=gpu_id,
                batch_size=judge_batch_size,
                base_dir=base_dir,
                num_rounds=judge_rounds,
                max_response_length=max_response_length,
                temperature=temperature,
                top_p=top_p,
            )
        
        # Step 3: Collect all judged pairs and calculate averages
        # raw_pairs already contains the judged results since we passed references to _judge_batch_with_model
        # Add pairs that didn't need judging
        judged_pairs = raw_pairs.copy()
        judged_pairs.extend(pairs_without_judges)
        
        # Calculate judge averages (required before computing weighted scores)
        judged_pairs = calculate_judge_averages_sparta(judged_pairs)

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
                freeze_ratings=freeze_ratings,
                debug=debug,
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
                freeze_ratings=freeze_ratings,
                debug=debug,
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
                freeze_ratings=freeze_ratings,
                debug=debug,
            )

        rating_history: List[Dict[str, Any]] = []
        for idx_pair, pair in enumerate(judged_pairs):
            # Map adapter paths in pair["models"] back to original model names for rating system
            # Create a copy to avoid modifying the original pair
            pair_for_rating = pair.copy()
            if "models" in pair_for_rating:
                original_models = []
                for model_path in pair_for_rating["models"]:
                    # Map adapter path back to original model name
                    original_name = path_to_name.get(model_path, model_path)
                    original_models.append(original_name)
                pair_for_rating["models"] = original_models
            
            rating_system.update_ratings_from_judges(pair_for_rating)
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
            # Map adapter paths in pair["models"] back to original model names for rating system
            # Create a copy to avoid modifying the original pair
            pair_for_pref = pair.copy()
            if "models" in pair_for_pref:
                original_models = []
                for model_path in pair_for_pref["models"]:
                    # Map adapter path back to original model name
                    original_name = path_to_name.get(model_path, model_path)
                    original_models.append(original_name)
                pair_for_pref["models"] = original_models
            
            pref = rating_system.select_preference_response(pair_for_pref)
            if pref is not None:
                preference_pairs.append(pref)

        old_len = len(preference_pairs)
        preference_pairs = filter_tie_sparta(preference_pairs)
        logger.info(
            f"[Sparta] Iter {iteration}: preference_pairs {old_len} -> {len(preference_pairs)} after tie filter"
        )

        dataset_dir = os.path.join(iter_dir, "dataset")
        pref_path = save_preference_pairs_to_json_sparta(preference_pairs, dataset_dir)

        # ------------------------- 9. DPO training -------------------------
        # DPO is performed to update models based on preference pairs
        if preference_pairs and pref_path:
            dpo_data_paths = [pref_path for _ in model_names]
            dpo_gpu_ids = gpu_ids[: len(model_names)] or [0]
            
            # Generate unique directory names for adapters based on model paths
            # Use a hash or the last part of the path to create a safe directory name
            def _get_model_dir_name(model_path: str) -> str:
                # Use the last part of the path, replacing / with _
                safe_name = model_path.replace("/", "_").replace("\\", "_")
                # Limit length to avoid filesystem issues
                if len(safe_name) > 100:
                    import hashlib
                    safe_name = hashlib.md5(model_path.encode()).hexdigest()[:16]
                return safe_name
            
            dpo_output_model_paths = [
                os.path.join(iter_dir, f"dpo_{_get_model_dir_name(current_model_paths.get(m, m))}")
                for m in model_names
            ]

            # Use current_model_paths to get actual model paths (supports adapters from previous iterations)
            # This allows DPO to be applied on top of previously trained adapters.
            dpo_hf_model_names = [
                current_model_paths.get(m, m) for m in model_names
            ]

            print(f"[Sparta] Iter {iteration}: Starting DPO for {model_names}")
            # Use default DPO hyperparameters: batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-6, epoch=1
            distributed_dpo.distributed_dpo(
                list_of_model_names=dpo_hf_model_names,
                list_of_dpo_data_paths=dpo_data_paths,
                list_of_gpu_ids=dpo_gpu_ids,
                list_of_output_model_paths=dpo_output_model_paths,
            )
            print(f"[Sparta] Iter {iteration}: DPO finished.")
            
            # Update current_model_paths to use DPO-trained adapter paths for next iteration
            # This mirrors the approach in text_multiagent_finetuning.py where
            # finetuned model paths replace the base model paths.
            for idx, m in enumerate(model_names):
                current_model_paths[m] = dpo_output_model_paths[idx]
                print(f"[Sparta] Iter {iteration}: Updated model path for {m} -> {dpo_output_model_paths[idx]}")

    # ------------------------- 10. Final evaluation: evaluate ALL adapters from ALL iterations on dev, then pick best for test -------------------------
    print(f"[Sparta] All iterations completed. Evaluating all adapters on dev set...")

    # 1) Collect all adapter paths from every iteration for every model
    #    We rely on the naming convention used in the DPO step:
    #    model_collaboration/logs/text_sparta/iteration_{k}/dpo_<safe_model_name>
    all_adapter_entries: List[Tuple[int, str, str]] = []  # (iteration, model_name, adapter_path)
    for it in range(start_iteration, start_iteration + num_iterations):
        iter_dir_eval = os.path.join(base_dir, f"iteration_{it}")
        if not os.path.isdir(iter_dir_eval):
            continue
        for m in model_names:
            safe_name = current_model_paths.get(m, m).replace("/", "_").replace("\\", "_")
            candidate_path = os.path.join(iter_dir_eval, f"dpo_{safe_name}")
            if os.path.isdir(candidate_path):
                all_adapter_entries.append((it, m, candidate_path))

    # If no adapters were found (e.g., no preference_pairs), fall back to using the final current_model_paths
    if not all_adapter_entries:
        print("[Sparta] No DPO adapters found across iterations; falling back to final models only.")
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        list_of_input_list = [dev_input_list for _ in model_names]
        final_model_paths = [current_model_paths.get(m, m) for m in model_names]
        list_of_output_list = distributed_generation.distributed_generation(
            final_model_paths,
            list_of_input_list,
            gpu_ids,
        )

        list_of_dev_scores = []
        for i in range(len(model_names)):
            dev_outputs = list_of_output_list[i]
            dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_score) / len(dev_score)
            list_of_dev_scores.append(avg_dev_score)
            print(f"[Sparta] Final model {model_names[i]}: dev {task} score: {avg_dev_score}")

        best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
        best_model_name = model_names[best_model_index]
        best_model_path = final_model_paths[best_model_index]
        best_model_iteration = start_iteration + num_iterations - 1
        print(
            f"[Sparta] Best model (no adapters case) selected for test evaluation: "
            f"{best_model_name} from iteration {best_model_iteration} "
            f"(dev score: {list_of_dev_scores[best_model_index]})"
        )
        per_model_dev_scores = {
            model_names[i]: list_of_dev_scores[i] for i in range(len(model_names))
        }
    else:
        # 2) Evaluate every (iteration, model, adapter_path) on dev set
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        list_of_input_list = [dev_input_list for _ in all_adapter_entries]
        adapter_paths = [entry[2] for entry in all_adapter_entries]

        list_of_output_list = distributed_generation.distributed_generation(
            adapter_paths,
            list_of_input_list,
            gpu_ids,
        )

        adapter_dev_scores: List[float] = []
        adapter_keys: List[str] = []
        per_model_dev_scores = {}

        for idx, ((it, m, path), outputs) in enumerate(
            zip(all_adapter_entries, list_of_output_list)
        ):
            dev_score = eval.get_scores(task, task_type, "dev", outputs)
            avg_dev_score = sum(dev_score) / len(dev_score)
            adapter_dev_scores.append(avg_dev_score)
            key = f"{m}_iter{it}"
            adapter_keys.append(key)
            per_model_dev_scores[key] = avg_dev_score
            print(
                f"[Sparta] Adapter {key} ({path}): dev {task} score: {avg_dev_score}"
            )

        # 3) Select the best adapter across all iterations and models
        best_idx = adapter_dev_scores.index(max(adapter_dev_scores))
        best_iter, best_model_name, best_model_path = all_adapter_entries[best_idx]
        print(
            f"[Sparta] Best adapter selected for test evaluation: {best_model_name} "
            f"from iteration {best_iter} (dev score: {adapter_dev_scores[best_idx]})"
        )
    
    # 4) Evaluate best model on test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    test_output_list = distributed_generation.distributed_generation(
        [best_model_path],
        [test_input_list],
        gpu_ids
    )
    final_output_list = test_output_list[0]
    
    # Evaluate the final outputs
    test_scores = eval.get_scores(task, task_type, "test", final_output_list)
    avg_test_score = sum(test_scores) / len(test_scores)
    print(f"[Sparta] Final test {task} score: {avg_test_score}")
    
    # 5) Save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_sparta",
        "model_names": model_names,
        "best_model": best_model_name,
        "best_model_iteration": int(best_iter) if 'best_iter' in locals() else int(start_iteration + num_iterations - 1),
        "best_model_path": best_model_path,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "dev_scores": per_model_dev_scores,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "output": final_output_list[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)
    
    # Save to a json file
    log_filename = f"model_collaboration/logs/{task}_{len(model_names)}_{round(avg_test_score, 4)}_text_sparta.json"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_logs, f, indent=4, ensure_ascii=False)
    print(f"[Sparta] Saved experiment logs to {log_filename}")

    return 0

if __name__ == "__main__":
    run_method()