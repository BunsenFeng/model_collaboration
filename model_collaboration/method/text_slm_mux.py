"""
SLM-MUX: orchestrating small language models for reasoning by training-free,
confidence-based routing.

For each test question:
  1. Generate k samples from each candidate model independently.
  2. Compute per-model confidence as the consistency of its k extracted answers
     (frequency of the majority answer = max_count / k).
  3. Select the model with the highest confidence; break ties by validation
     accuracy on the dev set (or by model order / randomly, configurable).
  4. Output the selected model's majority answer.

Reference:
    Wang et al., "SLM-MUX: Orchestrating Small Language Models for Reasoning",
    ICLR 2026. https://arxiv.org/abs/2510.05077
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from model_collaboration.data import eval
from model_collaboration.method import distributed_generation

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_split(task, split, ratio=1.0):
    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as f:
        data = json.load(f)[split]
    return data[: int(len(data) * ratio)]


def _extract_answers(task, task_type, split, outputs, ratio=1.0):
    """Extract a discrete answer string from each raw model output."""
    data = _load_split(task, split, ratio)
    extracted = []
    if task_type == "multiple_choice":
        assert "choices" in data[0], "Are you sure this is a multiple choice task?"
        for item, output in zip(data, outputs):
            options = [item["choices"][k] for k in item["choices"].keys()]
            chosen_letter, _ = eval.parse_model_response_mcq(output, options)
            extracted.append(chosen_letter if chosen_letter is not None else "")
    elif task_type in ("exact_match", "f1_match"):
        for output in outputs:
            extracted.append(eval.extract_answer_text(output) or "")
    return extracted


def _score_extracted(task, task_type, split, extracted_answers, ratio=1.0):
    """Score a list of already-extracted answers against the gold labels."""
    data = _load_split(task, split, ratio)
    scores = []
    if task_type == "multiple_choice":
        for item, ans in zip(data, extracted_answers):
            scores.append(1.0 if ans and item["answer"] == ans else 0.0)
    elif task_type == "exact_match":
        for item, ans in zip(data, extracted_answers):
            scores.append(eval.calculate_exact_match(ans, item["output"]))
    elif task_type == "f1_match":
        if task == "popqa":
            for item, ans in zip(data, extracted_answers):
                opts = item["output"].replace("[", "").replace("]", "").replace("\"", "").replace("'", "")
                opts = [o.strip() for o in opts.split(",")]
                scores.append(max((eval.calculate_f1_score(ans, o) for o in opts), default=0.0))
        else:
            for item, ans in zip(data, extracted_answers):
                scores.append(eval.calculate_f1_score(ans, item["output"]))
    return scores


def _consistency(extracted_k):
    """Consistency confidence for one model on one question.

    Returns (selected_answer, score, vote_counts) where score = max_count / k
    over non-empty answers; ties are broken randomly.
    """
    counts = defaultdict(int)
    for a in extracted_k:
        if a and a.strip():
            counts[a] += 1
    if not counts:
        return "", 0.0, {}
    total = sum(counts.values())
    max_c = max(counts.values())
    top = [a for a, c in counts.items() if c == max_c]
    return random.choice(top) if len(top) > 1 else top[0], max_c / total, dict(counts)


def _generate_k_samples(model_names, input_list, gpu_ids, k):
    """Generate k samples per model by stacking inputs k times in one call.

    Returns list-of-list-of-list: [model_idx][sample_idx][question_idx] -> output.
    Stacking inside one distributed_generation call avoids reloading each model k times.
    """
    n = len(input_list)
    stacked = [input_list * k for _ in model_names]  # each: n*k inputs
    flat = distributed_generation.distributed_generation(model_names, stacked, gpu_ids)
    out = []
    for mi in range(len(model_names)):
        per_sample = [flat[mi][s * n : (s + 1) * n] for s in range(k)]
        out.append(per_sample)
    return out


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    script_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(script_dir)

    assert task_type in ("multiple_choice", "exact_match", "f1_match"), (
        "text_slm_mux supports only multiple_choice, exact_match, and f1_match task types."
    )

    samples_per_model = int(hyperparameters.get("samples_per_model", 5))
    assert samples_per_model >= 1, "samples_per_model must be >= 1"
    tie_breaking = hyperparameters.get("tie", "dev-based")
    assert tie_breaking in ("dev-based", "model-order", "random"), (
        "tie must be one of 'dev-based', 'model-order', 'random'"
    )
    seed = int(hyperparameters.get("seed", 42))
    random.seed(seed)

    # 1. (Optional) compute per-model dev accuracy for tie-breaking
    model_accuracies = {}
    if tie_breaking == "dev-based":
        dev_inputs = eval.prepare_inputs(task, task_type, "dev")
        list_of_inputs = [dev_inputs for _ in model_names]
        dev_outputs = distributed_generation.distributed_generation(
            model_names, list_of_inputs, gpu_ids
        )
        for i, name in enumerate(model_names):
            scores = eval.get_scores(task, task_type, "dev", dev_outputs[i])
            acc = sum(scores) / len(scores) if scores else 0.0
            model_accuracies[name] = acc
            print("[SLM-MUX] dev accuracy for {}: {:.4f}".format(name, acc))

    # 2. Generate k samples per model on the test set
    test_inputs = eval.prepare_inputs(task, task_type, "test")
    print("[SLM-MUX] generating k={} samples per model on {} test items".format(
        samples_per_model, len(test_inputs)
    ))
    samples = _generate_k_samples(model_names, test_inputs, gpu_ids, samples_per_model)
    # samples[model_idx][sample_idx][question_idx] = raw output string

    # 3. Extract answers per (model, sample, question)
    extracted = []  # extracted[model_idx][sample_idx] = list of extracted answers
    for mi in range(len(model_names)):
        per_sample = []
        for s in range(samples_per_model):
            per_sample.append(_extract_answers(task, task_type, "test", samples[mi][s]))
        extracted.append(per_sample)

    # 4. Per question: per-model consistency, pick winner, tie-break
    final_extracted = []
    per_question_logs = []
    for qi in range(len(test_inputs)):
        per_model = []  # list of (model_name, selected_answer, score, vote_counts)
        for mi, name in enumerate(model_names):
            k_answers = [extracted[mi][s][qi] for s in range(samples_per_model)]
            sel, score, counts = _consistency(k_answers)
            per_model.append((name, sel, score, counts))

        max_score = max(p[2] for p in per_model)
        tied = [p for p in per_model if abs(p[2] - max_score) < 1e-9]
        if len(tied) == 1:
            winner = tied[0]
        elif tie_breaking == "model-order":
            winner = tied[0]  # already in model_names order
        elif tie_breaking == "random":
            winner = random.choice(tied)
        else:  # dev-based
            winner = max(tied, key=lambda p: model_accuracies.get(p[0], 0.0))

        final_extracted.append(winner[1])
        per_question_logs.append({
            "selected_model": winner[0],
            "selected_answer": winner[1],
            "confidence": winner[2],
            "per_model_confidence": {p[0]: p[2] for p in per_model},
            "per_model_selected": {p[0]: p[1] for p in per_model},
            "per_model_vote_counts": {p[0]: p[3] for p in per_model},
            "tied": len(tied) > 1,
        })

    # 5. Score and log
    test_scores = _score_extracted(task, task_type, "test", final_extracted)
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print("[SLM-MUX] final test {} score: {:.4f}".format(task, avg_test_score))

    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_slm_mux",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "model_accuracies": model_accuracies,
        "avg_test_score": avg_test_score,
        "logs": [],
    }
    for qi in range(len(test_inputs)):
        log_entry = {
            "input": test_inputs[qi],
            "output": final_extracted[qi],
            "score": test_scores[qi],
            **per_question_logs[qi],
        }
        experiment_logs["logs"].append(log_entry)

    log_filename = "model_collaboration/logs/{}_{}_{}_slm_mux.json".format(
        task, len(model_names), round(avg_test_score, 4)
    )
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0


if __name__ == "__main__":
    run_method()