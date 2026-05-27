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

    # 3. Extract answers and scores per (model, sample, question)
    extracted = []   # extracted[mi][s] = list of parsed answers per question
    ext_scores = []  # ext_scores[mi][s] = list of scores per question
    for mi in range(len(model_names)):
        per_sample_ans, per_sample_scores = [], []
        for s in range(samples_per_model):
            sc, ps = eval.get_scores(task, task_type, "test", samples[mi][s], return_output=True)
            per_sample_ans.append(ps)
            per_sample_scores.append(sc)
        extracted.append(per_sample_ans)
        ext_scores.append(per_sample_scores)

    # 4. Per question: per-model consistency, pick winner, tie-break
    final_extracted = []
    test_scores = []
    per_question_logs = []
    for qi in range(len(test_inputs)):
        per_model = []  # list of (model_name, selected_answer, confidence, vote_counts, score)
        for mi, name in enumerate(model_names):
            k_answers = [extracted[mi][s][qi] for s in range(samples_per_model)]
            sel, conf, counts = _consistency(k_answers)
            s_match = next((s for s in range(samples_per_model) if extracted[mi][s][qi] == sel), 0)
            per_model.append((name, sel, conf, counts, ext_scores[mi][s_match][qi]))

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
        test_scores.append(winner[4])
        per_question_logs.append({
            "selected_model": winner[0],
            "selected_answer": winner[1],
            "confidence": winner[2],
            "per_model_confidence": {p[0]: p[2] for p in per_model},
            "per_model_selected": {p[0]: p[1] for p in per_model},
            "per_model_vote_counts": {p[0]: p[3] for p in per_model},
            "tied": len(tied) > 1,
        })

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