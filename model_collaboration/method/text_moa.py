import json
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation


def _inject_references(question, references):
    """Build the MoA aggregator prompt: numbered reference list + original question."""
    prompt = (
        "You have been provided with a set of responses from various open-source models "
        "to the latest user query. Your task is to synthesize these responses into a single, "
        "high-quality response. It is crucial to critically evaluate the information provided "
        "in these responses, recognizing that some of it may be biased or incorrect. Your "
        "response should not simply replicate the given responses but should offer a refined, "
        "accurate and comprehensive reply to the instruction. Ensure your response is "
        "well-structured, coherent, and adheres to the highest standards of accuracy and "
        "reliability.\n\n"
        "Responses from models:\n"
    )
    for i, ref in enumerate(references, 1):
        prompt += f"{i}. {ref}\n"
    prompt += f"\nQuestion: {question}"
    return prompt


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    rounds = hyperparameters.get("round", 1)
    ratio = hyperparameters.get("ratio", 1.0)

    # --- Dev set: all models generate, pick best as aggregator ---
    dev_input_list = eval.prepare_inputs(task, task_type, "dev", ratio=ratio)
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        [dev_input_list for _ in model_names],
        gpu_ids
    )

    list_of_dev_scores = []
    for i, model in enumerate(model_names):
        score = eval.get_scores(task, task_type, "dev", list_of_output_list[i], ratio=ratio)
        avg = sum(score) / len(score)
        list_of_dev_scores.append(avg)
        print(f"Model: {model}, dev {task} score: {avg}")

    best_idx = list_of_dev_scores.index(max(list_of_dev_scores))
    aggregator = model_names[best_idx]
    print(f"Aggregator selected (best on dev): {aggregator}")

    # --- Test set: proposer rounds ---
    test_input_list = eval.prepare_inputs(task, task_type, "test", ratio=ratio)
    response_list = None

    for r in range(rounds):
        print(f"Proposer round {r+1}/{rounds}")
        if r == 0:
            list_of_input_list = [test_input_list for _ in model_names]
        else:
            # Inject previous-round references into each proposer's prompt
            list_of_input_list = []
            for i in range(len(model_names)):
                prompts = []
                for j in range(len(test_input_list)):
                    refs = [response_list[k][j] for k in range(len(model_names))]
                    prompts.append(_inject_references(test_input_list[j], refs))
                list_of_input_list.append(prompts)

        list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            list_of_input_list,
            gpu_ids
        )
        response_list = list_of_output_list

    # --- Aggregation: best model synthesizes all candidates ---
    agg_input_list = []
    for j in range(len(test_input_list)):
        refs = [response_list[i][j] for i in range(len(model_names))]
        agg_input_list.append(_inject_references(test_input_list[j], refs))

    agg_output_list = distributed_generation.distributed_generation(
        [aggregator],
        [agg_input_list],
        [gpu_ids[best_idx % len(gpu_ids)]]
    )
    final_outputs = agg_output_list[0]

    test_scores = eval.get_scores(task, task_type, "test", final_outputs, ratio=ratio)
    avg_test_score = sum(test_scores) / len(test_scores)
    print(f"Final Test {task} score after MoA ({rounds} proposer round(s), aggregator={aggregator}): {avg_test_score}")

    # --- Save logs ---
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "aggregator": aggregator,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": [
            {"input": test_input_list[i], "output": final_outputs[i], "score": test_scores[i]}
            for i in range(len(test_input_list))
        ]
    }

    log_filename = "model_collaboration/logs/{}_{}_{}_moa.json".format(
        task, len(model_names), round(avg_test_score, 4)
    )
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0
