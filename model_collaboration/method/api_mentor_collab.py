import json
from tqdm import tqdm
from model_collaboration.data import eval
from model_collaboration.utils.mentor_collab import MentorCollab

MENTOR_COLLAB_TRAIN_SUPPORT_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B-Base",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt"
]

TASK_TYPES = [
    "Math",
    "General"
]

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    if len(model_names) != 2 or len(gpu_ids) != 2:
        raise ValueError("MentorCollab requires exactly 2 models.")
    generator = model_names[0]
    mentor = model_names[1]
    generator_devices = f"cuda:{gpu_ids[0]}"
    mentor_devices = f"cuda:{gpu_ids[1]}"
    decision_proportion = hyperparameters.get("decision_proportion", 0.25)
    patch_size = hyperparameters.get("patch_size", 16)
    max_new_tokens = hyperparameters.get("max_response_length")
    mlp_task = hyperparameters.get("task", "General")  # Task type for MLP model (Math or General)
    mode = hyperparameters.get("mode", "free")
    mlp_threshold = hyperparameters.get("mlp_threshold", 0.5)

    if mode == "train":
        if generator not in MENTOR_COLLAB_TRAIN_SUPPORT_MODELS:
            raise NotImplementedError("Generator model {} is not supported for training-based mode.".format(generator))
    if mlp_task not in TASK_TYPES:
        raise NotImplementedError("MLP task type {} is not supported.".format(mlp_task))

    mentor_collab = MentorCollab(
        generator=generator,
        mentor=mentor,
        generator_devices=generator_devices,
        mentor_devices=mentor_devices,
        mode=mode,
        decision_proportion=decision_proportion,
        patch_size=patch_size,
        task=mlp_task,
        mlp_threshold=mlp_threshold
    )
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    outputs = []
    for input in tqdm(test_input_list, desc="Generating outputs"):
        output = mentor_collab.generate(input, max_new_tokens)
        outputs.append(output)
    
    test_scores = eval.get_scores(task, task_type, "test", outputs)
    avg_test_scores = sum(test_scores) / len(test_scores)
    print("Final test {} score after mentorcollab: {}".format(task, avg_test_scores))
    
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_scores,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_mentor_collab.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0