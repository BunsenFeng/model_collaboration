import json
from data import eval
from utils.mentor_collab import MentorCollab

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
    generator = hyperparameters.get("generator")
    mentor = hyperparameters.get("mentor")
    generator_devices = hyperparameters.get("generator_devices")
    mentor_devices = hyperparameters.get("mentor_devices")
    decision_proportion = hyperparameters.get("decision_proportion", 0.25)
    patch_size = hyperparameters.get("patch_size", 16)
    max_new_tokens = hyperparameters.get("max_response_length")
    task = hyperparameters.get("task", "General")
    mode = hyperparameters.get("mode", "free")
    mlp_threshold = hyperparameters.get("mlp_threshold", 0.5)

    if mode == "train":
        if generator not in MENTOR_COLLAB_TRAIN_SUPPORT_MODELS:
            raise NotImplementedError("Generator model {} is not supported for training-based mode.".format(generator))
    if task not in TASK_TYPES:
        raise NotImplementedError("Task type {} is not supported.".format(task))
    
    mentor_collab = MentorCollab(
        generator=generator, 
        mentor=mentor, 
        generator_devices=generator_devices, 
        mentor_devices=mentor_devices, 
        mode=mode,
        decision_proportion=decision_proportion, 
        patch_size=patch_size,
        task=task,
        mlp_threshold=mlp_threshold
    )
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    outputs = []
    for input in test_input_list:
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
    log_filename = "logs/{}_{}_{}_mentor_collab.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0