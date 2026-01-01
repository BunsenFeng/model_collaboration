import json
from data import eval
from model_collaboration.method import distributed_generation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # method-specific hyperparameters
    model_descriptions = hyperparameters.get("model_descriptions", None)
    if model_descriptions is None:
        raise ValueError("model_descriptions must be provided in hyperparameters")
    assert len(model_descriptions) == len(model_names), "Length of model_descriptions must match length of model_names"

    # selecting a model as the prompt-based router based on performance on the dev set
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    list_of_input_list = [dev_input_list for _ in model_names]

    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(avg_dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task_type, avg_dev_score))

    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    print("Best model selected for prompt-based routing: {}".format(best_model_name))

    # prompt-based routing: the best model routes each test input to the most suitable model based on model descriptions
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    routing_prompts = []
    for test_input in test_input_list:
        prompt = "You are an AI assistant that routes user questions to the most suitable AI model based on their descriptions.\n\n"
        prompt += "User question: {}\n\n".format(test_input)
        prompt += "Model descriptions:\n"
        for i in range(len(model_names)):
            prompt += "- {}: {}\n".format(i+1, model_descriptions[i])
        prompt += "\nBased on the above descriptions, which model is best suited to answer the user's question? Respond with the model number only."
        routing_prompts.append(prompt)
    
    routing_outputs = distributed_generation.distributed_generation(
        [best_model_name],
        [routing_prompts],
        [gpu_ids[0]]
    )[0]

    routed_model_indices = []
    for output in routing_outputs:
        found = False
        for i in range(len(model_names), 0, -1):
            if str(i) in output:
                routed_model_indices.append(i-1)
                found = True
                break
        if not found:
            routed_model_indices.append(best_model_index)  # default to best model if no valid index found
    
    final_responses = []
    list_of_input_list = [[] for _ in model_names]
    for i in range(len(test_input_list)):
        model_index = routed_model_indices[i]
        list_of_input_list[model_index].append(test_input_list[i])
    
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )

    for i in range(len(test_input_list)):
        model_index = routed_model_indices[i]
        response = list_of_output_list[model_index].pop(0)
        final_responses.append(response)
    
    test_scores = eval.get_scores(task, task_type, "test", final_responses)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score prompt-based routing: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "routed_model": model_names[routed_model_indices[i]],
            "output": final_responses[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_prompt_routing.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0
    
if __name__ == "__main__":
    run_method()