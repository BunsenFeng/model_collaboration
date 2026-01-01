import json
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # method-specific hyperparameters
    exclude_self = hyperparameters.get("exclude_self", True)

    # selecting a model as the final response generator based on performance on the dev set
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
        print("Model: {}, dev {} score: {}".format(model_names[i], task, avg_dev_score))

    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    print("Best model selected for final summarization: {}".format(best_model_name))

    # generaing relevant knowledge on the test set from all models
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    knowledge_generation_prompt = []
    for input_text in test_input_list:
        prompt = "You are an AI assistant tasked with generating relevant knowledge to help answer the user's question. Please provide relevant information that can assist in answering the following question:\n\n"
        prompt += "Question: {}\n\n".format(input_text)
        prompt += "Relevant Knowledge:"
        knowledge_generation_prompt.append(prompt)
    list_of_input_list = [knowledge_generation_prompt for _ in model_names]
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # len(model_names) * len(test_input_list)

    # final response generation using the best model
    final_response_prompts = []
    for i in range(len(test_input_list)):
        prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Other assistants have provided the following relevant knowledge to help answer the question:\n\n"
        prompt += "Question: {}\n\n".format(test_input_list[i])
        prompt += "Relevant Knowledge from other assistants:\n"
        for j in range(len(model_names)): 
            if exclude_self and j == best_model_index:
                continue
            knowledge_text = list_of_output_list[j][i]
            prompt += "- {}\n".format(knowledge_text)
        prompt += "\nPlease provide a comprehensive answer to the question with the help of the above knowledge."
        final_response_prompts.append(prompt)

    final_outputs = distributed_generation.distributed_generation(
        [best_model_name],
        [final_response_prompts],
        [gpu_ids[0]]
    )[0]

    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score after knowledge card: {}".format(task, avg_test_score))

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
            "output": final_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_knowledge_card.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()