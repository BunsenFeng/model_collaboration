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
    rounds = hyperparameters.get("round", 3)

    # selecting a model as the final summarizer based on performance on the dev set
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

    # multiagent refine on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    response_list = None # len(model_names) * len(test_input_list)
    for r in range(rounds):
        print("Round {}/{}".format(r+1, rounds))
        list_of_input_list = []
        if r == 0:
            for _ in model_names:
                list_of_input_list.append(test_input_list)
        else:
            assert response_list is not None, "Response list should not be None in round {}".format(r)
            for i in range(len(model_names)):
                refine_prompt_list = []
                for j in range(len(test_input_list)):
                    prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to refine and improve your own answer.\n\n"
                    prompt += "Question: {}\n\n".format(test_input_list[j])
                    prompt += "Your previous answer: {}\n\n".format(response_list[i][j])
                    prompt += "Other assistants' answers:\n"
                    for k in range(len(model_names)):
                        if k != i:
                            prompt += "- {}\n".format(response_list[k][j])
                    prompt += "\nPlease provide a refined answer to the question."
                    refine_prompt_list.append(prompt)
                list_of_input_list.append(refine_prompt_list)
        
        assert len(list_of_input_list) == len(model_names), "Length of input lists must match number of models"
        assert len(list_of_input_list[0]) == len(test_input_list), "Each input list must match number of test inputs"

        list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            list_of_input_list,
            gpu_ids
        )
        response_list = list_of_output_list
    
    # final summarization using the best model
    summarization_input_list = []
    for j in range(len(test_input_list)):
        prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to create a final, comprehensive answer.\n\n"
        prompt += "Question: {}\n\n".format(test_input_list[j])
        prompt += "Assistants' answers:\n"
        for i in range(len(model_names)):
            prompt += "- {}\n".format(response_list[i][j])
        prompt += "\nPlease provide a final, comprehensive answer to the question."
        summarization_input_list.append(prompt)
    
    list_of_input_list = [summarization_input_list]
    list_of_model_names = [best_model_name]
    list_of_gpu_ids = [gpu_ids[0]] # use the first GPU for final summarization
    list_of_output_list = distributed_generation.distributed_generation(
        list_of_model_names,
        list_of_input_list,
        list_of_gpu_ids
    )
    final_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final Test {} score after {} rounds of multiagent refine: {}".format(task, rounds, avg_test_score))

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
    log_filename = "model_collaboration/logs/{}_{}_{}_multiagent_refine.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()