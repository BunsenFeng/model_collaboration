import os
import json
import shutil
from model_collaboration.data import eval
from model_collaboration.utils import lora_check
from model_collaboration.utils.swarm import lora_merge
from model_collaboration.method import distributed_generation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    print("The model you are using are:")
    for model_name in model_names:
        print(model_name)
    print("Make sure they share the same model architecture, or expect errors.")

    # checking if the models are lora adapters
    model_names = lora_check.lora_to_full(model_names)

    # evaluating the models and rank them by dev set performance
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

    # a rank of model indices sorted by dev scores in descending order
    ranked_model_indices = sorted(range(len(model_names)), key=lambda k: list_of_dev_scores[k], reverse=True)
    print("Model ranking by dev {} score:".format(task))
    for rank, model_index in enumerate(ranked_model_indices):
        print("Rank {}: Model {}, dev {} score: {}".format(rank+1, model_names[model_index], task, list_of_dev_scores[model_index]))
    
    # select the greedy soup based on the dev set
    included_model_indices = [ranked_model_indices[0]]  # start with the best model
    current_best_score = list_of_dev_scores[ranked_model_indices[0]]
    for model_index in ranked_model_indices[1:]:
        current_selected_models = [model_names[i] for i in included_model_indices + [model_index]]
        current_weights = [1.0 / len(current_selected_models)] * len(current_selected_models)
        merged_model_path = "model_collaboration/logs/greedy_soup"

        # remove existing merged model path if any
        if os.path.exists(merged_model_path):
            shutil.rmtree(merged_model_path)

        lora_merge(
            weights=current_weights,
            lora_name_list=current_selected_models,
            output_path=merged_model_path,
            gpu_id=gpu_ids[0]
        )
        # evaluate the merged model on the dev set
        list_of_input_list = [dev_input_list]
        list_of_output_list = distributed_generation.distributed_generation(
            [merged_model_path],
            list_of_input_list,
            [gpu_ids[0]]
        )
        dev_outputs = list_of_output_list[0]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        print("Evaluating greedy soup with models {}: dev {} score: {}".format(current_selected_models, task, avg_dev_score))
        if avg_dev_score >= current_best_score:
            included_model_indices.append(model_index)
            current_best_score = avg_dev_score
            print("Included model {} in the greedy soup.".format(model_names[model_index]))
        else:
            print("Excluded model {} from the greedy soup.".format(model_names[model_index]))
        
    final_selected_models = [model_names[i] for i in included_model_indices]
    print("Final selected models in the greedy soup: {}".format(final_selected_models))
    # final weights are uniform among the selected models
    if len(final_selected_models) == 1:
        final_model_name = final_selected_models[0]
        print("Only one model selected, using the model: {}".format(final_model_name))

        # evaluate it on the test set
        test_input_list = eval.prepare_inputs(task, task_type, "test")
        list_of_input_list = [test_input_list]
        list_of_output_list = distributed_generation.distributed_generation(
            [final_model_name],
            list_of_input_list,
            [gpu_ids[0]]
        )
    else:
        final_weights = [1.0 / len(final_selected_models)] * len(final_selected_models)

        if os.path.exists("model_collaboration/logs/greedy_soup"):
            os.remove("model_collaboration/logs/greedy_soup")

        # save the final greedy soup model
        lora_merge(
            weights=final_weights,
            lora_name_list=final_selected_models,
            output_path="model_collaboration/logs/greedy_soup",
            gpu_id=gpu_ids[0]
        )

        # evaluate it on the test set
        test_input_list = eval.prepare_inputs(task, task_type, "test")
        list_of_input_list = [test_input_list]
        list_of_output_list = distributed_generation.distributed_generation(
            ["model_collaboration/logs/greedy_soup"],
            list_of_input_list,
            [gpu_ids[0]]
        )

    test_outputs = list_of_output_list[0]
    test_score = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_score) / len(test_score)
    print("Greedy soup test {} score: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "selected_models": final_selected_models,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_score[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_greedy_soup.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()