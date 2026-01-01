"""
Method: text_structure.py
"""
import json
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
import numpy as np

# optionally, `from utils import distributed_sft` if your approach finetunes multiple models
# optionally, `from utils import logit_arithmetic` if your approach is composing the logits of multiple models
# optionally, `from utils.numeric_swarm import NumericSwarm` if your approach optimizes continuous vectors of hyperparameters/weights
"""
{
    "method": "text_structure", // the name under method/ folder, names like <type>_<approach>.py
    "task": "agieval", // the name under data/ folder, see data/eval_readme.md
    "task_type": "multiple_choice", // see data/eval_readme.md
    "gpu_ids": [0,1,2], // a list of GPUs available
    "model_names": [
        "model_1_name",
        "model_2_name",
        "model_3_name"
    ], // a list of model identifiers, local or huggingface
    "hyperparameters": {
        "max_response_length": 512, // max generation length
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 8, // per GPU batch size
        // and then, method-specific hyperparameters
        "num_rounds": 3, // number of rounds of updates between models
        "structure_type": "your_structure_type", // chain, tree, star, circle, complete, other
        "structure_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]], // optional, only applicable if structure_type is "other", matrix of size num_models x num_models defining the structure
    }
}
"""


def generate_adj(n, graph_type, structure_matrix=None):
    if "complete" in graph_type:
        adj_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
    if "tree" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            # Add edges if left and right children are within bounds
            if left_child < n:
                adj_matrix[i][left_child] = 1
                adj_matrix[left_child][i] = 1
            if right_child < n:
                adj_matrix[i][right_child] = 1
                adj_matrix[right_child][i] = 1
    if "chain" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        # Set the values for a chain structure
        for i in range(n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
    if "star" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj_matrix[0][i] = 1
            adj_matrix[i][0] = 1
        for i in range(1, n - 1):
            adj_matrix[i][i + 1] = 1
            adj_matrix[i + 1][i] = 1
        adj_matrix[1][n - 1] = 1
        adj_matrix[n - 1][1] = 1
    if "circle" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj_matrix[i][(i + 1) % n] = 1
            adj_matrix[(i + 1) % n][i] = 1
    if "other" in graph_type:
        adj_matrix = np.array(structure_matrix)
    return adj_matrix

def re_generate_prompt(initial_prompt, model_response, neighbor_responses):
    prompt = f"Task: {initial_prompt}\n"
    prompt += "Based on your previous response and the responses from other AI assistants, provide an updated response.\n"
    prompt += f"Your previous response: {model_response}\n"
    prompt += "Responses from other AI assistants:\n"
    for i in range(len(neighbor_responses)):
        prompt += f"Assistant {i+1}: {neighbor_responses[i]}\n"
    prompt += "\nPlease consider the above information carefully and provide your updated response."
    return prompt

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Implement your approach here.
    Args:
        task: str, the name of the task
        task_type: str, the type of the task (e.g., "multiple_choice", "exact_match", etc.)
        gpu_ids: list of int, the GPU ids to use for distributed generation
        model_names: list of str, the names of the models to use
        hyperparameters: dict, method-specific hyperparameters
        You get these arguments from the config file that users pass in.
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # 1. extract the hyperparameters from the hyperparameters dict
    num_rounds = hyperparameters.get("num_rounds", None)
    assert type(num_rounds) == int and num_rounds > 0, "Please provide num_rounds (positive integer) in hyperparameters."
    if num_rounds == 1:
        print("Warning: num_rounds is 1, which means no interaction between models will happen.")
    structure_type = hyperparameters.get("structure_type", "not_supported")
    structure_matrix = hyperparameters.get("structure_matrix", None)
    assert structure_type in ["chain", "tree", "star", "circle", "complete", "other"], "Invalid structure type, please provide one of the supported types: chain, tree, star, circle, complete, other."
    if structure_type == "other":
        assert structure_matrix is not None, "Please provide a structure matrix (list of lists of size num_models x num_models) for 'other' structure type."
        assert len(structure_matrix) == len(model_names), "Structure matrix size must match number of models."
        assert all(len(row) == len(model_names) for row in structure_matrix), "All rows in structure matrix must have length equal to number of models."
        # check if all cells are either 0 or 1
        for i in range(len(structure_matrix)):
            for j in range(len(structure_matrix)):
                assert structure_matrix[i][j] in [0,1], "Structure matrix can only contain 0 or 1."
    else:
        print("Using predefined structure type: {}".format(structure_type))
        if structure_matrix is not None:
            print("Warning: structure_matrix is provided but will be ignored since structure_type is not 'other'.")
    structure_matrix = generate_adj(len(model_names), structure_type, structure_matrix)

    # print model names and indices like 0: model_name[0],...
    model_dict = {i: model_names[i] for i in range(len(model_names))}
    print("Model indices and names: {}".format(model_dict))
    print("Number of rounds: {}".format(num_rounds))
    print("Structure type: {}".format(structure_type))
    print("Structure matrix: {}".format(structure_matrix.tolist()))



    # 3. select best model for final score based on the dev set of the dataset
    print("Evaluating models on dev set to select the best model...")
    dev_input_list = eval.prepare_inputs(task, task_type, "dev") # grab the inputs for the dev set
    # evaluate every model on it through distributed generation
    list_of_input_list = [dev_input_list for _ in model_names] # replicate the dev inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(dev_input_list)

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs) # send the outputs to the eval module to get a list of per-input scores
        avg_dev_score = sum(dev_score) / len(dev_score)
        list_of_dev_scores.append(avg_dev_score)
        print("Model: {}, dev {} score: {}".format(model_names[i], task, avg_dev_score))

    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    print("Best model selected for final generation: {}".format(best_model_name))


    # start the multi-round interaction between models
    print("Round 0: Starting multi-round interaction by generating initial outputs with each model...")
    test_input_list = eval.prepare_inputs(task, task_type, "test") # grab the inputs for the test set
    # evaluate every model on it through distributed generation
    list_of_input_list = [test_input_list for _ in model_names] # replicate the test inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(test_input_list)

    
    all_input_list = [list_of_input_list]
    all_output_list = [list_of_output_list] # round * len(model_names) * len(test_input_list)
    for interaction_round in range(num_rounds - 1):
        print("Round {}/{} (with interaction)".format(interaction_round + 1, num_rounds - 1))
        new_list_of_input_list = [] # len(model_names) * len(test_input_list)
        # new_list_of_output_list = [] # len(model_names) * len(test_input_list)
        no_in_edges_model_idx = []
        for i in range(len(model_names)):
            model_name = model_names[i]
            in_edges = structure_matrix[:, i]
            in_idxs = np.nonzero(in_edges)[0]
            if len(in_idxs) == 0:
                print("Model {} has no incoming edges, skipping regeneration.".format(model_name))
                new_list_of_input_list.append([])
                no_in_edges_model_idx.append(i)
                # new_list_of_output_list.append(all_output_list[-1][i])
            else:
                model_response_list = all_output_list[-1][i]
                neighbor_response_lists = [all_output_list[-1][j] for j in in_idxs]
                cur_model_input_list = []
                for k in range(len(test_input_list)):
                    neighbor_responses = [neighbor_response_lists[m][k] for m in range(len(neighbor_response_lists))]
                    new_prompt = re_generate_prompt(test_input_list[k], model_response_list[k], neighbor_responses)
                    cur_model_input_list.append(new_prompt)
                new_list_of_input_list.append(cur_model_input_list)
        new_list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            new_list_of_input_list,
            gpu_ids
        ) # will be size len(model_names) x len(test_input_list)
        all_input_list.append(new_list_of_input_list)
        all_output_list.append(new_list_of_output_list)
        print("no_in_edges_model_idx: {}".format(no_in_edges_model_idx))
        for i in range(len(model_names)):
            if len(all_output_list[-1][i]) == 0 and len(all_output_list) > 1:
                print("Model {}: {} did not regenerate, keeping previous outputs.".format(i, model_names[i]))
                all_output_list[-1][i] = all_output_list[-2][i][:] # keep the previous outputs if no regeneration happened




#####################################

    test_scores_dict = {}
    final_output_list = []
    avg_test_score = 0
    test_scores = []
    for i in range(len(model_names)):
        # print("Final outputs from model {}: {}".format(model_names[i], all_output_list[-1][i][:2])) # print first 3 outputs as a sample
        cur_test_outputs = all_output_list[-1][i]
        cur_test_score = eval.get_scores(task, task_type, "test", cur_test_outputs) #
        cur_avg_test_score = sum(cur_test_score) / len(cur_test_score)
        test_scores_dict[model_names[i]] = cur_avg_test_score
        if model_names[i] == best_model_name:
            final_output_list = cur_test_outputs
            test_scores = cur_test_score
            avg_test_score = cur_avg_test_score
        print("Model_{}: {}, final test {} score: {}".format(i, model_names[i], task, cur_avg_test_score))
    print("Best model for final output: {}, avg test {} score: {}".format(best_model_name, task, avg_test_score))
    
    if not final_output_list:
        print("CRITICAL ERROR: Best model output was not captured.")
        return 1

    # 5. save the logs
    # please follow the exact same format here
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_structure", 
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": [] # score the response, score, and other method-specific info for each test input
    }
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "output": final_output_list[i],
            "score": test_scores[i]
            # optionally, add other method-specific info here
        }
        experiment_logs["logs"].append(log_entry)
    
    # save to a json file
    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_text_structure.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

    # after that, you can use "method": "your_approach_name" in the config file to run your approach
    # if you ever saves anything other than the final log, make sure to save it in `model_collaboration/logs/<your_method_name>/`!
    # hooray, that's pretty much it!
    # for documentation of all the helper functions we provide, see `method/developer_readme.md`

if __name__ == "__main__":
    run_method()