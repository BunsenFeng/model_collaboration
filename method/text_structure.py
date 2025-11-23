"""
A blank template for implementing your approach.
"""
import json
from data import eval
from method import distributed_generation
import numpy as np

# optionally, `from utils import distributed_sft` if your approach finetunes multiple models
# optionally, `from utils import logit_arithmetic` if your approach is composing the logits of multiple models
# optionally, `from utils.numeric_swarm import NumericSwarm` if your approach optimizes continuous vectors of hyperparameters/weights
"""
{
    "method": "api_prompt_routing", // the name under method/ folder, names like <type>_<approach>.py
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
        "structure_type": "your_structure_type", // chain, tree, star, circle, complete, other
        "structure_matrix": [] // optional, only applicable if structure_type is "other", matrix of size num_models x num_models defining the structure
        "output_model": ["model_1_name"] // optional, a list of model names to record final outputs from (default: all models)
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

    # 1. extract the general hyperparameters from the hyperparameters dict
    structure_type = hyperparameters.get("structure_type", "not_supported")
    structure_matrix = hyperparameters.get("structure_matrix", None)
    output_model = hyperparameters.get("output_model", model_names)
    assert structure_type in ["chain", "tree", "star", "circle", "complete", "other"], "Invalid structure type, please provide one of the supported types: chain, tree, star, circle, complete, other."
    if structure_type == "other":
        assert structure_matrix is not None, "Please provide a structure matrix (list of lists of size num_models x num_models) for 'other' structure type."
        assert len(structure_matrix) == len(model_names), "Structure matrix size must match number of models."
        assert len(structure_matrix[0]) == len(model_names), "Structure matrix must be square with size equal to number of models."
        # check if all cells are either 0 or 1
        for i in range(len(structure_matrix)):
            for j in range(len(structure_matrix)):
                assert structure_matrix[i][j] in [0,1], "Structure matrix can only contain 0 or 1."
    else:
        print("Using predefined structure type: {}".format(structure_type))
        if structure_matrix is not None:
            print("Warning: structure_matrix is provided but will be ignored since structure_type is not 'other'.")
    structure_matrix = generate_adj(len(model_names), structure_type, structure_matrix)
    assert all([model in model_names for model in output_model]), "All output models must be in the list of model names."

    # print model names and indices like 0: model_name[0],...
    model_dict = {i: model_names[i] for i in range(len(model_names))}
    print("Model indices and names: {}".format(model_dict))
    print("Structure type: {}".format(structure_type))
    print("Structure matrix: {}".format(structure_matrix.tolist()))
    print("Output models: {}".format(output_model))
    



    # max_response_length = hyperparameters.get("max_response_length")
    # temperature = hyperparameters.get("temperature")
    # top_p = hyperparameters.get("top_p")
    # batch_size = hyperparameters.get("batch_size")

    # 2. optionally, extract the method-specific hyperparameters from the hyperparameters dict
    # users would pass this in via the config file, through the "hyperparameters" dict
    # you need to tell users to set them in the readme
    # you should also provide a default value (in most cases)

    # method_specific_hyperparameter_a = hyperparameters.get("method_specific_hyperparameter_a", default_value_a)
    # method_specific_hyperparameter_b = hyperparameters.get("method_specific_hyperparameter_b", default_value_b)
    # no default value, will throw an error if the user doesn't provide it in the config file
    # method_specific_hyperparameter_c = hyperparameters.get("method_specific_hyperparameter_c")

    # 3. optionally, do something based on the dev set of the dataset
    # could be: selecting a model as the summarizer/evaluator/... based on dev performance
    # could be: finetuning the models somehow on the dev set
    # could be: setting some sort of hyperparameter/threshold based on the dev set
    # it's ok that your approach doesn't have this step, e.g. multiagent debate
    # if you ever saves anything during this step, make sure to save it in `logs/<your_method_name>/`!

    # a most simple example, select the best model based on dev set performance
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
    # you can then use best_model_name somehow in the next step

    # 4. evaluate the approach on the test set
    # based on the stuff you did in the dev set, you arrived at some final approach
    # generate responses with it, evaluate it, do logging

    test_input_list = eval.prepare_inputs(task, task_type, "test") # grab the inputs for the test set

    # let's implement a multi-agent summary approach
    # each model generates their own response
    # then the best model from dev set generates the final response based on all model responses

    list_of_input_list = [test_input_list for _ in model_names] # replicate the test inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(test_input_list)

    # now have the best model generate the final responses based on all model responses
    final_input_list = []
    for i in range(len(test_input_list)):
        prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to generate the final answer.\n\n"
        prompt += "Question: {}\n\n".format(test_input_list[i])
        prompt += "Assistants' answers:\n"
        for j in range(len(model_names)):
            prompt += "- {}\n".format(list_of_output_list[j][i])
        prompt += "\nPlease provide the final answer to the question."
        final_input_list.append(prompt)

    # you can generate with a single model using distributed_generation too
    # just pass [model], [input_list], [gpu_id] to it
    final_output_list = distributed_generation.distributed_generation(
        [best_model_name],
        [final_input_list],
        [gpu_ids[0]] # just use the first GPU for the final generation
    )[0] # get the only output list, [0] is important because the output is list of list and [0] takes the list out

    # evaluate the final outputs
    test_scores = eval.get_scores(task, task_type, "test", final_output_list)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score of the approach: {}".format(task, avg_test_score))

    # 5. save the logs
    # please follow the exact same format here
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "your_approach_name", # CHANGE!
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
    # CHANGE! your method name
    log_filename = "logs/{}_{}_{}_your_approach_name.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

    # after that, you can use "method": "your_approach_name" in the config file to run your approach
    # if you ever saves anything other than the final log, make sure to save it in `logs/<your_method_name>/`!
    # hooray, that's pretty much it!
    # for documentation of all the helper functions we provide, see `method/developer_readme.md`

if __name__ == "__main__":
    run_method()