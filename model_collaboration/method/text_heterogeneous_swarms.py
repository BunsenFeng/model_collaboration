import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from model_collaboration.data import eval
from multiprocessing import Pool
from model_collaboration.method import distributed_generation
from model_collaboration.utils.numeric_swarm import NumericSwarm

FIRST_INSTRUCTION = "Please answer the following question."
NON_LAST_INSTRUCTION = "Please answer the following question with the help of previous responses, feel free to ignore wrong or unhelpful responses."
LAST_INSTRUCTION = "Please answer the following question with the help of previous responses, feel free to ignore wrong or unhelpful responses. Make sure to provide a final answer."

def list_to_numpy_graph(list_of_floats):
    # turn a list of floats (size n*n) in the numeric swarm functionality
    # to a numpy adjacency matrix and set diagonal to zeros
    n = int(len(list_of_floats) ** 0.5)
    adjacency_matrix = np.array(list_of_floats).reshape((n, n))
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix

def softmax(probs):
    """
    Args:
        probs: np.ndarray, shape (n, ), probabilities.
    Returns:
        probs: np.ndarray, shape (n, ), probabilities after softmax.
    zero terms are not considered in the softmax operation.
    """
    probs = np.array(probs)
    probs = np.exp(probs)
    for i in range(len(probs)):
        if probs[i] == np.exp(0): # 0 means 0, no edge
            probs[i] = 0
    probs = probs / np.sum(probs)
    assert sum(probs) >= 0.999 and sum(probs) <= 1.001
    return probs

def top_p_sampling_selection(probs, top_p_threshold):
    """
    Args:
        probs: np.ndarray, shape (n, ), probabilities.
        top_p_threshold: float, threshold for top-p sampling-based operations.
    Returns:
        selected_indice: int, selected indice.
    sum up probs from high to low until reach top_p_threshold, renormalize probs with softmax, sample from probs and return the selected index.
    """
    assert sum(probs) >= 0.999 and sum(probs) <= 1.001
    prob_index_lists = list(zip(probs, range(len(probs))))
    prob_index_lists.sort(reverse = True)
    cum_prob = 0
    selected_indice = []
    for prob, index in prob_index_lists:
        cum_prob += prob
        selected_indice.append(index)
        if cum_prob > top_p_threshold:
            break
    selected_probs = [probs[i] for i in selected_indice]
    selected_probs = np.array(selected_probs)
    selected_probs = softmax(selected_probs)
    selected_index = np.random.choice(selected_indice, p = selected_probs)
    return selected_index

def graph_decode(adjacency_matrix, top_p_threshold = 0): # 0 for deterministic by default
    """
    Args:
        adjacency_matrix: np.ndarray, shape (n, n), continuous adjacency matrix.
        top_p_threshold: float, threshold for top-p sampling-based operations.
    Returns:
        discrete_adjacency_matrix: np.ndarray, shape (n, n), discrete adjacency matrix.
    """
    n = adjacency_matrix.shape[0]
    discrete_adjacency_matrix = np.zeros((n, n))
    remaining_nodes = list(range(n))
    existing_nodes = []

    # diagnoals must be zeros
    assert np.all(np.diag(adjacency_matrix) == 0)
    
    # select end point
    out_degrees = np.sum(adjacency_matrix, axis = 1)
    out_degrees = np.array([1 / value for value in out_degrees])
    out_degrees = softmax(out_degrees)
    end_point = top_p_sampling_selection(out_degrees, top_p_threshold)
    existing_nodes.append(end_point)
    remaining_nodes.remove(end_point)

    # iteratively selecting and adding one point
    while len(remaining_nodes) > 0:
        out_degrees = np.sum(adjacency_matrix, axis = 1)
        # existing node to 0
        for node in existing_nodes:
            out_degrees[node] = 0
        out_degrees = softmax(out_degrees)
        selected_node = top_p_sampling_selection(out_degrees, top_p_threshold)
        # select an existing node to connect to
        out_degree_to_existing_nodes = adjacency_matrix[selected_node]
        for node in remaining_nodes:
            out_degree_to_existing_nodes[node] = 0
        out_degree_to_existing_nodes = softmax(out_degree_to_existing_nodes)
        selected_existing_node = top_p_sampling_selection(out_degree_to_existing_nodes, top_p_threshold)
        # update the state
        discrete_adjacency_matrix[selected_node, selected_existing_node] = 1
        existing_nodes.append(selected_node)
        # print(selected_node, remaining_nodes)
        remaining_nodes.remove(selected_node)

    # the one with no out degrees is the end point
    # the one(s) with no in degrees is the start point
    return discrete_adjacency_matrix

def topological_sort(discrete_adjacency_matrix):
    """
    Args:
        discrete_adjacency_matrix: np.ndarray, shape (n, n), discrete adjacency matrix.
    Returns:
        topological_order: list, topological order of the graph.
    """
    n = discrete_adjacency_matrix.shape[0]
    in_degrees = np.sum(discrete_adjacency_matrix, axis = 0)
    out_degrees = np.sum(discrete_adjacency_matrix, axis = 1)
    topological_order = []
    while len(topological_order) < n:
        for node in range(n):
            if in_degrees[node] == 0:
                topological_order.append(node)
                in_degrees[node] = -1
                for i in range(n):
                    if discrete_adjacency_matrix[node, i] == 1:
                        in_degrees[i] -= 1
    return topological_order

def graph_generate(prompts, adjacency_matrix, model_names, gpu_id, max_response_length, temperature, top_p, batch_size, assignment=None, graph_top_p=0.7):
    if assignment is None:
        assignment = list(range(len(model_names)))
    assert len(adjacency_matrix) == len(assignment) == len(model_names)

    # decode graph
    graph_decoded = graph_decode(adjacency_matrix, graph_top_p)
    topological_order = topological_sort(graph_decoded)

    # bookkeeping
    intermediate_outputs = ["" for _ in range(len(adjacency_matrix))]

    for i in range(len(topological_order)):
        node = topological_order[i]
        model_name = model_names[assignment[node]]
        
        prompts_now = []
        if sum(graph_decoded[:, node]) == 0: # start node
            for _ in range(len(prompts)):
                prompts_now.append(FIRST_INSTRUCTION + "\n" + prompts[_])
        elif i == len(topological_order) - 1: # end node
            assert sum(graph_decoded[node, :]) == 0
            previous_nodes = [j for j in range(len(adjacency_matrix)) if graph_decoded[j, node] == 1]
            concatenated_previous_outputs = []
            for _ in range(len(prompts)):
                concat_output = ""
                for prev_node in previous_nodes:
                    concat_output += "Previous response from node " + str(prev_node + 1) + ": " + intermediate_outputs[prev_node][_]
                    concat_output += "\n\n"
                concatenated_previous_outputs.append(concat_output)
                prompts_now.append(LAST_INSTRUCTION + "\n" + concatenated_previous_outputs[_] + prompts[_])
        else: # non-starting and non-ending node
            previous_nodes = [j for j in range(len(adjacency_matrix)) if graph_decoded[j, node] == 1]
            concatenated_previous_outputs = []
            for _ in range(len(prompts)):
                concat_output = ""
                for prev_node in previous_nodes:
                    concat_output += "Previous response from node " + str(prev_node + 1) + ": " + intermediate_outputs[prev_node][_]
                    concat_output += "\n\n"
                concatenated_previous_outputs.append(concat_output)
                prompts_now.append(NON_LAST_INSTRUCTION + "\n" + concatenated_previous_outputs[_] + prompts[_])

        # generate outputs
        outputs_now = distributed_generation.batch_generate_text(
            model_names[assignment[node]],
            gpu_id,
            prompts_now,
            max_response_length,
            temperature,
            top_p,
            batch_size
        ) # size: len(prompts)

        # outputs_now = distributed_generation.distributed_generation(
        #     [model_names[assignment[node]]],
        #     [prompts_now],
        #     [gpu_id]
        # )[0] # size: len(prompts)

        # store outputs
        intermediate_outputs[node] = outputs_now

    # final outputs
    final_outputs = intermediate_outputs[topological_order[-1]]
    return final_outputs

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    max_response_length = hyperparameters.get("max_response_length")
    temperature = hyperparameters.get("temperature")
    top_p = hyperparameters.get("top_p")
    batch_size = hyperparameters.get("batch_size")

    # method-specific hyperparameters
    population = hyperparameters.get("population", 5)
    max_iterations = hyperparameters.get("max_iterations", 5)

    starting_velocity_mode = hyperparameters.get("starting_velocity_mode", "random")
    weight_randomness = hyperparameters.get("weight_randomness", True)
    inertia = hyperparameters.get("inertia", 0.2)
    cognitive_coeff = hyperparameters.get("cognitive_coeff", 0.3)
    social_coeff = hyperparameters.get("social_coeff", 0.4)
    repel_coeff = hyperparameters.get("repel_coeff", 0.05)
    repel_term = hyperparameters.get("repel_term", True)
    step_length = hyperparameters.get("step_length", 0.5)
    step_length_factor = hyperparameters.get("step_length_factor", 0.95)
    minimum_step_length = hyperparameters.get("minimum_step_length", 0.1)
    patience = hyperparameters.get("patience", 5)
    restart_patience = hyperparameters.get("restart_patience", 3)

    # initialize the swarm
    swarm = NumericSwarm(
        dimension=len(model_names) * len(model_names), # adjacency matrix size
        population=population,
        starting_velocity_mode=starting_velocity_mode,
        weight_randomness=weight_randomness,
        inertia=inertia,
        cognitive_coeff=cognitive_coeff,
        social_coeff=social_coeff,
        repel_coeff=repel_coeff,
        step_length=step_length,
        repel_term=repel_term,
        step_length_factor=step_length_factor,
        minimum_step_length=minimum_step_length,
        patience=patience,
        restart_patience=restart_patience
    )

    # optimize the graph structure on the dev set
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    for iteration in range(max_iterations):
        population_of_graphs = swarm.get_particles() # [tensor(len(model_names)*len(model_names)), ...]
        list_of_adjacency_matrices = [list_to_numpy_graph(graph.tolist()) for graph in population_of_graphs]

        list_of_output_list = [] # size: population * len(dev_input_list)

        for i in range(0, len(list_of_adjacency_matrices), len(gpu_ids)):

            generation_args = []

            for j in range(len(gpu_ids)):
                if i + j < len(list_of_adjacency_matrices):
                    generation_args.append((
                        dev_input_list,
                        list_of_adjacency_matrices[i + j],
                        model_names,
                        gpu_ids[j],
                        max_response_length,
                        temperature,
                        top_p,
                        batch_size
                    ))
            
            pool = Pool(len(generation_args))
            output = pool.starmap(graph_generate, generation_args) # size len(generation_args) * len(dev_input_list)
            pool.close()
            pool.join()

            for out in output:
                list_of_output_list.append(out)

        assert len(list_of_output_list) == len(list_of_adjacency_matrices)
        assert len(list_of_output_list[0]) == len(dev_input_list)

        # evaluate the outputs
        list_of_scores = []
        for i in range(len(list_of_output_list)):
            outputs = list_of_output_list[i]
            score = eval.get_scores(task, task_type, "dev", outputs)
            list_of_scores.append(sum(score) / len(score)) # average score over the dev set
        assert len(list_of_scores) == len(list_of_adjacency_matrices)

        # update the swarm
        terminal_signal = swarm.update(list_of_scores)
        if terminal_signal:
            print("Early stopping at iteration {}".format(iteration))
            break
    
    # evaluate the best graph on the test set
    best_graph = swarm.get_global_best_particle().tolist()
    best_adjacency_matrix = list_to_numpy_graph(best_graph)
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    test_outputs = graph_generate(
        test_input_list,
        best_adjacency_matrix,
        model_names,
        gpu_ids[0],
        max_response_length,
        temperature,
        top_p,
        batch_size
    )

    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("H-Swarm test {} score: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "best_adjacency_matrix": best_adjacency_matrix.tolist(),
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        experiment_logs["logs"].append({
            "input": test_input_list[i],
            "output": test_outputs[i],
            "score": test_scores[i]
        })

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_heterogeneous_swarms.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()

#     distributed_generation.update_generation_hyperparameters(50, 0.7, 0.9, 4)

#     prompts = ["What is the capital of France?", "Who wrote 'Pride and Prejudice'?", "What is the largest planet in our solar system?"]
#     adjacency_matrix = np.random.rand(3, 3)
#     np.fill_diagonal(adjacency_matrix, 0) # no self-loops
#     model_names = [
#        "bunsenfeng/yuru_qw_wizardlm",
#        "bunsenfeng/yuru_qw_sharegpt",
#        "bunsenfeng/yuru_qw_oasst1"
#    ]
#     gpu_id = 0
#     outputs = graph_generate(prompts, adjacency_matrix, model_names, gpu_id)
#     print(outputs)