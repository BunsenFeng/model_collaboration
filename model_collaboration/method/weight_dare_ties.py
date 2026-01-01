import os
import json
import random
from model_collaboration.data import eval
import torch.nn.functional as F
from model_collaboration.utils import lora_check
from model_collaboration.method import distributed_generation
from model_collaboration.utils.numeric_swarm import NumericSwarm

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

    # check if the models are lora adapters
    model_names = lora_check.lora_to_full(model_names)

    # method-specific hyperparameters

    base_model_name = hyperparameters.get("base_model_name")

    population = hyperparameters.get("population", 5)
    max_iterations = hyperparameters.get("max_iterations", 5)
    mode = hyperparameters.get("mode", "average") # optimized or average
    dare_ties_base_path = hyperparameters.get("dare_ties_base_path", "model_collaboration/logs/dare_ties/")

    dare_ties_base_path = dare_ties_base_path[:-1] + "_" + task + "/"
    if os.path.exists(dare_ties_base_path):
        dare_ties_base_path = dare_ties_base_path[:-1] + str(random.randint(0, 10000000)) + "/"
        print("Warning: dare_ties base path already exists. Using new path: {}".format(dare_ties_base_path))
        # raise ValueError("dare_ties_base_path {} already exists. Please specify a new path to avoid overwriting.".format(dare_ties_base_path))
    os.makedirs(dare_ties_base_path)

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

    if mode == "optimized":

        swarm = NumericSwarm(
            dimension=len(model_names),
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

        # optimize the weights of model merging on the dev set
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        for iteration in range(max_iterations):
            population_of_weights = swarm.get_particles() # [tensor(len(model_names)), ...]
            # softmax to ensure weights sum to 1
            list_of_normalized_weights = [F.softmax(weights, dim=0) for weights in population_of_weights]
            # turn it into list of lists
            list_of_weights = [weights.tolist() for weights in list_of_normalized_weights]

            # mergekit the models
            gpu_id = gpu_ids[0]
            for i in range(len(list_of_weights)):
                weight = list_of_weights[i]
                merged_model_path = dare_ties_base_path + "dare_ties_{}".format(i)
                with open(dare_ties_base_path + "dare_ties.yml", "w") as f:
                    f.write("models:\n")
                    for j in range(len(model_names)):
                        f.write("  - model: " + model_names[j] + "\n")
                        f.write("    parameters:\n")
                        f.write("      weight: " + str(weight[j]) + "\n")
                    f.write("merge_method: dare_ties\n")
                    f.write("base_model: " + base_model_name + "\n")
                    f.write("dtype: float16\n")
                
                os.system("mergekit-yaml " + dare_ties_base_path + "dare_ties.yml " + merged_model_path + " --cuda --device cuda:" + str(gpu_id))

            # evaluate the merged models on the dev set
            list_of_input_list = [dev_input_list for _ in range(len(list_of_weights))]
            list_of_model_names = [dare_ties_base_path + "dare_ties_{}".format(i) for i in range(len(list_of_weights))]
            list_of_output_list = distributed_generation.distributed_generation(
                list_of_model_names,
                list_of_input_list,
                gpu_ids
            )

            dev_scores = []
            for i in range(len(list_of_output_list)):
                dev_output = list_of_output_list[i]
                dev_score = eval.get_scores(task, task_type, "dev", dev_output)
                avg_dev_score = sum(dev_score) / len(dev_score)
                dev_scores.append(avg_dev_score)
                print("Iteration {}, particle {}: dev {} score: {}".format(iteration, i, task, avg_dev_score))

            # update the swarm
            terminate_signal = swarm.update(dev_scores)
            if terminate_signal:
                print("Swarm optimization reached patience at iteration {}.".format(iteration))
                break

        # get the best weights
        best_weights = swarm.get_global_best_particle()
        normalized_best_weights = F.softmax(best_weights, dim=0).tolist()
        print("Best weights found by dare-ties: {}".format(normalized_best_weights))
    
    elif mode == "average":
        # uniform weights
        normalized_best_weights = [1.0 / len(model_names)] * len(model_names)
        print("Using uniform weights for dare-ties: {}".format(normalized_best_weights))

    # merge the final model
    merged_model_path = dare_ties_base_path + "final_model"
    with open(dare_ties_base_path + "dare_ties.yml", "w") as f:
        f.write("models:\n")
        for j in range(len(model_names)):
            f.write("  - model: " + model_names[j] + "\n")
            f.write("    parameters:\n")
            f.write("      weight: " + str(normalized_best_weights[j]) + "\n")
        f.write("merge_method: dare_ties\n")
        f.write("base_model: " + base_model_name + "\n")
        f.write("dtype: float16\n")

    os.system("mergekit-yaml " + dare_ties_base_path + "dare_ties.yml " + merged_model_path)
    # os.system("mergekit-yaml " + dare_ties_base_path + "dare_ties.yml " + merged_model_path + " --cuda --device cuda:" + str(gpu_ids[0]))
    
    # evaluate it on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    list_of_input_list = [test_input_list]
    list_of_output_list = distributed_generation.distributed_generation(
        [merged_model_path],
        list_of_input_list,
        [gpu_ids[0]]
    )

    test_outputs = list_of_output_list[0]
    test_score = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_score) / len(test_score)
    print("dare-ties test {} score: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "best_weights": normalized_best_weights,
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
    log_filename = "model_collaboration/logs/{}_{}_{}_dare_ties.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()