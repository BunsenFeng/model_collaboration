import os
import json
import random
from model_collaboration.data import eval
from model_collaboration.utils import swarm
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

    # method-specific hyperparameters
    swarm_base_path = hyperparameters.get("swarm_base_path", "model_collaboration/logs/model_swarms/")
    swarm_base_path = swarm_base_path[:-1] + "_" + task + "/"
    if os.path.exists(swarm_base_path):
        swarm_base_path = swarm_base_path[:-1] + str(random.randint(0, 10000000)) + "/"
        print("Warning: Swarm base path already exists. Using new path: {}".format(swarm_base_path))
        # raise ValueError("Swarm base path {} already exists. Please specify a new path to avoid overwriting.".format(swarm_base_path))
    base_model = hyperparameters.get("base_model", None)
    if base_model is None:
        print("Taking the first model as the base model. Hopefully it is the one with the largest vocabulary size and thus embedding/lm head matrices.")
        print("Please specify the base_model in hyperparameters to avoid this warning.")
        base_model = model_names[0]
    fast_merge_flag = hyperparameters.get("fast_merge_flag", False)
    weight_randomness_flag = hyperparameters.get("weight_randomness_flag", True)
    inertia = hyperparameters.get("inertia", 0.2)
    cognitive_coefficient = hyperparameters.get("cognitive_coefficient", 0.3)
    social_coefficient = hyperparameters.get("social_coefficient", 0.4)
    repel_coefficient = hyperparameters.get("repel_coefficient", 0.05)
    step_length = hyperparameters.get("step_length", 0.5)
    repel_term_flag = hyperparameters.get("repel_term_flag", True)
    step_length_factor = hyperparameters.get("step_length_factor", 0.95)
    minimum_step_length = hyperparameters.get("minimum_step_length", 0.1)
    patience = hyperparameters.get("patience", 5)
    restart_patience = hyperparameters.get("restart_patience", 3)
    max_iterations = hyperparameters.get("max_iterations", 10)

    # initialize the swarm
    model_swarm = swarm.Swarm(
        swarm_base_path = swarm_base_path,
        model_paths = model_names,
        base_model = base_model,
        fast_merge = fast_merge_flag,
        starting_velocity_mode = "random",
        weight_randomness = weight_randomness_flag,
        inertia = inertia,
        cognitive_coeff = cognitive_coefficient,
        social_coeff = social_coefficient,
        repel_coeff = repel_coefficient,
        step_length = step_length,
        repel_term = repel_term_flag,
        step_length_factor = step_length_factor,
        minimum_step_length = minimum_step_length,
        gpus = gpu_ids,
        patience = patience,
        restart_patience = restart_patience
    )

    # run the swarm optimization
    for iter in range(max_iterations):
        print("Swarm optimization iteration {}/{}".format(iter+1, max_iterations))
        
        # evaluate the swarm of models on the dev set
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        model_paths = model_swarm.get_model_paths()
        list_of_input_list = [dev_input_list for _ in model_paths]
        list_of_output_list = distributed_generation.distributed_generation(
            model_paths,
            list_of_input_list,
            gpu_ids
        )
        list_of_dev_scores = []
        for i in range(len(model_paths)):
            dev_outputs = list_of_output_list[i]
            dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
            avg_dev_score = sum(dev_score) / len(dev_score)
            list_of_dev_scores.append(avg_dev_score)
            print("Model: {}, dev {} score: {}".format(model_paths[i], task, avg_dev_score))
        
        # update the swarm based on the dev scores
        termination_flag = model_swarm.update(list_of_dev_scores)
        if termination_flag:
            print("Early stopping triggered. Ending swarm optimization.")
            break
        
    model_swarm.clean_up()

    # evaluate the global best model on the test set
    best_model_path = model_swarm.get_global_best_path()
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    test_outputs = distributed_generation.distributed_generation(
        [best_model_path],
        [test_input_list],
        [gpu_ids[0]]
    )
    test_outputs = test_outputs[0]

    test_scores = eval.get_scores(task, task_type, "test", test_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)

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
            "output": test_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)
    
    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_model_swarms.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()