"""
A blank template for implementing your approach.
"""
import json
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
# optionally, `from utils import distributed_sft` if your approach finetunes multiple models
# optionally, `from utils import logit_arithmetic` if your approach is composing the logits of multiple models
# optionally, `from utils.numeric_swarm import NumericSwarm` if your approach optimizes continuous vectors of hyperparameters/weights

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

    # preamble, to be compatible with pypi package format
    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # 1. optionally, extract the general hyperparameters from the hyperparameters dict
    # these four are included in any config file by default
    # this is useful if you are handling generation/finetuning without the (amazing) helper functions provided
    # these generation args will be auto-configured for the helper functions

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
    # if you ever saves anything during this step, make sure to save it in `model_collaboration/logs/<your_method_name>/`!

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
    log_filename = "model_collaboration/logs/{}_{}_{}_your_approach_name.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

    # after that, you can use "method": "your_approach_name" in the config file to run your approach
    # if you ever saves anything other than the final log, make sure to save it in `model_collaboration/logs/<your_method_name>/`!
    # hooray, that's pretty much it!
    # for documentation of all the helper functions we provide, see `method/developer_readme.md`

if __name__ == "__main__":
    run_method()