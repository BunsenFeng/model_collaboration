"""
A blank template for implementing your approach.
"""
import json
from data import eval
from method import distributed_generation

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    assert task_type in ["multiple_choice", "exact_match", "f1_match"], "This method only supports multiple_choice, exact_match, and f1_match types of tasks."

    # evaluate on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test", 0.1) # grab the inputs for the test set

    list_of_input_list = [test_input_list for _ in model_names] # replicate the test inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(test_input_list)

    list_of_extracted_answers = []
    for output_list in list_of_output_list:
        extracted_answers = eval.get_extracted_answers(task, task_type, "test", output_list)
        list_of_extracted_answers.append(extracted_answers)

    majority_vote_answers = []
    for i in range(len(test_input_list)):
        extracted_answers = [list_of_extracted_answers[j][i] for j in range(len(model_names))]
        majority_vote_answers.append(max(set(extracted_answers), key=extracted_answers.count))

    # evaluate the final outputs
    test_scores = eval.get_scores(task, task_type, "test", majority_vote_answers)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score of majority vote: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_majority_vote",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "raw_output": list_of_extracted_answers[i],
            "output": majority_vote_answers[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)
    
    # save to a json file
    log_filename = "logs/{}_{}_{}_text_majority_vote.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()