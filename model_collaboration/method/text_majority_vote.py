import json
import os
import random
from collections import defaultdict
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def get_extracted_answers(task, task_type, split, outputs, ratio=1.0):

    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as f:
        data = json.load(f)[split]
        data = data[:int(len(data)*ratio)]

    extracted_answers = []

    if task_type == "multiple_choice":
        assert "choices" in data[0], "Are you sure this is a multiple choice task?"
        for item, output in zip(data, outputs):
            options = []
            for option in item["choices"].keys():
                options.append(item["choices"][option])
            chosen_letter, _ = eval.parse_model_response_mcq(output, options)
            extracted_answers.append(chosen_letter)
    if task_type == "exact_match":
        for output in outputs:
            extracted_answer = eval.extract_answer_text(output)
            extracted_answers.append(extracted_answer)
    if task_type == "f1_match":
        for output in outputs:
            extracted_answer = eval.extract_answer_text(output)
            extracted_answers.append(extracted_answer)
    
    return extracted_answers

def get_scores_from_extracted_answers(task, task_type, split, extracted_answers, ratio=1.0):

    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as f:
        data = json.load(f)[split]
        data = data[:int(len(data)*ratio)]

    scores = []

    if task_type == "multiple_choice":
        assert "choices" in data[0], "Are you sure this is a multiple choice task?"
        for item, extracted_answer in zip(data, extracted_answers):
            if extracted_answer is None:
                scores.append(0.0)
            else:
                if item["answer"] == extracted_answer:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
    if task_type == "exact_match":
        for item, extracted_answer in zip(data, extracted_answers):
            em_score = eval.calculate_exact_match(extracted_answer, item["output"])
            scores.append(em_score)
    if task_type == "f1_match":
        if task == "popqa":
            # parse string of list "[\"Akkineni Nagarjuna\", \"Nagarjuna Akkineni\", \"Nagarjuna\", \"Akkineni Nagarjuna Rao\"]" into a list
            for item, extracted_answer in zip(data, extracted_answers):
                string_of_list = item["output"]
                string_of_list = string_of_list.replace("[", "").replace("]", "").replace("\"", "").replace("'", "")
                options = [option.strip() for option in string_of_list.split(",")]
                max_f1_match = 0.0
                for option in options:
                    f1_match = eval.calculate_f1_score(extracted_answer, option)
                    if f1_match > max_f1_match:
                        max_f1_match = f1_match
                scores.append(max_f1_match)
        else:
            for item, extracted_answer in zip(data, extracted_answers):
                f1_score = eval.calculate_f1_score(extracted_answer, item["output"])
                scores.append(f1_score)
    return scores

def evaluate_models_on_dev(task, task_type, model_indices, model_names, gpu_ids, dev_scores_cache, dev_input_list):

    # find models that need evaluation
    models_to_evaluate = [i for i in model_indices if i not in dev_scores_cache]
    if not models_to_evaluate:
        return
    
    # evaluate only the models that need evaluation
    models_to_evaluate_names = [model_names[i] for i in models_to_evaluate]
    
    list_of_input_list = [dev_input_list for _ in models_to_evaluate_names]
    list_of_output_list = distributed_generation.distributed_generation(
        models_to_evaluate_names,
        list_of_input_list,
        gpu_ids
    )
    
    # cache the scores
    for idx, model_idx in enumerate(models_to_evaluate):
        dev_outputs = list_of_output_list[idx]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        avg_dev_score = sum(dev_score) / len(dev_score)
        dev_scores_cache[model_idx] = avg_dev_score
        print("Model: {} (index {}), dev {} score: {}".format(model_names[model_idx], model_idx, task, avg_dev_score))

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    assert task_type in ["multiple_choice", "exact_match", "f1_match"], "This method only supports multiple_choice, exact_match, and f1_match types of tasks."

    tie_breaking = hyperparameters.get("tie", "random")
    assert tie_breaking in ["random", "dev-based"], "tie parameter must be either 'random' or 'dev-based'"

    if tie_breaking == "dev-based":
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        dev_scores_cache = {}

    # evaluate on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test") # grab the inputs for the test set

    list_of_input_list = [test_input_list for _ in model_names] # replicate the test inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(test_input_list)
    
    list_of_extracted_answers = []
    for output_list in list_of_output_list:
        extracted_answers = get_extracted_answers(task, task_type, "test", output_list)
        list_of_extracted_answers.append(extracted_answers)
    
    majority_vote_answers = []
    for i in range(len(test_input_list)):
        extracted_answers = [answers[i] for answers in list_of_extracted_answers]
        
        answer_counts = defaultdict(list)
        for j, answer in enumerate(extracted_answers):
            answer_counts[answer].append(j)
        
        max_count = max(len(indices) for indices in answer_counts.values())
        tied_answers = [answer for answer, indices in answer_counts.items() if len(indices) == max_count]
        
        if len(tied_answers) == 1:
            majority_vote_answers.append(tied_answers[0])
        elif tie_breaking == "random":
            majority_vote_answers.append(random.choice(tied_answers))
        elif tie_breaking == "dev-based":
            # get all model indices that voted for tied answers
            tied_model_indices = [idx for answer in tied_answers for idx in answer_counts[answer]]
            
            # evaluate tied models on dev set (only if not already cached)
            evaluate_models_on_dev(
                task, task_type, tied_model_indices, model_names, gpu_ids, 
                dev_scores_cache, dev_input_list
            )
            
            # find the best-performing model (on dev set) among those that voted for tied answers
            best_tied_model_index = max(tied_model_indices, key=lambda idx: dev_scores_cache[idx])
            majority_vote_answers.append(extracted_answers[best_tied_model_index])
    
    # evaluate the final outputs
    test_scores = get_scores_from_extracted_answers(task, task_type, "test", majority_vote_answers)
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
            "raw_output": [extracted_answers[i] for extracted_answers in list_of_extracted_answers],
            "output": majority_vote_answers[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)
    
    # save to a json file
    log_filename = "model_collaboration/logs/{}_{}_{}_majority_vote.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()