"""
LLM API level cascade
"""
import json
from model_collaboration.data import eval
import torch
from model_collaboration.method import distributed_generation

def just_ask_prompt(input):
    # ask the model to say whether they are unconfident about their answers
    prompt = f"User: ### Question: {input}\n"
    prompt += "For the above question, please answer it to your best effort.\n"
    prompt += "If you are not confident of your answer, please say you are unconfident after your answer.\n"
    return prompt

def score_confidence(logit_score):
    if len(logit_score) != 0:
        return sum(logit_score) / len(logit_score) 
    else:
        return 0.0

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    print("The model you are using are:")
    for i in range(len(model_names)):
        print(f"Index {i + 1}: {model_names[i]}")
    print("Make sure they are in cascading structure.")

    # extract the general hyperparameters from the hyperparameters dict
    max_response_length = hyperparameters.get("max_response_length", 256)
    temperature = hyperparameters.get("temperature", 0.7)
    top_p = hyperparameters.get("top_p", 0.9)
    batch_size = hyperparameters.get("batch_size", 8)

    # extract the method-specific hyperparameters from the hyperparameters dict
    mode = hyperparameters.get("mode", "logit") # confidence: `logit` or `just_ask`
    assert mode == "logit" or mode == "just_ask", "unsupported mode"
    percentage = hyperparameters.get("percentage", 0.5) # threshold percentage, default Top 50%

    print(f"Current mode: {mode}")

    # Determine threshold
    if mode == "logit":
        threshold_list = []
        print(f"Finding threshold in dev set under logit mode")
        dev_input_list = eval.prepare_inputs(task, task_type, "dev")
        for i in range(len(model_names)-1): # find threshld for every model except last model
            output_list, list_logit_scores_list = distributed_generation.batch_generate_text_with_score(
                        model_name=model_names[0],
                        gpu_id=gpu_ids[0],
                        input_list=dev_input_list,
                        max_response_length=max_response_length,
                        temperature=temperature,
                        top_p=top_p,
                        batch_size=batch_size,
                    )
            # record logit scores
            dev_scores = []
            for k in range(len(output_list)):
                dev_scores.append(score_confidence(list_logit_scores_list[k]))
            # select threshold
            print(f"For model {model_names[i]}, dev set max confidence {max(dev_scores)}, min confidence {min(dev_scores)}")
            threshold = torch.quantile(torch.tensor(dev_scores), percentage) # Top (percentage * 100) % threshold
            print(f"For model {model_names[i]}, select threshold {threshold} for Top {percentage * 100} % elements")
            threshold_list.append(threshold)

    # prepare test set input
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    # record each problem deferral state
    unsolved_dict = {i: True for i in range(len(test_input_list))}
    answer_dict = {i: "" for i in range(len(test_input_list))}
    # let's implement a simple LLM cascade pipeline
    # 1. the first model (weak) generates its response
    # 2. deferral rule: 
    # - `logit`: calculate from output sequence each token probability
    # - `just_ask`: add prompt asking weak LLM to say if unconfident about answer, if unconfident, defer
    # 3. the second model (strong) generates its response
    # then the best model from dev set generates the final response based on all model responses
    for i in range(len(model_names)):
        print(f"Cascade Level {i}, Model {model_names[i]}")
        # current unsolved problems
        current_test_input_list = []
        for index, unsolved in unsolved_dict.items():
            if unsolved:
                current_test_input_list.append(test_input_list[index]) 
        print(f"Current problems number: {len(current_test_input_list)}")
        # not the final model
        if i != len(model_names) - 1:
            if mode == "logit":
                # return output and logit scores list
                output_list, list_logit_scores_list = distributed_generation.batch_generate_text_with_score(
                    model_name=model_names[i],
                    gpu_id=gpu_ids[0],
                    input_list=current_test_input_list,
                    max_response_length=max_response_length,
                    temperature=temperature,
                    top_p=top_p,
                    batch_size=batch_size,
                )
                # need_defer_test_list = []
                count = 0
                for index, unsolved in unsolved_dict.items():
                    if unsolved:
                        current_threshld = threshold_list[i]
                        if current_threshld and score_confidence(list_logit_scores_list[count]) >= current_threshld: # solved
                            unsolved_dict[index] = False
                            answer_dict[index] = output_list[count] # accept as final answer
                            count += 1
                        else:
                            count += 1
                
            elif mode == "just_ask":
                # add just ask prompt
                input_list_with_ask = []
                for k in range(len(current_test_input_list)):
                    input_list_with_ask.append(just_ask_prompt(current_test_input_list[k]))

                # generate responses
                list_of_output_list = distributed_generation.distributed_generation(
                    [model_names[i]],
                    [input_list_with_ask],
                    [gpu_ids[0]]
                )
                output_list = list_of_output_list[0]

                # find problems need deferral
                count = 0
                for index, unsolved in unsolved_dict.items():
                    if unsolved:
                        if "unconfident" in output_list[count]: # need next model
                            count += 1
                        else:
                            unsolved_dict[index] = False
                            answer_dict[index] = output_list[count] # accept as final answer
                            count += 1
        # final model
        else:
            # generate responses
            list_of_output_list = distributed_generation.distributed_generation(
                [model_names[i]],
                [current_test_input_list],
                [gpu_ids[0]]
            )
            output_list = list_of_output_list[0]

            # find problems need deferral
            count = 0
            for index, unsolved in unsolved_dict.items():
                if unsolved:
                    unsolved_dict[index] = False
                    answer_dict[index] = output_list[count] # accept as final answer
                    count += 1
    
    # combine first and second round answers to final output
    final_output_list = []
    for index, answer in answer_dict.items():
        if unsolved_dict[index] == False:
            final_output_list.append(answer)
    assert len(test_input_list) == len(final_output_list), "length of test_input_list and final_output_list is not same"

    # evaluate the final outputs
    test_scores = eval.get_scores(task, task_type, "test", final_output_list)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score of the approach: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "api_cascade", # CHANGE!
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "output": final_output_list[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)
    
    # save to a json file
    log_filename = "model_collaboration/logs/{}_{}_{}_api_cascade.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()