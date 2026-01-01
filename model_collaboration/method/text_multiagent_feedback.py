import json
import random
from model_collaboration.data import eval
from model_collaboration.method import distributed_generation

def feedback_prompt_generator(input, output):
    prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Below is one assistant's answer to the question. Provide constructive feedback to help improve the answer.\n\n"
    prompt += "Question: {}\n\n".format(input)
    prompt += "Assistant's Answer: {}\n\n".format(output)
    prompt += "Please provide your feedback."
    return prompt

def refine_based_on_feedback_prompt_generator(input, output, feedbacks):
    prompt = "You are part of a team of AI assistants collaborating to answer the user's question. Below is your previous answer along with feedback from other assistants. Use the feedback to refine and improve your answer.\n\n"
    prompt += "Question: {}\n\n".format(input)
    prompt += "Your Previous Answer: {}\n\n".format(output)
    prompt += "Feedback from other assistants:\n"
    for feedback in feedbacks:
        prompt += "- {}\n".format(feedback)
    prompt += "\nPlease provide a refined answer to the question."
    return prompt

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # method-specific hyperparameters
    rounds = hyperparameters.get("round", 3)
    feedback_count = hyperparameters.get("feedback_count", 3)

    # selecting a model as the final summarizer based on performance on the dev set
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
        print("Model: {}, dev {} score: {}".format(model_names[i], task_type, avg_dev_score))

    best_model_index = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_index]
    print("Best model selected for final summarization: {}".format(best_model_name))

    # multiagent feedback on the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    response_list = None # len(model_names) * len(test_input_list)
    for r in range(rounds):
        print("Round {}/{}".format(r+1, rounds))
        list_of_input_list = []
        if r == 0:
            for _ in model_names:
                list_of_input_list.append(test_input_list)
            list_of_output_list = distributed_generation.distributed_generation(
                model_names,
                list_of_input_list,
                gpu_ids
            )
            response_list = list_of_output_list
        else:
            assert response_list is not None, "Response list should not be None in round {}".format(r)
            # randomly select feedback_count other models to provide feedback on one model's response
            feedback_list = [] # len(model_names) * len(test_input_list) * feedback_count
            feedback_indices_list = [] # len(model_names) * len(test_input_list) * feedback_count
            for i in range(len(model_names)):
                feedback_indices = []
                for _ in range(len(test_input_list)):
                    other_indices = list(range(len(model_names)))
                    other_indices.remove(i)
                    selected_indices = random.sample(other_indices, min(feedback_count, len(other_indices)))
                    feedback_indices.append(selected_indices)
                feedback_indices_list.append(feedback_indices)
            
            for i in range(len(model_names)): # the model that receives feedback
                list_of_input_list = [[] for _ in range(len(model_names))]
                for j in range(len(test_input_list)):
                    for k in feedback_indices_list[i][j]: # the models that provide feedback
                        prompt = feedback_prompt_generator(test_input_list[j], response_list[i][j])
                        list_of_input_list[k].append(prompt)
                list_of_output_list = distributed_generation.distributed_generation(
                    model_names,
                    list_of_input_list,
                    gpu_ids
                ) # len(model_names) * varies
                # collect feedbacks
                feedbacks = [] # len(test_input_list) * feedback_count
                for j in range(len(test_input_list)):
                    feedback_for_input = []
                    for k in feedback_indices_list[i][j]:
                        model_index = k
                        feedback_for_input.append(list_of_output_list[model_index].pop(0))
                    feedbacks.append(feedback_for_input)
                feedback_list.append(feedbacks)
            
            # refine responses based on feedbacks
            list_of_input_list = [] # len(model_names) * len(test_input_list)
            for i in range(len(model_names)):
                refine_prompt_list = []
                for j in range(len(test_input_list)):
                    prompt = refine_based_on_feedback_prompt_generator(
                        test_input_list[j],
                        response_list[i][j],
                        feedback_list[i][j]
                    )
                    refine_prompt_list.append(prompt)
                list_of_input_list.append(refine_prompt_list)
            
            list_of_output_list = distributed_generation.distributed_generation(
                model_names,
                list_of_input_list,
                gpu_ids
            )
            response_list = list_of_output_list

    # final summarization using the best model
    summarization_input_list = []
    for i in range(len(test_input_list)):
        prompt = "ou are part of a team of AI assistants collaborating to answer the user's question. Each assistant provides their own answer: use their answers to create a final, comprehensive answer.\n\n"
        prompt += "Question: {}\n\n".format(test_input_list[i])
        prompt += "Assistants' answers:\n"
        for j in range(len(model_names)):
            prompt += "- {}\n".format(response_list[j][i])
        prompt += "\nPlease provide a final, comprehensive answer to the question."
        summarization_input_list.append(prompt)
    
    list_of_input_list = [summarization_input_list]
    list_of_model_names = [best_model_name]
    list_of_gpu_ids = [gpu_ids[0]] # use the first GPU for final summarization
    list_of_output_list = distributed_generation.distributed_generation(
        list_of_model_names,
        list_of_input_list,
        list_of_gpu_ids
    )
    final_outputs = list_of_output_list[0]
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final Test {} score after {} rounds of multiagent feedback: {}".format(task, rounds, avg_test_score))

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
            "output": final_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # fill name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_multiagent_feedback.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()