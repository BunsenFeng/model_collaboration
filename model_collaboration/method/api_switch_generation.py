import os
import json
import torch
import random
from tqdm import tqdm
from model_collaboration.data import eval
from peft import LoraConfig
from collections import Counter
from datasets import load_dataset
from model_collaboration.method import distributed_generation
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

def selector_model_prompt(generation_log, model_list):

    selector_prompt = generation_log["query"]
    assert len(generation_log["generated_segments"]) == len(generation_log["segment_model_index"]), "Generated segments and model indices must match"
    for i in range(len(generation_log["generated_segments"])):
        selector_prompt += " <model " + str(generation_log["segment_model_index"][i]) + " begins> " + generation_log["generated_segments"][i] + " <model " + str(generation_log["segment_model_index"][i]) + " ends>"
    selector_prompt += " Which model should generate the next segment? Please respond with a number from 0 to " + str(len(model_list) - 1) + ". The answer is model "

    return selector_prompt

def switch_generation(prompts, model_names, selector_model_name, gpu_ids, patch_size, max_response_length, random_selection=False, objective_flag=False, wait_flag=True):

    assert selector_model_name is not None or random_selection, "Either selector model or random selection must be provided."

    generation_logs = []
    for i in range(len(prompts)):
        generation_logs.append({
            "query": prompts[i],
            "generated_segments": [],
            "segment_model_index": [],
            "generated_sequence": ""
        })

    for round_id in tqdm(range(max_response_length // patch_size)):
        # selector model decides which model generates the next segment
        which_model = []
        if random_selection:
            which_model = [random.randint(0, len(model_names) - 1) for _ in range(len(prompts))]
        else:
            selector_prompts = [selector_model_prompt(generation_logs[i], model_names) for i in range(len(prompts))]
            selector_outputs = distributed_generation.distributed_generation(
                [selector_model_name],
                [selector_prompts],
                [gpu_ids[0]]
            )[0]
            for output in selector_outputs:
                found = False
                for i in range(len(model_names)-1, -1, -1):
                    if str(i) in output:
                        which_model.append(i)
                        found = True
                        break
                if not found:
                    which_model.append(random.randint(0, len(model_names) - 1))  # fallback to random selection
            
        # assemble list_of_prompt_list by model
        list_of_prompt_list = [[] for _ in model_names]
        for i in range(len(prompts)):
            if round_id == max_response_length // patch_size - 1 and objective_flag:
                generation_logs[i]["generated_segments"][-1] += " The final answer is"
                generation_logs[i]["generated_sequence"] += " The final answer is"
            # the s1 inference-time scaling recipe, if the last patch is shorter than patch_size characters (the average character-per-token is about 4, less for math)
            if round_id != max_response_length // patch_size - 1 and round_id > 0 and len(generation_logs[i]["generated_segments"][-1]) < patch_size:
                if wait_flag: # force to continue generation
                    generation_logs[i]["generated_segments"][-1] += " Wait,"
                    generation_logs[i]["generated_sequence"] += " Wait,"
                else: # EOSed
                    generation_logs[i]["generated_segments"][-1] += "<end>"
                    generation_logs[i]["generated_sequence"] += "<end>"
            list_of_prompt_list[which_model[i]].append(
                generation_logs[i]["query"] + "<begin>" + generation_logs[i]["generated_sequence"]
            )

        # generate a patch from the selected models distributedly
        list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            list_of_prompt_list,
            gpu_ids,
            max_response_length=patch_size
        )

        # update generation logs
        for i in range(len(prompts)):
            patch = list_of_output_list[which_model[i]][0]
            list_of_output_list[which_model[i]].pop(0)
            generation_logs[i]["generated_segments"].append(patch)
            generation_logs[i]["segment_model_index"].append(which_model[i])
            generation_logs[i]["generated_sequence"] += patch + " "
        
        assert all(len(output) == 0 for output in list_of_output_list), "All outputs should be consumed"

    final_outputs = [generation_logs[i]["generated_sequence"].strip() for i in range(len(prompts))]
    # truncate at <end> token if exists
    for i in range(len(final_outputs)):
        final_outputs[i] = final_outputs[i].split("<end>")[0].strip()

    return final_outputs, generation_logs

def load_reward_model(gpu_id, model_name):
    device = "cuda:{}".format(gpu_id) if gpu_id >= 0 else "cpu"
    global rm, rm_tokenizer
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

def reward_model_scores(gpu_id, list_of_input, list_of_output):
    assert len(list_of_input) == len(list_of_output), "Input and output lists must have the same length"
    scores = []
    for i in range(len(list_of_input)):
        conv = [{"role": "user", "content": list_of_input[i]}, {"role": "assistant", "content": list_of_output[i]}]
        conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to("cuda:{}".format(gpu_id) if gpu_id >= 0 else "cpu")
        with torch.no_grad():
            score = rm(conv_tokenized).logits[0][0].item()
        scores.append(score)
    return scores

def get_all_inputs(task=None):
    list_of_all_inputs = []
    files = os.listdir("data/")
    for file in files:
        try:
            if file.endswith(".json") and (task is None or file.startswith(task)):
                with open(os.path.join("data/", file), "r") as f:
                    task_type = json.load(f)["task_type"]
                inputs = eval.prepare_inputs(file[:-5], task_type, "dev")
                list_of_all_inputs.extend(inputs)
        except:
            print("Error processing file: {}".format(file))
    return list_of_all_inputs
            
def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    max_response_length = hyperparameters.get("max_response_length")
    batch_size = hyperparameters.get("batch_size")

    # method-specific hyperparameters
    patch_size = hyperparameters.get("patch_size", 50)
    selector_model_name = hyperparameters.get("selector_model_name", None)
    selector_base_model = hyperparameters.get("selector_base_model", "Qwen/Qwen2.5-7B-Instruct")
    objective_flag = hyperparameters.get("objective_flag", False)
    training_instance_num = hyperparameters.get("training_instance_num", 250)
    rollout_per_instance = hyperparameters.get("rollout_per_instance", 16)
    reward_model_gpu_id = hyperparameters.get("reward_model_gpu_id", gpu_ids[0])
    reward_model_name = hyperparameters.get("reward_model_name", "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    wait_flag = hyperparameters.get("wait_flag", False)

    if selector_model_name is None:
        # train a selector model
        print("Training selector model...")
        input_list = get_all_inputs(task)
        sft_data_points = []

        for iter in range(training_instance_num // batch_size):
            prompts = random.sample(input_list, batch_size)
            # sample the shared prefix
            context_count = random.randint(1, max_response_length // patch_size - 3)
            context_list, generation_log_list = switch_generation(
                prompts,
                model_names,
                None,
                gpu_ids,
                patch_size,
                context_count * patch_size,
                random_selection=True,
                wait_flag=wait_flag
            )
            # sample the rollouts
            input_now = [prompts[i] + context_list[i] for i in range(batch_size)]
            full_input_now = []
            for i in range(len(input_now)):
                full_input_now = full_input_now + [input_now[i]] * rollout_per_instance
            
            full_new_context_list, full_new_generation_log_list = switch_generation(
                full_input_now,
                model_names,
                None,
                gpu_ids,
                patch_size,
                max_response_length - context_count * patch_size,
                random_selection=True,
                wait_flag=wait_flag
            )

            # scoring the rollouts
            load_reward_model(reward_model_gpu_id, reward_model_name)
            for i in range(batch_size):
                input_now = full_input_now[i * rollout_per_instance]
                new_context_list = full_new_context_list[i * rollout_per_instance:(i + 1) * rollout_per_instance]
                new_generation_log_list = full_new_generation_log_list[i * rollout_per_instance:(i + 1) * rollout_per_instance]
                generation_log = generation_log_list[i]

                utility = [0] * len(model_names)
                for j in range(len(model_names)):
                    scores = []
                    for k in range(rollout_per_instance):
                        if new_generation_log_list[k]["segment_model_index"][0] == j:
                            full_output = input_now + new_context_list[k]
                            scores.append(
                                reward_model_scores(reward_model_gpu_id, [input_now], [full_output])[0]
                            )
                    if scores:
                        utility[j] = sum(scores) / len(scores)
                    else:
                        utility[j] = -1000.0 # didn't test, shouldn't select

                # select the model with the highest utility for sft
                sft_model_id = utility.index(max(utility))
                sft_input = selector_model_prompt(generation_log, model_names)
                sft_output = str(sft_model_id)
                sft_data_points.append(
                    {"prompt": sft_input, "completion": sft_output}
                )

                # print({"prompt": sft_input, "completion": sft_output})

        # save the sft data
        sft_filename = "model_collaboration/logs/selector_model_sft_data_{}_{}.jsonl".format(task, len(model_names))
        with open(sft_filename, "w") as f:
            for data_point in sft_data_points:
                f.write(json.dumps(data_point) + "\n")

        # fine-tuning the switcher
        dataset = load_dataset("json", data_files=sft_filename, split="train")
        model_name = selector_base_model
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:{}".format(gpu_ids[0]), trust_remote_code=True)

        peft_config = LoraConfig(
            r=64,  # the rank of the LoRA matrices
            lora_alpha=16, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )

        training_args = SFTConfig(
            output_dir= "model_collaboration/logs/selector_sft_{}".format(task),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            bf16=True,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_ratio = 0.1,
            gradient_checkpointing=True,
            eval_strategy="epoch",
            num_train_epochs=5,
            # logging strategies 
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            max_seq_length=4096,
            run_name="selector_sft_{}".format(task),
        )

        trainer = SFTTrainer(
            model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            peft_config=peft_config,
        )

        trainer.train()
        trainer.save_model("model_collaboration/logs/selector_sft_{}".format(task))
        # save tokenizer as well
        tokenizer.save_pretrained("model_collaboration/logs/selector_sft_{}".format(task))

        del model
        del tokenizer
        torch.cuda.empty_cache()

        selector_model_name = "model_collaboration/logs/selector_sft_{}".format(task)
    else:
        print("Using pre-defined selector model: {}".format(selector_model_name))
    test_inputs = eval.prepare_inputs(task, task_type, "test")
    final_outputs, generation_logs = switch_generation(
        test_inputs,
        model_names,
        selector_model_name,
        gpu_ids,
        patch_size,
        max_response_length,
        random_selection=False,
        objective_flag=objective_flag,
        wait_flag=wait_flag
    )

    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score switch generation: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "selector_model_name": selector_model_name,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(final_outputs)):
        log_entry = generation_logs[i]
        log_entry["input"] = test_inputs[i]
        log_entry["output"] = final_outputs[i]
        log_entry["score"] = test_scores[i]
        experiment_logs["logs"].append(log_entry)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_switch_generation.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()
    # load_reward_model(gpu_id=0)
    # scores = reward_model_scores(0, ["What is the capital of France?"], ["The capital of France is Beijing."])
    # print(scores)