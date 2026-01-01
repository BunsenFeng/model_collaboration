import os
import json
import torch
import shutil
import random
from model_collaboration.data import eval
from peft import LoraConfig
from datasets import load_dataset
from model_collaboration.method import distributed_generation
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

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

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # method-specific hyperparameters
    router_base_model = hyperparameters.get("router_base_model", "Qwen/Qwen2.5-7B-Instruct")
    model_descriptions = hyperparameters.get("model_descriptions", None)
    reward_model_gpu_id = hyperparameters.get("reward_model_gpu_id", gpu_ids[0])
    reward_model_name = hyperparameters.get("reward_model_name", "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

    # preparing router SFT data
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    list_of_input_list = [dev_input_list for _ in model_names]

    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )

    list_of_dev_scores = [] # len(model_names) * len(dev_input_list)
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_score = eval.get_scores(task, task_type, "dev", dev_outputs)
        list_of_dev_scores.append(dev_score)

    # reward modeling scoring, in case there is a tie on the task metric
    load_reward_model(reward_model_gpu_id, reward_model_name)
    list_of_dev_reward_scores = [] # len(model_names) * len(dev_input_list)
    for i in range(len(model_names)):
        dev_outputs = list_of_output_list[i]
        dev_reward_score = reward_model_scores(reward_model_gpu_id, dev_input_list, dev_outputs)
        list_of_dev_reward_scores.append(dev_reward_score)
    
    best_model_index = [] # len(dev_input_list)
    for j in range(len(dev_input_list)):
        best_score = -float("inf")
        best_index = -1
        for i in range(len(model_names)):
            if list_of_dev_scores[i][j] > best_score:
                best_score = list_of_dev_scores[i][j]
                best_index = i
            elif list_of_dev_scores[i][j] == best_score:
                # tie-breaking with reward model score
                if list_of_dev_reward_scores[i][j] > list_of_dev_reward_scores[best_index][j]:
                    best_index = i
        best_model_index.append(best_index)

    sft_data_points = []
    for i in range(len(dev_input_list)):
        if model_descriptions is not None:
            sft_input = "You are an AI assistant tasked with choosing the best model to answer the user's question based on the model descriptions.\n\n"
            sft_input += "Model Descriptions:\n"
            for m in range(len(model_names)):
                sft_input += "- Model {}: {}\n".format(m+1, model_descriptions[m])
            sft_input += "\nQuestion: {}\n\n".format(dev_input_list[i])
            sft_input += "Based on the model descriptions, choose the best model (1-{}) to answer the question.".format(len(model_names))
        else:
            sft_input = "You are an AI assistant tasked with choosing the best model to answer the user's question.\n\n"
            sft_input += "Question: {}\n\n".format(dev_input_list[i])
            sft_input += "Based on your knowledge, choose the best model (1-{}) to answer the question.".format(len(model_names))
        sft_output = "Model {}".format(best_model_index[i] + 1)
        sft_data_points.append({"prompt": sft_input, "completion": sft_output})

    # save the SFT data
    sft_filename = "model_collaboration/logs/router_model_sft_data_{}_{}.jsonl".format(task, len(model_names))
    with open(sft_filename, "w") as f:
        for data_point in sft_data_points:
            f.write(json.dumps(data_point) + "\n")

    # fine-tuning the router
    dataset = load_dataset("json", data_files=sft_filename, split="train")
    model_name = router_base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    router_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:{}".format(gpu_ids[0]) if gpu_ids[0] >= 0 else "cpu",
    )

    if os.path.exists("model_collaboration/logs/router_sft_{}".format(task)):
        shutil.rmtree("model_collaboration/logs/router_sft_{}".format(task))

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
        output_dir= "model_collaboration/logs/router_sft_{}".format(task),
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
        run_name="router_sft_{}".format(task),
    )

    trainer = SFTTrainer(
        router_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("model_collaboration/logs/router_sft_{}".format(task))
    # save tokenizer as well
    tokenizer.save_pretrained("model_collaboration/logs/router_sft_{}".format(task))

    del router_model
    del tokenizer
    torch.cuda.empty_cache()

    router_model_name = "model_collaboration/logs/router_sft_{}".format(task)

    # using the router for the test set
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    selected_model_indices = []

    router_prompts = []
    for i in range(len(test_input_list)):
        if model_descriptions is not None:
            prompt = "You are an AI assistant tasked with choosing the best model to answer the user's question based on the model descriptions.\n\n"
            prompt += "Model Descriptions:\n"
            for m in range(len(model_names)):
                prompt += "- Model {}: {}\n".format(m+1, model_descriptions[m])
            prompt += "\nQuestion: {}\n\n".format(test_input_list[i])
            prompt += "Based on the model descriptions, choose the best model (1-{}) to answer the question.".format(len(model_names))
        else:
            prompt = "You are an AI assistant tasked with choosing the best model to answer the user's question.\n\n"
            prompt += "Question: {}\n\n".format(test_input_list[i])
            prompt += "Based on your knowledge, choose the best model (1-{}) to answer the question.".format(len(model_names))
        router_prompts.append(prompt)
    
    list_of_input_list = [router_prompts]
    list_of_output_list = distributed_generation.distributed_generation(
        [router_model_name],
        list_of_input_list,
        [gpu_ids[0]]
    )[0]
    for output in list_of_output_list:
        output = output.strip()
        found = False
        for id in range(len(model_names)-1, -1, -1):
            if "{}".format(id + 1) in output:
                model_index = id
                found = True
                break
        if not found:
            model_index = random.randint(0, len(model_names) - 1)
        selected_model_indices.append(model_index)
    assert max(selected_model_indices) < len(model_names), "Selected model index out of range"
    assert min(selected_model_indices) >= 0, "Selected model index out of range"

    # generating final outputs
    list_of_input_list = []
    for i in range(len(model_names)):
        model_input_list = []
        for j in range(len(test_input_list)):
            if selected_model_indices[j] == i:
                model_input_list.append(test_input_list[j])
        list_of_input_list.append(model_input_list)

    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    )
    final_outputs = []
    for j in range(len(test_input_list)):
        final_outputs.append(list_of_output_list[selected_model_indices[j]][0])
        # pop the used output
        list_of_output_list[selected_model_indices[j]].pop(0)

    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores)
    print("Final test {} score after trained router: {}".format(task, avg_test_score))

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
            "selected_model": model_names[selected_model_indices[i]],
            "output": final_outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_trained_router.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    run_method()