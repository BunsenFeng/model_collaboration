"""
The helper functions for distributed DPO.
"""
import os
import torch
import shutil
import random
from tqdm import tqdm
from peft import LoraConfig, AutoPeftModelForCausalLM
from multiprocessing import Pool
from datasets import load_dataset
from model_collaboration.method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, DataCollatorForCompletionOnlyLM
from model_collaboration.utils import lora_check

def single_dpo(model_name, dpo_data_path, gpu_id, output_model_path, batch_size=1, gradient_accumulation_steps=16,
               learning_rate=1e-6, epoch=1):
    """
    DPO training of a single model on a single GPU.
    model_name: the name of the model you want to DPO.
    dpo_data_path: the path of the DPO data, should be JSONL with each line {"prompt":..., "chosen":..., "rejected":...}
    gpu_id: the GPU id you want to use.
    output_model_path: the path to save the DPO model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    dataset = load_dataset("json", data_files=dpo_data_path, split="train")
    # Sparta exports use "instruction" as the prompt field; DPOTrainer expects "prompt" by default.
    # Rename here so existing preference_pairs.json can be used directly.
    if "instruction" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("instruction", "prompt")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check if model_name is a LoRA adapter. If so, merge it into base model first.
    # This ensures we train a new adapter on top of the merged model, rather than
    # stacking multiple adapters (which can cause issues).
    is_adapter = lora_check.is_lora_adapter_peft(model_name)
    
    if is_adapter:
        # Load adapter and merge into base model
        print(f"[DPO] Detected adapter at {model_name}, merging into base model first...")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model = peft_model.merge_and_unload()
        del peft_model
        torch.cuda.empty_cache()
        print(f"[DPO] Adapter merged successfully.")
    else:
        # Load base model directly
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    if os.path.exists(output_model_path):
        print(f"Model path {output_model_path} exists. Deleting it to avoid conflicts.")
        shutil.rmtree(output_model_path)
    
    peft_config = LoraConfig(
        r=64,  # the rank of the LoRA matrices
        lora_alpha=16, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
        modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    )

    training_args = DPOConfig(
        output_dir= output_model_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.1,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        num_train_epochs=epoch,
        # logging strategies 
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        # Suppress label_names warning for PEFT models
        label_names=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()

def distributed_dpo(list_of_model_names, list_of_dpo_data_paths, list_of_gpu_ids, list_of_output_model_paths,
                    batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-6, epoch=1):
    """
    Distributed DPO of multiple models on multiple GPUs.
    list_of_model_names: the list of model names you want to DPO.
    list_of_dpo_data_paths: the list of paths of the DPO data, should be JSONL with each line {"prompt":..., "chosen":..., "rejected":...}
    list_of_gpu_ids: the list of GPU ids you want to use.
    list_of_output_model_paths: the list of paths to save the DPO models.
    """

    num_models = len(list_of_model_names)
    
    for i in range(0, len(list_of_model_names), len(list_of_gpu_ids)):
        dpo_args = []
        for j in range(len(list_of_gpu_ids)):
            if i + j < num_models:
                dpo_args.append((
                    list_of_model_names[i + j],
                    list_of_dpo_data_paths[i + j],
                    list_of_gpu_ids[j],
                    list_of_output_model_paths[i + j],
                    batch_size,
                    gradient_accumulation_steps,
                    learning_rate,
                    epoch
                ))
        
        pool = Pool(len(dpo_args))
        pool.starmap(single_dpo, dpo_args)
        pool.close()
        pool.join()

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    # Example usage

    list_of_model_names = [
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    list_of_dpo_data_paths = [
        "logs/router_model_sft_data_agieval_3.jsonl",
        "logs/router_model_sft_data_agieval_3.jsonl",
        "logs/router_model_sft_data_agieval_3.jsonl"
    ]

    list_of_gpu_ids = [0, 1, 2]

    list_of_output_model_paths = [
        "logs/temp_model_dpo_gemma",
        "logs/temp_model_dpo_llama",
        "logs/temp_model_dpo_qwen"
    ]

    distributed_dpo(
        list_of_model_names,
        list_of_dpo_data_paths,
        list_of_gpu_ids,
        list_of_output_model_paths
    )

    list_of_model_names = [
        "logs/temp_model_dpo_gemma",
        "logs/temp_model_dpo_llama",
        "logs/temp_model_dpo_qwen"
    ]

    list_of_input_list = [
        ["What is the capital of France?", "Who is the president of the China?"],
        ["What is the capital of France?", "Who is the president of the China?"],
        ["What is the capital of France?", "Who is the president of the China?"]
    ]

    list_of_gpu_ids = [0, 1, 2]

    distributed_generation.update_generation_hyperparameters(100, 0.7, 0.9, 8)

    list_of_output_list = distributed_generation.distributed_generation(
        list_of_model_names,
        list_of_input_list,
        list_of_gpu_ids
    )

    print(list_of_output_list)

    # single_dpo(
    #     model_name="google/gemma-3-4b-it",
    #     dpo_data_path="logs/router_model_dpo_data_agieval_3.jsonl",
    #     gpu_id=0,
    #     output_model_path="logs/temp_model_dpo",
    #     epoch=1
    # )

    # list_of_model_name = ["logs/temp_model_dpo"]
    # list_of_input_list = [["What is the capital of France?", "Who is the president of the China?"]]
    # list_of_gpu_id = [0]

    # distributed_generation.update_generation_hyperparameters(100, 0.7, 0.9, 8)

    # list_of_output_list = distributed_generation.distributed_generation(
    #     list_of_model_name,
    #     list_of_input_list,
    #     list_of_gpu_id
    # )