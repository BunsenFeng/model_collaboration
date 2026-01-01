import os
import torch
import random
from tqdm import tqdm
from torch import _dynamo
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer

# global hyperparameters for generation
MAX_RESPONSE_LENGTH = None
TEMPERATURE = None
TOP_P = None
BATCH_SIZE = None
BIG_MODEL_MODE = None

def update_generation_hyperparameters(max_response_length, temperature, top_p, batch_size, big_model_mode=False):
    global MAX_RESPONSE_LENGTH, TEMPERATURE, TOP_P, BATCH_SIZE, BIG_MODEL_MODE
    MAX_RESPONSE_LENGTH = max_response_length
    TEMPERATURE = temperature
    TOP_P = top_p
    BATCH_SIZE = batch_size
    BIG_MODEL_MODE = big_model_mode

def batch_generate_text(model_name, gpu_id, input_list, max_response_length, temperature, top_p, batch_size):
    # Load model and tokenizer
    if not BIG_MODEL_MODE:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}", trust_remote_code=True)
    else:
        # ensure that gpu_id is a list
        if not isinstance(gpu_id, list):
            raise ValueError("In BIG_MODEL_MODE, gpu_id should be a list of GPU ids.")
        # set CUDA_VISIBLE_DEVICES
        gpu_id_str = ",".join([str(i) for i in gpu_id])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except:
        # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=True)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        raise ValueError("Tokenizer loading failed. Please check the model name. If it is a lora module, upload your tokenizer to the huggingface repo too.")
    output_list = []
    for i in tqdm(range(0, len(input_list), batch_size)):
        batch_inputs = input_list[i:i+batch_size]
        # try to apply chat template
        try:
            chat_inputs = []
            for input in batch_inputs:
                if "<begin>" in input:
                    question, partial_response = input.split("<begin>", 1)
                    chat = [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ]
                    chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    chat_input += partial_response
                    chat_inputs.append(chat_input)
                else:
                    chat = [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input}
                    ]
                    chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    chat_inputs.append(chat_input)
        except:
            chat_inputs = batch_inputs
        
        inputs = tokenizer(chat_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_response_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # thinking model compatibility
        for idx in range(len(decoded_outputs)):
            if "</think>" in decoded_outputs[idx]:
                decoded_outputs[idx] = decoded_outputs[idx].split("</think>")[-1].strip()

        output_list.extend(decoded_outputs)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    _dynamo.reset_code_caches()
    return output_list

def distributed_generation(list_of_model_name, list_of_input_list, list_of_gpu_id, max_response_length=None):
    """
    Generate text using multiple models in a distributed manner
    Args:
        list_of_model_name (list): List of model names or paths, size n
        list_of_input_list (list): List of input lists for each model, size n * any
        list_of_gpu_id (list): List of GPU IDs available, size any, e.g. [0,1,2,3]
        max_response_length (int): Maximum response length for generation, potentially overriding the global value
    """

    assert len(list_of_model_name) == len(list_of_input_list), "Length of model names and input lists must be the same"

    for i in range(len(list_of_model_name)):
        assert isinstance(list_of_input_list[i], list), "Each element in input lists must be a list"
        # assert len(list_of_input_list[i]) > 0, "Each input list must contain at least one input"
    
    if not BIG_MODEL_MODE:

        list_of_output_list = []

        for i in range(0, len(list_of_model_name), len(list_of_gpu_id)):

            generation_args = []

            for j in range(len(list_of_gpu_id)):
                if i + j < len(list_of_model_name):
                    generation_args.append((
                        list_of_model_name[i + j],
                        list_of_gpu_id[j],
                        list_of_input_list[i + j],
                        MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length,
                        TEMPERATURE,
                        TOP_P,
                        BATCH_SIZE
                    ))
            
            pool = Pool(len(generation_args))
            output = pool.starmap(batch_generate_text, generation_args) # size len(generation_args) * any
            pool.close()
            pool.join()

            for out in output:
                list_of_output_list.append(out)

    else:
        
        list_of_output_list = []

        for i in range(len(list_of_model_name)):

            gpu_id = list_of_gpu_id

            output = batch_generate_text(
                list_of_model_name[i],
                gpu_id,
                list_of_input_list[i],
                MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length,
                TEMPERATURE,
                TOP_P,
                BATCH_SIZE
            )

            list_of_output_list.append(output)
    
    assert len(list_of_output_list) == len(list_of_model_name), "Output list length mismatch"
    for i in range(len(list_of_output_list)):
        assert len(list_of_output_list[i]) == len(list_of_input_list[i]), "Output and input list length mismatch for model {}".format(list_of_model_name[i])
    
    return list_of_output_list

def batch_generate_text_with_score(model_name, gpu_id, input_list, max_response_length, temperature, top_p, batch_size):

    # Load model and tokenizer
    if not BIG_MODEL_MODE:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}", trust_remote_code=True)
    else:
        # ensure that gpu_id is a list
        if not isinstance(gpu_id, list):
            raise ValueError("In BIG_MODEL_MODE, gpu_id should be a list of GPU ids.")
        # set CUDA_VISIBLE_DEVICES
        gpu_id_str = ",".join([str(i) for i in gpu_id])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except:
        raise ValueError("Tokenizer loading failed. Please check the model name. If it is a lora module, upload your tokenizer to the huggingface repo too.")
    output_list = []
    logit_scores = [] # list of list, each list is a logit history
    for i in tqdm(range(0, len(input_list), batch_size)):
        batch_inputs = input_list[i:i+batch_size]
        # try to apply chat template
        try:
            chat_inputs = []
            for input in batch_inputs:
                chat = [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input}
                ]
                chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                chat_inputs.append(chat_input)
        except:
            chat_inputs = batch_inputs
        
        inputs = tokenizer(chat_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # generate response with logit information in each token
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_response_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # `output` is dict, with `sequences`(texts), `scores`(each step's logit scores), ...
        # Text, sequences
        output_sequences_id = outputs["sequences"]
        # output texts
        decoded_outputs = tokenizer.batch_decode(output_sequences_id[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # logit in each token
        logit_score = outputs["scores"] # num_steps length, each element is [bs, vocab_size]
        scores = [[] for _ in range(batch_size)] # list to store each response logit record
        for i in range(len(logit_score)):
            # each step logit
            current_logit_score = logit_score[i] # [bs, vocab_size]
            # select top-k, default k=5
            current_logit_score_top_k, _ = torch.topk(current_logit_score, k=5, dim=-1) # [bs, 5]
            # softmax
            probs = torch.softmax(current_logit_score_top_k, dim=-1) # [bs, 5]
            # each response stores max prob [0, 1]
            for j in range(probs.shape[0]):
                scores[j].append(max(probs[j]))
        logit_scores.extend(scores)
        output_list.extend(decoded_outputs)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    _dynamo.reset_code_caches()
    return output_list, logit_scores

if __name__ == "__main__":

    update_generation_hyperparameters(50, 0.7, 0.9, 4)

    # output_list = batch_generate_text("allenai/Llama-3.1-Tulu-3-8B", 0, ["Hello, how are you?", "What is the capital of France?"] * 4)
    # print(output_list)

    list_of_model_name = ["meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B"]
    list_of_input_list = [
        ["Hello, how are you?", "What is the capital of France?"] * 4,
        ["Explain the theory of relativity.", "What is quantum computing?"] * 3,
        ["Describe the process of photosynthesis.", "What are black holes?"] * 2
    ]
    list_of_gpu_id = [0,1,2]

    output = distributed_generation(list_of_model_name, list_of_input_list, list_of_gpu_id)
    print(output)