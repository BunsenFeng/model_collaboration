import json
import torch
import torch.nn.functional as F
from model_collaboration.data import eval
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)

class Nudging:
    def __init__(
        self,
        model_names,
        model_devices,
        system_prompt = None,
        model_kwargs = None
    ):
        assert len(model_names) >= 2, "len(model_names) should be equal to or larger than 2"
        self.models = [
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(model_devices[i]) for i, model_name in enumerate(model_names)
        ]
        for model in self.models:
            model.eval()
        self.tokenizers = [AutoTokenizer.from_pretrained(model_name, use_fast=True) for model_name in model_names]
        for tokenizer in self.tokenizers:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        self.base_model_id = 0
        self.nudging_model_id = 1
        self.system_prompt = system_prompt
        self.model_devices = model_devices

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs
    
    def decode_token(self, tokenizer, prefix_ids, new_id):
        """
        Returns the human-readable text added by `new_id` when appended to `prefix_ids`,
        in a way that is consistent with tokenizer.decode(prefix_ids + [new_id]).
        """
        # convert prefix_ids to list
        prefix_ids = prefix_ids.tolist()
        
        # IMPORTANT: disable cleanup so spaces are not collapsed/stripped
        old = tokenizer.decode(prefix_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
        new = tokenizer.decode(prefix_ids + [new_id], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)

        if not new.startswith(old):
            # Rare edge cases (special tokens, weird normalization). Fallback: suffix diff.
            # This is conservative and still usually correct.
            # Find the longest common prefix.
            i = 0
            m = min(len(old), len(new))
            while i < m and old[i] == new[i]:
                i += 1
            return new[i:]
        return new[len(old):]

    def get_next_token_ids(self,logits, temperature, do_sample):
        if temperature != 1.0:
            logits = logits / temperature
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_ids = torch.argmax(logits, dim=-1)
        return next_token_ids
    
    def forward(
        self,
        inputs,
        return_dict = None
    ):  
        assert len(inputs) == 2, "inputs must be a list of length 2, inputs[0] for base model and inputs[1] for nudging model"
        base_model_input = inputs[0]
        nudging_model_input = inputs[1]
        # Move inputs to correct device before calling models, and return a list of outputs
        outputs = []
        for idx, (model, model_input, device) in enumerate([
            (self.models[self.base_model_id], base_model_input, self.model_devices[self.base_model_id]),
            (self.models[self.nudging_model_id], nudging_model_input, self.model_devices[self.nudging_model_id]),
        ]):
            model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in model_input.items()}
            outputs.append(model(**model_inputs, return_dict=return_dict))
        return outputs[0], outputs[1]

    @torch.inference_mode()
    def generate(
        self,
        base_chat_prompts = None,
        nudging_chat_prompts = None,
        max_new_tokens = 100,
        do_sample = True,
        temperature = 1.0,
        gamma = 0.4,
        stopping_criteria = None,
        debug = False,
        **kwargs
    ):
        """
        Generate text using nudging.

        Args:
            base_chat_prompts: The chat prompts for the base model.
            nudging_chat_prompts: The chat prompts for the nudging model.
            max_new_tokens: The maximum number of new tokens to generate.
            do_sample: Whether to sample from the distribution, false for greedy decoding.
            temperature: The temperature for sampling.
            gamma: The top-1 base model probability threshold for nudging.
            stopping_criteria: The stopping criteria.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated texts.
        """
        # # Tokenize separately for each model
        base_tokenizer = self.tokenizers[self.base_model_id]
        nudging_tokenizer = self.tokenizers[self.nudging_model_id]
        input_ids_device = 'cpu'
        batch_size = len(base_chat_prompts)
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids_device)
        base_eos_token_id_tensor = torch.tensor(base_tokenizer.eos_token_id).to(input_ids_device)
        nudging_eos_token_id_tensor = torch.tensor(nudging_tokenizer.eos_token_id).to(input_ids_device)
        
        decoded_outputs = ["" for _ in range(batch_size)]
        do_nudging = [True for _ in range(batch_size)]
        accepted_nudging_tokens = [0 for _ in range(batch_size)]
        
        for step in tqdm(range(max_new_tokens)):
            # --- OPTIMIZATION START: Get active indices ---
            # We only process sequences that are NOT finished (unfinished_sequences == 1)
            active_indices = torch.nonzero(unfinished_sequences).squeeze(-1).tolist()
            
            # If all sequences are finished, stop immediately
            if not active_indices:
                break
                
            current_batch_size = len(active_indices)
            
            # Prepare inputs ONLY for active indices
            # Note: We map back to the original lists using 'idx'
            active_base_prompts = [base_chat_prompts[idx] + decoded_outputs[idx] for idx in active_indices]
            active_nudging_prompts = [nudging_chat_prompts[idx] + decoded_outputs[idx] for idx in active_indices]

            # Tokenize the smaller batch
            base_encodings = base_tokenizer(active_base_prompts, return_tensors="pt", padding=True, truncation=True)
            nudging_encodings = nudging_tokenizer(active_nudging_prompts, return_tensors="pt", padding=True, truncation=True)
            
            base_input_ids = base_encodings.input_ids
            nudging_input_ids = nudging_encodings.input_ids

            # Include attention masks!
            base_inputs = {
                'input_ids': base_input_ids,
                'attention_mask': base_encodings.attention_mask
            }
            nudging_inputs = {
                'input_ids': nudging_input_ids,
                'attention_mask': nudging_encodings.attention_mask
            }

            inputs = [base_inputs, nudging_inputs]
            
            # Forward pass on smaller batch
            base_output, nudging_output = self.forward(inputs, return_dict=True) 
            base_logits = base_output.logits[..., -1, :]
            nudging_logits = nudging_output.logits[..., -1, :]
            base_logits = base_logits.to(input_ids_device)
            nudging_logits = nudging_logits.to(input_ids_device)
            
            base_probs = F.softmax(base_logits, dim=-1)
            base_probs_top_1 = torch.topk(base_probs, k=1, dim=-1).values
            
            base_next_token_ids = self.get_next_token_ids(base_logits, temperature, do_sample)
            nudging_next_token_ids = self.get_next_token_ids(nudging_logits, temperature, do_sample)
            
            # Decode tokens (batched decoding for the active set)
            # note: base_input_ids[k] corresponds to active_indices[k]
            base_next_tokens = [self.decode_token(base_tokenizer, base_input_ids[k], base_next_token_ids[k]) for k in range(current_batch_size)]
            nudging_next_tokens = [self.decode_token(nudging_tokenizer, nudging_input_ids[k], nudging_next_token_ids[k]) for k in range(current_batch_size)]
            
            # --- Update Loop: Iterate over the ACTIVE batch ---
            for k, original_idx in enumerate(active_indices):
                # Check EOS for this specific sequence
                # Note: We check if EITHER model produced an EOS
                is_base_eos = (base_next_token_ids[k] == base_eos_token_id_tensor).item()
                is_nudge_eos = (nudging_next_token_ids[k] == nudging_eos_token_id_tensor).item()
                
                if is_base_eos or is_nudge_eos:
                    unfinished_sequences[original_idx] = 0
                    # Don't append the token if it's EOS (optional, but cleaner)
                    continue 

                # Retrieve state for the original index
                nudging = do_nudging[original_idx]
                top_1_base_prob = base_probs_top_1[k].item()
                accepted_nudging_token = accepted_nudging_tokens[original_idx]
                
                base_next_token = base_next_tokens[k]
                nudging_next_token = nudging_next_tokens[k]
                space_in_nudging_next_token = nudging_next_token.count(" ")
                
                next_token = ""
                
                # Nudging logic
                if nudging and (accepted_nudging_token == 0 or space_in_nudging_next_token == 0 or top_1_base_prob < gamma):
                    # Three conditions to keep nudging when in nudging mode:
                    # 1. No accepted nudging token yet (always nudge the first token)
                    # 2. No space in the next token (haven't reached a word boundary), we keep nudging until we accept a full nudging word
                    # 3. Top-1 base model probability is less than gamma (the next token is still uncertain)
                    accepted_nudging_tokens[original_idx] += 1
                    next_token = nudging_next_token
                elif top_1_base_prob >= gamma:
                    # Switch back to the base model
                    do_nudging[original_idx] = False
                    accepted_nudging_tokens[original_idx] = 0
                    next_token = base_next_token
                elif not nudging and top_1_base_prob < gamma:
                    # Switch to the nudging model
                    do_nudging[original_idx] = True
                    accepted_nudging_tokens[original_idx] = 1
                    next_token = nudging_next_token
                else:
                    raise ValueError("Invalid nudging condition: nudging={}, top_1_base_prob={}, accepted_nudging_token={}, space_in_nudging_next_token={}".format(nudging, top_1_base_prob, accepted_nudging_token, space_in_nudging_next_token))
                
                decoded_outputs[original_idx] += next_token
                
                if debug:
                    print("Step {}, active indices: {}".format(step, active_indices))
                    print("Step {}: base_next_token={}, nudging_next_token={}, next_token={}".format(step, base_next_token, nudging_next_token, next_token))
                    print("Nudging: {}, Top-1 base prob: {}, Accepted nudging token: {}, Space in nudging next token: {}".format(nudging, top_1_base_prob, accepted_nudging_token, space_in_nudging_next_token))
                    print("Decoded output: {}".format(decoded_outputs[original_idx]))
                    print("--------------------------------")

            # if all sequences are finished, stop
            if unfinished_sequences.max() == 0:
                break
        return decoded_outputs

    def batch_generate(
        self,
        prompts,
        batch_size = 1,
        max_new_tokens = 100,
        do_sample = True,
        temperature = 1.0,
        gamma = 0.4,
        base_model_id = 0,
        nudging_model_id = 1,
    ):
        # Always reset the base and nudging model ids
        self.base_model_id = base_model_id
        self.nudging_model_id = nudging_model_id
        
        # Get tokenizers for base and nudging models
        base_tokenizer = self.tokenizers[self.base_model_id]
        nudging_tokenizer = self.tokenizers[self.nudging_model_id]
        
        # Prepare chat prompts separately for each model
        base_chat_prompts = []
        nudging_chat_prompts = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}] if self.system_prompt is None else [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            # Apply chat template for base model
            try:
                base_chat_prompt = base_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except:
                base_chat_prompt = prompt
            base_chat_prompts.append(base_chat_prompt)
            
            # Apply chat template for nudging model
            try:
                nudging_chat_prompt = nudging_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except:
                nudging_chat_prompt = prompt
            nudging_chat_prompts.append(nudging_chat_prompt)

        outputs = []
        for i in tqdm(range(0, len(base_chat_prompts), batch_size)):
            batch_base_chat_prompts = base_chat_prompts[i:i+batch_size]
            batch_nudging_chat_prompts = nudging_chat_prompts[i:i+batch_size]
            
            outputs.extend(self.generate(
                base_chat_prompts=batch_base_chat_prompts,
                nudging_chat_prompts=batch_nudging_chat_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                gamma=gamma
            ))
        return outputs

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    batch_size = hyperparameters.get("batch_size")
    max_new_tokens = hyperparameters.get("max_response_length")
    temperature = hyperparameters.get("temperature")
    do_sample = hyperparameters.get("do_sample", True)

    # method-specific hyperparameters
    gamma = hyperparameters.get("gamma", 0.4) # top-1 base model probability threshold for nudging
    base_model_id = hyperparameters.get("base_model_id", 0)
    nudging_model_id = hyperparameters.get("nudging_model_id", 1)
    search_nudging = hyperparameters.get("search_nudging", False)
    search_gamma = hyperparameters.get("search_gamma", False)
    
    nudging_object = Nudging(
        model_names=model_names,
        model_devices=["cuda:{}".format(gpu_id) for gpu_id in gpu_ids],
    )
    
    dev_input_list = eval.prepare_inputs(task, task_type, "dev")
    
    gamma_list = [gamma]
    nudging_model_id_list = [nudging_model_id]

    if search_gamma:
        # Search over gamma
        gamma_list = [0.2, 0.3, 0.4, 0.5]
    if search_nudging:
        # Search over all available nudging model ids (excluding base_model_id)
        nudging_model_id_list = [i for i in range(len(model_names)) if i != base_model_id]

    search_list = []
    for gamma in gamma_list:
        for nudging_model_id in nudging_model_id_list:
            search_list.append((gamma, nudging_model_id))

    best_gamma = gamma
    best_nudging_model_id = nudging_model_id
    best_dev_score = 0
    if len(search_list) > 1:
        print("Searching over gamma and nudging model id...")
        print("Total number of searches on the dev set: {}".format(len(search_list)))
        print("Search list (gamma, nudging model id): {}".format(search_list))
        for gamma, nudging_model_id in search_list:
            outputs = nudging_object.batch_generate(
                prompts=dev_input_list,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                gamma=gamma,
                base_model_id=base_model_id,
                nudging_model_id=nudging_model_id
            )
            dev_score = eval.get_scores(task, task_type, "dev", outputs)
            avg_dev_score = sum(dev_score) / len(dev_score)
            if avg_dev_score > best_dev_score:
                best_gamma = gamma
                best_nudging_model_id = nudging_model_id
                best_dev_score = avg_dev_score
            print("Gamma: {}, Nudging model: {}, Dev score: {}".format(gamma, model_names[nudging_model_id], avg_dev_score))
        print("Best gamma: {}, Best nudging model: {}, Best dev score: {}".format(best_gamma, model_names[best_nudging_model_id], best_dev_score))
    else:
        print("No search, using default gamma: {}, base model: {}, nudging model: {}".format(best_gamma, model_names[base_model_id], model_names[best_nudging_model_id]))

    test_input_list = eval.prepare_inputs(task, task_type, "test")
    outputs = nudging_object.batch_generate(
        prompts=test_input_list,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        gamma=best_gamma,
        base_model_id=base_model_id,
        nudging_model_id=best_nudging_model_id
    )

    test_scores = eval.get_scores(task, task_type, "test", outputs)
    avg_test_scores = sum(test_scores) / len(test_scores)
    print("Final test {} score after nudging: {}".format(task, avg_test_scores))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_scores,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": outputs[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "logs/{}_{}_{}_nudging.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    instance = Nudging(
        model_names=[
            "allenai/Llama-3.1-Tulu-3-8B",
            "google/gemma-2-2b-it",
            "allenai/Llama-3.1-Tulu-3-8B-DPO",
            "allenai/Llama-3.1-Tulu-3-8B-SFT",
        ],
        model_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )
    prompts = [
        "Explain the theory of relativity in simple terms. Explain in one sentence.",
        "What are the benefits of using renewable energy sources?",
    ]
    for nudging_model_id in [1, 2, 3]:
        outputs = instance.batch_generate(
            prompts,
            batch_size=2,
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            gamma=0.4,
            base_model_id=0,
            nudging_model_id=nudging_model_id,
        )
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}\nOutput: {output}\n")