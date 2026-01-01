import json
import torch
import torch.nn.functional as F
from model_collaboration.data import eval
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any, List
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)

# Assuming data.eval exists in your environment as per your snippet
try:
    from data import eval
except ImportError:
    # Placeholder for standalone testing if data.eval is missing
    class eval:
        @staticmethod
        def prepare_inputs(task, task_type, split): return ["Test prompt"]
        @staticmethod
        def get_scores(task, task_type, split, outputs): return [0.0] * len(outputs)

class Nudging:
    def __init__(
        self,
        model_names,
        model_devices,
        system_prompt = None,
        model_kwargs = None
    ):
        assert len(model_names) > 1, "len(model_names) should be larger than 1"
        self.models = [
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(model_devices[i]) for i, model_name in enumerate(model_names)
        ]
        for model in self.models:
            model.eval()
        self.tokenizers = [AutoTokenizer.from_pretrained(model_name, use_fast=True) for model_name in model_names]
        for tokenizer in self.tokenizers:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        self.base_model_id = 0
        self.nudging_model_id = 1
        self.system_prompt = system_prompt
        self.model_devices = model_devices
    
    def decode_token(self, tokenizer, prefix_ids, new_id):
        """
        Returns the human-readable text added by `new_id` when appended to `prefix_ids`,
        in a way that is consistent with tokenizer.decode(prefix_ids + [new_id]).
        """
        # 1. Filter out Padding tokens from the prefix
        # We convert to list and remove any ID that matches the pad token
        if isinstance(prefix_ids, torch.Tensor):
            prefix_ids = prefix_ids.tolist()
        prefix_ids = [
            pid for pid in prefix_ids 
            if pid != tokenizer.pad_token_id
        ]
        
        # 2. Check if the NEW token is a pad token (shouldn't happen with valid logic, but safety first)
        if new_id == tokenizer.pad_token_id:
            return ""

        # 3. Decode as before
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
    
    def forward_step(
        self,
        model_idx,
        input_ids,
        attention_mask,
        past_key_values
    ):
        """
        Helper to run a single model step with KV cache.
        """
        model = self.models[model_idx]
        device = self.model_devices[model_idx]
        
        # Move inputs to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        return outputs.logits, outputs.past_key_values

    @torch.inference_mode()
    def generate_with_kv_cache(
        self,
        base_chat_prompts: List[str],
        nudging_chat_prompts: List[str],
        max_new_tokens = 100,
        do_sample = True,
        temperature = 1.0,
        gamma = 0.4,
        debug = False,
        **kwargs
    ):
        batch_size = len(base_chat_prompts)
        
        # 1. INITIALIZATION
        base_tokenizer = self.tokenizers[self.base_model_id]
        nudging_tokenizer = self.tokenizers[self.nudging_model_id]
        
        # Tokenize initial full prompts
        # Note: We must disable adding special tokens inside the loop, so we do it correctly here once
        base_encodings = base_tokenizer(base_chat_prompts, return_tensors="pt", padding=True, truncation=True)
        nudging_encodings = nudging_tokenizer(nudging_chat_prompts, return_tensors="pt", padding=True, truncation=True)

        # Current Input IDs (will be updated to just the new token in the loop)
        base_input_ids = base_encodings.input_ids
        nudging_input_ids = nudging_encodings.input_ids
        
        # Attention Masks (Must grow as sequence grows)
        base_attention_mask = base_encodings.attention_mask
        nudging_attention_mask = nudging_encodings.attention_mask

        # KV Caches
        base_past_key_values = None
        nudging_past_key_values = None

        # Tracking state
        # We need to keep track of the *full* sequence ids for decoding purposes 
        # (decode_token needs context to handle spacing correctly)
        base_full_sequence_ids = base_input_ids.clone().cpu().tolist()
        nudging_full_sequence_ids = nudging_input_ids.clone().cpu().tolist()

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device='cpu')
        base_eos_token_id = base_tokenizer.eos_token_id
        nudging_eos_token_id = nudging_tokenizer.eos_token_id
        
        decoded_outputs = ["" for _ in range(batch_size)]
        do_nudging = [True for _ in range(batch_size)]
        accepted_nudging_tokens = [0 for _ in range(batch_size)]

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            
            # --- 2. FORWARD PASS (BASE) ---
            base_logits, base_past_key_values = self.forward_step(
                self.base_model_id, 
                base_input_ids, 
                base_attention_mask, 
                base_past_key_values
            )
            
            # --- 3. FORWARD PASS (NUDGING) ---
            nudging_logits, nudging_past_key_values = self.forward_step(
                self.nudging_model_id, 
                nudging_input_ids, 
                nudging_attention_mask, 
                nudging_past_key_values
            )

            # Get logits for the last token only
            next_token_base_logits = base_logits[:, -1, :].cpu()
            next_token_nudging_logits = nudging_logits[:, -1, :].cpu()

            # --- 4. SELECTION LOGIC ---
            # Calculate probabilities for base model (needed for gamma threshold)
            base_probs = F.softmax(next_token_base_logits, dim=-1)
            base_probs_top_1 = torch.topk(base_probs, k=1, dim=-1).values

            # Sample/Greedy select next token IDs
            base_next_token_ids = self.get_next_token_ids(next_token_base_logits, temperature, do_sample)
            nudging_next_token_ids = self.get_next_token_ids(next_token_nudging_logits, temperature, do_sample)

            # Decode to string to handle tokenizer mismatch
            # We use the full sequence context from CPU for accurate decoding
            base_next_texts = [
                self.decode_token(base_tokenizer, base_full_sequence_ids[k], base_next_token_ids[k].item()) 
                for k in range(batch_size)
            ]
            nudging_next_texts = [
                self.decode_token(nudging_tokenizer, nudging_full_sequence_ids[k], nudging_next_token_ids[k].item()) 
                for k in range(batch_size)
            ]

            next_texts_to_append = []
            
            for k in range(batch_size):
                if unfinished_sequences[k] == 0:
                    next_texts_to_append.append("") # Already done
                    continue

                nudging = do_nudging[k]
                top_1_base_prob = base_probs_top_1[k].item()
                accepted_count = accepted_nudging_tokens[k]
                
                b_text = base_next_texts[k]
                n_text = nudging_next_texts[k]
                space_in_nudge = n_text.count(" ")
                
                is_base_eos = (base_next_token_ids[k] == base_eos_token_id).item()
                is_nudge_eos = (nudging_next_token_ids[k] == nudging_eos_token_id).item()

                selected_text = ""
                
                # Nudging Heuristics
                if nudging and (accepted_count == 0 or space_in_nudge == 0 or top_1_base_prob < gamma):
                    # Check Nudging EOS
                    if is_nudge_eos:
                        unfinished_sequences[k] = 0
                    else:
                        accepted_nudging_tokens[k] += 1
                        selected_text = n_text
                elif top_1_base_prob >= gamma:
                    # Check Base EOS
                    if is_base_eos:
                        unfinished_sequences[k] = 0
                    else:
                        do_nudging[k] = False
                        accepted_nudging_tokens[k] = 0
                        selected_text = b_text  
                elif not nudging and top_1_base_prob < gamma:
                    # Check Nudging EOS
                    if is_nudge_eos:
                        unfinished_sequences[k] = 0
                    else:
                        do_nudging[k] = True
                        accepted_nudging_tokens[k] = 1
                        selected_text = n_text
                else:
                    # Fallback / Invalid state
                    raise ValueError("Invalid nudging condition: nudging={}, top_1_base_prob={}, accepted_nudging_token={}, space_in_nudging_next_token={}".format(nudging, top_1_base_prob, accepted_count, space_in_nudge))
                
                decoded_outputs[k] += selected_text
                next_texts_to_append.append(selected_text)
                if debug:
                    print(f"Step {step}, Sequence {k}, base_next_texts={base_next_texts[k]}, nudging_next_texts={nudging_next_texts[k]}, selected_text={selected_text}")
                    print(f"Nudging: {do_nudging[k]}, Top-1 Base Prob: {top_1_base_prob}, Accepted Nudging Tokens: {accepted_nudging_tokens[k]}, Space in Nudge: {space_in_nudge}")
                    print(f"Decoded output: {decoded_outputs[k]}")
                    print(f"Base full sequence ids: {base_full_sequence_ids[k]}")
                    print(f"Nudging full sequence ids: {nudging_full_sequence_ids[k]}")
                    # print(f"Base decoded ids: {base_tokenizer.decode(base_full_sequence_ids[k])}")
                    # print(f"Nudging decoded ids: {nudging_tokenizer.decode(nudging_full_sequence_ids[k])}")
                    print("================================================")

            if unfinished_sequences.max() == 0:
                break

            # --- 5. PREPARE INPUTS FOR NEXT STEP ---
            # We must re-tokenize the `selected_text` for BOTH models.
            # This aligns the KV cache of both models to the single chosen path.
            
            new_base_input_ids_list = []
            new_nudging_input_ids_list = []
            
            for k, text in enumerate(next_texts_to_append):
                if unfinished_sequences[k] == 0 or text == "":
                    # If finished, just feed a dummy token (e.g. EOS or Pad) to keep shapes valid
                    # The mask will ignore it, but we need valid tensor shapes
                    new_base_input_ids_list.append(torch.tensor([base_tokenizer.pad_token_id]))
                    new_nudging_input_ids_list.append(torch.tensor([nudging_tokenizer.pad_token_id]))
                else:
                    # Encode the text segment. 
                    # add_special_tokens=False is CRITICAL here as we are mid-sentence.
                    b_ids = base_tokenizer.encode(text, add_special_tokens=False)
                    n_ids = nudging_tokenizer.encode(text, add_special_tokens=False)
                    
                    # Safety: if tokenizer produces empty list (rare but possible with weird chars), use a dummy
                    if len(b_ids) == 0: b_ids = [base_tokenizer.pad_token_id] # Should not happen usually
                    if len(n_ids) == 0: n_ids = [nudging_tokenizer.pad_token_id]

                    new_base_input_ids_list.append(torch.tensor(b_ids))
                    new_nudging_input_ids_list.append(torch.tensor(n_ids))

            # Manual Left-Padding for the Increments
            # (torch.nn.utils.rnn.pad_sequence only does Right Padding)
            def left_pad_tensor_list(tensor_list, pad_token_id, device):
                max_len = max(len(x) for x in tensor_list)
                padded_batch = []
                for x in tensor_list:
                    if isinstance(x, torch.Tensor):
                        x = x.tolist()
                    # Calculate how many pads we need on the LEFT
                    pad_amt = max_len - len(x)
                    padded = torch.tensor([pad_token_id] * pad_amt + x, device=device)
                    padded_batch.append(padded)
                return torch.stack(padded_batch)

            # Create the micro-batch for the next step
            base_input_ids = left_pad_tensor_list(new_base_input_ids_list, base_tokenizer.pad_token_id, self.model_devices[self.base_model_id])
            nudging_input_ids = left_pad_tensor_list(new_nudging_input_ids_list, nudging_tokenizer.pad_token_id, self.model_devices[self.nudging_model_id])
            
            # Update Attention Masks (1 for real, 0 for pad)
            base_new_mask = (base_input_ids != base_tokenizer.pad_token_id).long().to(base_attention_mask.device)
            nudging_new_mask = (nudging_input_ids != nudging_tokenizer.pad_token_id).long().to(nudging_attention_mask.device)
            
            base_attention_mask = torch.cat([base_attention_mask, base_new_mask], dim=-1)
            nudging_attention_mask = torch.cat([nudging_attention_mask, nudging_new_mask], dim=-1)

            # Update Full Sequence Trackers (CPU)
            # We simply concatenate the raw list values, ignoring the padding we just used for the forward pass
            for k in range(batch_size):
                if unfinished_sequences[k] == 0:
                    continue
                
                new_b_ids = base_input_ids[k] 
                new_n_ids = nudging_input_ids[k]
                
                # Simply extend the history list
                base_full_sequence_ids[k].extend(new_b_ids.tolist())
                nudging_full_sequence_ids[k].extend(new_n_ids.tolist())

        return decoded_outputs

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
                # Note: We check if the selected model produced an EOS
                is_base_eos = (base_next_token_ids[k] == base_eos_token_id_tensor).item()
                is_nudge_eos = (nudging_next_token_ids[k] == nudging_eos_token_id_tensor).item()
                
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
                    if is_nudge_eos:
                        unfinished_sequences[original_idx] = 0
                        continue
                    accepted_nudging_tokens[original_idx] += 1
                    next_token = nudging_next_token
                elif top_1_base_prob >= gamma:
                    # Switch back to the base model
                    if is_base_eos:
                        unfinished_sequences[original_idx] = 0
                        continue
                    do_nudging[original_idx] = False
                    accepted_nudging_tokens[original_idx] = 0
                    next_token = base_next_token
                elif not nudging and top_1_base_prob < gamma:
                    # Switch to the nudging model
                    if is_nudge_eos:
                        unfinished_sequences[original_idx] = 0
                        continue
                    do_nudging[original_idx] = True
                    accepted_nudging_tokens[original_idx] = 1
                    next_token = nudging_next_token
                else:
                    raise ValueError("Invalid nudging condition: nudging={}, top_1_base_prob={}, accepted_nudging_token={}, space_in_nudging_next_token={}".format(nudging, top_1_base_prob, accepted_nudging_token, space_in_nudging_next_token))
                
                decoded_outputs[original_idx] += next_token
                
                if debug:
                    print("Step {}, active indices: {}".format(step, active_indices))
                    print("Step {}: base_next_token={}, nudging_next_token={}, next_token={}".format(step, base_next_token, nudging_next_token, next_token))
                    print("Nudging: {}, Top-1 base prob: {}, Accepted nudging token: {}, Space in nudging next token: {}".format(do_nudging[original_idx], top_1_base_prob, accepted_nudging_tokens[original_idx], space_in_nudging_next_token))
                    print("Decoded output: {}".format(decoded_outputs[original_idx]))
                    print("Base full sequence ids: {}".format(base_input_ids[original_idx].tolist()))
                    print("Nudging full sequence ids: {}".format(nudging_input_ids[original_idx].tolist()))
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
        use_kv_cache = False,
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
            if use_kv_cache:
                outputs.extend(self.generate_with_kv_cache(
                    base_chat_prompts=batch_base_chat_prompts,
                    nudging_chat_prompts=batch_nudging_chat_prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    gamma=gamma
                ))
            else:
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

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

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
    use_kv_cache = hyperparameters.get("use_kv_cache", False)
    
    if use_kv_cache:
        print("Using KV cache..., outputs can be different from the one without KV cache")
    else:
        print("Not using KV cache...")
    
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
                nudging_model_id=nudging_model_id,
                use_kv_cache=use_kv_cache,
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
        nudging_model_id=best_nudging_model_id,
        use_kv_cache=use_kv_cache,
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
    log_filename = "model_collaboration/logs/{}_{}_{}_nudging.json".format(task, len(model_names), round(avg_test_scores, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

if __name__ == "__main__":
    model_names = [
        "allenai/Llama-3.1-Tulu-3-8B",
        "google/gemma-2-2b-it",
        # "allenai/Llama-3.1-Tulu-3-8B-DPO",
        # "allenai/Llama-3.1-Tulu-3-8B-SFT",
    ]
    model_devices = ["cuda:{}".format(i) for i in range(len(model_names))]
    instance = Nudging(
        model_names=model_names,
        model_devices=model_devices,
    )
    prompts = [
        "Explain the theory of relativity in simple terms. Explain in one sentence.",
        "What are the benefits of using renewable energy sources?",
        "Another man might have thrown up his.hands\u2014but not Nawabdin. His twelve daughters.acted as a spur to his genius, and he looked with.Line satisfaction in the mirror each morning at the face of.a warrior going out to do battle. Nawab of course.knew that he must proliferate his sources of.revenue\u2014the salary he received from K. K. Harouni.for tending the tube wells would not even begin to.suffice. He set up a little one-room flour mill, run off.a condemned electric motor\u2014condemned by him..He tried his hand at fish-farming in a little pond at.the edge of his master\u2019s fields. He bought broken.radios, fixed them, and resold them. He did not.demur even when asked to fix watches, though that.enterprise did spectacularly badly, and in fact earned.him more kicks than kudos, for no watch he took.apart ever kept time again..K. K. Harouni rarely went to his farms, but lived.mostly in Lahore. Whenever the old man visited,.Nawab would place himself night and day at the door.leading from the servants\u2019 sitting area into the walled.grove of ancient banyan trees where the old.farmhouse stood. Grizzled, his peculiar aviator.glasses bent and smudged, Nawab tended the.household machinery, the air conditioners, water.heaters, refrigerators, and water pumps, like an.engineer tending the boilers on a foundering steamer.in an Atlantic gale. By his superhuman efforts he.almost managed to maintain K. K. Harouni in the.same mechanical cocoon, cooled and bathed and.lighted and fed, that the landowner enjoyed in.Lahore..Harouni of course became familiar with this.ubiquitous man, who not only accompanied him on.his tours of inspection, but morning and night could.be found standing on the master bed rewiring the.light fixture or in the bathroom poking at the water.heater. Finally, one evening at teatime, gauging the.psychological moment, Nawab asked if he might say.a word. The landowner, who was cheerfully filing his.nails in front of a crackling rosewood fire, told him.to go ahead..\u201cSir, as you know, your lands stretch from here to.the Indus, and on these lands are fully seventeen tube.wells, and to tend these seventeen tube wells there is.but one man, me, your servant. In your service I have.earned these gray hairs\u201d\u2014here he bowed his head to.show the gray\u2014\u201cand now I cannot fulfill my duties.as I should. Enough, sir, enough. I beg you, forgive.me my weakness. Better a darkened house and proud.hunger within than disgrace in the light of day..Release me, I ask you, I beg you.\u201d.The old man, well accustomed to these sorts of.speeches, though not usually this florid, filed away at.his nails and waited for the breeze to stop..\u201cWhat\u2019s the matter, Nawabdin?\u201d.Unauthorized copying or reuse of any part of this page is illegal. **22 CONTINUE**.\u201cMatter, sir? O what could be the matter in your.service. I\u2019ve eaten your salt for all my years. But sir,.on the bicycle now, with my old legs, and with the.many injuries I\u2019ve received when heavy machinery.fell on me\u2014I cannot any longer bicycle about like a.bridegroom from farm to farm, as I could when I.first had the good fortune to enter your employment..I beg you, sir, let me go.\u201d.\u201cAnd what\u2019s the solution?\u201d asked Harouni, seeing.that they had come to the crux. He didn\u2019t particularly.care one way or the other, except that it touched on.his comfort\u2014a matter of great interest to him..\u201cWell, sir, if I had a motorcycle, then I could.somehow limp along, at least until I train up some.younger man.\u201d.The crops that year had been good, Harouni felt.expansive in front of the fire, and so, much to the.disgust of the farm managers, Nawab received a.brand-new motorcycle, a Honda 70. He even.managed to extract an allowance for gasoline..The motorcycle increased his status, gave him.weight, so that people began calling him \u201cUncle,\u201d and.asking his opinion on world affairs, about which he.knew absolutely nothing. He could now range.further, doing a much wider business. Best of all,.now he could spend every night with his wife, who.had begged to live not on the farm but near her.family in Firoza, where also they could educate at.least the two eldest daughters. A long straight road.ran from the canal headworks near Firoza all the way.to the Indus, through the heart of the K. K. Harouni.lands. Nawab would fly down this road on his new.machine, with bags and cloths hanging from every.knob and brace, so that the bike, when he hit a bump,.seemed to be flapping numerous small vestigial.wings; and with his grinning face, as he rolled up to.whichever tube well needed servicing, with his ears.almost blown off, he shone with the speed of his.arrival. It can reasonably be inferred from the passage that Harouni provides Nawab with a motorcycle mainly because?",
    ]
    for nudging_model_id in range(len(model_names)):
        if nudging_model_id == instance.base_model_id:
            continue
        outputs = instance.batch_generate(
            prompts,
            batch_size=3,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            gamma=0.4,
            base_model_id=0,
            nudging_model_id=nudging_model_id,
        )
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}\nOutput: {output}\n")
            
        outputs = instance.batch_generate(
            prompts,
            batch_size=3,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            gamma=0.4,
            base_model_id=0,
            nudging_model_id=nudging_model_id,
            use_kv_cache=True,
        )
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}\nOutput: {output}\n")  