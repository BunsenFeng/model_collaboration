import json
import math
import sys
import os
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Any
from torchvision.ops import MLP
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoConfig
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)

MENTOR_COLLAB_TRAIN_DICT = {
    "Qwen/Qwen3-1.7B": "Qwen3_1.7B",
    "Qwen/Qwen3-8B-Base": "qwen3_8B_base",
    "meta-llama/Llama-3.1-8B": "llama3_1_8B",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3_2_3B",
    "google/gemma-3-4b-it": "gemma3_4b_it",
    "google/gemma-3-4b-pt": "gemma3_4b_pt"
}

def generating_next_token_with_probability(model, prompt, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    logprobs = torch.log_softmax(logits, dim=-1).squeeze(0)
    topk_logprobs, topk_indices = torch.topk(logprobs, 1)
    next_token_id = topk_indices[0].item()
    next_token_prob = math.exp(topk_logprobs[0].item())
    return next_token_id, next_token_prob

def complete_next_word(model, prompt, tokenizer, generator_next_token, tolerence = 10):
    process = 0
    _generator_next_token = generator_next_token
    while process < tolerence:
        if not _generator_next_token:
            break
        generator_next_token_id, generator_next_token_prob = generating_next_token_with_probability(model, prompt + _generator_next_token, tokenizer)
        buffer_token = tokenizer.decode(generator_next_token_id)
        if not buffer_token:
            break
        if buffer_token[0] == ' ' or buffer_token[0] == '\n':
            break
        else:
            _generator_next_token += buffer_token
        process += 1
    return _generator_next_token

def complete_segment(model, prompt, tokenizer, patch_size = 16):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=patch_size,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, input_len:]
    segment = tokenizer.decode(new_ids, skip_special_tokens=True)
    return segment

def generate_query_prompt(generator_segment, mentor_segment):
    return f"\nNow I will choose the next sequence that could lead to the correct answer. Option (A): {generator_segment}, Option (B): {mentor_segment}. My choice: Option ("


class BranchPredictionMLP(nn.Module):
    """
    MLP model for predicting which branch to take (A or B).

    Architecture:
        Input (hidden_size)
        → MLP (hidden_size → 2*hidden_size → hidden_size → hidden_size//2)
        → Linear (hidden_size//2 → 1)
        → Sigmoid

    Output:
        - Score between 0 and 1
        - > 0.5 means choose Option A
        - ≤ 0.5 means choose Option B
    """

    def __init__(self, hidden_size, dropout=0.1):
        """
        Args:
            hidden_size: Hidden size of the language model
            dropout: Dropout probability (default: 0.1)
        """
        super(BranchPredictionMLP, self).__init__()
        self.hidden_size = hidden_size
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=[2 * hidden_size, hidden_size, hidden_size // 2],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        """
        Forward pass.

        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size] or [batch_size, 1, hidden_size]

        Returns:
            Tensor of shape [batch_size] with scores between 0 and 1
        """
        if hidden_states.dim() == 3:
            batch_size, seq_length, hidden_size = hidden_states.size()
            if seq_length != 1:
                raise ValueError(f"Expected seq_length=1, got {seq_length}")
            hidden_states = hidden_states.squeeze(1)
        elif hidden_states.dim() == 2:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {hidden_states.shape}")
        mlp_output = self.mlp(hidden_states)
        logits = self.linear(mlp_output)
        scores = self.sigmoid(logits).squeeze(-1)
        return scores

    def predict(self, hidden_states, threshold=0.5):
        """
        Predict branch choice (A or B).

        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size]
            threshold: Threshold for choosing A vs B (default: 0.5)

        Returns:
            List of 'A' or 'B' predictions
        """
        scores = self.forward(hidden_states)
        predictions = []
        for score in scores:
            if score.item() > threshold:
                predictions.append('A')
            else:
                predictions.append('B')
        return predictions


class MentorCollab:
    def __init__(
        self,
        generator,
        mentor,
        generator_devices,
        mentor_devices,
        mode = "free",
        decision_proportion = 25,
        patch_size = 16,
        task = "General",
        mlp_threshold = 0.5
    ):
        self.generator_name = generator  # Store the model name before loading
        self.mentor_name = mentor
        self.generator = AutoModelForCausalLM.from_pretrained(generator, torch_dtype=torch.bfloat16).to(generator_devices)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator)
        self.mentor = AutoModelForCausalLM.from_pretrained(mentor, torch_dtype=torch.bfloat16).to(mentor_devices)
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor)

        # Set pad_token to avoid warnings
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        if self.mentor_tokenizer.pad_token is None:
            self.mentor_tokenizer.pad_token = self.mentor_tokenizer.eos_token

        # Clear generation config for greedy decoding to avoid warnings
        self.generator.generation_config.do_sample = False
        self.generator.generation_config.temperature = None
        self.generator.generation_config.top_p = None
        self.generator.generation_config.top_k = None
        self.mentor.generation_config.do_sample = False
        self.mentor.generation_config.temperature = None
        self.mentor.generation_config.top_p = None
        self.mentor.generation_config.top_k = None

        self.generator.eval()
        self.mentor.eval()
        self.decision_proportion = decision_proportion
        self.patch_size = patch_size
        self.mode = mode
        self.task = task
        self.generator_devices = generator_devices
        self.mentor_devices = mentor_devices
        self.mlp_model = self.load_mlp_model() if self.mode == "train" else None
        self.mlp_threshold = mlp_threshold

    def load_mlp_model(self):
        # Load MLP model
        config = AutoConfig.from_pretrained(self.generator_name)
        # Handle different config structures (Gemma3 has text_config, others have hidden_size directly)
        if hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        else:
            raise ValueError(f"Cannot find hidden_size in config for {self.generator_name}")

        # Initialize MLP model
        mlp_model = BranchPredictionMLP(hidden_size=hidden_size)

        # Download MLP weights from HuggingFace
        base_model = MENTOR_COLLAB_TRAIN_DICT[self.generator_name]
        subfolder = f"{base_model}/{self.task}"
        mlp_path = hf_hub_download(
            repo_id="SeanWang0027/MentorCollab-MLP",
            filename="branch_mlp.pth",
            subfolder=subfolder,
        )

        # Load state dict
        mlp_model.load_state_dict(torch.load(mlp_path, map_location='cpu'))
        mlp_model = mlp_model.to(self.generator_devices)
        mlp_model.eval()

        return mlp_model
    
    def extract_hidden_state_at_branch(self, query_prompt):
        branch_marker = "My choice: Option ("
        if branch_marker not in query_prompt:
            raise ValueError(f"Branch marker '{branch_marker}' not found in prompt")
        text_before_branch = query_prompt[:query_prompt.index(branch_marker)]
        input_ids = self.generator_tokenizer(text_before_branch, return_tensors="pt").input_ids
        device = next(self.generator.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = self.generator(
                input_ids=input_ids, 
                output_hidden_states=True,
                return_dict=True
            )
            hidden_state = outputs.hidden_states[-1]
            hidden_state = hidden_state[0, -1, :].cpu()
        return hidden_state
    
    def mlp_judgement(self, query_prompt):
        hidden_state = self.extract_hidden_state_at_branch(query_prompt)
        device = next(self.mlp_model.parameters()).device
        with torch.no_grad():
            hidden_state = hidden_state.unsqueeze(0).float().to(device)
            score = self.mlp_model(hidden_state).item()
            print(f"MLP score: {score}")
        choice = 'A' if score > self.mlp_threshold else 'B'
        return choice, score

    def generate(
        self,
        prompt,
        max_new_tokens = 100,
    ):
        generated_tokens = []
        token_choices = []
        current_prompt = prompt
        original_prompt = prompt
        generated_text = ""
        while True:
            newly_generated_text = current_prompt[len(original_prompt):]
            newly_generated_length = len(self.generator_tokenizer.encode(newly_generated_text, add_special_tokens=False))
            if newly_generated_length >= max_new_tokens:
                break
            generator_next_token_id, generator_next_token_prob = generating_next_token_with_probability(self.generator, current_prompt, self.generator_tokenizer)
            generator_next_token = self.generator_tokenizer.decode(generator_next_token_id)
            generator_next_token = complete_next_word(self.generator, current_prompt, self.generator_tokenizer, generator_next_token)
            random_decision = random.randint(1,100)
            if random_decision >= self.decision_proportion:
                next_token = generator_next_token
            else:
                mentor_next_token_id, mentor_next_token_prob = generating_next_token_with_probability(self.mentor, current_prompt, self.mentor_tokenizer)
                mentor_next_token = self.mentor_tokenizer.decode(mentor_next_token_id)
                mentor_next_token = complete_next_word(self.mentor, current_prompt, self.mentor_tokenizer, mentor_next_token)
                if generator_next_token == mentor_next_token:
                    next_token = generator_next_token
                else:
                    generator_next_segment = complete_segment(self.generator, current_prompt, self.generator_tokenizer, self.patch_size)
                    mentor_next_segment = complete_segment(self.mentor, current_prompt, self.mentor_tokenizer, self.patch_size)
                    query_prompt = current_prompt + generate_query_prompt(generator_next_segment, mentor_next_segment)
                    if self.mode == "train":
                        choice, score = self.mlp_judgement(query_prompt)
                        query_token = choice
                    else:
                        query_id, query_prob = generating_next_token_with_probability(self.generator, query_prompt, self.generator_tokenizer)
                        query_token = self.generator_tokenizer.decode(query_id)
                    if query_token == 'B':
                        next_token = mentor_next_segment
                    else:
                        next_token = generator_next_segment
            generated_text += next_token
            current_prompt = original_prompt + generated_text
        return generated_text

if __name__ == "__main__":
    generator = "Qwen/Qwen3-8B-Base"
    mentor = "meta-llama/Llama-3.1-8B-Instruct"
    generator_devices = "cuda:0"
    mentor_devices = "cuda:1"
    mode = "free"
    decision_proportion = 25
    patch_size = 16
    task = "General"
    mlp_threshold = 0.5
    mentor_collab = MentorCollab(generator, mentor, generator_devices, mentor_devices, mode, decision_proportion, patch_size, task, mlp_threshold)
    print(mentor_collab.generate("Hello, how are you?", max_new_tokens=256))