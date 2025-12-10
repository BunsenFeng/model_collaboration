import json
import math
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)


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
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, input_len:]
    segment = tokenizer.decode(new_ids, skip_special_tokens=True)
    return segment

def generate_query_prompt(generator_segment, mentor_segment):
    return f"\nNow I will choose the next sequence that could lead to the correct answer. Option (A): {generator_segment}, Option (B): {mentor_segment}. My choice: Option ("

class MentorCollab:
    def __init__(
        self, 
        generator, 
        mentor,
        generator_devices,
        mentor_devices,
        decision_proportion = 25,
        patch_size = 16,
    ):
        self.generator = AutoModelForCausalLM.from_pretrained(generator, torch_dtype=torch.bfloat16).to(generator_devices)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator)
        self.mentor = AutoModelForCausalLM.from_pretrained(mentor, torch_dtype=torch.bfloat16).to(mentor_devices)
        self.mentor_tokenizer = AutoTokenizer.from_pretrained(mentor)
        self.generator.eval()
        self.mentor.eval()
        self.decision_proportion = decision_proportion
        self.patch_size = patch_size

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
                    query_id, query_prob = generating_next_token_with_probability(self.generator, query_prompt, self.generator_tokenizer)
                    query_token = self.generator_tokenizer.decode(query_id)
                    print(f"Query token: {query_token}")
                    if query_token == 'B':
                        next_token = mentor_next_segment
                    else:
                        next_token = generator_next_segment
            generated_text += next_token
            current_prompt = original_prompt + generated_text
        return generated_text

if __name__ == "__main__":
    generator = "meta-llama/Llama-3.1-8B"
    mentor = "meta-llama/Llama-3.1-8B-Instruct"
    generator_devices = "cuda:0"
    mentor_devices = "cuda:1"
    mentor_collab = MentorCollab(generator, mentor, generator_devices, mentor_devices)
    print(mentor_collab.generate("Hello, how are you?"))