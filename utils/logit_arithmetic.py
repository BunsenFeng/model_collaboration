import json
import torch
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

def average_logits(logits_list):
    summed_logits = torch.zeros_like(logits_list[0])
    for logits in logits_list:
        summed_logits += logits
    avg_logits = summed_logits / len(logits_list)
    return avg_logits

class LogitArithmetic:
    def __init__(
        self,
        model_names,
        model_devices,
        tokenizer,
        system_prompt = None,
        model_kwargs = None
    ):
        self.models = [
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(model_devices[i]) for i, model_name in enumerate(model_names)
        ]
        for model in self.models:
            model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
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
        
    def forward(
        self,
        inputs,
        return_dict = None
    ):
        list_of_outputs = []
        for i, model in enumerate(self.models):
            # Create a copy of inputs for each model to avoid modifying the original
            model_inputs = {k: v.to(self.model_devices[i]) if hasattr(v, 'to') else v for k, v in inputs.items()}
            list_of_outputs.append(
                self.models[i](**model_inputs, return_dict=return_dict)
            )
        return list_of_outputs

    def generate(
        self,
        arithmetic_func = average_logits,
        input_ids = None,
        max_new_tokens = 100,
        do_sample = True,
        temprature = 1.0,
        logits_processor = None,
        stopping_criteria = None,
        **kwargs
    ):
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor(self.tokenizer.eos_token_id).to(input_ids.device)

        for step in tqdm(range(max_new_tokens)):
            inputs = {'input_ids': input_ids, **kwargs}
            list_of_outputs = self.forward(inputs, return_dict=True) 
            logits_list = [output.logits[..., -1, :] for output in list_of_outputs]
            # Move all logits to the same device as input_ids
            for i in range(len(logits_list)):
                logits_list[i] = logits_list[i].to(input_ids.device)
            merged_logits = arithmetic_func(logits_list)
            if logits_processor is not None:
                merged_logits = logits_processor(input_ids, merged_logits)
            if temprature != 1.0:
                merged_logits = merged_logits / temprature
            if do_sample:
                probs = F.softmax(merged_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(merged_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if stopping_criteria and stopping_criteria(input_ids, None):
                break
            
            # update unfinished sequences
            # unfinished_sequences = unfinished_sequences.mul(
            #     next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            # )

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(eos_token_id_tensor).long()
            )

            # if all sequences are finished, stop
            if unfinished_sequences.max() == 0:
                break
        return input_ids

    def batch_generate(
        self,
        prompts,
        tokenizer,
        batch_size = 1,
        max_new_tokens = 100,
        do_sample = True,
        temprature = 1.0,
        arithmetic_func = average_logits
    ):
        chat_prompts = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            chat_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        
        input_ids = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

        outputs = []
        for i in tqdm(range(0, input_ids.shape[0], batch_size)):
            batch_input_ids = input_ids[i:i+batch_size]
            generated_ids = self.generate(
                input_ids=batch_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temprature=temprature,
                arithmetic_func=arithmetic_func
            )
            for j in range(generated_ids.shape[0]):
                output_text = tokenizer.decode(
                    generated_ids[j][batch_input_ids[j].shape[0]:],
                    skip_special_tokens=True,
                )
                outputs.append(output_text)
        return outputs

if __name__ == "__main__":
    instance = LogitArithmetic(
        model_names=[
            "allenai/Llama-3.1-Tulu-3-8B-SFT",
            "allenai/Llama-3.1-Tulu-3-8B-DPO",
            "allenai/Llama-3.1-Tulu-3-8B"
        ],
        model_devices=["cuda:0", "cuda:1", "cuda:2"],
        tokenizer=AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B"),
    )
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "What are the benefits of using renewable energy sources?",
    ]
    outputs = instance.batch_generate(
        prompts,
        tokenizer=instance.tokenizer,
        batch_size=1,
        max_new_tokens=100,
        do_sample=True,
        temprature=1.0,
    )
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nOutput: {output}\n")
