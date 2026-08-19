import os
import shutil
import uuid
from peft import PeftConfig, AutoPeftModelForCausalLM
from peft.utils import PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_lora_adapter_peft(model_id: str) -> bool:
    """
    Checks if a model on the Hugging Face Hub is a LoRA adapter.

    Args:
        model_id: The ID of the model on the Hugging Face Hub.

    Returns:
        True if it is a LoRA adapter, False otherwise.
    """
    try:
        # Load the configuration from the Hub
        config = PeftConfig.from_pretrained(model_id)
        
        # Directly check if the PEFT type is LORA
        return config.peft_type == PeftType.LORA
        
    except Exception:
        # If PeftConfig.from_pretrained fails (e.g., no config file, 
        # missing 'adapter_config.json', or a non-PEFT model), 
        # it is not a recognizable PEFT adapter, so we return False.
        return False

def lora_to_full(model_names):
    """
    Converts any LoRA-adapter entries in model_names to full merged models on disk.

    Returns:
        tuple: (model_names, converted_dirs) where converted_dirs is the list of newly
               created full-model directories. Callers are responsible for removing these
               once they're done using them (e.g. after the final merge is saved) --
               otherwise each run permanently leaks a full merged checkpoint per LoRA input.
    """
    converted_dirs = []
    for i in range(len(model_names)):
        if is_lora_adapter_peft(model_names[i]):
            model = AutoPeftModelForCausalLM.from_pretrained(model_names[i], torch_dtype="bfloat16")
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(model_names[i])
            # Unique per call (not just per model name): two concurrent MoCo
            # runs that both reference this same LoRA model would otherwise
            # race on an identical shared path -- one run's rmtree/makedirs
            # can wipe another's in-progress save_pretrained() out from under it.
            full_model_name = "model_collaboration/logs/" + model_names[i].split("/")[-1] + "_full_" + uuid.uuid4().hex[:8]
            os.makedirs(full_model_name, exist_ok=True)
            model.save_pretrained(full_model_name)
            tokenizer.save_pretrained(full_model_name)
            model_names[i] = full_model_name
            converted_dirs.append(full_model_name)
    return model_names, converted_dirs