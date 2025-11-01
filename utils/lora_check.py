import os
import shutil
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
    for i in range(len(model_names)):
        if is_lora_adapter_peft(model_names[i]):
            model = AutoPeftModelForCausalLM.from_pretrained(model_names[i], torch_dtype="bfloat16")
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(model_names[i])
            full_model_name = "logs/" + model_names[i].split("/")[-1] + "_full"
            if os.path.exists(full_model_name):
                shutil.rmtree(full_model_name)
            model.save_pretrained(full_model_name)
            tokenizer.save_pretrained(full_model_name)
            model_names[i] = full_model_name
    return model_names