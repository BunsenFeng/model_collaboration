from __future__ import annotations

import gc
import os
from multiprocessing import get_context, Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import _dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftConfig

try:
    from model_collaboration.utils import lora_check
except Exception:  # pragma: no cover
    lora_check = None


# global hyperparameters for generation
MAX_RESPONSE_LENGTH = None
TEMPERATURE = None
TOP_P = None
BATCH_SIZE = None
BIG_MODEL_MODE = None
LOAD_IN_8BIT = False

# Models to exclude from 8-bit quantization (fall back to bf16 for these)
NO_8BIT_MODELS = {
    "openai/gpt-oss-20b",       # trust_remote_code conflicts with bitsandbytes
    "google/gemma-3-12b-it",    # CUDA device-side assert with 8-bit
}

# Qwen3 models support enable_thinking=False in apply_chat_template
QWEN3_MODELS = {
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
}

# gpt-oss-20b emits a spurious "assistantfinal" prefix before the answer due to chat template
STRIP_ASSISTANT_FINAL_MODELS = {
    "openai/gpt-oss-20b",
}

def _strip_assistant_final(text: str) -> str:
    # gpt-oss-20b appends "assistantfinal<ANSWER>" at the end; extract just the answer
    idx = text.lower().find("assistantfinal")
    if idx >= 0:
        return text[idx + len("assistantfinal"):].strip()
    return text

def _register_embedding_long_hook(model):
    """Recover Long embedding indices at the embedding boundary.
    Loading bfloat16 models can cause input_ids to be cast to a float dtype under
    some transformers/PEFT versions, breaking nn.Embedding lookups. Only float32/
    float64 indices are recovered; fp16/bf16 indices are rejected as they are lossy."""
    import torch.nn as nn
    def _cast_to_long(module, args):
        def to_long(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                if x.dtype in (torch.float16, torch.bfloat16):
                    raise TypeError(
                        f"nn.Embedding received a {x.dtype} index tensor; casting to "
                        f"Long would silently corrupt token IDs."
                    )
                return x.long()
            return x
        return tuple(to_long(x) for x in args)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.register_forward_pre_hook(_cast_to_long)


def _load_model(model_name, device_map, load_in_8bit=False):
    if load_in_8bit and model_name not in NO_8BIT_MODELS:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config,
            device_map=device_map, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map=device_map, trust_remote_code=True
        )
    _register_embedding_long_hook(model)
    return model


def update_generation_hyperparameters(max_response_length, temperature, top_p, batch_size, big_model_mode=False, load_in_8bit=False):
    global MAX_RESPONSE_LENGTH, TEMPERATURE, TOP_P, BATCH_SIZE, BIG_MODEL_MODE, LOAD_IN_8BIT
    MAX_RESPONSE_LENGTH = max_response_length
    TEMPERATURE = temperature
    TOP_P = top_p
    BATCH_SIZE = batch_size
    BIG_MODEL_MODE = big_model_mode
    LOAD_IN_8BIT = load_in_8bit


def _is_lora_adapter(path_or_name: str) -> bool:
    if lora_check is not None:
        try:
            return bool(lora_check.is_lora_adapter_peft(path_or_name))
        except Exception:
            pass
    return os.path.exists(os.path.join(path_or_name, "adapter_config.json"))


def _normalize_visible_devices(
    gpu_id: Union[int, Sequence[int], str, None],
) -> Tuple[Optional[str], str]:
    if gpu_id is None:
        return None, "cuda:0"
    if isinstance(gpu_id, str):
        visible = gpu_id
    elif isinstance(gpu_id, (list, tuple)):
        visible = ",".join(str(item) for item in gpu_id)
    else:
        visible = str(gpu_id)
    return visible, "cuda:0"


def _tokenizer_source(model_name_or_path: str) -> str:
    # Use adapter tokenizer files when present, otherwise fall back to its base model
    local_tokenizer_markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
    )
    if any(os.path.exists(os.path.join(model_name_or_path, marker)) for marker in local_tokenizer_markers):
        return model_name_or_path
    if _is_lora_adapter(model_name_or_path):
        try:
            return str(PeftConfig.from_pretrained(model_name_or_path).base_model_name_or_path)
        except Exception:
            pass
    return model_name_or_path


def build_model_and_tokenizer_for_generation(
    model_name_or_path: str,
    gpu_id: Union[int, Sequence[int], str, None] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    # Load directly onto the physical GPU index instead of restricting
    # CUDA_VISIBLE_DEVICES + device_map={"": 0}: CUDA caches the visible
    # device list at first initialization in a process, and if anything
    # (e.g. a peft/bitsandbytes import side effect, triggered simply by
    # importing this module in a freshly-spawned worker, before this
    # function's os.environ assignment ever runs) touches CUDA first, the
    # CUDA_VISIBLE_DEVICES restriction is silently a no-op -- every
    # worker's device_map={"": 0} then resolves to the SAME literal
    # physical GPU 0 regardless of the intended per-worker gpu_id.
    # Targeting the real physical index directly sidesteps this ordering
    # fragility entirely (confirmed via instrumentation: device_count()
    # reported 4, not 1, and all workers shared one physical GPU UUID).
    target_device_index: Union[int, str] = 0
    if isinstance(gpu_id, int):
        target_device_index = gpu_id
    else:
        visible_devices, _ = _normalize_visible_devices(gpu_id)
        if visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto" if BIG_MODEL_MODE else {"": target_device_index}

    if _is_lora_adapter(model_name_or_path):
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(model_name_or_path),
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def _apply_chat_template_or_raw(tokenizer: Any, prompts: List[str], model_name: str = "") -> List[str]:
    rendered: List[str] = []
    tmpl_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_name in QWEN3_MODELS:
        tmpl_kwargs["enable_thinking"] = False
    try:
        for text in prompts:
            if "<begin>" in text:
                question, partial_response = text.split("<begin>", 1)
                chat = [{"role": "user", "content": question}]
                prefix = tokenizer.apply_chat_template(chat, **tmpl_kwargs)
                rendered.append(prefix + partial_response)
            else:
                chat = [{"role": "user", "content": text}]
                rendered.append(tokenizer.apply_chat_template(chat, **tmpl_kwargs))
        return rendered
    except Exception:
        return prompts


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _generation_kwargs(
    max_response_length: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
) -> Dict[str, Any]:
    do_sample = float(temperature) > 0.0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_response_length),
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    return kwargs


def batch_generate_text_adapter_aware(
    model_name: str,
    gpu_id: Union[int, Sequence[int], str, None],
    input_list: List[str],
    max_response_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> List[str]:
    max_response_length = MAX_RESPONSE_LENGTH if max_response_length is None else int(max_response_length)
    temperature = TEMPERATURE if temperature is None else float(temperature)
    top_p = TOP_P if top_p is None else float(top_p)
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model, tokenizer = build_model_and_tokenizer_for_generation(model_name, gpu_id)
    outputs_all: List[str] = []
    try:
        short_name = os.path.basename(model_name.rstrip("/"))
        for start in tqdm(range(0, len(input_list), batch_size), desc=f"generate:{short_name[:32]}"):
            prompts = input_list[start : start + batch_size]
            rendered = _apply_chat_template_or_raw(tokenizer, prompts, model_name=model_name)
            inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    **_generation_kwargs(
                        max_response_length=max_response_length,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    ),
                )
            decoded = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            outputs_all.extend(_strip_thinking(text) for text in decoded)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dynamo.reset_code_caches()
    return outputs_all


def batch_generate_text(
    model_name: str,
    gpu_id: Union[int, Sequence[int], str, None],
    input_list: List[str],
    max_response_length: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> List[str]:
    """Legacy-compatible wrapper."""
    return batch_generate_text_adapter_aware(
        model_name=model_name,
        gpu_id=gpu_id,
        input_list=input_list,
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )


def distributed_generation_adapter_aware(
    list_of_model_name: List[str],
    list_of_input_list: List[List[str]],
    list_of_gpu_id: List[int],
    max_response_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_parallel_models: Optional[int] = None,
    multiprocessing_start_method: str = "spawn",
) -> List[List[str]]:
    if len(list_of_model_name) != len(list_of_input_list):
        raise ValueError("Length of model names and input lists must match.")
    if any(not isinstance(inputs, list) for inputs in list_of_input_list):
        raise TypeError("Each input collection must be a list.")
    if not list_of_gpu_id:
        list_of_gpu_id = [0]

    # Resolve unset args from the module globals HERE, in the parent process, so
    # concrete values reach spawn-ed workers (children re-import the module and
    # reset these globals to None). Legacy callers pass none of these and rely on
    # update_generation_hyperparameters() having populated the globals.
    max_response_length = MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length
    temperature = TEMPERATURE if temperature is None else temperature
    top_p = TOP_P if top_p is None else top_p
    batch_size = BATCH_SIZE if batch_size is None else batch_size

    if BIG_MODEL_MODE:
        results = [
            batch_generate_text_adapter_aware(
                model_name,
                list_of_gpu_id,
                inputs,
                max_response_length=max_response_length,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
            )
            for model_name, inputs in zip(list_of_model_name, list_of_input_list)
        ]
    else:
        if max_parallel_models is None:
            max_parallel_models = len(list_of_gpu_id)
        max_parallel_models = max(
            1,
            min(int(max_parallel_models), len(list_of_gpu_id), max(len(list_of_model_name), 1)),
        )
        results: List[List[str]] = []
        context = get_context(multiprocessing_start_method)
        for start in range(0, len(list_of_model_name), max_parallel_models):
            end = min(start + max_parallel_models, len(list_of_model_name))
            arguments = []
            for local_index, global_index in enumerate(range(start, end)):
                arguments.append(
                    (
                        list_of_model_name[global_index],
                        list_of_gpu_id[local_index % len(list_of_gpu_id)],
                        list_of_input_list[global_index],
                        max_response_length,
                        temperature,
                        top_p,
                        batch_size,
                    )
                )
            if not arguments:
                continue
            with context.Pool(processes=len(arguments), maxtasksperchild=1) as pool:
                results.extend(pool.starmap(batch_generate_text_adapter_aware, arguments))

    if len(results) != len(list_of_model_name):
        raise RuntimeError("Output list length mismatch.")
    for model_name, outputs, inputs in zip(list_of_model_name, results, list_of_input_list):
        if len(outputs) != len(inputs):
            raise RuntimeError(f"Output/input length mismatch for {model_name}.")
    return results


def distributed_generation(
    list_of_model_name: List[str],
    list_of_input_list: List[List[str]],
    list_of_gpu_id: List[int],
    max_response_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_parallel_models: Optional[int] = None,
    multiprocessing_start_method: str = "spawn",
) -> List[List[str]]:
    """Legacy name retained for all existing callers."""
    return distributed_generation_adapter_aware(
        list_of_model_name=list_of_model_name,
        list_of_input_list=list_of_input_list,
        list_of_gpu_id=list_of_gpu_id,
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        max_parallel_models=max_parallel_models,
        multiprocessing_start_method=multiprocessing_start_method,
    )


def batch_generate_text_with_score(
    model_name: str,
    gpu_id: Union[int, Sequence[int], str, None],
    input_list: List[str],
    max_response_length: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> Tuple[List[str], List[List[float]]]:
    """Legacy scored generation, fixed for partial final batches and PEFT adapters."""
    model, tokenizer = build_model_and_tokenizer_for_generation(model_name, gpu_id)
    output_list: List[str] = []
    logit_scores: List[List[float]] = []
    try:
        for start in tqdm(range(0, len(input_list), batch_size), desc=f"generate-score:{model_name}"):
            prompts = input_list[start : start + batch_size]
            rendered = _apply_chat_template_or_raw(tokenizer, prompts, model_name=model_name)
            inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
            kwargs = _generation_kwargs(
                max_response_length=max_response_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            kwargs.update(return_dict_in_generate=True, output_scores=True)
            with torch.no_grad():
                generated = model.generate(**inputs, **kwargs)
            sequences = generated.sequences
            decoded = tokenizer.batch_decode(
                sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            actual_batch_size = len(decoded)
            scores_by_example: List[List[float]] = [[] for _ in range(actual_batch_size)]
            for step_logits in generated.scores:
                top_values, _ = torch.topk(step_logits, k=min(5, step_logits.shape[-1]), dim=-1)
                top_probs = torch.softmax(top_values, dim=-1)
                for row_index in range(actual_batch_size):
                    scores_by_example[row_index].append(float(top_probs[row_index].max().item()))
            output_list.extend(_strip_thinking(text) for text in decoded)
            logit_scores.extend(scores_by_example)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
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