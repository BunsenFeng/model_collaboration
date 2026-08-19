"""
Helper functions for distributed GRPO.
"""

import inspect
import json
import os
import random
import re
import shutil
import threading
from multiprocessing import Pool, get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from accelerate.utils import gather, gather_object, is_peft_model
except Exception:  # pragma: no cover
    gather = None
    gather_object = None
    is_peft_model = lambda _: False

try:
    from trl.models import unwrap_model_for_generation
except Exception:  # pragma: no cover
    unwrap_model_for_generation = None

try:
    from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
except Exception:  # pragma: no cover
    apply_chat_template = None
    maybe_apply_chat_template = None
    is_conversational = lambda _: False

try:
    from model_collaboration.utils import lora_check
except Exception:  # pragma: no cover
    lora_check = None


_SCORE_PATTERNS = [
    r'{\s*"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*}',
    r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    r'score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
    r'Score:\s*([0-9]+(?:\.[0-9]+)?)',
    r'([0-9]+(?:\.[0-9]+)?)\s*/\s*10',
]


def _is_lora_adapter(path_or_name: str) -> bool:
    if lora_check is not None:
        try:
            return bool(lora_check.is_lora_adapter_peft(path_or_name))
        except Exception:
            pass
    return os.path.exists(os.path.join(path_or_name, "adapter_config.json"))


def _completion_to_text(completion: Any) -> str:
    """Handle standard or conversational TRL completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        # Conversational completion: [{'role':'assistant', 'content':'...'}]
        if isinstance(completion[0], dict):
            return str(completion[0].get("content", ""))
        return "".join(str(x) for x in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def _extract_score_1_to_10(text: Optional[str], default: float = 1.0) -> float:
    if text is None:
        return float(default)
    s = str(text).strip()
    try:
        payload = json.loads(s)
        if isinstance(payload, dict) and "score" in payload:
            v = float(payload["score"])
            if 1.0 <= v <= 10.0:
                return v
    except Exception:
        pass
    for pattern in _SCORE_PATTERNS:
        for match in re.findall(pattern, s, flags=re.IGNORECASE):
            try:
                v = float(match)
                if 1.0 <= v <= 10.0:
                    return v
            except Exception:
                continue
    return float(default)


def _build_judge_prompt(question: str, response: str) -> str:
    return f"""
Please judge the following response based on the question and the response to be evaluated.

Question: {question}

Response to be evaluated: {response}

Operation: Output ONLY a JSON object with one score in this exact format. Score must be in the range of 1 to 10.
Your output should be like this:
{{"score": score}}
""".strip()


def _apply_chat_template_or_raw(tokenizer: Any, prompts: List[str]) -> List[str]:
    rendered: List[str] = []
    try:
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            rendered.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        return rendered
    except Exception:
        return prompts


def _disable_cache_for_training(model: Any) -> Any:
    """
    Disable KV cache anywhere it may live: model wrapper, PEFT wrapper,
    base model, config, and generation_config.
    """
    seen = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, "config") and obj.config is not None:
            obj.config.use_cache = False

        if hasattr(obj, "generation_config") and obj.generation_config is not None:
            obj.generation_config.use_cache = False

        # PEFT / Accelerate / wrapper variants.
        for attr in ["base_model", "model", "module"]:
            try:
                visit(getattr(obj, attr, None))
            except Exception:
                pass

        try:
            if hasattr(obj, "get_base_model"):
                visit(obj.get_base_model())
        except Exception:
            pass

    visit(model)
    return model


def _build_model_for_training(model_name: str, torch_dtype: torch.dtype = torch.bfloat16) -> Any:
    """Load a base model or merge an incoming PEFT adapter before adding a fresh LoRA adapter."""
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    if _is_lora_adapter(model_name):
        print(f"[GRPO] Detected adapter at {model_name}; merging before new GRPO LoRA training.")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = peft_model.merge_and_unload()
        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return _disable_cache_for_training(model)


def _build_model_and_tokenizer_for_judge(
    model_name: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any]:
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if device.startswith("cuda") and torch.cuda.is_available():
        local_index = int(device.split(":", 1)[1]) if ":" in device else 0
        kwargs["device_map"] = {"": local_index}
    if _is_lora_adapter(model_name):
        model = AutoPeftModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def _score_with_one_judge(
    judge_model_path: str,
    judge_device: str,
    prompts: List[str],
    completions: List[str],
    batch_size: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[float], List[str]]:
    judge_prompts = [_build_judge_prompt(q, r) for q, r in zip(prompts, completions)]
    model, tokenizer = _build_model_and_tokenizer_for_judge(judge_model_path, judge_device)
    all_scores: List[float] = []
    all_raw_outputs: List[str] = []
    try:
        for start in range(0, len(judge_prompts), batch_size):
            chunk = judge_prompts[start : start + batch_size]
            rendered = _apply_chat_template_or_raw(tokenizer, chunk)
            inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_response_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            all_raw_outputs.extend(decoded)
            all_scores.extend([_extract_score_1_to_10(x) for x in decoded])
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_scores, all_raw_outputs


class ReputationWeightedPeerJudgeReward:
    """Callable reward function passed into GRPOTrainer.

    The reward is computed online from the freshly generated completions:
      reward = sum_j reputation[j] * score_j(prompt, completion) / sum_j reputation[j]

    The raw judge score is in [1, 10]. Returned reward is either raw [1,10] or
    normalized [0,1], depending on reward_scale.
    """

    def __init__(
        self,
        model_name: str,
        judge_model_paths_by_name: Dict[str, str],
        judge_weights_by_name: Dict[str, float],
        log_path: str,
        cache_path: Optional[str] = None,
        judge_device: str = "cuda:0",
        judge_batch_size: int = 4,
        judge_max_response_length: int = 64,
        judge_temperature: float = 1e-5,
        judge_top_p: float = 1.0,
        reward_scale: str = "zero_one",
        default_score: float = 1.0,
    ):
        self.model_name = model_name
        self.judge_model_paths_by_name = dict(judge_model_paths_by_name)
        self.judge_weights_by_name = {k: float(v) for k, v in judge_weights_by_name.items()}
        self.log_path = log_path
        self.cache_path = cache_path
        self.judge_device = judge_device
        self.judge_batch_size = int(judge_batch_size)
        self.judge_max_response_length = int(judge_max_response_length)
        self.judge_temperature = float(judge_temperature)
        self.judge_top_p = float(judge_top_p)
        self.reward_scale = reward_scale
        self.default_score = float(default_score)
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                except Exception:
                    self._cache = {}

    def _cache_key(self, judge_name: str, prompt: str, completion: str) -> str:
        import hashlib
        payload = json.dumps(
            {"judge": judge_name, "prompt": prompt, "completion": completion},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _flush_cache(self) -> None:
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    def _parse_judge_names(self, value: Any) -> List[str]:
        if value is None:
            return list(self.judge_model_paths_by_name.keys())
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
        return list(self.judge_model_paths_by_name.keys())

    def _normalize_reward(self, raw_score: float) -> float:
        raw_score = max(1.0, min(10.0, float(raw_score)))
        if self.reward_scale in ["raw", "one_to_ten", "1_10"]:
            return raw_score
        if self.reward_scale in ["minus_one_one", "-1_1"]:
            return 2.0 * ((raw_score - 1.0) / 9.0) - 1.0
        return (raw_score - 1.0) / 9.0

    def __call__(
        self,
        prompts: List[str],
        completions: List[Any],
        raw_prompt: Optional[List[Any]] = None,
        judge_names_json: Optional[List[Any]] = None,
        duel_id: Optional[List[Any]] = None,
        instruction_idx: Optional[List[Any]] = None,
        opponent_model: Optional[List[Any]] = None,
        trainer_state: Any = None,
        **kwargs: Any,
    ) -> List[float]:
        completion_texts = [_completion_to_text(x) for x in completions]

        if raw_prompt is not None:
            prompt_texts = [str(x) for x in raw_prompt]
        else:
            prompt_texts = [str(x) for x in prompts]

        n = len(completion_texts)
        if len(prompt_texts) != n:
            prompt_texts = (prompt_texts * n)[:n]

        per_sample_judges: List[List[str]] = []
        for i in range(n):
            value = judge_names_json[i] if judge_names_json is not None and i < len(judge_names_json) else None
            names = [j for j in self._parse_judge_names(value) if j in self.judge_model_paths_by_name]
            if not names:
                names = list(self.judge_model_paths_by_name.keys())
            per_sample_judges.append(names)

        # Score in batches grouped by judge so each judge model is loaded at most once per reward call.
        judge_scores_by_sample: List[Dict[str, float]] = [dict() for _ in range(n)]
        judge_raw_by_sample: List[Dict[str, str]] = [dict() for _ in range(n)]

        for judge_name in sorted({j for names in per_sample_judges for j in names}):
            judge_model_path = self.judge_model_paths_by_name[judge_name]
            sample_indices = [i for i, names in enumerate(per_sample_judges) if judge_name in names]
            uncached_indices: List[int] = []
            uncached_prompts: List[str] = []
            uncached_completions: List[str] = []

            for i in sample_indices:
                key = self._cache_key(judge_name, prompt_texts[i], completion_texts[i])
                cached = self._cache.get(key)
                if cached is not None:
                    judge_scores_by_sample[i][judge_name] = float(cached.get("score", self.default_score))
                    judge_raw_by_sample[i][judge_name] = str(cached.get("raw_output", "<cache>"))
                else:
                    uncached_indices.append(i)
                    uncached_prompts.append(prompt_texts[i])
                    uncached_completions.append(completion_texts[i])

            if uncached_indices:
                try:
                    scores, raw_outputs = _score_with_one_judge(
                        judge_model_path=judge_model_path,
                        judge_device=self.judge_device,
                        prompts=uncached_prompts,
                        completions=uncached_completions,
                        batch_size=self.judge_batch_size,
                        max_response_length=self.judge_max_response_length,
                        temperature=self.judge_temperature,
                        top_p=self.judge_top_p,
                    )
                except Exception as e:
                    print(f"[GRPO reward] Judge {judge_name} failed: {e}. Using default scores.")
                    scores = [self.default_score] * len(uncached_indices)
                    raw_outputs = [f"ERROR: {repr(e)}"] * len(uncached_indices)

                for i, score, raw_output in zip(uncached_indices, scores, raw_outputs):
                    judge_scores_by_sample[i][judge_name] = float(score)
                    judge_raw_by_sample[i][judge_name] = str(raw_output)
                    key = self._cache_key(judge_name, prompt_texts[i], completion_texts[i])
                    self._cache[key] = {"score": float(score), "raw_output": str(raw_output)}

        rewards: List[float] = []
        log_rows: List[Dict[str, Any]] = []
        global_step = getattr(trainer_state, "global_step", None) if trainer_state is not None else None

        for i in range(n):
            numerator = 0.0
            denominator = 0.0
            for judge_name, score in judge_scores_by_sample[i].items():
                weight = max(0.0, float(self.judge_weights_by_name.get(judge_name, 1.0)))
                if weight <= 0.0:
                    continue
                numerator += weight * float(score)
                denominator += weight
            raw_score = numerator / denominator if denominator > 0.0 else self.default_score
            reward = self._normalize_reward(raw_score)
            rewards.append(float(reward))

            log_rows.append({
                "model_name": self.model_name,
                "opponent_model": opponent_model[i] if opponent_model is not None and i < len(opponent_model) else None,
                "duel_id": int(duel_id[i]) if duel_id is not None and i < len(duel_id) and str(duel_id[i]).lstrip("-").isdigit() else duel_id[i] if duel_id is not None and i < len(duel_id) else None,
                "instruction_idx": int(instruction_idx[i]) if instruction_idx is not None and i < len(instruction_idx) and str(instruction_idx[i]).lstrip("-").isdigit() else instruction_idx[i] if instruction_idx is not None and i < len(instruction_idx) else None,
                "prompt": prompt_texts[i],
                "completion": completion_texts[i],
                "reward": float(reward),
                "raw_score_1_to_10": float(raw_score),
                "judge_scores": judge_scores_by_sample[i],
                "judge_raw_outputs": judge_raw_by_sample[i],
                "global_step": global_step,
            })

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for row in log_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._flush_cache()

        return rewards


def _make_grpo_config(**kwargs: Any) -> GRPOConfig:
    """Filter kwargs against the installed TRL GRPOConfig signature."""
    params = inspect.signature(GRPOConfig.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[GRPO] Dropping unsupported GRPOConfig kwargs for this TRL version: {dropped}")
    return GRPOConfig(**filtered)


def _set_visible_devices(training_gpu_id: int, judge_gpu_ids: Optional[Sequence[int]]) -> str:
    visible = [int(training_gpu_id)]
    for gid in judge_gpu_ids or []:
        gid = int(gid)
        if gid not in visible:
            visible.append(gid)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    if len(visible) > 1 and torch.cuda.is_available():
        return "cuda:1"
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _get_trainer_tokenizer(trainer: Any) -> Any:
    tokenizer = getattr(trainer, "processing_class", None)
    if tokenizer is None:
        tokenizer = getattr(trainer, "_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Could not find tokenizer on GRPOTrainer.")
    return tokenizer


def _get_trainer_model(trainer: Any) -> Any:
    # In your single-GPU spawned setup, trainer.model.generate works and matched your probe.
    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("Could not find model on GRPOTrainer.")
    return model


@torch.no_grad()
def _compute_completion_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Compute old-policy per-token logprobs for generated completion tokens.

    prompt_ids:      [B, P]
    completion_ids:  [B, C]
    returns:         [B, C]
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = input_ids.ne(pad_token_id).long()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits

    prompt_len = prompt_ids.shape[1]

    # Token at position t is predicted by logits at position t-1.
    # Completion tokens occupy positions [prompt_len, prompt_len + C - 1],
    # so their predicting logits are [prompt_len - 1, ..., prompt_len + C - 2].
    completion_logits = logits[:, prompt_len - 1 : -1, :]

    log_probs = torch.log_softmax(completion_logits, dim=-1)
    token_logprobs = log_probs.gather(
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)

    # Avoid meaningful logprobs on padding.
    token_logprobs = token_logprobs.masked_fill(completion_ids.eq(pad_token_id), 0.0)
    return token_logprobs.detach()


def make_stable_transformers_rollout_func(
    max_completion_length: int,
    do_sample: bool = False,
    temperature: float = 1e-5,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    log_decoded_completions_path: Optional[str] = None,
):
    """
    Custom rollout_func for TRL GRPOTrainer.

    This intentionally uses the same simple generate pattern that worked in your
    sanity checks, instead of TRL's internal rollout path.

    Important:
      - prompts received here are already the dataset prompt strings.
      - the function must repeat each prompt num_generations times.
      - it must return prompt_ids, completion_ids, and logprobs.
    """

    def stable_rollout_func(prompts: List[str], trainer: Any) -> Dict[str, Any]:
        model = _get_trainer_model(trainer)
        tokenizer = _get_trainer_tokenizer(trainer)

        num_generations = int(getattr(trainer, "num_generations", trainer.args.num_generations))
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

        if pad_token_id is None:
            pad_token_id = eos_token_id

        # TRL gives the raw per-process prompt slice with no duplication.
        # We must duplicate each prompt G times.
        expanded_prompts: List[str] = []
        original_prompt_index: List[int] = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_generations):
                expanded_prompts.append(prompt)
                original_prompt_index.append(i)

        model_was_training = model.training
        model.eval()

        try:
            tokenized = tokenizer(
                expanded_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(trainer, "max_prompt_length", None),
            )

            device = next(model.parameters()).device
            prompt_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            gen_kwargs = {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(max_completion_length),
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "use_cache": False,
            }

            if do_sample:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": max(float(temperature), 1e-5),
                        "top_p": float(top_p),
                    }
                )
            else:
                gen_kwargs.update({"do_sample": False})

            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = float(repetition_penalty)

            generated = model.generate(**gen_kwargs)

            # generated is [B, P + C]. Slice off the padded prompt length.
            completion_ids = generated[:, prompt_ids.shape[1] :]

            # Defensive: if model.generate returns no completion tokens, create one pad token.
            if completion_ids.shape[1] == 0:
                completion_ids = torch.full(
                    (prompt_ids.shape[0], 1),
                    fill_value=pad_token_id,
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )

            logprobs = _compute_completion_logprobs(
                model=model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                pad_token_id=pad_token_id,
            )

            decoded_completions = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )

            if log_decoded_completions_path is not None:
                os.makedirs(os.path.dirname(log_decoded_completions_path), exist_ok=True)
                with open(log_decoded_completions_path, "a", encoding="utf-8") as f:
                    for prompt_idx, prompt, completion in zip(
                        original_prompt_index,
                        expanded_prompts,
                        decoded_completions,
                    ):
                        f.write(
                            json.dumps(
                                {
                                    "prompt_index": prompt_idx,
                                    "prompt": prompt,
                                    "completion": completion,
                                    "do_sample": bool(do_sample),
                                    "temperature": float(temperature),
                                    "top_p": float(top_p),
                                    "max_completion_length": int(max_completion_length),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            return {
                "prompt_ids": prompt_ids.detach(),
                "completion_ids": completion_ids.detach(),
                "logprobs": logprobs.detach(),

                # Extra per-completion fields. TRL forwards extra fields to reward funcs
                # in current versions. These are just for debugging; your reward func
                # can ignore them through **kwargs.
                "rollout_prompt_text": expanded_prompts,
                "rollout_completion_text": decoded_completions,
                "rollout_original_prompt_index": original_prompt_index,
            }

        finally:
            if model_was_training:
                model.train()

    return stable_rollout_func




def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move nested tensors to a device without calling GRPOTrainer._prepare_inputs."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    return batch


def _first_parameter_device(model: Any, fallback: Optional[torch.device] = None) -> torch.device:
    """Best-effort device lookup for wrapped/PEFT/Accelerate models."""
    try:
        return next(model.parameters()).device
    except Exception:
        pass

    # Some Accelerate/device_map models expose hf_device_map.
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                if isinstance(dev, str) and dev not in {"cpu", "disk", "meta"}:
                    return torch.device(dev)
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
    except Exception:
        pass

    if fallback is not None:
        return fallback
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _call_get_per_token_logps_on_model_device(
    trainer: Any,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    return_device: torch.device,
) -> torch.Tensor:
    """
    Call trainer._get_per_token_logps with input tensors on the same device as
    the model, then move the result back to return_device.

    This fixes CPU/GPU mismatch errors when ref_model is on CPU or a different
    CUDA device than the generated token tensors.
    """
    model_device = _first_parameter_device(model, fallback=return_device)
    input_ids_model = input_ids.to(model_device)
    attention_mask_model = attention_mask.to(model_device)

    # Different TRL versions have slightly different signatures. Prefer the
    # common positional call, then fall back to a keyword call if needed.
    try:
        out = trainer._get_per_token_logps(
            model,
            input_ids_model,
            attention_mask_model,
            logits_to_keep,
        )
    except TypeError:
        out = trainer._get_per_token_logps(
            model=model,
            input_ids=input_ids_model,
            attention_mask=attention_mask_model,
            logits_to_keep=logits_to_keep,
        )
    return out.to(return_device)


def _get_reward_weights_tensor(trainer: Any, device: torch.device, num_reward_funcs: int) -> torch.Tensor:
    """Return reward weights as a 1D tensor on the requested device."""
    weights = getattr(trainer, "reward_weights", None)
    if weights is None:
        return torch.ones(num_reward_funcs, dtype=torch.float32, device=device)
    if isinstance(weights, torch.Tensor):
        return weights.to(device=device, dtype=torch.float32)
    return torch.tensor(list(weights), dtype=torch.float32, device=device)


def _unwrap_for_generation_context(model: Any, accelerator: Any, args: Any):
    """
    Return a context manager yielding an unwrapped model for generation.
    Handles older/newer TRL unwrap_model_for_generation signatures.
    """
    if unwrap_model_for_generation is None:
        class _SimpleCtx:
            def __enter__(self_inner):
                return accelerator.unwrap_model(model)
            def __exit__(self_inner, exc_type, exc, tb):
                return False
        return _SimpleCtx()

    kwargs = {}
    try:
        sig = inspect.signature(unwrap_model_for_generation)
        if "gather_deepspeed3_params" in sig.parameters:
            kwargs["gather_deepspeed3_params"] = getattr(args, "ds3_gather_for_generation", False)
    except Exception:
        pass
    return unwrap_model_for_generation(model, accelerator, **kwargs)


def _examples_are_grouped_for_grpo(inputs: List[Dict[str, Any]], num_generations: int) -> bool:
    """
    Check whether inputs already look like [p0, p0, ..., p1, p1, ...].
    Older TRL normally uses a repeated sampler, but some configurations may not.
    """
    G = int(num_generations)
    if G <= 1:
        return True
    if len(inputs) % G != 0:
        return False
    for start in range(0, len(inputs), G):
        block = inputs[start:start + G]
        if not block:
            continue
        first_prompt = block[0].get("prompt")
        first_raw = block[0].get("raw_prompt", None)
        first_idx = block[0].get("instruction_idx", None)
        for ex in block[1:]:
            if ex.get("prompt") != first_prompt:
                return False
            # If metadata exists, it should also agree.
            if first_raw is not None and ex.get("raw_prompt", None) != first_raw:
                return False
            if first_idx is not None and ex.get("instruction_idx", None) != first_idx:
                return False
    return True


def _ensure_repeated_inputs_for_grpo(inputs: List[Dict[str, Any]], num_generations: int) -> List[Dict[str, Any]]:
    """
    Ensure the batch is grouped into num_generations completions per prompt.
    If TRL's sampler already did this, leave inputs unchanged. Otherwise repeat.
    """
    if _examples_are_grouped_for_grpo(inputs, num_generations):
        return inputs
    print(
        "[StableRolloutGRPOTrainer] Inputs were not grouped by num_generations; "
        "repeating each example inside subclass."
    )
    repeated: List[Dict[str, Any]] = []
    for ex in inputs:
        for _ in range(int(num_generations)):
            repeated.append(dict(ex))
    return repeated


class StableRolloutGRPOTrainer(GRPOTrainer):
    """
    Backward-compatible GRPOTrainer subclass for older TRL versions that do not
    support the constructor-level rollout_func argument.

    It overrides the old TRL generation/scoring hook (_prepare_inputs in
    TRL ~=0.15.x) and replaces only the generation path with a stable direct
    model.generate call. The loss computation remains the parent GRPO loss.
    """

    def __init__(
        self,
        *args: Any,
        stable_rollout_config: Optional[Dict[str, Any]] = None,
        stable_rollout_log_path: Optional[str] = None,
        **kwargs: Any,
    ):
        self.stable_rollout_config = dict(stable_rollout_config or {})
        self.stable_rollout_log_path = stable_rollout_log_path
        if self.stable_rollout_log_path:
            os.makedirs(os.path.dirname(self.stable_rollout_log_path), exist_ok=True)
        super().__init__(*args, **kwargs)

    def _prompt_to_text(self, example: Dict[str, Any]) -> str:
        """Return prompt text without re-templating strings that were already formatted."""
        prompt = example.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if maybe_apply_chat_template is not None:
            try:
                return maybe_apply_chat_template(example, self.processing_class)["prompt"]
            except Exception:
                pass
        return str(prompt)

    def _stable_generate_prompt_completion_ids(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Stable direct generate path, intentionally similar to the sanity checks."""
        cfg = self.stable_rollout_config
        do_sample = bool(cfg.get("do_sample", False))
        temperature = max(float(cfg.get("temperature", 1e-5)), 1e-5)
        top_p = float(cfg.get("top_p", 1.0))
        repetition_penalty = float(cfg.get("repetition_penalty", 1.0))
        max_completion_length = int(cfg.get("max_completion_length", self.max_completion_length))

        gen_kwargs = dict(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            use_cache=False,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        model_was_training = self.model.training
        try:
            self.model.eval()
            with _unwrap_for_generation_context(self.model, self.accelerator, self.args) as unwrapped_model:
                # Move inputs to the actual generation model device. This avoids
                # cuda:0/cpu or cuda:0/cuda:1 mismatches after unwrapping.
                gen_device = _first_parameter_device(unwrapped_model, fallback=self.accelerator.device)
                local_kwargs = dict(gen_kwargs)
                local_kwargs["input_ids"] = local_kwargs["input_ids"].to(gen_device)
                local_kwargs["attention_mask"] = local_kwargs["attention_mask"].to(gen_device)
                prompt_completion_ids = unwrapped_model.generate(**local_kwargs)
        finally:
            if model_was_training:
                self.model.train()

        return prompt_completion_ids

    def _maybe_log_stable_rollouts(
        self,
        prompts_text: List[str],
        completion_ids: torch.Tensor,
        completions_text: List[str],
    ) -> None:
        if not self.stable_rollout_log_path:
            return
        cfg = self.stable_rollout_config
        rows = []
        for prompt, completion in zip(prompts_text, completions_text):
            rows.append({
                "global_step": int(getattr(self.state, "global_step", -1)),
                "prompt": prompt,
                "completion": completion,
                "do_sample": bool(cfg.get("do_sample", False)),
                "temperature": float(cfg.get("temperature", 1e-5)),
                "top_p": float(cfg.get("top_p", 1.0)),
                "max_completion_length": int(cfg.get("max_completion_length", self.max_completion_length)),
            })
        with open(self.stable_rollout_log_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _prepare_inputs(self, inputs: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Override for older TRL versions where GRPO generation + reward scoring
        happen inside _prepare_inputs.
        """
        device = self.accelerator.device
        inputs = _ensure_repeated_inputs_for_grpo(inputs, int(self.num_generations))

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [self._prompt_to_text(example) for example in inputs]

        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        # Do NOT call super()._prepare_inputs here: GRPOTrainer._prepare_inputs
        # would recursively enter TRL's original generation path.
        prompt_inputs = _move_batch_to_device(prompt_inputs, device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Stable generation path replacing TRL's internal generation_config path.
        prompt_completion_ids = self._stable_generate_prompt_completion_ids(prompt_ids, prompt_mask)
        prompt_completion_ids = prompt_completion_ids.to(device)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length].to(device)
        completion_ids = prompt_completion_ids[:, prompt_length:].to(device)

        if completion_ids.numel() == 0 or completion_ids.shape[1] == 0:
            completion_ids = torch.full(
                (prompt_ids.shape[0], 1),
                fill_value=self.processing_class.pad_token_id,
                dtype=prompt_ids.dtype,
                device=device,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after first EOS token.
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if is_eos.any(dim=1).any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int().to(device)

        attention_mask = torch.cat([prompt_mask.to(device), completion_mask], dim=1).to(device)
        logits_to_keep = completion_ids.size(1)
        prompt_completion_ids = prompt_completion_ids.to(device)

        # If beta == 0, KL/reference logprobs do not affect the loss. Avoid the
        # entire reference path because it is the most common source of CPU/GPU
        # mismatch in older TRL + PEFT + Accelerate combinations.
        beta_value = float(getattr(self, "beta", getattr(self.args, "beta", 0.0)))
        with torch.inference_mode():
            if beta_value == 0.0:
                ref_per_token_logps = torch.zeros(
                    completion_ids.shape,
                    dtype=torch.float32,
                    device=device,
                )
            elif self.ref_model is not None:
                ref_per_token_logps = _call_get_per_token_logps_on_model_device(
                    trainer=self,
                    model=self.ref_model,
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=logits_to_keep,
                    return_device=device,
                )
            else:
                unwrapped = self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped, "disable_adapter"):
                    with unwrapped.disable_adapter():
                        ref_per_token_logps = _call_get_per_token_logps_on_model_device(
                            trainer=self,
                            model=self.model,
                            input_ids=prompt_completion_ids,
                            attention_mask=attention_mask,
                            logits_to_keep=logits_to_keep,
                            return_device=device,
                        )
                else:
                    ref_per_token_logps = torch.zeros(
                        completion_ids.shape,
                        dtype=torch.float32,
                        device=device,
                    )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self._maybe_log_stable_rollouts(prompts_text, completion_ids, completions_text)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                try:
                    prompt_copy = list(prompt)
                    bootstrap = prompt_copy.pop()["content"] if prompt_copy and prompt_copy[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
                except Exception:
                    completions.append([{"role": "assistant", "content": completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if apply_chat_template is None:
                    raise RuntimeError("Reward model path requires trl.data_utils.apply_chat_template")
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = _move_batch_to_device(reward_inputs, device)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0].to(device)
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example.get(key) for example in inputs] for key in keys}
                try:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        trainer_state=self.state,
                        **reward_kwargs,
                    )
                except TypeError:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        **reward_kwargs,
                    )
                if len(output_reward_func) != len(prompts):
                    raise ValueError(
                        f"Reward function returned {len(output_reward_func)} rewards for {len(prompts)} completions."
                    )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if gather is not None:
            rewards_per_func = gather(rewards_per_func)

        reward_weights = _get_reward_weights_tensor(self, device, len(self.reward_funcs))
        rewards = (rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)

        if rewards.numel() % int(self.num_generations) != 0:
            raise ValueError(
                f"StableRolloutGRPOTrainer expected number of rewards ({rewards.numel()}) "
                f"to be divisible by num_generations={self.num_generations}. "
                "This usually means the GRPO sampler did not repeat prompts as expected."
            )

        # Compute group-wise advantages.
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        std_grouped_rewards = grouped_rewards.std(dim=1, unbiased=False)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice local process part after gather.
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice].to(device)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)

            metric_key = f"rewards/{reward_func_name}"
            self._metrics.setdefault(metric_key, []).append(reward_per_func[i].item())

        self._metrics.setdefault("reward", []).append(rewards.mean().item())
        self._metrics.setdefault("reward_std", []).append(std_grouped_rewards.mean().item())


        return {
            "prompt_ids": prompt_ids.to(device),
            "prompt_mask": prompt_mask.to(device),
            "completion_ids": completion_ids.to(device),
            "completion_mask": completion_mask.to(device),
            "ref_per_token_logps": ref_per_token_logps.to(device),
            "old_per_token_logps": None,
            "advantages": advantages.to(device),
        }

def single_grpo_with_judges(
    model_name: str,
    train_data_path: str,
    gpu_id: int,
    output_model_path: str,
    original_model_name: str,
    judge_model_paths_by_name: Dict[str, str],
    judge_weights_by_name: Dict[str, float],
    judge_gpu_ids: Optional[Sequence[int]] = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,
    epoch: float = 1.0,
    num_generations: int = 4,
    max_prompt_length: Optional[int] = None,
    max_completion_length: int = 32,
    beta: float = 0.0,
    epsilon: float = 0.2,
    scale_rewards: Union[bool, str] = "group",
    loss_type: str = "grpo",
    reward_scale: str = "zero_one",
    judge_batch_size: int = 4,
    judge_max_response_length: int = 64,
    temperature: float = 0.7,
    force_greedy_rollout: bool = True,
    top_p: float = 0.8,
    judge_temperature: float = 1e-5,
    judge_top_p: float = 1.0,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    save_steps: int = 1000,
    logging_steps: int = 10,
    seed: int = 42,
    # Default False under trl>=1.0: MoCo's custom stable_rollout_func (trl's
    # experimental rollout_func hook) has an internal count mismatch in trl
    # 1.10.0 -- the reward function receives completions correctly expanded
    # by num_generations but prompts/dataset-extra-columns un-expanded,
    # which trl's own reward-count validation then rejects. trl's built-in
    # (non-custom) rollout path does not hit this. If custom_rollout was
    # originally added to work around a different trl<1.0 issue, that issue
    # is untested here -- flip this back to True if problems resurface on
    # an older trl.
    use_custom_rollout: bool = False,
    rollout_do_sample: bool = True,
    rollout_temperature: float = 0.7,
    rollout_top_p: float = 0.8,
    rollout_repetition_penalty: float = 1.0,
) -> str:
    """Run online GRPO for one model with a reputation-weighted peer-judge reward."""
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(train_data_path)

    judge_device = _set_visible_devices(gpu_id, judge_gpu_ids)
    random.seed(seed)
    torch.manual_seed(seed)

    effective_batch = int(batch_size) * int(gradient_accumulation_steps)
    if effective_batch % int(num_generations) != 0:
        raise ValueError(
            "For TRL GRPO, batch_size * gradient_accumulation_steps must be divisible "
            f"by num_generations. Got {batch_size} * {gradient_accumulation_steps} "
            f"% {num_generations} != 0."
        )

    def _format_policy_prompt(tokenizer, prompt: str) -> str:
        try:
            if "<begin>" in prompt:
                question, partial_response = prompt.split("<begin>", 1)
                chat = [{"role": "user", "content": question}]
                chat_input = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return chat_input + partial_response

            chat = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[GRPO] apply_chat_template failed for tokenizer={type(tokenizer)}: {repr(e)}")
            return (
                "You are a helpful assistant. Answer the following multiple-choice "
                "question. Choose the correct letter and give a brief explanation.\n\n"
                f"{prompt}\n\nAnswer:"
            )

    dataset = load_dataset("json", data_files=train_data_path, split="train")

    if "instruction" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("instruction", "prompt")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Keep original question for the judge.
    if "raw_prompt" not in dataset.column_names:
        dataset = dataset.map(lambda ex: {"raw_prompt": ex["prompt"]})

    # Use chat-formatted prompt for the policy model.
    dataset = dataset.map(
        lambda ex: {"prompt": _format_policy_prompt(tokenizer, ex["raw_prompt"])}
    )

    if os.path.exists(output_model_path):
        print(f"[GRPO] Output path {output_model_path} exists; deleting it.")
        shutil.rmtree(output_model_path)
    os.makedirs(output_model_path, exist_ok=True)

    reward_log_dir = os.path.join(output_model_path, "reward_logs")
    os.makedirs(reward_log_dir, exist_ok=True)
    reward_log_path = os.path.join(reward_log_dir, "online_rewards.jsonl")
    reward_cache_path = os.path.join(reward_log_dir, "judge_cache.json")

    reward_func = ReputationWeightedPeerJudgeReward(
        model_name=original_model_name,
        judge_model_paths_by_name=judge_model_paths_by_name,
        judge_weights_by_name=judge_weights_by_name,
        log_path=reward_log_path,
        cache_path=reward_cache_path,
        judge_device=judge_device,
        judge_batch_size=judge_batch_size,
        judge_max_response_length=judge_max_response_length,
        judge_temperature=judge_temperature,
        judge_top_p=judge_top_p,
        reward_scale=reward_scale,
    )
    reward_func.__name__ = "reputation_weighted_peer_judge_reward"


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = _build_model_for_training(model_name)
    model = _disable_cache_for_training(model)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=None,
    )

    generation_kwargs = None
    safe_temperature = max(float(temperature), 1e-5)

    if force_greedy_rollout:
        generation_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    args = _make_grpo_config(
        output_dir=output_model_path,
        per_device_train_batch_size=int(batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        learning_rate=float(learning_rate),
        temperature=float(safe_temperature),
        generation_kwargs=generation_kwargs,
        top_p=float(top_p),
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=True,
        gradient_checkpointing=True,
        num_train_epochs=float(epoch),
        logging_strategy="steps",
        logging_steps=int(logging_steps),
        save_strategy="steps",
        save_steps=int(save_steps),
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=[],
        num_generations=int(num_generations),
        max_prompt_length=max_prompt_length,
        max_completion_length=int(max_completion_length),
        beta=float(beta),
        epsilon=float(epsilon),
        scale_rewards=scale_rewards,
        loss_type=loss_type,
        seed=int(seed),
    )

    rollout_func = None
    rollout_log_path = os.path.join(reward_log_dir, "custom_rollout_completions.jsonl")
    trainer_cls = GRPOTrainer
    use_constructor_rollout_func = False
    use_subclass_rollout_override = False

    if use_custom_rollout:
        rollout_func = make_stable_transformers_rollout_func(
            max_completion_length=max_completion_length,
            do_sample=bool(rollout_do_sample),
            temperature=float(rollout_temperature),
            top_p=float(rollout_top_p),
            repetition_penalty=float(rollout_repetition_penalty),
            log_decoded_completions_path=rollout_log_path,
        )
        if "rollout_func" in inspect.signature(GRPOTrainer.__init__).parameters:
            use_constructor_rollout_func = True
            print("[GRPO] Using constructor rollout_func supported by installed TRL.")
        else:
            use_subclass_rollout_override = True
            trainer_cls = StableRolloutGRPOTrainer
            print("[GRPO] Installed TRL has no rollout_func; using StableRolloutGRPOTrainer._prepare_inputs override.")
        
    model = _disable_cache_for_training(model)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if use_constructor_rollout_func and rollout_func is not None:
        trainer_kwargs["rollout_func"] = rollout_func
    elif use_subclass_rollout_override:
        trainer_kwargs["stable_rollout_config"] = {
            "max_completion_length": int(max_completion_length),
            "do_sample": bool(rollout_do_sample),
            "temperature": float(rollout_temperature),
            "top_p": float(rollout_top_p),
            "repetition_penalty": float(rollout_repetition_penalty),
        }
        trainer_kwargs["stable_rollout_log_path"] = rollout_log_path

    trainer = trainer_cls(**trainer_kwargs)
    # _set_visible_devices() above deliberately exposes multiple GPUs here --
    # the training GPU plus every judge GPU, since judge models need to be
    # reachable directly in-process for reward computation. But that also
    # means HF Trainer's n_gpu reads torch.cuda.device_count() > 1 (with
    # local_rank left at its -1 default) and wraps the TRAINABLE model in
    # nn.DataParallel across all of those visible GPUs -- including the ones
    # meant to hold judge models, not replicas of the training model -- which
    # OOMs on backward. Overriding _n_gpu=1 disables that wrap (same fix as
    # text_agglm.py's separate GRPO training path, api_trained_router.py's
    # and api_switch_generation.py's SFT).
    trainer.args._n_gpu = 1
    trainer.model = _disable_cache_for_training(trainer.model)
    trainer.train()
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Save enough metadata for aggregation/debugging.
    with open(os.path.join(output_model_path, "grpo_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "original_model_name": original_model_name,
                "input_model_name": model_name,
                "train_data_path": train_data_path,
                "reward_log_path": reward_log_path,
                "judge_model_paths_by_name": judge_model_paths_by_name,
                "judge_weights_by_name": judge_weights_by_name,
                "num_generations": num_generations,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    del trainer, model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_model_path



def _single_grpo_with_judges_star(arg_tuple: Tuple[Any, ...]) -> str:
    (
        model_name,
        train_data_path,
        gpu_id,
        output_model_path,
        original_model_name,
        judge_paths,
        judge_weights,
        local_judge_gpu_ids,
        kwargs,
    ) = arg_tuple
    
    return single_grpo_with_judges(
        model_name=model_name,
        train_data_path=train_data_path,
        gpu_id=gpu_id,
        output_model_path=output_model_path,
        original_model_name=original_model_name,
        judge_model_paths_by_name=judge_paths,
        judge_weights_by_name=judge_weights,
        judge_gpu_ids=local_judge_gpu_ids,
        **kwargs,
    )

def _make_gpu_groups(
    training_gpu_ids: Sequence[int],
    judge_gpu_ids: Optional[Sequence[int]] = None,
    gpus_per_job: int = 2,
) -> List[Tuple[int, List[int]]]:
    """
    Return groups like:
      [(train_gpu, [judge_gpu]), ...]

    For 4 GPUs and gpus_per_job=2:
      [0,1,2,3] -> [(0, [1]), (2, [3])]
    """
    all_gpus = list(judge_gpu_ids) if judge_gpu_ids is not None else list(training_gpu_ids)
    all_gpus = [int(x) for x in all_gpus]

    if gpus_per_job <= 1:
        return [(int(g), []) for g in training_gpu_ids]

    groups = []
    for start in range(0, len(all_gpus), gpus_per_job):
        chunk = all_gpus[start : start + gpus_per_job]
        if not chunk:
            continue
        train_gpu = chunk[0]
        local_judges = chunk[1:]
        groups.append((train_gpu, local_judges))

    if not groups:
        groups = [(int(training_gpu_ids[0]), [])]

    return groups

def distributed_grpo_with_judges(
    list_of_model_names: List[str],
    list_of_train_data_paths: List[str],
    list_of_gpu_ids: List[int],
    list_of_output_model_paths: List[str],
    list_of_original_model_names: List[str],
    judge_model_paths_by_name: Dict[str, str],
    judge_weights_by_name: Dict[str, float],
    judge_gpu_ids: Optional[Sequence[int]] = None,
    parallel_training: bool = False,
    **grpo_kwargs: Any,
) -> List[str]:
    """Run GRPO for many models.

    Default is sequential because online peer judging loads extra LMs during reward
    computation; parallel training can easily OOM or make several processes fight
    over judge GPUs. Set parallel_training=True only when you have enough GPUs.
    """
    n = len(list_of_model_names)
    assert len(list_of_train_data_paths) == n
    assert len(list_of_output_model_paths) == n
    assert len(list_of_original_model_names) == n
    if not list_of_gpu_ids:
        list_of_gpu_ids = [0]

    gpus_per_grpo_job = int(grpo_kwargs.pop("gpus_per_grpo_job", 2))
    gpu_groups = _make_gpu_groups(
        training_gpu_ids=list_of_gpu_ids,
        judge_gpu_ids=judge_gpu_ids,
        gpus_per_job=gpus_per_grpo_job,
    )

    args = []
    for idx in range(n):
        train_gpu, local_judge_gpu_ids = gpu_groups[idx % len(gpu_groups)]
        args.append((
            list_of_model_names[idx],
            list_of_train_data_paths[idx],
            train_gpu,
            list_of_output_model_paths[idx],
            list_of_original_model_names[idx],
            judge_model_paths_by_name,
            judge_weights_by_name,
            local_judge_gpu_ids,
            grpo_kwargs,
        ))

    def _run_one_grpo_job_spawned(arg):
        ctx = get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.map(_single_grpo_with_judges_star, [arg])[0]

    max_parallel_jobs = int(grpo_kwargs.pop("max_parallel_grpo_jobs", len(gpu_groups)))

    if parallel_training:
        ctx = get_context("spawn")
        with ctx.Pool(min(n, max_parallel_jobs, len(gpu_groups))) as pool:
            return pool.map(_single_grpo_with_judges_star, args)

    return [_run_one_grpo_job_spawned(arg) for arg in args]
