"""
Full PhatGoose-style token- and module-level routing for LoRA experts.

This module implements a complete version aligned with the summarized paper:

1) Per-token × per-module top-k routing. At every injected LoRA site, we compute
   normalized dot-product scores between the current token activation and each expert's
   gate vector; select top-k and combine experts by softmax weights.

2) Gate learning. For each expert and each LoRA site, we add a sigmoid gate scalar
   sigma(v^T u_t) in front of that expert's LoRA delta during training. We freeze base
   and LoRA weights, and only update gate vectors using the same (SFT) objective. We
   train gates for each expert sequentially in a single process.

3) Inference. For each LoRA site, we combine multiple experts simultaneously in the
   same forward: W u_t + sum_{z in top-k} w_{t,z} B_z A_z u_t.

Interface: provides run_method(task, task_type, gpu_ids, model_names, hyperparameters)
to fit into the model_collaboration framework.

Key modes (hyperparameters["mode"]):
- "train_and_infer": (RECOMMENDED) Train gates for all experts, then run inference.
- "train_all_gates": Train gate vectors for all experts sequentially (no inference).
- "infer_moe_full": Load pre-trained gates and run inference only.

Usage:
- Specify LoRA experts via model_names (supports HuggingFace Hub IDs or local paths).
- The base_model is auto-detected from the first adapter's config file.
- Example: model_names = ["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt"]

Gate checkpoint format (per expert): a torch file containing a dict
  {
    'meta': {...},
    'gates': { module_path: tensor[d_in] (Parameter tensor) }
  }
Where module_path matches the linear submodule path in the base model where the LoRA
adapter applied (e.g., 'model.layers.0.self_attn.q_proj').

Limitations/Notes:
- Works with CausalLMs whose LoRA adapters target nn.Linear modules.
- We extract LoRA tensors (A, B, alpha) directly from adapter state_dict, and do not
  rely on PeftModel forward merging; this allows simultaneous multi-expert combination.
- If you trained adapters with uncommon target_modules, ensure they are nn.Linear.
"""

import os
import json
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from peft import PeftModel
from model_collaboration.data import eval as eval_mod


# -----------------------------
# Small helpers
# -----------------------------

def _now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _stage(msg: str):
    print(f"[weight_phatgoose][{_now_str()}] {msg}")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _apply_chat_template_if_available(tokenizer, texts: List[str]) -> List[str]:
    try:
        chats = []
        for t in texts:
            chat = [{"role": "user", "content": t}]
            chats.append(
                tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            )
        return chats
    except Exception:
        return texts


def _configure_tokenizer(tokenizer, context_label: str = ""):
    changed = []
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        changed.append("pad_token=eos_token")
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
        changed.append("padding_side='left'")
    if changed:
        label = f" ({context_label})" if context_label else ""
        print(f"[weight_phatgoose] Decoder-only fix{label}: set " + ", ".join(changed))


# -----------------------------
# LoRA utilities
# -----------------------------

@dataclass
class LoraDelta:
    # Shapes: A: [r, d_in], B: [d_out, r]
    A: torch.Tensor
    B: torch.Tensor
    alpha: float  # lora_alpha; effective scale is alpha / r

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])


def _try_load_file(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    return {}


def _download_adapter_if_needed(adapter_path_or_id: str) -> str:
    """
    Download adapter from HuggingFace Hub if it's not a local path.
    Returns the local directory path.
    """
    if os.path.exists(adapter_path_or_id):
        return adapter_path_or_id
    
    # Try to download from HuggingFace Hub
    try:
        from huggingface_hub import snapshot_download
        _stage(f"Downloading adapter from HuggingFace Hub: {adapter_path_or_id}")
        # Download to a cache directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(script_dir, "logs", "adapter_cache")
        _ensure_dir(cache_dir)
        
        # Create a safe directory name from the repo ID
        safe_name = adapter_path_or_id.replace("/", "_")
        local_path = os.path.join(cache_dir, safe_name)
        
        # Check if already downloaded
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "adapter_config.json")):
            _stage(f"Using cached adapter at {local_path}")
            return local_path
        
        # Download
        downloaded_path = snapshot_download(
            repo_id=adapter_path_or_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
        _stage(f"Downloaded adapter to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        raise FileNotFoundError(
            f"Adapter '{adapter_path_or_id}' not found locally and failed to download from HuggingFace Hub: {e}"
        )


def _infer_base_model_from_adapter(adapter_path: str) -> Optional[str]:
    """
    Try to read base_model_name_or_path from adapter_config.json.
    Returns None if not found.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", None)
    except Exception:
        return None


def _load_adapter_state(adapter_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load a PEFT adapter state dict from a directory. Supports common filenames:
    adapter_model.bin, adapter_model.safetensors (if available), or model.safetensors.
    Also supports HuggingFace Hub IDs - will download if needed.
    """
    # Download from HuggingFace Hub if needed
    adapter_path = _download_adapter_if_needed(adapter_path)
    
    # Try PyTorch bin
    pt = _try_load_file(os.path.join(adapter_path, "adapter_model.bin"), device)
    if pt:
        if isinstance(pt, dict) and "model" in pt and isinstance(pt["model"], dict):
            return pt["model"]
        return pt
    # Try safetensors
    try:
        from safetensors.torch import load_file as safe_load

        st_path = None
        for name in ["adapter_model.safetensors", "model.safetensors"]:
            p = os.path.join(adapter_path, name)
            if os.path.exists(p):
                st_path = p
                break
        if st_path is not None:
            return safe_load(st_path, device=str(device))
    except Exception:
        pass
    raise FileNotFoundError(f"No adapter state found in {adapter_path}")


def _split_before(substring: str, s: str) -> str:
    idx = s.find(substring)
    if idx == -1:
        return s
    return s[:idx]


def _normalize_module_path_candidate(raw: str) -> List[str]:
    """
    Generate candidate dotted paths from a raw key prefix. We progressively strip
    leading common wrappers so resolution on the base model succeeds across variants.
    """
    # Remove trailing .weight if present
    if raw.endswith(".weight"):
        raw = raw[: -len(".weight")]
    # strip known PEFT wrappers around the target module path
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.model.",
        "transformer.model.",
        "transformer.",
        "model.",
    ]
    cands = set()
    cands.add(raw)
    for p in prefixes:
        if raw.startswith(p):
            cands.add(raw[len(p) :])
    # progressively drop leading segments to be robust
    parts = raw.split(".")
    for i in range(1, min(4, len(parts))):
        cands.add(".".join(parts[i:]))
    return list(cands)


def _get_module_by_path(root: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    cur = root
    if dotted_path == "":
        return None
    for name in dotted_path.split("."):
        if name.isdigit():
            # indexing into ModuleList/Sequential
            idx = int(name)
            if isinstance(cur, (nn.ModuleList, nn.Sequential)):
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
            else:
                return None
        else:
            if not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        if not isinstance(cur, nn.Module):
            return None
    return cur


def _resolve_module_path(base_model: nn.Module, raw_prefix: str) -> Optional[str]:
    for cand in _normalize_module_path_candidate(raw_prefix):
        mod = _get_module_by_path(base_model, cand)
        if isinstance(mod, nn.Linear):
            return cand
    return None


def _parse_single_adapter(
    base_model: nn.Module, adapter_state: Dict[str, torch.Tensor]
) -> Dict[str, LoraDelta]:
    """
    Return mapping: module_path -> LoraDelta, extracted from one adapter.
    """
    # Collect A/B by module raw prefix
    tmp: Dict[str, Dict[str, torch.Tensor]] = {}
    alpha_map: Dict[str, float] = {}
    for k, v in adapter_state.items():
        if ".lora_alpha" in k:
            raw = _split_before(".lora_alpha", k)
            alpha_map[raw] = float(v.item()) if v.numel() == 1 else float(v.flatten()[0].item())
        elif ".lora_A" in k and k.endswith(".weight"):
            raw = _split_before(".lora_A", k)
            tmp.setdefault(raw, {})["A"] = v
        elif ".lora_B" in k and k.endswith(".weight"):
            raw = _split_before(".lora_B", k)
            tmp.setdefault(raw, {})["B"] = v

    out: Dict[str, LoraDelta] = {}
    for raw_prefix, parts in tmp.items():
        if "A" not in parts or "B" not in parts:
            continue
        mod_path = _resolve_module_path(base_model, raw_prefix)
        if mod_path is None:
            continue
        A = parts["A"].float()
        B = parts["B"].float()
        alpha = alpha_map.get(raw_prefix, float(A.shape[0]))
        out[mod_path] = LoraDelta(A=A, B=B, alpha=alpha)
    return out


def _collect_all_experts_loras(
    base_model: nn.Module, expert_paths: List[str], device: torch.device
) -> Tuple[Dict[str, List[LoraDelta]], List[str]]:
    """
    For N experts, return:
      - lora_map: { module_path: [LoraDelta_e0, LoraDelta_e1, ...] }
      - module_paths: sorted list of module_paths where LoRA is present
    """
    per_exp_maps: List[Dict[str, LoraDelta]] = []
    for p in expert_paths:
        sd = _load_adapter_state(p, device)
        per_exp_maps.append(_parse_single_adapter(base_model, sd))

    # unify keys
    all_paths = set()
    for m in per_exp_maps:
        all_paths.update(m.keys())
    module_paths = sorted(list(all_paths))

    lora_map: Dict[str, List[LoraDelta]] = {}
    for mp in module_paths:
        lst: List[LoraDelta] = []
        for em in per_exp_maps:
            if mp in em:
                lst.append(em[mp])
            else:
                # A missing expert for a module: use zeros of correct shape to keep indexing consistent
                # We need base module to infer shapes; fetch from any existing delta
                # If none exists, skip the module entirely
                # Here, we simply skip adding this module if any missing; better behavior is to fill zeros.
                lst = None
                break
        if lst is not None:
            lora_map[mp] = lst
    # Filter module_paths to those present in lora_map
    module_paths = sorted(list(lora_map.keys()))
    return lora_map, module_paths


# -----------------------------
# Routing/gating linear wrapper
# -----------------------------


class WeightedLoraMoELinear(nn.Module):
    """
    Wrap a base nn.Linear with multi-expert LoRA deltas and routing.

    Modes:
    - inference: token × module top-k routing among E experts with softmax weights
      over normalized dot products.
    - train_single: enable only one expert's delta with a sigmoid gate scalar
      sigma(v^T u_t) multiplying BA(u_t); base W, A, B frozen.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        expert_loras: List[LoraDelta],
        gate_vectors: Optional[List[torch.Tensor]] = None,
        top_k: int = 2,
        score_type: str = "cosine",
        eps: float = 1e-6,
        train_single_expert: Optional[int] = None,
        init_train_gate_vec: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.expert_loras = expert_loras
        self.num_experts = len(expert_loras)
        self.top_k = max(1, min(top_k, self.num_experts))
        self.score_type = score_type
        self.eps = eps
        self.train_single_expert = train_single_expert

        # Freeze base params (safety)
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Register LoRA tensors as buffers for device/dtype handling
        self.A_list = nn.ParameterList([])
        self.B_list = nn.ParameterList([])
        self.alpha_list: List[float] = []
        for ld in expert_loras:
            A = nn.Parameter(
                ld.A.detach().clone().to(dtype=self.base.weight.dtype, device=self.base.weight.device),
                requires_grad=False,
            )
            B = nn.Parameter(
                ld.B.detach().clone().to(dtype=self.base.weight.dtype, device=self.base.weight.device),
                requires_grad=False,
            )
            self.A_list.append(A)
            self.B_list.append(B)
            self.alpha_list.append(float(ld.alpha))

        # Inference gate vectors (fixed) or training gate vector (single expert)
        self.register_buffer("dummy", torch.tensor(0.0))  # for device
        self.infer_gate_vecs: Optional[nn.ParameterList] = None
        self.train_gate_param: Optional[nn.Parameter] = None
        self.train_gate_expert_idx: Optional[int] = None

        if train_single_expert is not None:
            self.train_gate_expert_idx = int(train_single_expert)
            if init_train_gate_vec is None:
                init = torch.zeros(self.in_features, dtype=self.base.weight.dtype, device=self.base.weight.device)
            else:
                init = init_train_gate_vec.to(dtype=self.base.weight.dtype, device=self.base.weight.device)
            self.train_gate_param = nn.Parameter(init)
        else:
            if gate_vectors is not None and len(gate_vectors) == self.num_experts:
                self.infer_gate_vecs = nn.ParameterList([
                    nn.Parameter(
                        g.detach().clone().to(dtype=self.base.weight.dtype, device=self.base.weight.device),
                        requires_grad=False,
                    )
                    for g in gate_vectors
                ])
            else:
                self.infer_gate_vecs = nn.ParameterList([
                    nn.Parameter(
                        torch.zeros(self.in_features, dtype=self.base.weight.dtype, device=self.base.weight.device),
                        requires_grad=False,
                    )
                    for _ in range(self.num_experts)
                ])

    def extra_repr(self) -> str:
        mode = (
            f"train_single={self.train_gate_expert_idx}"
            if self.train_gate_expert_idx is not None
            else f"infer_top_k={self.top_k}"
        )
        return f"{mode}, score={self.score_type}, in={self.in_features}, out={self.out_features}, experts={self.num_experts}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base forward
        base_out = F.linear(x, self.base.weight, self.base.bias)

        if self.train_gate_expert_idx is not None:
            # Only one expert contributes with sigmoid gate
            e = self.train_gate_expert_idx
            A = self.A_list[e]
            B = self.B_list[e]
            r = float(A.shape[0])
            scale = float(self.alpha_list[e]) / max(r, 1.0)

            # Gate scalar per token (following PhatGoose paper implementation)
            # Gate applied to input, not output: gate * x then pass through LoRA
            # v = self.train_gate_param
            # gate_scores = (x * v).sum(dim=-1, keepdim=True)  # without sqrt(d) to preserve gradient
            # gate = torch.sigmoid(gate_scores)  # [..., 1]
            
            # # Apply gate to input, then pass through LoRA
            # gated_x = x * gate  # [..., d_in]
            # mid = torch.matmul(gated_x, A.t())  # [..., r]
            # delta = torch.matmul(mid, B.t())  # [..., out]
            # return base_out + scale * delta

            x32 = x.to(torch.float32)
            v32 = self.train_gate_param.to(torch.float32)

            gate_scores = (x32 * v32).sum(dim=-1, keepdim=True) / math.sqrt(x32.shape[-1])
            gate_scores = gate_scores.clamp_(-20.0, 20.0)

            gate = torch.sigmoid(gate_scores).to(x.dtype)
            gated_x = x * gate
            mid = torch.matmul(gated_x, A.t())
            delta = torch.matmul(mid, B.t()) * scale
            return base_out + delta

        # Inference: multi-expert top-k weighted sum
        assert self.infer_gate_vecs is not None

        if self.score_type == "cosine":
            x_norm = x / (x.norm(dim=-1, keepdim=True) + self.eps)
            v_mat = torch.stack([g / (g.norm() + self.eps) for g in self.infer_gate_vecs], dim=0)  # [E, d]
            scores = torch.matmul(x_norm, v_mat.t())  # [..., E]
        else:  # dot
            v_mat = torch.stack([g for g in self.infer_gate_vecs], dim=0)  # [E, d]
            scores = torch.matmul(x, v_mat.t())

        # stabilize
        scores = scores / math.sqrt(x.shape[-1])

        # top-k selection
        topk_scores, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        weights = torch.softmax(topk_scores, dim=-1)  # [..., k]

        # sum weighted deltas for selected experts
        out = base_out
        # Flatten leading dims for computation: [..., d] -> [N, d]
        x_flat = x.reshape(-1, x.shape[-1])
        w_flat = weights.reshape(-1, self.top_k)
        idx_flat = topk_idx.reshape(-1, self.top_k)

        # Compute contributions per selected expert
        # Loop over k for memory friendliness
        d_out = out.shape[-1]
        contrib = 0.0
        for j in range(self.top_k):
            ej = idx_flat[:, j]  # [N]
            wj = w_flat[:, j].unsqueeze(-1)  # [N, 1]
            # Gather A/B per sample: not vectorized; instead compute unique experts batch-wise
            # For efficiency and simplicity, group by expert id in this mini-batch
            unique_e = torch.unique(ej)
            accum = torch.zeros(x_flat.size(0), d_out, device=x.device, dtype=base_out.dtype)
            for e in unique_e.tolist():
                mask = (ej == e)
                if not mask.any():
                    continue
                A = self.A_list[e]
                B = self.B_list[e]
                r = float(A.shape[0])
                scale = float(self.alpha_list[e]) / max(r, 1.0)
                x_sel = x_flat[mask]
                mid = torch.matmul(x_sel, A.t())
                delta = torch.matmul(mid, B.t()) * scale
                accum[mask] = delta
            contrib += (accum * wj)

        contrib = contrib.reshape(*x.shape[:-1], d_out)
        return out + contrib


# -----------------------------
# Gate checkpoints (per expert)
# -----------------------------


def _save_gate_checkpoint(save_path: str, gates: Dict[str, torch.Tensor], meta: Dict[str, Any]):
    payload = {
        "meta": meta,
        "gates": {k: v.detach().cpu() for k, v in gates.items()},
    }
    _ensure_dir(os.path.dirname(save_path))
    torch.save(payload, save_path)


def _load_gate_checkpoint(load_path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(load_path, map_location=device)
    gates = {k: v.to(device) for k, v in payload.get("gates", {}).items()}
    meta = payload.get("meta", {})
    return gates, meta


# -----------------------------
# Wrapping helpers
# -----------------------------


def _replace_module(root: nn.Module, dotted_path: str, new_module: nn.Module):
    parts = dotted_path.split(".")
    parent = root
    for name in parts[:-1]:
        if name.isdigit():
            parent = parent[int(name)]
        else:
            parent = getattr(parent, name)
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        parent[idx] = new_module
    else:
        setattr(parent, last, new_module)


def _wrap_model_for_inference(
    base_model: nn.Module,
    lora_map: Dict[str, List[LoraDelta]],
    gates_per_expert: List[Dict[str, torch.Tensor]],
    top_k: int,
    score_type: str,
) -> nn.Module:
    # For each module path, gather gate vectors ordered by expert index
    device = next(base_model.parameters()).device
    for mp, loras in lora_map.items():
        gate_vecs: List[torch.Tensor] = []
        for e, _ld in enumerate(loras):
            gv = gates_per_expert[e].get(mp, torch.zeros(loras[e].A.shape[1], device=device))
            gate_vecs.append(gv)
        orig = _get_module_by_path(base_model, mp)
        wrapped = WeightedLoraMoELinear(
            base_linear=orig,
            expert_loras=loras,
            gate_vectors=gate_vecs,
            top_k=top_k,
            score_type=score_type,
        )
        _replace_module(base_model, mp, wrapped)
    return base_model


def _wrap_model_for_single_expert_training(
    base_model: nn.Module,
    lora_map: Dict[str, List[LoraDelta]],
    expert_idx: int,
    init_gates: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, Dict[str, nn.Parameter]]:
    train_params: Dict[str, nn.Parameter] = {}
    for mp, loras in lora_map.items():
        orig = _get_module_by_path(base_model, mp)
        init_vec = None
        if init_gates is not None and mp in init_gates:
            init_vec = init_gates[mp]
        wrapped = WeightedLoraMoELinear(
            base_linear=orig,
            expert_loras=loras,
            train_single_expert=expert_idx,
            init_train_gate_vec=init_vec,
        )
        # collect param
        assert wrapped.train_gate_param is not None
        train_params[mp] = wrapped.train_gate_param
        _replace_module(base_model, mp, wrapped)
    return base_model, train_params


# -----------------------------
# SFT-style gate training helpers
# -----------------------------


def _prepare_sft_tuples_from_task(task: str, task_type: str, ratio: float = 1.0) -> List[Tuple[str, str]]:
    """
    Prepare (prompt, answer) pairs from task dev set for gate training.
    Uses eval module to get inputs and constructs answers based on task type.
    """
    def _to_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return str(v)

    def _extract_gold(item: Dict[str, Any]) -> str:
        for k in ["answer", "output"]:
            if k in item and item.get(k) is not None:
                s = _to_text(item.get(k)).strip()
                if s:
                    return s
        return ""

    # Get dev inputs
    dev_inputs = eval_mod.prepare_inputs(task, task_type, "dev", ratio=ratio)
    
    # Load task data file to get answers (use absolute path from script location)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_data_path = os.path.join(script_dir, "data", f"{task}.json")
    if not os.path.exists(task_data_path):
        raise FileNotFoundError(f"Task data file not found: {task_data_path}")
    
    with open(task_data_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    
    dev_data = task_data.get("dev", [])
    if len(dev_data) == 0:
        raise ValueError(f"No dev data found for task {task}")
    
    # Apply ratio if needed
    if ratio < 1.0:
        import math
        n = max(1, math.floor(len(dev_data) * ratio))
        dev_data = dev_data[:n]
    
    # Build (question, answer) pairs based on task type
    pairs: List[Tuple[str, str]] = []
    for i, item in enumerate(dev_data):
        if i >= len(dev_inputs):
            break
        
        question = dev_inputs[i]
        answer = _extract_gold(item)
        
        if question and answer:
            pairs.append((question, answer))
    
    return pairs


def _build_sft_batch(
    tokenizer,
    pairs: List[Tuple[str, str]],
    max_length: int,
    device: torch.device,
):
    # Tokenize prompt and completion separately to mask prompt tokens
    batch_prompts = [p for p, _ in pairs]
    batch_comps = [c for _, c in pairs]
    tok_p = tokenizer(
        batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    tok_c = tokenizer(
        batch_comps, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )

    input_ids = []
    labels = []
    attention_mask = []
    for i in range(len(batch_prompts)):
        p_ids = tok_p["input_ids"][i].tolist()
        c_ids = tok_c["input_ids"][i].tolist()
        # Remove BOS of completion if duplicated; keep simple
        if len(c_ids) > 0 and tokenizer.bos_token_id is not None and c_ids[0] == tokenizer.bos_token_id:
            c_ids = c_ids[1:]
        # ids = p_ids + c_ids
        # # Truncate
        # if len(ids) > max_length:
        #     ids = ids[-max_length:]
        # attn = [1] * len(ids)
        # # Mask prompt part
        # num_p = min(len(p_ids), len(ids))
        # lbl = [-100] * num_p + ids[num_p:]

        seq = p_ids + c_ids
        total = len(seq)
        L = min(max_length, total)
        start = total - L
        ids = seq[start:]
        prompt_kept = max(0, len(p_ids) - start)
        prompt_kept = min(prompt_kept, len(ids))
        attn = [1] * len(ids)
        lbl = [-100] * prompt_kept + ids[prompt_kept:]
        if all(x == -100 for x in lbl):
            continue

        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lbl, dtype=torch.long))
        attention_mask.append(torch.tensor(attn, dtype=torch.long))

    # Pad to max len in batch (right padding)
    maxlen = max(len(t) for t in input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        return torch.cat([t, torch.full((maxlen - t.size(0),), fill, dtype=t.dtype)], dim=0)
    input_ids = torch.stack([pad(t, pad_id) for t in input_ids], dim=0).to(device)
    labels = torch.stack([
        pad(t, -100) if t.size(0) < maxlen else t for t in labels
    ], dim=0).to(device)
    attention_mask = torch.stack([pad(t, 0) for t in attention_mask], dim=0).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _train_single_expert_gates(
    base_model_name: str,
    tokenizer_name: str,
    expert_path: str,
    expert_name: str,
    task: str,
    task_type: str,
    device_str: str,
    steps: int = 100,
    batch_size: int = 1,
    lr: float = 5e-3,
    max_length: int = 512,
    grad_accum: int = 1,
    seed: int = 42,
    init_gate_path: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    torch.manual_seed(seed)
    device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # For training, right-padding is fine
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"

    _stage("Load base model for gate training")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base.train()
    for p in base.parameters():
        p.requires_grad_(False)

    # Parse LoRA deltas for this expert
    lora_map, module_paths = _collect_all_experts_loras(base, [expert_path], device)
    # Optionally load existing gates (warm start)
    init_gates = None
    if init_gate_path is not None and os.path.exists(init_gate_path):
        init_gates, _ = _load_gate_checkpoint(init_gate_path, device)

    # Wrap modules
    base, train_params = _wrap_model_for_single_expert_training(
        base, lora_map, expert_idx=0, init_gates=init_gates
    )

    # Optimizer over gate params only
    opt = torch.optim.AdamW(train_params.values(), lr=lr, weight_decay=0.01)

    # Load gate training data from task dev set (use all dev data)
    pairs = _prepare_sft_tuples_from_task(task, task_type, ratio=1.0)
    if len(pairs) == 0:
        raise RuntimeError(f"No training data for gate training from task {task} dev set.")

    _stage(f"Start gate training for expert {expert_name} | modules={len(module_paths)} | steps={steps}")
    idx = 0
    pbar = tqdm(range(steps), desc=f"Training gates for {expert_name}")
    for step in pbar:
        batch_pairs = pairs[idx : idx + batch_size]
        if len(batch_pairs) < batch_size:
            idx = 0
            continue
        idx += batch_size
        inputs = _build_sft_batch(tokenizer, batch_pairs, max_length=max_length, device=next(base.parameters()).device)
        out = base(**inputs)
        loss = out.loss / max(1, grad_accum)
        loss.backward()
        if (step + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        # Update progress bar with loss info
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Collect learned gates
    learned: Dict[str, torch.Tensor] = {mp: p.detach().clone().to("cpu") for mp, p in train_params.items()}
    meta = {
        "expert_name": expert_name,
        "base_model": base_model_name,
        "tokenizer_name": tokenizer_name,
        "task": task,
        "task_type": task_type,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "created": _now_str(),
        "module_paths": module_paths,
    }
    return learned, meta


# -----------------------------
# Generation utilities (MOE-wrapped model)
# -----------------------------


def _batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: str,
) -> List[str]:
    outputs: List[str] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[i : i + batch_size]
            batch_prompts = _apply_chat_template_if_available(tokenizer, batch_prompts)
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            try:
                target_device = next(model.get_input_embeddings().parameters()).device
            except Exception:
                target_device = device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.extend(decoded)
    return outputs


# -----------------------------
# Entry point (model_collaboration interface)
# -----------------------------


def _infer_moe_full(
    task: str,
    task_type: str,
    base_model: str,
    tokenizer_name: str,
    model_names: List[str],
    gate_paths: List[str],
    hyperparameters: Dict[str, Any],
    device: str,
    output_log_path: Optional[str] = None,
):
    _stage("Prepare test inputs")
    test_inputs = eval_mod.prepare_inputs(task, task_type, "test")
    
    # Extract hyperparameters
    max_new_tokens = int(hyperparameters.get("max_response_length", 128))
    temperature = float(hyperparameters.get("temperature", 0.7))
    top_p = float(hyperparameters.get("top_p", 0.9))
    batch_size = int(hyperparameters.get("batch_size", 8))
    top_k = int(hyperparameters.get("top_k", 2))
    score_type = hyperparameters.get("score_type", "cosine")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _configure_tokenizer(tokenizer, context_label="infer_moe_full")

    _stage("Load base model and build MOE wrappers")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    device_t = next(base.parameters()).device
    # Download experts if needed and load LoRA deltas
    expert_paths = [_download_adapter_if_needed(name) for name in model_names]
    lora_map, module_paths = _collect_all_experts_loras(base, expert_paths, device_t)
    if len(lora_map) == 0:
        raise RuntimeError("No common LoRA modules found across experts.")
    # Load gates
    gates_per_expert: List[Dict[str, torch.Tensor]] = []
    for gp in gate_paths:
        g, _ = _load_gate_checkpoint(gp, device_t)
        gates_per_expert.append(g)
    if len(gates_per_expert) != len(expert_paths):
        raise RuntimeError("gate_paths must match expert_paths length")

    base = _wrap_model_for_inference(
        base, lora_map, gates_per_expert, top_k=top_k, score_type=score_type
    )

    _stage("Run generation with MOE routing")
    outputs = _batch_generate(
        base, tokenizer, test_inputs, max_new_tokens, temperature, top_p, batch_size, device
    )
    scores = eval_mod.get_scores(task, task_type, "test", outputs)
    avg_score = float(sum(scores) / max(1, len(scores)))

    # Simplified log format (similar to weight_model_swarms)
    logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_score,
        "logs": [
            {"input": test_inputs[i], "output": outputs[i], "score": scores[i]}
            for i in range(len(test_inputs))
        ],
    }
    if output_log_path is None:
        # Ensure logs are saved relative to the method script location
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_log_path = os.path.join(
            script_dir, "logs", f"{task}_{len(model_names)}_{avg_score:.4f}_phatgoose.json"
        )
    _ensure_dir(os.path.dirname(output_log_path))
    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    _stage(f"Saved logs to {output_log_path}")


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    # device visibility
    physical_gpu_info = ""
    if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])
        physical_gpu_info = f"cuda:{gpu_ids[0]}"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    mode = hyperparameters.get("mode", "train_and_infer")
    
    # Auto-infer base_model from first LoRA adapter if not specified
    base_model = hyperparameters.get("base_model", None)
    if base_model is None:
        if not model_names or len(model_names) == 0:
            raise ValueError("Either base_model must be specified in hyperparameters or model_names must be provided")
        # Try to infer from first adapter's config
        first_adapter_local = _download_adapter_if_needed(model_names[0])
        inferred = _infer_base_model_from_adapter(first_adapter_local)
        if inferred is not None:
            base_model = inferred
            _stage(f"Auto-detected base model from adapter config: {base_model}")
        else:
            print("[weight_phatgoose] Warning: Could not auto-detect base model from adapter config.")
            print(f"[weight_phatgoose] Using first model as base: {model_names[0]}")
            print("[weight_phatgoose] If this is incorrect, please specify 'base_model' in hyperparameters.")
            base_model = model_names[0]
    
    tokenizer_name = hyperparameters.get("tokenizer_name", base_model)

    _stage(
        f"Start run_method | task={task} task_type={task_type} mode={mode} device={physical_gpu_info}"
    )

    if mode == "train_all_gates" or mode == "train_and_infer":
        # Use model_names as expert paths (like other weight methods)
        if not model_names or len(model_names) == 0:
            raise ValueError("model_names must be provided")
        expert_paths: List[str] = model_names
        # Generate expert names from paths/IDs
        expert_names: List[str] = [
            p.replace("/", "_") if "/" in p else os.path.basename(p.rstrip("/"))
            for p in expert_paths
        ]
        steps = int(hyperparameters.get("gate_steps", 100))
        gate_batch_size = int(hyperparameters.get("gate_batch_size", 1))
        gate_lr = float(hyperparameters.get("gate_lr", 5e-3))
        max_length = int(hyperparameters.get("max_length", 512))
        grad_accum = int(hyperparameters.get("grad_accum", 1))
        # Ensure logs are saved relative to the method script location
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_out_dir = os.path.join(script_dir, "logs", "phatgoose", _now_str(), "gates")
        out_dir = hyperparameters.get("gate_output_dir", default_out_dir)
        _ensure_dir(out_dir)
        gate_paths: List[str] = []
        for ep, en in zip(expert_paths, expert_names):
            _stage(f"Train gate for expert {en}")
            gates, meta = _train_single_expert_gates(
                base_model_name=base_model,
                tokenizer_name=tokenizer_name,
                expert_path=ep,
                expert_name=en,
                task=task,
                task_type=task_type,
                device_str=device,
                steps=steps,
                batch_size=gate_batch_size,
                lr=gate_lr,
                max_length=max_length,
                grad_accum=grad_accum,
            )
            save_path = os.path.join(out_dir, f"gate_{en}.pt")
            _save_gate_checkpoint(save_path, gates, meta)
            gate_paths.append(save_path)
        _stage(f"Saved all gates under {out_dir}")

        if mode == "train_all_gates":
            return 0
        # else (train_and_infer) continue to inference
        hyperparameters = dict(hyperparameters)
        hyperparameters["gate_paths"] = gate_paths
        mode = "infer_moe_full"

    if mode == "infer_moe_full":
        # Use model_names as expert paths (like other weight methods)
        if not model_names or len(model_names) == 0:
            raise ValueError("model_names must be provided")
        gate_paths: List[str] = hyperparameters["gate_paths"]
        out_path = hyperparameters.get("output_log_path", None)

        _infer_moe_full(
            task,
            task_type,
            base_model,
            tokenizer_name,
            model_names,
            gate_paths,
            hyperparameters,
            device,
            out_path,
        )
        return 0

    raise ValueError(f"Unknown mode: {mode}")
    