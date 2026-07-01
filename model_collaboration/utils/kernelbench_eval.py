"""
Kernel evaluation helpers adapted from KernelBench (https://arxiv.org/abs/2502.10517).
Supports CUDA backend only. No kernelbench package dependency.
"""

import importlib
import importlib.util
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KernelExecResult:
    compiled: bool = False
    correctness: bool = False
    metadata: dict = field(default_factory=dict)
    runtime: float = -1.0        # mean kernel time in ms
    runtime_stats: dict = field(default_factory=dict)
    ref_runtime: float = -1.0    # mean PyTorch reference time in ms
    ref_runtime_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_error_name(e: Exception) -> str:
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _get_tolerance(precision: torch.dtype) -> float:
    return {
        torch.float32: 1e-4,
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
    }[precision]


def _process_input(x, device, precision):
    if not isinstance(x, torch.Tensor):
        return x
    return x.to(dtype=precision, device=device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_reference(src: str, context: dict):
    """Execute reference script and return (Model, get_init_inputs, get_inputs)."""
    try:
        compile(src, "<string>", "exec")
    except SyntaxError as e:
        print(f"[kernelbench] Syntax error in reference: {e}")
        return None
    try:
        exec(src, context)
    except Exception as e:
        print(f"[kernelbench] Error executing reference: {e}")
        return None
    return context.get("Model"), context.get("get_init_inputs"), context.get("get_inputs")


def _load_custom(src: str, context: dict, build_dir: Optional[str] = None):
    """Compile and exec the generated CUDA kernel, return ModelNew class or None."""
    if build_dir:
        src = f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n" + src
    try:
        compile(src, "<string>", "exec")
        exec(src, context)
    except SyntaxError as e:
        print(f"[kernelbench] Syntax/compilation error in custom kernel: {e}")
        return None
    except Exception as e:
        print(f"[kernelbench] Error compiling custom kernel: {e}")
        return None
    return context.get("ModelNew")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _clear_l2_cache(device):
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def _time_cuda_event(
    fn: callable,
    args: list,
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    device=None,
) -> list:
    """CUDA-event timing. Returns list of elapsed times in ms."""
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        for _ in range(num_warmup):
            fn(*args)
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        elapsed_times = []
        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            _clear_l2_cache(device=device)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize(device=device)
            if trial >= discard_first:
                elapsed_times.append(start.elapsed_time(end))

    return elapsed_times


def _timing_stats(elapsed_times: list, device=None) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }
    if device is not None:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)
    return stats


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _check_correctness(
    ref_model: nn.Module,
    new_model: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_trials: int,
    seed: int,
    device,
    precision: torch.dtype,
) -> KernelExecResult:
    torch.manual_seed(seed)
    seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)]
    pass_count = 0

    with torch.no_grad():
        for trial, trial_seed in enumerate(seeds):
            _set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [_process_input(x, device, precision) for x in inputs]

            ref = ref_model.to(device=device, dtype=precision)
            new = new_model.to(device=device, dtype=precision)

            out_ref = ref(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                out_new = new(*inputs)
                torch.cuda.synchronize(device=device)
            except Exception as e:
                metadata["runtime_error"] = str(e)
                metadata["runtime_error_name"] = _get_error_name(e)
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

            if out_ref.shape != out_new.shape:
                metadata["correctness_issue"] = f"Shape mismatch: expected {out_ref.shape}, got {out_new.shape}"
                return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

            tol = _get_tolerance(precision)
            if not torch.allclose(out_ref, out_new, atol=tol, rtol=tol):
                max_diff = torch.max(torch.abs(out_ref - out_new)).item()
                avg_diff = torch.mean(torch.abs(out_ref - out_new)).item()
                metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                metadata["correctness_issue"] = "Output mismatch"
            else:
                pass_count += 1

    metadata["correctness_trials"] = f"({pass_count} / {num_trials})"
    correct = pass_count == num_trials
    return KernelExecResult(compiled=True, correctness=correct, metadata=metadata)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def eval_kernel_against_ref(
    ref_src: str,
    custom_src: str,
    device: int = 0,
    seed: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    precision: torch.dtype = torch.float32,
    build_dir: Optional[str] = None,
) -> KernelExecResult:
    """
    Evaluate a generated CUDA kernel (ModelNew) against the reference PyTorch model.

    Args:
        ref_src:    Source code of the reference problem (defines Model, get_inputs, get_init_inputs)
        custom_src: Source code of the generated kernel (defines ModelNew)
        device:     CUDA device index
        seed:       Random seed for reproducibility
        num_correct_trials: Number of random-input trials for correctness check
        num_perf_trials:    Number of timed runs for performance measurement
        precision:  torch.dtype for computation
        build_dir:  Optional directory for CUDA extension build cache

    Returns:
        KernelExecResult with compiled, correctness, runtime, ref_runtime fields populated
    """
    assert torch.cuda.is_available(), "CUDA required for kernel evaluation"
    torch.cuda.set_device(device)

    metadata = {
        "hardware": torch.cuda.get_device_name(device=device),
        "device": str(device),
    }

    # --- Load reference ---
    ref_context = {}
    result = _load_reference(ref_src, ref_context)
    if result is None:
        metadata["compilation_error"] = "Failed to load reference model"
        return KernelExecResult(compiled=False, metadata=metadata)
    Model, get_init_inputs_fn, get_inputs_fn = result

    _set_seed(seed)
    init_inputs = get_init_inputs_fn()
    init_inputs = [_process_input(x, device, precision) for x in init_inputs]

    with torch.no_grad():
        _set_seed(seed)
        ref_model = Model(*init_inputs)
        ref_model = ref_model.to(device=device, dtype=precision)

    # --- Load and compile custom kernel ---
    custom_context = {}
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    stdout_buf = StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            ModelNew = _load_custom(custom_src, custom_context, build_dir)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        if "lock" in str(e) or "No such file or directory" in str(e):
            print(f"[kernelbench] Lock error during compilation, retry: {e}")
            torch.cuda.empty_cache()
            return None
        metadata["compilation_error_name"] = _get_error_name(e)
        metadata["compilation_error"] = str(e)
        torch.cuda.empty_cache()
        return KernelExecResult(compiled=False, metadata=metadata)

    if ModelNew is None:
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = "ModelNew not found or syntax error in generated code"
        torch.cuda.empty_cache()
        return KernelExecResult(compiled=False, metadata=metadata)

    # --- Instantiate custom model ---
    try:
        with torch.no_grad():
            _set_seed(seed)
            custom_model = ModelNew(*init_inputs)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
    except RuntimeError as e:
        metadata["runtime_error"] = str(e)
        metadata["runtime_error_name"] = _get_error_name(e)
        torch.cuda.empty_cache()
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    # --- Correctness ---
    result = _check_correctness(
        ref_model, custom_model, get_inputs_fn,
        metadata=metadata,
        num_trials=num_correct_trials,
        seed=seed,
        device=device,
        precision=precision,
    )

    if not result.correctness:
        torch.cuda.empty_cache()
        return result

    # --- Performance ---
    torch.cuda.synchronize(device=device)
    _set_seed(seed)
    perf_inputs = get_inputs_fn()
    perf_inputs = [_process_input(x, device, precision) for x in perf_inputs]

    custom_model = custom_model.to(device=device, dtype=precision)
    ref_model = ref_model.to(device=device, dtype=precision)

    try:
        custom_times = _time_cuda_event(custom_model, perf_inputs, num_trials=num_perf_trials, device=device)
        result.runtime = _timing_stats(custom_times, device)["mean"]
        result.runtime_stats = _timing_stats(custom_times, device)

        ref_times = _time_cuda_event(ref_model, perf_inputs, num_trials=num_perf_trials, device=device)
        result.ref_runtime = _timing_stats(ref_times, device)["mean"]
        result.ref_runtime_stats = _timing_stats(ref_times, device)
    except Exception as e:
        result.metadata["perf_error"] = str(e)

    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SPEEDUP_CAPS = {1: 10.0, 2: 5.0, 3: 2.0}


def score_kernel_result(result: KernelExecResult, level: int) -> float:
    """
    Convert a KernelExecResult to a scalar score in [0, 1].

    Score is 0 if: not compiled, not correct, or not faster than baseline.
    Score scales linearly from 0 → 1 as speedup goes from 1× → cap[level]×.
    """
    if result is None or not result.compiled or not result.correctness:
        return 0.0
    if result.runtime <= 0 or result.ref_runtime <= 0:
        return 0.0

    speedup = result.ref_runtime / result.runtime
    if speedup < 1.01:
        return 0.0

    cap = SPEEDUP_CAPS.get(level, 2.0)
    return min(1.0, (speedup - 1.0) / (cap - 1.0))
