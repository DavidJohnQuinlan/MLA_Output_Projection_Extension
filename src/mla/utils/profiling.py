import statistics
import time

import torch
from calflops import calculate_flops
from omegaconf import DictConfig
from torch import nn
from transformers import PreTrainedTokenizerBase

from mla.utils.setup import get_device

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def calculate_flop_metrics(model: nn.Module, config: DictConfig) -> tuple[float, float, float]:
    """
    Calculate the number of FLOPs, MACs and trainable parameters.
    """
    # Create a dummy batch that looks exactly like your training data
    dummy_ids = torch.ones((config.batch_size, config.max_seq_length), dtype=torch.long).to("cpu")
    dummy_mask = torch.ones((config.batch_size, config.max_seq_length), dtype=torch.long).to("cpu")

    # Pass this "Controlled" batch to the counter
    flops, macs, params = calculate_flops(
        model=model,
        kwargs={"input_ids": dummy_ids, "attention_mask": dummy_mask},
        print_results=False,
        print_detailed=False,
        output_as_string=False
    )
    return flops, macs, params


def compute_training_compute(config: DictConfig, fwd_flops: float) -> dict:
    """
    Training-compute axis for iso-FLOP comparison.

    fwd_flops: total forward FLOPs for one controlled batch
               (batch_size x max_seq_length), from calculate_flop_metrics.
    """
    train_tokens = (config.max_steps * config.batch_size * config.gradient_accumulation_steps * config.max_seq_length)
    f_fwd_per_token = fwd_flops / (config.batch_size * config.max_seq_length)
    train_flops = 3.0 * f_fwd_per_token * train_tokens
    return {"train_tokens": train_tokens, "train_flops": train_flops}


def measure_inference_cost(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    n_runs: int = 100,
    dtype: torch.dtype | str | None = None,
) -> dict:
    """
    Measure inference cost -- median forward latency (ms) and peak memory (MB) --
    at a specific precision, sharing a single setup and forward loop.

    dtype-invariance: rather than timing whatever precision the model's weights
    happen to be stored in (which may silently be fp32 even under a fp16 training
    config), we run the forward under torch.autocast at an explicit dtype. Every
    variant is then compared under identical numeric conditions, so the timing
    reflects the architecture, not an ambient cast state. Pass dtype=None to time
    the model exactly as-is.

    Uses CUDA events + per-iteration synchronize so timings reflect real GPU
    execution, not asynchronous kernel-launch/dispatch time (which over-penalizes
    the multi-kernel output-subspace path).

    Returns:
        {"median_ms": float, "peak_mem_mb": float | None}
    """
    device = get_device()
    model.eval()

    if isinstance(dtype, str):
        dtype = _DTYPES.get(dtype)

    # autocast is supported on cuda/cpu; skip it elsewhere (e.g. mps) or when off.
    use_autocast = dtype is not None and device.type in ("cuda", "cpu")
    if dtype is not None and not use_autocast:
        print(f"WARNING: autocast unavailable on {device.type}; timing model as-is.")
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if use_autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    dummy_input = tokenizer(
        "This is a test sentence for inference speed measurement.",
        return_tensors="pt",
        padding="max_length",
        max_length=config.max_seq_length,
        truncation=True,
    ).to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    samples = []
    with torch.no_grad(), autocast_ctx:
        # warmup, inside autocast so kernels match
        for _ in range(20):
            model(**dummy_input)
        sync()

        # Reset the high-water mark AFTER warmup so peak = weights + activations
        # of a steady-state forward (workspaces already resident).
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(n_runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(**dummy_input)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(**dummy_input)
                sync()
                samples.append((time.perf_counter() - t0) * 1e3)

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else None
    )
    return {"median_ms": statistics.median(samples), "peak_mem_mb": peak_mem_mb}


def compute_compression_ratio(config: DictConfig) -> float:
    """
    Returns the KV compression ratio relative to hidden size. 1.0 for MHA (no compression).
    """
    if config.kv_compression_dim is None:
        return 1.0
    return config.kv_compression_dim / config.hidden_size
