import csv
import functools
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb.integration.torch.wandb_torch as wandb_torch
from calflops import calculate_flops
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn
from transformers import PreTrainedTokenizerBase

import wandb


def get_device() -> torch.device:
    """
    Define the device of compute resources.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int=42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch.

    This helps make experiments more reproducible by configuring
    deterministic behavior where supported by PyTorch and CuDNN.

    Args:
        seed: Random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb() -> str:
    """
    Login and use W&B online otherwise, if password is missing use offline.
    """
    wandb_pw = os.getenv("WANDB_PW")
    if wandb_pw:
        wandb.login(key=wandb_pw)
        return "online"
    return "offline"


def safe_callback(grad, log_track, self_instance, name):
    # Catch missing or unallocated gradients safely
    if grad is None:
        return

    # Check if tensor is valid and contains actual data elements
    if not isinstance(grad, torch.Tensor) or grad.numel() == 0:
        return

    # Check wandb's internal logging frequency rules
    if not wandb_torch.log_track_update(log_track):
        return

    # Use .detach() instead of .data to remain compliant with torch.compile
    self_instance.log_tensor_stats(grad.detach(), name)


def safe_hook_variable_gradient_stats(self, var, name, log_track):
    # If the variable doesn't track gradients, back out early
    if not getattr(var, "requires_grad", False):
        return

    # Ensure the internal hook dictionary exists to prevent an AttributeError
    if not hasattr(self, "_hook_handles"):
        self._hook_handles = {}

    # Use partial to cleanly seal variables instead of a leaky lambda
    callback = functools.partial(
        safe_callback, log_track=log_track, self_instance=self, name=name
    )

    handle = var.register_hook(callback)
    self._hook_handles[name] = handle
    return handle


def setup_logging(level: str = "INFO") -> None:
    """
    Define the base logging formatting.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        print_detailed=False
    )
    return flops, macs, params


def measure_inference_speed(model: nn.Module, tokenizer: PreTrainedTokenizerBase, config: DictConfig, n_runs: int=100):
    """
    Measure the inference speed of the model averaged over n runs.
    """
    device = get_device()
    model.eval()
    dummy_input = tokenizer(
        "This is a test sentence for inference speed measurement.",
        return_tensors="pt",
        padding="max_length",
        max_length=config.max_seq_length,
        truncation=True
    ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**dummy_input)

    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(**dummy_input)
    elapsed_time = time.time() - start_time
    ms_per_sample = (elapsed_time / n_runs) * 1000
    return ms_per_sample


def append_to_results_csv(results: dict, csv_path: Path) -> None:
    """
    Save the results to a central csv file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def print_output_table(title: str, results: dict) -> None:
    """
    Prepare and print the table to screen.
    """
    results = {k: "N/A" if v is None else v for k, v in results.items()}
    table = tabulate(results.items(), tablefmt="rounded_outline")
    width = len(table.splitlines()[0])
    print("\n", title.center(width))
    print(table)
