import csv
import functools
import logging
import os
import random
import time
import datatime
from pathlib import Path
from calflops import calculate_flops
from torch.utils.data import DataLoader

import numpy as np
import torch
import wandb.integration.torch.wandb_torch as wandb_torch
from omegaconf import DictConfig
from torch import nn
from transformers import PreTrainedTokenizerBase
from mla.utils.attention_hooks import collect_attention_head_activations
from mla.utils.attention_utils import compute_model_cka

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
        print_results=True,
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
        max_length=config.max_length,
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


def build_pretrain_results(
        pretrainer: ModelPreTraining, 
        tokenizer: PreTrainedTokenizerBase,
        val_loader: DataLoader, 
        config: DictConfig
    ) -> dict:
    """
    Build a results summary dictionary for a pretraining run.
    """
    flops, macs, params = calculate_flop_metrics(pretrainer.model, config)
    ms_per_sample = measure_inference_speed(pretrainer.model, tokenizer, config)
    activations_list = collect_attention_head_activations(pretrainer, val_loader)
    avg_cka = compute_model_cka(activations_list)

    return {
        "model_name": config.pretrained_model_name,
        "attention_mechanism": config.attention_mechanism,
        "dataset": config.dataset_config_name,
        "n_params": params,
        "GFLOPS": flops,
        "GMACS": macs,
        "Inf (ms)": f"{ms_per_sample:.2f}",
        "kv": config.kv_compression_dim,
        "q": config.q_compression_dim,
        "o": config.output_compression_dim,
        "pre_training_validation_loss": f"{pretrainer.best_eval_loss:.4f}",
        "pre_training_top1_accuracy": f"{pretrainer.best_eval_metrics.accuracy:.4f}",
        "pre_training_top5_accuracy": f"{pretrainer.best_eval_metrics.top5_accuracy:.4f}",
        "pre_training_cka": f"{avg_cka:.4f}",
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "timestamp": datetime.now().isoformat(),
    }


def build_finetune_results(results: list, config: DictConfig):
    """
    Build a results summary dictionary across multiple finetuning runs.
    """
    return {
        "model_name": config.pretrained_model_name,
        "attention_mechanism": config.attention_mechanism,
        "dataset": config.dataset_config_name,
        "kv": config.kv_compression_dim,
        "q": config.q_compression_dim,
        "o": config.output_compression_dim,
        "avg_finetune_validation_loss": f"{np.mean(results["loss"]):.4f} +/- {np.std(results["loss"]):.4f}",
        "avg_finetune_accuracy_loss": f"{np.mean(results["accuracy"]):.4f} +/- {np.std(results["accuracy"]):.4f}",
        "avg_finetune_f1_loss": f"{np.mean(results["f1"]):.4f} +/- {np.std(results["f1"]):.4f}",
    }


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
