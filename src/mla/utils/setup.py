import functools
import logging
import os
import random

import numpy as np
import torch
import wandb.integration.torch.wandb_torch as wandb_torch

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


def configure_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(filename)s:%(lineno)d  %(message)s",
        datefmt="%H:%M:%S",
    )
    for h in logging.getLogger().handlers:
        h.setFormatter(fmt)
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)
