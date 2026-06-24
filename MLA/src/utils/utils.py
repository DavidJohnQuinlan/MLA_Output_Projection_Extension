import random 
import torch 
import functools
import time
import wandb.integration.torch.wandb_torch as wandb_torch
import numpy as np


def set_all_seeds(seed=42):
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

def count_params(model):
    """
    Computes the total number of parameters and the number of trainable parameters in a PyTorch module.

    This utility iterates through all parameters within the model's computation graph. It is particularly 
    useful when validating structural setups, analyzing parameter freezing strategies (such as keeping a 
    BERT encoder static while training a top head), or estimating overall hardware VRAM requirements.

    Args:
        model (nn.Module): The target PyTorch network or structural sub-module to audit.

    Returns:
        Tuple[int, int]: A tuple containing:
            - **total** (int): The absolute count of all scalar values across all parameter tensors.
            - **trainable** (int): The count of parameters actively participating in gradient calculation 
              and backpropagation updates (`requires_grad=True`).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_speed(model, tokenizer, n_runs=100):
    model.eval()
    dummy_input = tokenizer(
        "This is a test sentence for inference speed measurement.",
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    ).to("cuda")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**dummy_input)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(**dummy_input)
    elapsed = time.time() - start
    
    ms_per_sample = (elapsed / n_runs) * 1000
    print(f"Inference: {ms_per_sample:.2f}ms per sample")
    return ms_per_sample

    
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

# # Create a dummy batch that looks exactly like your training data
# dummy_ids = torch.ones((hp.batch_size, hp.max_seq_length), dtype=torch.long).to("cpu")
# dummy_mask = torch.ones((hp.batch_size, hp.max_seq_length), dtype=torch.long).to("cpu")

# # Pass this "Controlled" batch to the counter
# flops, macs, params = calculate_flops(
#     model=bert_model,
#     kwargs={"input_ids": dummy_ids, "attention_mask": dummy_mask},
#     print_results=True,
#     print_detailed=False
# )

# # Measure the inference speed
# inf_time = measure_inference_speed(pretrainer.model, tokenizer)
# inf_time