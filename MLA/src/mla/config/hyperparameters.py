from dataclasses import dataclass


@dataclass
class PreTrainingHyperparameters:
    """
    Configuration for training constraints, optimization mechanics, and hardware orchestration.

    Attributes:
        batch_size (int): Micro-batch capacity processed per backward training step per worker device.
        eval_batch_size (int): Evaluation batch size, should be bigger than training batch size.
        gradient_accumulation_steps (int): Step iterations aggregated before evaluating optimization steps.
        max_seq_length (int): Absolute spatial truncation cutoff metric applied to incoming sequences.
        max_steps (int): Boundary parameter defining the absolute number of dataset training steps.
        warmup (bool): Uses an initial linear learning rate escalation envelope if True.
        warmup_rate (float): Relative percentage of global step arrays allocated to the optimization warmup phase.
        max_norm (float): Absolute gradient clipping threshold ceiling value to avoid exploding gradients.
        learning_rate (float): Peak optimizer scalar factor assigned to initial parameter step matrices.
        weight_decay (float): L2 regularization factor multiplier to scale back non-bias parameters.
        mixed_precision (str): Precision format for training execution ("no", "fp16", or "bf16").
        parallel_processes (int): Number of active hardware units or GPU instances allocated across the runtime context.
        num_workers (int): Total auxiliary worker processes spun up for data generation tasks.
        pin_memory (bool): Pins processed host RAM directly to maximize high-speed GPU device transfers.
        mlm (bool): Configures data streams and heads to execute a Masked Language Modeling objective.
        mlm_probability (float): Percentage level of masking.
        drop_last (bool): Filters out unaligned trailing dataset slices to enforce symmetric batch dimensions.
    """
    kv_compression_dim: int=None
    q_compression_dim: int=None
    output_compression_dim: int=None

    batch_size: int=8
    eval_batch_size: int=64
    gradient_accumulation_steps: int=1 #32
    max_seq_length: int=128
    max_position_embeddings: int=128
    max_steps: int=10# 25000
    warmup: bool=True
    warmup_rate: float=0.05
    max_norm: int=1.0
    learning_rate: float=5e-4
    weight_decay: float=0.01
    mixed_precision: str="no"
    parallel_processes: int=8
    
    num_workers: int=0
    pin_memory: bool=False
    mlm: bool=True
    mlm_probability: float=0.15
    drop_last: bool=True
    text_column: str= "text"
    mode: str="training"

    train_eval_steps: int=5 # 10
    eval_steps: int=5 # 500

@dataclass
class FineTuneHyperparameters:
    """
    Configuration for training, optimization, and hardware utilization.

    Args:
        batch_size: Batch size per step.
        gradient_accumulation_steps: Number of steps to accumulate gradients before performing a weight update. 
            Effective batch size = batch_size * accumulation_steps.
        max_seq_length: Maximum number of tokens per sequence after tokenization.
        max_epoch: Total number of full passes through the training data.
        warmup: Whether to use a learning rate scheduler with a warmup phase.
        warmup_rate: The ratio of total training steps used for the warmup phase (e.g., 0.05 = 5% of steps).
        max_norm: Maximum gradient norm for clipping. Set to np.inf to disable clipping.
        learning_rate: Initial peak learning rate for the optimizer.
        weight_decay: L2 regularization coefficient applied to non-bias/norm weights.
        mixed_precision: Precision format for training (e.g., "bf16", "fp16", or "no").
        parallel_processes: Number of distributed processes or GPUs to utilize.
        num_workers: Number of subprocesses for data loading (0 means main process).
        pin_memory: If True, the data loader copies Tensors into device-pinned memory for faster GPU transfer.
        mlm: Boolean flag to enable Masked Language Modeling objective logic.
        drop_last: If True, drops the last incomplete batch in an epoch.
        eval_steps: Frequency of evaluation, measured in global optimizer steps.
    """

    kv_compression_dim: int=None
    q_compression_dim: int=None
    output_compression_dim: int=None

    batch_size: int=32
    eval_batch_size: int=872
    gradient_accumulation_steps: int=1
    hidden_size: int=312
    max_seq_length: int=128
    max_position_embeddings: int=128
    max_steps: int=10#2104*5
    dropout_rate: float=0.1
    warmup: bool=True
    warmup_rate: float=0.1
    max_norm: float=1.0
    learning_rate: float=2e-5
    weight_decay: float=0.01
    mixed_precision: str="no"
    parallel_processes: int=8
    num_labels: int=2
    num_fine_tune_runs: int=5
    
    num_workers: int=0
    pin_memory: bool=False
    mlm: bool=False
    drop_last: bool=True
    text_column: str="sentence"
    mode: str="fine-tuning"

    eval_steps: int = 5#100
    train_eval_steps: int=1#10