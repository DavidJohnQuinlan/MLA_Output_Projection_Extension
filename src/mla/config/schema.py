from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

_CONFIG_REQUIRED_DIMS = {
    "MHAE": ["output_compression_dim"],
    "MLA":  ["kv_compression_dim", "q_compression_dim"],
    "MLAE": ["kv_compression_dim", "q_compression_dim", "output_compression_dim"],
}


def validate_compression_dims(config: DictConfig) -> None:
    required = _CONFIG_REQUIRED_DIMS.get(config.attention_mechanism, [])
    missing = [k for k in required if config.get(k) is None]
    if missing:
        raise ValueError(
            f"{config.attention_mechanism} requires {missing}; "
            f"pass them on the CLI, e.g. {missing[0]}=128"
        )


@dataclass
class PreTrainingConfig:
    """Training loop parameters for BERT/GPT2 pretraining."""

    # ExperimentConfig
    experiment_name: str = MISSING
    base_model: str = MISSING
    attention_mechanism: str = MISSING
    job_type: str = MISSING
    experiment_project: str = MISSING
    model_config_name: str = MISSING
    kv_compression_dim: int | None = None
    q_compression_dim: int | None = None
    output_compression_dim: int | None = None

    # Dataset
    dataset_name: str = MISSING
    dataset_config_name: str | None = ""
    max_load_pct: int | None = None
    train_token_budget: int | None = None
    val_token_budget: int | None = None
    drop_last: bool = MISSING
    sentence_keys: list = MISSING
    task_type: str = MISSING
    parallel_processes: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING

    # OptimizerConfig
    max_steps: int = MISSING
    warmup: bool = MISSING
    warmup_rate: float = MISSING
    max_norm: float = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING

    # Model
    n_layer: int = MISSING
    n_head: int = MISSING
    hidden_size: int = MISSING
    intermediate_size: int = MISSING
    max_seq_length: int = MISSING
    max_position_embeddings: int = MISSING
    mixed_precision: str = MISSING
    loss_type: str | None = None

    # Training
    batch_size: int = MISSING
    eval_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    mlm: bool = False
    mlm_probability: float = 0.0
    train_eval_steps: int = MISSING
    eval_steps: int = MISSING
    eval_metric: str = MISSING
    pre_training_seeds: list[int] = MISSING


@dataclass
class FineTuningConfig:
    """Training loop parameters for BERT/GPT2 downstream classification fine-tuning."""

    # ExperimentConfig
    experiment_name: str = MISSING
    base_model: str = MISSING
    attention_mechanism: str = MISSING
    job_type: str = MISSING
    experiment_project: str = MISSING
    model_config_name: str = MISSING
    kv_compression_dim: int | None = None
    q_compression_dim: int | None = None
    output_compression_dim: int | None = None

    # Dataset
    max_load_pct: int | None = None
    train_token_budget: int | None = None
    val_token_budget: int | None = None
    dataset_name: str = MISSING
    dataset_config_name: str | None = ""
    drop_last: bool = MISSING
    sentence_keys: list = MISSING
    task_type: str = MISSING
    parallel_processes: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING

    # OptimizerConfig
    max_steps: int = MISSING
    warmup: bool = MISSING
    warmup_rate: float = MISSING
    max_norm: float = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING

    # Model
    hidden_size: int = MISSING
    max_seq_length: int = MISSING
    num_labels: int = MISSING
    mixed_precision: str = MISSING
    loss_type: str | None = None

    # Training
    batch_size: int = MISSING
    eval_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    train_eval_steps: int = MISSING
    eval_steps: int = MISSING
    eval_metric: str = MISSING
    pre_training_seed: int | None = None
    fine_tuning_seeds: list[int] = MISSING


cs = ConfigStore.instance()
cs.store(name="pre_training_schema", node=PreTrainingConfig)
cs.store(name="fine_tuning_schema", node=FineTuningConfig)
