from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
cs = ConfigStore.instance()


@dataclass
class ExperimentConfig:
    experiment_name: str = MISSING
    attention_mechanism: str = MISSING
    dataset_name: str = MISSING
    dataset_config_name: str = MISSING
    job_type: str = MISSING
    experiment_project: str = MISSING
    tiny_bert_model_name: str = MISSING
    kv_compression_dim: int | None = None
    q_compression_dim: int | None = None
    output_compression_dim: int | None = None
    root_dir: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class OptimizerConfig:
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_steps: int = MISSING
    warmup: bool = MISSING
    warmup_rate: float = MISSING
    max_norm: float = MISSING


@dataclass
class PreTrainingConfig:
    batch_size: int = MISSING
    eval_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    max_seq_length: int = MISSING
    max_position_embeddings: int = MISSING
    hidden_size: int = MISSING
    mixed_precision: str = MISSING
    train_eval_steps: int = MISSING
    eval_steps: int = MISSING
    parallel_processes: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING
    drop_last: bool = MISSING
    text_column: str = MISSING
    mlm: bool = MISSING
    mlm_probability: float = MISSING
    mode: str = MISSING


@dataclass
class FineTuningConfig:
    batch_size: int = MISSING
    eval_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    max_seq_length: int = MISSING
    max_position_embeddings: int = MISSING
    hidden_size: int = MISSING
    mixed_precision: str = MISSING
    train_eval_steps: int = MISSING
    eval_steps: int = MISSING
    parallel_processes: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING
    drop_last: bool = MISSING
    text_column: str = MISSING
    num_labels: int = MISSING
    num_fine_tune_runs: int = MISSING
    pretrained_checkpoint: str = ""


@dataclass
class PreTrainRunConfig:
    experiment: ExperimentConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    training: PreTrainingConfig = MISSING


@dataclass
class FineTuneRunConfig:
    experiment: ExperimentConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    training: FineTuningConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="pretrain_schema", node=PreTrainRunConfig)
cs.store(name="finetune_schema", node=FineTuneRunConfig)
cs.store(group="experiment", name="bert_mha_baseline", node=ExperimentConfig)
cs.store(group="optimizer", name="adamw", node=OptimizerConfig)
cs.store(group="training", name="pretraining", node=PreTrainingConfig)
cs.store(group="training", name="finetuning", node=FineTuningConfig)
