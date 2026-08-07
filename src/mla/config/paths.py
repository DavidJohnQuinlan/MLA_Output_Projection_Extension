import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

TRAINING_DATA_DIR = "training_data"
TRAINING_MODELS_DIR = "training_models"

@dataclass
class Paths:
    """Paths to data and model directories for a given experiment."""
    tokenized_data_path: Path
    model_file_path: Path
    pretrained_model_path: Path | None = None


def build_dataset_name(config: DictConfig) -> str:
    """
    Cache-dir name that is unique to the dataset and its processing params.
    """
    dataset_name = config.dataset_name.replace("/", "__")
    dataset_config_name = config.dataset_config_name if config.dataset_config_name else ""
    keys = ["dataset_name", "dataset_config_name", "max_seq_length", "train_token_budget", "val_token_budget", "max_load_pct"]
    payload = json.dumps({k: config.get(k) for k in keys}, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode()).hexdigest()[:10]
    return f"{dataset_name}_{dataset_config_name}_{digest}"


def build_model_name(config: DictConfig, seed: int | None = None) -> str:
    """
    Deterministic, collision-free model name from the config that defines the model.
    """
    parts = [config.base_model, f"h{config.hidden_size}", f"l{config.n_layer}", f"a{config.n_head}", config.attention_mechanism]
    if config.kv_compression_dim:     parts.append(f"kv{config.kv_compression_dim}")
    if config.q_compression_dim:      parts.append(f"q{config.q_compression_dim}")
    if config.output_compression_dim: parts.append(f"o{config.output_compression_dim}")
    parts += [(config.dataset_config_name or config.dataset_name).replace("/", "__"), f"seq{config.max_seq_length}"]
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(str(p) for p in parts)


def get_pretrain_paths(root_dir: Path, config: DictConfig, seed: int | None = None) -> Paths:
    """
    Builds and returns paths for a pretraining experiment.

    Args:
        config (DictConfig): Experiment configuration containing dataset and model name settings.
        root_dir (Path): Root directory of the project.

    Returns:
        Paths: Populated paths for pretraining data and model checkpoint.
    """
    return Paths(
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / config.experiment_project / "pretraining" / build_dataset_name(config),
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / f"{build_model_name(config, seed)}.th",
    )


def get_finetune_paths(root_dir: Path, config: DictConfig) -> Paths:
    """
    Builds and returns paths for a fine-tuning experiment.

    Args:
        config (DictConfig): Experiment configuration containing dataset and model name settings.
        root_dir (Path): Root directory of the project.

    Returns:
        Paths: Populated paths for fine-tuning data, pretrained checkpoint, and fine-tuned model.
    """
    return Paths(
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / config.experiment_project / "finetuning" / build_dataset_name(config),
        pretrained_model_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / f"{config.pretrained_model_name}.th",
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / config.dataset_config_name / f"{config.fine_tuned_model_name}.th",
    )
