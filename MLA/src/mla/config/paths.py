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


def get_pretrain_paths(root_dir: Path, config: DictConfig) -> Paths:
    """
    Builds and returns paths for a pretraining experiment.

    Args:
        config (DictConfig): Experiment configuration containing dataset and model name settings.
        root_dir (Path): Root directory of the project.

    Returns:
        Paths: Populated paths for pretraining data and model checkpoint.
    """
    return Paths(
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / "pretraining" / config.dataset_config_name,
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / f"{config.pretrained_model_name}.th",
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
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / "finetuning" / config.dataset_config_name,
        pretrained_model_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / f"{config.pretrained_model_name}.th",
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / config.dataset_config_name / f"{config.fine_tuned_model_name}.th",
    )
