from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass


@dataclass
class Paths:
    data_path: Path
    tokenized_data_path: Path
    model_file_path: Path
    pretrained_model_path: Path | None = None

def get_pretrain_paths(config: DictConfig, root_dir: Path) -> Paths:
    return Paths(
        data_path = root_dir / "training_data",
        tokenized_data_path = root_dir / "training_data" / "pretraining" / config.dataset_config_name,
        model_file_path = root_dir / "training_models" / "pretraining" / f"{config.pretrained_model_name}.th",
    )

def get_finetune_paths(config: DictConfig, root_dir: Path) -> Paths:
    return Paths(
        data_path = root_dir / "training_data",
        tokenized_data_path = root_dir / "training_data" / "finetuning" / config.dataset_config_name,
        pretrained_model_path = root_dir / "training_models" / "pretraining" / f"{config.pretrained_model_name}.th",
        model_file_path = root_dir / "training_models" / "finetuning" / f"{config.fine_tuned_model_name}.th",
    )
