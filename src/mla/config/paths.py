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
    Create a dataset name that is unique to the dataset and its processing params.
    """
    dataset_name = config.dataset_name.replace("/", "__")
    dataset_name_id = f"{dataset_name}_{config.dataset_config_name}" if config.dataset_config_name else dataset_name
    keys = ["dataset_name", "dataset_config_name", "max_seq_length", "train_token_budget", "val_token_budget", "max_load_pct"]
    payload = json.dumps({k: config.get(k) for k in keys}, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode()).hexdigest()[:10]
    return f"{dataset_name_id}_{digest}"


def build_model_name(config: DictConfig, seed: int | None = None) -> str:
    """
    Create a deterministic, collision-free model name base on the models configuration.
    """
    parts = [config.base_model, f"h{config.hidden_size}", f"l{config.n_layer}", f"a{config.n_head}", config.attention_mechanism]
    if config.kv_compression_dim:     parts.append(f"kv{config.kv_compression_dim}")
    if config.q_compression_dim:      parts.append(f"q{config.q_compression_dim}")
    if config.output_compression_dim: parts.append(f"o{config.output_compression_dim}")
    parts += [(config.dataset_config_name or config.dataset_name).replace("/", "__"), f"seq{config.max_seq_length}"]
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(str(p) for p in parts)


def discover_pre_trained_checkpoints(root_dir, config) -> list[tuple[str, Path]]:
    """
    Find all pretrained checkpoints whose variant matches the given config.

    Scans the experiment's pretraining directory for checkpoint sidecars
    (*.json), each of which stores the model config and pretrain seed written
    at save time. A checkpoint matches when its attention mechanism and kv/q/o
    compression dims equal those in config (None == None for MHA), and its paired
    .th weights file exists. This is how fine-tuning discovers which pretrained
    models to evaluate - the filesystem is the source of truth, so every seed that
    was actually trained is picked up without being declared in config.

    Args:
        root_dir (Path): Root directory of the project.
        config (DictConfig): Experiment configuration; matched on attention_mechanism
            and the kv/q/o compression dims.

    Returns:
        list[tuple[int, Path]]: (pretrain_seed, checkpoint_path) pairs, sorted by seed,
            one per matching checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint matches the variant in config.
    """
    pre_training_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining"

    def matches(c: dict) -> bool:
        return (c["attention_mechanism"] == config.attention_mechanism
                and c.get("kv_compression_dim") == config.kv_compression_dim
                and c.get("q_compression_dim") == config.q_compression_dim
                and c.get("output_compression_dim") == config.output_compression_dim)

    found_checkpoints: list[tuple[int, Path]] = []
    for sidecar in pre_training_path.glob("*.json"):
        metadata = json.loads(sidecar.read_text())
        checkpoint = sidecar.with_suffix(".th")
        if matches(metadata["config"]) and checkpoint.exists():
            found_checkpoints.append((int(metadata["seed"]), checkpoint))
    if not found_checkpoints:
        raise FileNotFoundError(
            f"No pretrained checkpoint for attention='{config.attention_mechanism}' "
            f"(kv={config.kv_compression_dim}, q={config.q_compression_dim} "
            f"o={config.output_compression_dim}) in {pre_training_path}")
    return sorted(found_checkpoints)


def resolve_checkpoint(root_dir: Path, config: DictConfig, pre_training_seed: int) -> Path:
    """
    Select the one discovered checkpoint matching this variant and a specific pretrain seed.
    """
    for seed, checkpoint_path in discover_pre_trained_checkpoints(root_dir, config):
        if seed == pre_training_seed:
            return checkpoint_path
    raise FileNotFoundError(
        f"No pretrained checkpoint with seed={pre_training_seed} for attention='{config.attention_mechanism}' "
        f"(kv={config.kv_compression_dim}, o={config.output_compression_dim})."
    )

def get_pretrain_paths(root_dir: Path, config: DictConfig, pre_training_seed: int | None = None) -> Paths:
    """
    Builds and returns paths for a pretraining experiment.

    Args:
        config (DictConfig): Experiment configuration containing dataset and model name settings.
        root_dir (Path): Root directory of the project.
        pre_training_seed (int): The pretraining models run seed.

    Returns:
        Paths: Populated paths for pretraining data and model checkpoint.
    """
    return Paths(
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / config.experiment_project / "pretraining" / build_dataset_name(config),
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / f"{build_model_name(config, pre_training_seed)}.th",
    )


def get_finetune_paths(
        root_dir: Path,
        config: DictConfig,
        pre_trained_model_path: Path,
        fine_tuning_seed: int | None
    ) -> Paths:
    """
    Builds and returns paths for a fine-tuning experiment.

    Args:
        config (DictConfig): Experiment configuration containing dataset and model name settings.
        root_dir (Path): Root directory of the project.
        pre_trained_model_path (Path): Pre-trained model checkpoint path.
        finetuning_seed (int | None): Seed for the finetuning model run.

    Returns:
        Paths: Populated paths for fine-tuning data, pretrained checkpoint, and fine-tuned model.
    """
    return Paths(
        tokenized_data_path = root_dir / TRAINING_DATA_DIR / config.experiment_project / "finetuning" / build_dataset_name(config),
        pretrained_model_path = pre_trained_model_path,
        model_file_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / config.dataset_config_name /  f"{pre_trained_model_path.stem}__{config.dataset_config_name}_ft{fine_tuning_seed}.th",
    )
