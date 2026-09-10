import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from mla.config.paths import TRAINING_MODELS_DIR, Paths, discover_pretraining_checkpoints, get_finetuning_paths, resolve_checkpoint
from mla.config.schema import validate_compression_dims
from mla.model_training.model_training import ModelFineTuning
from mla.model_training.strategies import _FINETUNING_STRATEGIES, FineTuningStrategy
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.model_utils import LossMeter, MetricEvaluationProtocol
from mla.utils.utils import append_to_results_csv, configure_logging, print_output_table, set_all_seeds

logger = logging.getLogger(__name__)

config_path = str(Path(__file__).parent.parent / "config" / "experiments")


@dataclass
class FineTuneRun:
    """Validation result + provenance for a single fine-tuning run (one seed)."""
    loss: LossMeter
    metrics: dict
    finetuning_seed: int
    wandb_id: str | None


def prepare_finetuning_data(config: DictConfig, paths: Paths) -> tuple[DataLoader, DataLoader]:
    """
    Loads and tokenizes the fine-tuning dataset and returns train and validation dataloaders.

    Args:
        config (DictConfig): Experiment configuration containing dataset and tokenizer settings.
        paths (Paths): Paths to data directories.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)
    datasets = import_and_prepare_data(tokenizer, config, paths.tokenized_data_path)
    train_dataloader, validation_dataloader = prepare_dataloaders(datasets, tokenizer, config, DataCollatorWithPadding)

    return train_dataloader, validation_dataloader


def model_finetuning(
        model: nn.Module,
        optimizer: type[Optimizer],
        metric_fn: type[MetricEvaluationProtocol],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        config: DictConfig,
        paths: Paths,
        finetuning_seed: int,
    ) -> FineTuneRun:
    """
    Runs a single fine-tuning experiment for a given seed and returns validation results.

    Args:
        model (nn.Module): Compiled classification model to fine-tune.
        train_dataloader (DataLoader): Training dataloader.
        validation_dataloader (DataLoader): Validation dataloader.
        config (DictConfig): Experiment configuration.
        paths (Paths): Paths to model checkpoints.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[LossMeter, dict]: Validation loss and validation metrics.
    """
    # Fine tune the model
    finetuning = ModelFineTuning(model, optimizer, metric_fn, config, paths, finetuning_seed)
    finetuning.model_finetuning(training_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

    # Output the loss/metric values
    validation_loss, validation_metrics = finetuning.eval_model(validation_dataloader=validation_dataloader, training_eval=False)

    return FineTuneRun(
        loss=validation_loss,
        metrics=validation_metrics,
        finetuning_seed=finetuning_seed,
        wandb_id=finetuning.wandb_id,
    )


def finetuning_checkpoint(
        config: DictConfig,
        strategy: FineTuningStrategy,
        root_dir: Path,
        results_csv: Path,
        checkpoint_path: Path,
        pretraining_seed: int,
    ) -> None:
    """
    Run every finetuning seed against one pretraining checkpoint; write one aggregated row.
    """
    runs: list[FineTuneRun] = []
    for finetuning_seed in config.finetuning_seeds:
        paths = get_finetuning_paths(root_dir, config, checkpoint_path, finetuning_seed)
        set_all_seeds(finetuning_seed)
        train_dataloader, validation_dataloader = prepare_finetuning_data(config, paths)
        logger.info("pretraining_seed=%d finetuning_seed=%d  Train batches: %d  Val batches: %d",
                    pretraining_seed, finetuning_seed, len(train_dataloader), len(validation_dataloader))

        torch._dynamo.reset()
        model = strategy.build_model(config, paths)
        logger.info("pretraining_seed=%d finetuning_seed=%d: Loading checkpoint: %s", pretraining_seed,
                    finetuning_seed, checkpoint_path.name)
        run = model_finetuning(
            model=model,
            optimizer=AdamW,
            metric_fn=strategy.metric_cls(config),
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            config=config,
            paths=paths,
            finetuning_seed=finetuning_seed
        )
        logger.info("pretraining_seed=%d finetuning_seed=%d  loss=%.4f %s=%.4f", pretraining_seed, finetuning_seed,
                    run.loss.avg, config.eval_metric, run.metrics[config.eval_metric])
        runs.append(run)

    aggregated = {
        "pretraining_model_name": checkpoint_path.stem,
        "loss":     [r.loss.avg for r in runs],
        "accuracy": [r.metrics["accuracy"] for r in runs],
        "f1":       [r.metrics["f1"] for r in runs],
        "mcc":      [r.metrics["mcc"] for r in runs],
        "seed_to_wandb": {r.finetuning_seed: r.wandb_id for r in runs},
    }
    results = strategy.build_results(aggregated, config, pretraining_seed)
    append_to_results_csv(results, results_csv)
    print_output_table(title=f"Fine Tuning Complete (pretraining seed {pretraining_seed})", results=results)
    logger.info("pretraining_seed=%d  all finetuning seeds complete for %s", pretraining_seed, config.experiment_name)


def _finetuning_setup(config: DictConfig) -> tuple[Path, FineTuningStrategy, Path]:
    """
    Shared per-entry setup: logging, root dir, strategy, results path.
    """
    configure_logging()
    validate_compression_dims(config)
    root_dir = Path(hydra.utils.get_original_cwd())
    strategy = _FINETUNING_STRATEGIES[config.base_model]
    results_csv = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / "finetuning_results.csv"
    return root_dir, strategy, results_csv


@hydra.main(version_base=None, config_path=config_path, config_name="GPT2/finetuning/RTE/tinygpt2_mha_rte")
def run_finetuning(config: DictConfig) -> None:
    """
    Finetune all discovered pretraining checkpoint for the model variant as defined by the supplied config.
    """
    root_dir, strategy, results_csv = _finetuning_setup(config)
    if config.get("pretraining_seed") is not None:
        checkpoints = [(config.pretraining_seed, resolve_checkpoint(root_dir, config, config.pretraining_seed))]
    else:
        checkpoints = discover_pretraining_checkpoints(root_dir, config)
    logger.info("Discovered %d pretraining checkpoint(s): seeds=%s", len(checkpoints), [s for s, _ in checkpoints])
    for pretraining_seed, checkpoint_path in checkpoints:
        finetuning_checkpoint(config, strategy, root_dir, results_csv, checkpoint_path, pretraining_seed)


if __name__ == "__main__":
    run_finetuning()
