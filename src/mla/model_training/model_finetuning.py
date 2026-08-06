import logging
from pathlib import Path

import hydra
import torch
from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from mla.config.paths import TRAINING_MODELS_DIR, Paths, get_finetune_paths
from mla.model_training.model_training import ModelFineTuning
from mla.model_training.strategies import _FINETUNING_STRATEGIES
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.model_utils import LossMeter, MetricEvaluationProtocol
from mla.utils.utils import append_to_results_csv, configure_logging, print_output_table, set_all_seeds

logger = logging.getLogger(__name__)

config_path = str(Path(__file__).parent.parent / "config" / "experiments")


@dataclass
class FineTuneRun:
    """Validation result + provenance for a single fine-tuning run (one seed)."""
    seed: int
    loss: LossMeter
    metrics: dict
    wandb_id: str | None


def prepare_fine_tune_data(config: DictConfig, paths: Paths) -> tuple[DataLoader, DataLoader]:
    """
    Loads and tokenizes the fine-tuning dataset and returns train and validation dataloaders.

    Args:
        config (DictConfig): Experiment configuration containing dataset and tokenizer settings.
        paths (Paths): Paths to data directories.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """
    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, config, paths)
    train_dataloader, validation_dataloader = prepare_dataloaders(dataset, tokenizer, config, DataCollatorWithPadding)

    return train_dataloader, validation_dataloader


def model_fine_tuning(
        model: nn.Module,
        optimizer: type[Optimizer],
        metric_fn: type[MetricEvaluationProtocol],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        config: DictConfig,
        paths: Paths,
        seed: int,
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
    # Set the seed
    set_all_seeds(seed)

    # Fine tune the model
    fine_tuner = ModelFineTuning(model, optimizer, metric_fn, config, paths, seed)
    fine_tuner.fine_tune_model(training_dataloader=train_dataloader, validation_dataloader=validation_dataloader)

    # Output the loss/metric values
    validation_loss, validation_metrics = fine_tuner.eval_model(validation_dataloader=validation_dataloader, training_eval=False)

    return FineTuneRun(
        seed=seed,
        loss=validation_loss,
        metrics=validation_metrics,
        wandb_id=fine_tuner.wandb_id,
    )


def run_model_fine_tuning(config: DictConfig) -> tuple[LossMeter, dict]:
    """
    Entry point for running a single fine-tuning experiment.

    Prepares the data and model from config, runs fine-tuning with the first
    seed defined in config, and logs results to W&B.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    configure_logging()
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(root_dir, config)
    strategy = _FINETUNING_STRATEGIES[config.base_model]

    # Prepare the data and model
    train_dataloader, validation_dataloader = prepare_fine_tune_data(config, paths)
    logger.info("Train batches: %d  Val batches: %d", len(train_dataloader), len(validation_dataloader))
    model = strategy.build_model(config, paths)
    logger.info("Loading checkpoint: %s", paths.pretrained_model_path)
    run = model_fine_tuning(
        model=model,
        optimizer=AdamW,
        metric_fn=strategy.metric_cls(),
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
        paths=paths,
        seed=config.seeds[0]
    )
    logger.info("Seed %d — loss=%.4f  %s=%.4f", config.seeds[0], run.loss.avg, config.eval_metric, run.metrics[config.eval_metric])

    return run.loss, run.metrics


@hydra.main(version_base=None, config_path=config_path, config_name="GPT2/finetuning/RTE/tinygpt2_mha_rte")
def run_multiple_fine_tunings(config: DictConfig) -> None:
    """
    Runs fine-tuning across multiple seeds and aggregates results.

    Prepares data once, then rebuilds the model from the pretrained checkpoint
    for each seed, running an independent fine-tuning experiment each time.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    configure_logging()
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(root_dir, config)
    strategy = _FINETUNING_STRATEGIES[config.base_model]

    # Prepare the data and model
    train_dataloader, validation_dataloader = prepare_fine_tune_data(config, paths)
    logger.info("Train batches: %d  Val batches: %d", len(train_dataloader), len(validation_dataloader))

    all_run_results = {"loss": [], "accuracy": [], "f1": [], "mcc": [], "wandb_ids": []}

    runs: list[FineTuneRun] = []
    for seed in config.seeds:
        logger.info("Starting fine-tuning seed=%d", seed)
        torch._dynamo.reset()
        model = strategy.build_model(config, paths)
        logger.info("Loading checkpoint: %s", paths.pretrained_model_path)
        run = model_fine_tuning(
            model=model,
            optimizer=AdamW,
            metric_fn=strategy.metric_cls(),
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            config=config,
            paths=paths,
            seed=seed
        )
        logger.info("Seed %d — loss=%.4f  %s=%.4f", seed, run.loss.avg, config.eval_metric, run.metrics[config.eval_metric])
        runs.append(run)

    aggregated = {
        "loss":     [r.loss.avg for r in runs],
        "accuracy": [r.metrics["accuracy"] for r in runs],
        "f1":       [r.metrics["f1"] for r in runs],
        "mcc":      [r.metrics["mcc"] for r in runs],
        "seed_to_wandb": {r.seed: r.wandb_id for r in runs},
    }

    # Save results to central CSV
    results = strategy.build_results(aggregated, config)
    append_to_results_csv(results, root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / "finetune_results.csv")
    print_output_table(title="Fine tuning Complete", results=results)
    logger.info("All seeds complete for %s", config.experiment_name)


if __name__ == "__main__":
    run_multiple_fine_tunings()
