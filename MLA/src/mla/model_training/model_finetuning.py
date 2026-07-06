from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from mla.config.paths import TRAINING_MODELS_DIR, Paths, get_finetune_paths
from mla.model_training.model_training import ModelFineTuning, ModelPreTraining
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import (
    BERTModelForClassification,
    BertModelForMLM,
)
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.model_utils import ClassificationMetricEvaluation, LossMeter
from mla.utils.utils import append_to_results_csv, print_output_table, set_all_seeds, setup_logging

config_path = str(Path(__file__).parent.parent / "config" / "experiments" / "finetuning")


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
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, config, DataCollatorWithPadding)

    return train_loader, val_loader


def prepare_fine_tune_model(config: DictConfig, paths: Paths) -> nn.Module:
    """
    Loads a pretrained checkpoint and converts it to a classification model.

    Args:
        config (DictConfig): Experiment configuration containing model and attention settings.
        paths (Paths): Paths to the pretrained model checkpoint.

    Returns:
        nn.Module: Compiled classification model ready for fine-tuning.
    """
    # Load the Bert Configuration
    bert_config = BertConfig.from_pretrained(
        config.model_config_name,
        attention_mechanism=config.attention_mechanism,
        kv_compression_dim=config.kv_compression_dim,
        q_compression_dim=config.q_compression_dim,
        output_compression_dim=config.output_compression_dim,
    )

    # Load the pretrained BERT model
    bert_model = ModelPreTraining.load_model(
        model_class=BertModelForMLM,
        model_config=bert_config,
        checkpoint_path=paths.pretrained_model_path,
    ).bert

    # Convert MLM BERT model to classification BERT
    bert_classifier = BERTModelForClassification(bert_model=bert_model, config=config)

    # Compile the BERT classifier
    compiled_bert_classifier = torch.compile(bert_classifier)
    return compiled_bert_classifier


def model_fine_tuning(
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: DictConfig,
        paths: Paths,
        seed: int,
    ) -> tuple[LossMeter, dict]:
    """
    Runs a single fine-tuning experiment for a given seed and returns evaluation results.

    Args:
        model (nn.Module): Compiled classification model to fine-tune.
        train_dataloader (DataLoader): Training dataloader.
        eval_dataloader (DataLoader): Validation dataloader.
        config (DictConfig): Experiment configuration.
        paths (Paths): Paths to model checkpoints.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[LossMeter, dict]: Validation loss and evaluation metrics.
    """
    # Set the seed
    set_all_seeds(seed)

    # Fine tune the model
    fine_tuner = ModelFineTuning(model, AdamW, ClassificationMetricEvaluation, config, paths, seed)
    fine_tuner.fine_tune_model(training_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # Output the loss/metric values
    validation_loss, validation_metrics = fine_tuner.eval_model(eval_dataloader=eval_dataloader, training_eval=False)

    return validation_loss, validation_metrics


def run_model_fine_tuning(config: DictConfig) -> tuple[LossMeter, dict]:
    """
    Entry point for running a single fine-tuning experiment.

    Prepares the data and model from config, runs fine-tuning with the first
    seed defined in config, and logs results to W&B.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    setup_logging()
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(root_dir, config)

    # Prepare the data and model
    train_loader, val_loader = prepare_fine_tune_data(config, paths)
    bert_model = prepare_fine_tune_model(config, paths)
    validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, config, paths, config.seeds[0])

    return validation_loss, validation_metrics


@hydra.main(version_base=None, config_path=config_path, config_name="sst2/tinybert_mha_sst2")
def run_multiple_fine_tunings(config: DictConfig) -> None:
    """
    Runs fine-tuning across multiple seeds and aggregates results.

    Prepares data once, then rebuilds the model from the pretrained checkpoint
    for each seed, running an independent fine-tuning experiment each time.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    setup_logging()
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(root_dir, config)
    train_loader, val_loader = prepare_fine_tune_data(config, paths)

    all_run_results = {"loss": [], "accuracy": [], "f1": []}

    for seed in config.seeds:

        # Initialise the model
        bert_model = prepare_fine_tune_model(config, paths)
        validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, config, paths, seed)

        # Add results
        all_run_results["loss"].append(validation_loss.avg)
        all_run_results["accuracy"].append(validation_metrics["accuracy"])
        all_run_results["f1"].append(validation_metrics["f1"])

    # Save results to central CSV
    results = build_finetune_results(all_run_results, config)
    append_to_results_csv(results, root_dir / TRAINING_MODELS_DIR / config.experiment_project / "finetuning" / "finetune_results.csv")
    print_output_table(title="Fine tuning Complete", results=results)


def build_finetune_results(results: dict, config: DictConfig):
    """
    Build a results summary dictionary across multiple finetuning runs.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.pretrained_model_name,
        "attention_mechanism": config.attention_mechanism,
        "dataset": config.dataset_config_name,
        "kv": config.kv_compression_dim,
        "q": config.q_compression_dim,
        "o": config.output_compression_dim,
        "avg_finetune_validation_loss": f"{np.mean(results["loss"]):.4f} +/- {np.std(results["loss"]):.4f}",
        "avg_finetune_accuracy": f"{np.mean(results["accuracy"]):.4f} +/- {np.std(results["accuracy"]):.4f}",
        "avg_finetune_f1": f"{np.mean(results["f1"]):.4f} +/- {np.std(results["f1"]):.4f}",
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
    }


if __name__ == "__main__":
    run_multiple_fine_tunings()
