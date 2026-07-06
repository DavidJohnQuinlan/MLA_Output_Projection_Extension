import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mla.config.paths import TRAINING_MODELS_DIR, get_pretrain_paths
from mla.model_training.model_training import ModelPreTraining
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import BertModelForMLM
from mla.utils.attention_hooks import collect_attention_head_activations
from mla.utils.attention_utils import compute_model_cka
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.model_utils import MetricEvaluation
from mla.utils.utils import append_to_results_csv, calculate_flop_metrics, measure_inference_speed, print_output_table

logger = logging.getLogger(__name__)


config_path = str(Path(__file__).parent.parent / "config" / "experiments" / "pretraining")


@hydra.main(version_base=None, config_path=config_path, config_name="tinybert_mha")
def model_pretraining(config: DictConfig) -> None:
    """
    Entry point for pre-training a BERT-based model using Masked Language Modeling.

    Loads the tokenizer, prepares the dataset and dataloaders, builds the model from
    config, and runs the pre-training loop via ModelPreTraining.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_pretrain_paths(root_dir, config)

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, config, paths)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, config, collator_fn=None)
    logger.info("Train batches: %d  Val batches: %d", len(train_loader), len(val_loader))

    # Load the model Configuration
    logger.info("Loading checkpoint config: %s", config.model_config_name)
    bert_config = BertConfig.from_pretrained(
        config.model_config_name,
        attention_mechanism=config.attention_mechanism,
        kv_compression_dim=config.kv_compression_dim,
        q_compression_dim=config.q_compression_dim,
        output_compression_dim=config.output_compression_dim,
    )

    # Prepare the BERT Model
    bert_model = BertModelForMLM(bert_config)

    # Compile the model
    compiled_bert_model = torch.compile(bert_model)

    # Initialize pretraining class
    pretrainer = ModelPreTraining(
       model=compiled_bert_model,
       optimizer=AdamW,
       metric_fn=MetricEvaluation,
       config=config,
       paths=paths,
    )

    # Start the engine
    logger.info("Starting pretraining — attention=%s  lr=%s  max_steps=%d",
                config.attention_mechanism, config.learning_rate, config.max_steps)
    pretrainer.train_model(training_dataloader=train_loader, eval_dataloader=val_loader)

    # Save results to central CSV
    results = build_pretrain_results(pretrainer, bert_model, tokenizer, val_loader, config)
    append_to_results_csv(results, root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / "pretrain_results.csv")
    print_output_table(title="Pretraining Complete", results=results)
    logger.info("Pretraining complete — best_loss=%.4f", pretrainer.best_eval_loss)


def build_pretrain_results(
    pretrainer: ModelPreTraining,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    val_loader: DataLoader,
    config: DictConfig
) -> dict:
    """
    Build a results summary dictionary for a pretraining run.
    """
    flops, macs, params = calculate_flop_metrics(model, config)
    ms_per_sample = measure_inference_speed(pretrainer.model, tokenizer, config)
    activations_list = collect_attention_head_activations(pretrainer, val_loader)
    avg_cka = compute_model_cka(activations_list)
    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.pretrained_model_name,
        "attention_mechanism": config.attention_mechanism,
        "dataset": config.dataset_config_name,
        "n_params": params,
        "GFLOPS": flops,
        "GMACS": macs,
        "Inf (ms)": f"{ms_per_sample:.2f}",
        "kv": config.kv_compression_dim,
        "q": config.q_compression_dim,
        "o": config.output_compression_dim,
        "pre_training_validation_loss": f"{pretrainer.best_eval_loss:.4f}",
        "pre_training_top1_mlm_accuracy": f"{pretrainer.best_eval_metrics['accuracy']:.4f}",
        "pre_training_top5_mlm_accuracy": f"{pretrainer.best_eval_metrics['top5_accuracy']:.4f}",
        "pre_training_cka": f"{avg_cka:.4f}",
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
    }


if __name__ == "__main__":
    model_pretraining()
