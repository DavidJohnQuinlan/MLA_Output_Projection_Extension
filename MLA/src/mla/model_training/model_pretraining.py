import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from mla.config.paths import TRAINING_MODELS_DIR, get_pretrain_paths
from mla.model_training.model_training import ModelPreTraining
from mla.model_training.strategies import _STRATEGIES

from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.utils import append_to_results_csv, print_output_table

logger = logging.getLogger(__name__)


config_path = str(Path(__file__).parent.parent / "config" / "experiments")


@hydra.main(version_base=None, config_path=config_path, config_name="BERT/pretraining/tinybert_mha")
def model_pretraining(config: DictConfig) -> None:
    """
    Entry point for pre-training a BERT/GPT2 model.

    Loads the tokenizer, prepares the dataset and dataloaders, builds the model from
    config, and runs the pre-training loop via ModelPreTraining.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_pretrain_paths(root_dir, config)
    strategy = _STRATEGIES[config.base_model]

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, config, paths)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, config, collator_fn=None)
    logger.info("Train batches: %d  Val batches: %d", len(train_loader), len(val_loader))

    # Load and compile the model
    logger.info("Loading checkpoint config: %s", config.model_config_name)
    model = strategy.build_model(config)
    compiled_model = torch.compile(model)

    # Initialize pretraining class
    pretrainer = ModelPreTraining(
       model=compiled_model,
       optimizer=AdamW,
       metric_fn=strategy.metric_cls(),
       config=config,
       paths=paths,
    )

    # Start training
    logger.info("Starting pretraining — attention=%s  lr=%s  max_steps=%d",
                config.attention_mechanism, config.learning_rate, config.max_steps)
    pretrainer.train_model(training_dataloader=train_loader, eval_dataloader=val_loader)

    # Save results to central CSV
    results = strategy.build_results(pretrainer, model, tokenizer, val_loader, config)
    append_to_results_csv(results, root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / "pretrain_results.csv")
    print_output_table(title="Pretraining Complete", results=results)
    logger.info("Pretraining complete — best_loss=%.4f", pretrainer.best_eval_loss)


if __name__ == "__main__":
    model_pretraining()
