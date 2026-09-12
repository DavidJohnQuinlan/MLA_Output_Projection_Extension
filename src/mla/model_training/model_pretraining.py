import logging
from pathlib import Path

import hydra
import torch.distributed as dist
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from mla.config.paths import TRAINING_MODELS_DIR, get_pretraining_paths
from mla.config.schema import validate_compression_dims
from mla.model_training.model_training import ModelPreTraining
from mla.model_training.strategies import _PRETRAINING_STRATEGIES
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.reporting import append_to_results_csv, print_output_table
from mla.utils.setup import configure_logging, set_all_seeds

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


config_path = str(Path(__file__).parent.parent / "config" / "experiments")


@hydra.main(version_base=None, config_path=config_path, config_name="GPT2/pretraining/tinygpt2_mha")
def model_pretraining(config: DictConfig) -> None:
    """
    Entry point for pre-training a BERT/GPT2 model.

    Loads the tokenizer, prepares the dataset and dataloaders, builds the model from
    config, and runs the pre-training loop via ModelPreTraining.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    configure_logging()
    validate_compression_dims(config)
    root_dir = Path(hydra.utils.get_original_cwd())
    strategy = _PRETRAINING_STRATEGIES[config.base_model]
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    data_paths = get_pretraining_paths(root_dir, config)
    datasets = import_and_prepare_data(tokenizer, config, data_paths.tokenized_data_path)
    results_csv_path = root_dir / TRAINING_MODELS_DIR / config.experiment_project / "pretraining" / "pretraining_results.csv"

    # Independent pretraining run per seed
    for seed in config.pretraining_seeds:
        paths = get_pretraining_paths(root_dir, config, seed)
        if paths.best_checkpoint_file_path.exists():
            print(f"Skipping seed={seed} — checkpoint exists: {paths.best_checkpoint_file_path}\n")
            continue
        set_all_seeds(seed)
        train_loader, validation_loader = prepare_dataloaders(datasets, tokenizer, config, collator_fn=None)
        logger.info(f"(Seed={seed}): Train batches: {len(train_loader)}  Val batches: {len(validation_loader)}")

        logger.info(f"(Seed={seed}): Loading checkpoint config: {config.model_config_name}")
        model = strategy.build_model(config)

        pretraining = ModelPreTraining(
            model=model,
            optimizer=AdamW,
            metric_fn=strategy.metric_cls(),
            config=config,
            paths=paths,
            seed=seed,
        )

        logger.info(f"(Seed={seed}): Starting pretraining — attention={config.attention_mechanism} lr={config.learning_rate} max_steps={config.max_steps}")
        pretraining.model_training(training_dataloader=train_loader, validation_dataloader=validation_loader)

        # Save results to central CSV (main process)
        results = strategy.build_results(pretraining, model, tokenizer, validation_loader, config, seed)
        if pretraining.accelerator.is_main_process:
            append_to_results_csv(results, results_csv_path)
            print_output_table(title="Pretraining Complete", results=results)
            logger.info(f"(Seed={seed}): Pretraining complete — best_loss={pretraining.best_validation_loss:4f}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    model_pretraining()
