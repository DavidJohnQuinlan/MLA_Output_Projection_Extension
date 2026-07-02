from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

from mla.config.paths import get_pretrain_paths, TRAINING_MODELS_DIR
from mla.model_training.model_training import ModelPreTraining
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import BertModelForMLM
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
from mla.utils.model_utils import MetricEvaluation
from mla.utils.utils import setup_logging, build_pretrain_results, append_to_results_csv


config_path = str(Path(__file__).parent.parent / "config" / "experiments" / "pretraining")


@hydra.main(version_base=None, config_path=config_path, config_name="tinybert_mha_baseline")
def model_pretraining(config: DictConfig) -> None:
    """
    Entry point for pre-training a BERT-based model using Masked Language Modeling.

    Loads the tokenizer, prepares the dataset and dataloaders, builds the model from
    config, and runs the pre-training loop via ModelPreTraining.

    Args:
        config (DictConfig): Hydra config containing all experiment, optimizer, and training parameters.
    """
    setup_logging()
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_pretrain_paths(config, root_dir)

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, config, paths)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, config, collator_fn=None)
    
    # Load the model Configuration
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
    pretrainer.train_model(training_dataloader=train_loader, eval_dataloader=val_loader)

    # Save results to central CSV
    results = build_pretrain_results(pretrainer, tokenizer, val_loader, config)
    append_to_results_csv(results, root_dir / TRAINING_MODELS_DIR / "pretrain_results.csv")

if __name__ == "__main__":
    model_pretraining()
