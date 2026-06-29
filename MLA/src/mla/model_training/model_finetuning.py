import torch
import wandb
from torch.optim import AdamW
from transformers import DataCollatorWithPadding, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from torch._dynamo.eval_frame import OptimizedModule

from mla.config.paths import get_finetune_paths, Paths
from mla.model_training.model_training import ModelPreTraining, ModelFineTuning
from mla.models.BERT.bert_model.bert_heads import BertModelForMLM, BERTModelForClassification
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.utils.utils import set_all_seeds
from mla.utils.model_utils import ClassificationMetricEvaluation
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders
config_path = str(Path(__file__).parent.parent / "config" / "experiments" / "finetuning")


def prepare_fine_tune_data(config: DictConfig, paths: Paths) -> Tuple[DataLoader, DataLoader]:
    """
    """
    # Define the TinyBert Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config_name)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, config, paths)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, config, DataCollatorWithPadding)

    return train_loader, val_loader
    
def prepare_fine_tune_model(config: DictConfig, paths: Paths) -> torch.nn.Module:
    """
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

    # Compile the BERT classifer
    compiled_bert_classifier = torch.compile(bert_classifier)
    return compiled_bert_classifier

def model_fine_tuning(
        model: torch.nn.Module,
        train_dataloader: DataLoader, 
        eval_dataloader: DataLoader, 
        config: DictConfig, 
        paths: Paths, 
        seed: int,
    ) -> torch.nn.Module:
    """
    """
    # Prepare the datasets and model
    set_all_seeds(seed)
    
    # Fine tune the model
    fine_tuner = ModelFineTuning(model, AdamW, ClassificationMetricEvaluation, config, paths, seed)
    fine_tuner.fine_tune_model(training_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # Output the loss/metric values
    validation_loss, validation_metrics = fine_tuner.eval_model(eval_dataloader=eval_dataloader, training_eval=False)

    return validation_loss, validation_metrics

@hydra.main(version_base=None, config_path=config_path, config_name="bert_mha_sst2")
def run_model_fine_tuning(config: DictConfig) -> None:
    """
    """
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(config, root_dir)

    # Prepare the data and model
    train_loader, val_loader = prepare_fine_tune_data(config, paths)
    bert_model = prepare_fine_tune_model(config, paths)
    validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, config, paths, config.seeds[0])

    return validation_loss, validation_metrics

@hydra.main(version_base=None, config_path=config_path, config_name="bert_mha_sst2")
def run_multiple_fine_tunings(config: DictConfig) -> None:
    """
    """
    root_dir = Path(hydra.utils.get_original_cwd())
    paths = get_finetune_paths(config, root_dir)

    # Prepare the data and model
    train_loader, val_loader = prepare_fine_tune_data(config, paths)
    bert_model = prepare_fine_tune_model(config, paths)

    all_run_results = {"loss": [], "accuracy": [], "f1": []}

    for seed in config.seeds:

        # Prepare the datasets and model
        set_all_seeds(seed)
        
        validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, config, paths, seed)

        # Add results
        all_run_results["loss"].append(validation_loss.avg)
        all_run_results["accuracy"].append(validation_metrics["accuracy"])
        all_run_results["f1"].append(validation_metrics["f1"]) 

    return all_run_results


if __name__ == "__main__":
    run_model_fine_tuning()
