import torch
import wandb
from torch.optim import AdamW
from transformers import DataCollatorWithPadding, AutoTokenizer

from model_training import ModelPreTraining
from models.BERT.bert_model.bert_heads import BertModelForMLM, BERTModelForClassification
from models.BERT.bert_model.bert_config import BertConfig
from models.BERT.configuration.configuration import PreTrainConfig, FineTuneConfig
from models.BERT.configuration.hyperparameters import PreTrainingHyperparameters, FineTuneHyperparameters
from model_training import ModelFineTuning
from utils.utils import set_all_seeds
from utils.model_utils import ClassificationMetricEvaluation
from utils.data_preparation import import_and_prepare_data, prepare_dataloaders


def prepare_fine_tune_data(ft_cfg, ft_hp):

    # Define the TinyBert Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ft_cfg.tiny_bert_model_name, local_files_only=False)

    # Import the datasets and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, ft_cfg, ft_hp)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, ft_hp, DataCollatorWithPadding)

    return train_loader, val_loader
    
def prepare_fine_tune_model(cfg, hp, ft_hp):
    
    # Load the TinyBert Configuration
    tiny_bert_config = BertConfig.from_pretrained(
        cfg.tiny_bert_model_name,
        attention_mechanism=cfg.attention_mechanism,
        kv_compression_dim=hp.kv_compression_dim,
        q_compression_dim=hp.q_compression_dim, 
        output_compression_dim=hp.output_compression_dim, 
    )

    # Load the pretrained BERT model
    bert_model = ModelPreTraining.load_model(
        model_class=BertModelForMLM,
        model_config=tiny_bert_config,
        checkpoint_path=cfg.paths.local.model_file_path
    ).bert
    
    # Convert MLM BERT model to classification BERT
    bert_classifier = BERTModelForClassification(bert_model=bert_model, hp=ft_hp)

    # Compile the BERT classifer
    compiled_bert_classifier = torch.compile(bert_classifier)

    return compiled_bert_classifier

def model_fine_tuning(model, train_dataloader, eval_dataloader, ft_cfg, ft_hp, seed):

    # Prepare the datasets and model
    set_all_seeds(seed)
    
    # Fine tune the model
    fine_tuner = ModelFineTuning(model, AdamW, ClassificationMetricEvaluation, ft_cfg, ft_hp)
    fine_tuner.fine_tune_model(training_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # Output the loss/metric values
    validation_loss, validation_metrics = fine_tuner.eval_model(eval_dataloader=eval_dataloader, training_eval=False)

    return validation_loss, validation_metrics

def run_model_fine_tuning():

    # Define the config and hp
    hp = PreTrainingHyperparameters()
    cfg = PreTrainConfig(
        hp.kv_compression_dim, 
        hp.q_compression_dim, 
        hp.output_compression_dim
    )
    ft_hp = FineTuneHyperparameters()
    ft_cfg = FineTuneConfig(
        ft_hp.kv_compression_dim, 
        ft_hp.q_compression_dim, 
        ft_hp.output_compression_dim
    )

    # Prepare the data and model
    train_loader, val_loader = prepare_fine_tune_data(ft_cfg, ft_hp)
    bert_model = prepare_fine_tune_model(cfg, hp, ft_hp)

    # Record the experiment
    wandb.init(
        project=ft_cfg.experiment_project,
        group=ft_cfg.experiment_name, 
        name=f"MHA_FT_Seed_0",
        job_type=ft_cfg.job_type, 
        config={**vars(ft_cfg), **vars(ft_hp)}#, **vars(tiny_bert_config)}
    )
    validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, ft_cfg, ft_hp, seed=42)

    return validation_loss, validation_metrics

def run_multiple_fine_tunings():

    # Define the config and hp
    hp = PreTrainingHyperparameters()
    cfg = PreTrainConfig(
        hp.kv_compression_dim, 
        hp.q_compression_dim, 
        hp.output_compression_dim
    )
    ft_hp = FineTuneHyperparameters()
    ft_cfg = FineTuneConfig(
        ft_hp.kv_compression_dim, 
        ft_hp.q_compression_dim, 
        ft_hp.output_compression_dim
    )

    # Prepare the data and model
    train_loader, val_loader = prepare_fine_tune_data(ft_cfg, ft_hp)
    bert_model = prepare_fine_tune_model(cfg, hp, ft_hp)

    seeds = [42, 123, 456, 789, 999]
    all_run_results = {"loss": [], "accuracy": [], "f1": []}

    for run_id, seed in enumerate(seeds):

        # Prepare the datasets and model
        set_all_seeds(seed)

        # Record the experiment
        wandb.init(
            project=ft_cfg.experiment_project, 
            group=ft_cfg.experiment_name,
            name=f"MLA_PT_Concat_WO__kv_{ft_hp.kv_compression_dim}_q_{ft_hp.q_compression_dim}_o{ft_hp.output_compression_dim}__seed_{run_id}",
            job_type=ft_cfg.job_type, 
            config={**vars(ft_cfg), **vars(ft_hp)}#, **vars(tiny_bert_config)}
        )

        validation_loss, validation_metrics = model_fine_tuning(bert_model, train_loader, val_loader, ft_cfg, ft_hp, seed)

        # Add results
        all_run_results["loss"].append(validation_loss.avg)
        all_run_results["accuracy"].append(validation_metrics["accuracy"])
        all_run_results["f1"].append(validation_metrics["f1"]) 

        return all_run_results

    # sst2_optimizer = AdamW(optimizer_grouped_parameters, lr=ft_hp.learning_rate)

if __name__ == "__main__":
    run_model_fine_tuning()
