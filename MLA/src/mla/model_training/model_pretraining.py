import torch
from transformers import AutoTokenizer
from torch.optim import AdamW

from mla.config.hyperparameters import PreTrainingHyperparameters
from mla.config.configuration import PreTrainConfig
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import BertModelForMLM
from mla.model_training import ModelPreTraining
from mla.utils.utils import count_params
from mla.utils.model_utils import MetricEvaluation
from mla.utils.data_preparation import import_and_prepare_data, prepare_dataloaders


def model_pretraining():
    """
    """

    # Initalise the config and hyperparameters
    hp = PreTrainingHyperparameters()
    cfg = PreTrainConfig(hp.kv_compression_dim, hp.q_compression_dim, hp.output_compression_dim)

    # Define the TinyBert Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tiny_bert_model_name)

    # Import the datasetes and prepare dataloaders
    dataset = import_and_prepare_data(tokenizer, cfg, hp)
    train_loader, val_loader = prepare_dataloaders(dataset, tokenizer, hp, collator_fn=None)
    
    # Load the TinyBert Configuration
    tiny_bert_config = BertConfig.from_pretrained(
        cfg.tiny_bert_model_name,
        attention_mechanism=cfg.attention_mechanism,
        kv_compression_dim=hp.kv_compression_dim,
        q_compression_dim=hp.q_compression_dim, 
        output_compression_dim=hp.output_compression_dim, 
    )

    # Prepare the Bert Model
    bert_model = BertModelForMLM(tiny_bert_config)

    # Compile the model
    compiled_bert_model = torch.compile(bert_model)

    # Count the number of parameters (trainable) 
    total, trainable = count_params(compiled_bert_model)

    # Initialize pretraining class
    pretrainer = ModelPreTraining(
       model=compiled_bert_model, 
       optimizer=AdamW,
       metric_fn=MetricEvaluation, 
       config=cfg, 
       hyperparameters=hp)

    # Start the engine
    pretrainer.train_model(training_dataloader=train_loader, eval_dataloader=val_loader)

if __name__ == "__main__":
    model_pretraining()
