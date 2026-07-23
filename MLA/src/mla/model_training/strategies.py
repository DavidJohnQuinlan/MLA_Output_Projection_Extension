import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mla.config.paths import Paths
from mla.model_training.model_training import ModelPreTraining
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import BERTModelForClassification, BertModelForMLM
from mla.models.GPT2.config import GPT2Config
from mla.models.GPT2.heads import GPT2ForSequenceClassification, GPT2LMHeadModel
from mla.utils.attention_hooks import collect_attention_head_activations
from mla.utils.attention_utils import compute_model_cka
from mla.utils.model_utils import CausalLMMetricEvaluation, ClassificationMetricEvaluation, MetricEvaluation, MetricEvaluationProtocol
from mla.utils.utils import calculate_flop_metrics, measure_inference_speed


class TrainingStrategy(ABC):
    @abstractmethod
    def build_model(self, config: DictConfig) -> nn.Module: ...

    @abstractmethod
    def metric_cls(self) -> type[MetricEvaluationProtocol]: ...

    @abstractmethod
    def build_results(
        self,
        pretrainer: ModelPreTraining,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        val_loader: DataLoader,
        config: DictConfig
        ) -> dict: ...


class FineTuningStrategy(TrainingStrategy):
    @abstractmethod
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module: ...

    @abstractmethod
    def build_results(self, results: dict, config: DictConfig) -> dict: ...


class BERTPretrainingStrategy(TrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = BertConfig.from_pretrained(
            config.model_config_name,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )
        return BertModelForMLM(model_config)

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return MetricEvaluation

    def build_results(
        self,
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
            "pre_training_validation_loss": f"{pretrainer.best_validation_loss:.4f}",
            "pre_training_top1_mlm_accuracy": f"{pretrainer.best_validation_metrics['accuracy']:.4f}",
            "pre_training_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class GPT2PretrainingStrategy(TrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = GPT2Config(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.hidden_size,
            n_positions=config.max_position_embeddings,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )
        model = GPT2LMHeadModel(model_config)
        model.loss_type = config.loss_type
        return model

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return CausalLMMetricEvaluation

    def build_results(
        self,
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
            "pre_training_validation_loss": f"{pretrainer.best_validation_loss:.4f}",
            "pre_training_validation_perplexity": f"{math.exp(pretrainer.best_validation_loss):.4f}",
            "pre_training_next_token_accuracy": f"{pretrainer.best_validation_metrics['accuracy']:.4f}",
            "pre_training_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class BERTFineTuningStrategy(FineTuningStrategy):
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module:
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

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return ClassificationMetricEvaluation

    def build_results(self, results: dict, config: DictConfig) -> dict[str, Any]:
        """
        Build a results summary dictionary for a finetuning run.
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
            "avg_finetune_mcc": f"{np.mean(results["mcc"]):.4f} +/- {np.std(results["mcc"]):.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class GPT2FineTuningStrategy(FineTuningStrategy):
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module:
        model_config = GPT2Config(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.hidden_size,
            n_positions=config.max_position_embeddings,
            num_labels=config.num_labels,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )

        # Load the pretrained GPT2 model
        gpt2_model = ModelPreTraining.load_model(
            model_class=GPT2LMHeadModel,
            model_config=model_config,
            checkpoint_path=paths.pretrained_model_path,
        )

        # Convert GPT2 model to classification GPT2
        gpt2_classifier = GPT2ForSequenceClassification.from_pretrained_lm(lm_model=gpt2_model, config=model_config)

        # Compile the GPT2 classifier
        compiled_gpt2_classifier = torch.compile(gpt2_classifier)
        return compiled_gpt2_classifier

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return ClassificationMetricEvaluation

    def build_results(self, results: dict, config: DictConfig) -> dict[str, Any]:
        """
        Build a results summary dictionary for a finetuning run.
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
            "avg_finetune_mcc": f"{np.mean(results["mcc"]):.4f} +/- {np.std(results["mcc"]):.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }

_PRETRAINING_STRATEGIES: dict[str, TrainingStrategy] = {
    "BERT": BERTPretrainingStrategy(),
    "GPT2": GPT2PretrainingStrategy(),
}

_FINETUNING_STRATEGIES: dict[str, TrainingStrategy] = {
    "BERT": BERTFineTuningStrategy(),
    "GPT2": GPT2FineTuningStrategy(),
}
