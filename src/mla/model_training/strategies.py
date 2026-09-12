import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mla.config.paths import Paths, build_model_name
from mla.model_training.model_training import ModelPreTraining
from mla.models.BERT.config import BertConfig
from mla.models.BERT.heads import BertForMaskedLM, BertForSequenceClassification
from mla.models.GPT2.config import GPT2Config
from mla.models.GPT2.heads import GPT2ForSequenceClassification, GPT2LMHeadModel
from mla.utils.attention_hooks import collect_attention_head_activations
from mla.utils.attention_utils import compute_model_cka
from mla.utils.model_utils import CausalLMMetricEvaluation, ClassificationMetricEvaluation, MetricEvaluation, MetricEvaluationProtocol
from mla.utils.profiling import calculate_flop_metrics, compute_training_compute, measure_inference_cost
from mla.utils.reporting import get_run_metadata

class PreTrainingStrategy(ABC):
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
        config: DictConfig,
        pretraining_seed: int
        ) -> dict: ...


class FineTuningStrategy(ABC):
    @abstractmethod
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module: ...

    @abstractmethod
    def metric_cls(self, config) -> type[MetricEvaluationProtocol]: ...

    @abstractmethod
    def build_results(self, results: dict, config: DictConfig, pretraining_seed: int) -> dict: ...


class BERTPretrainingStrategy(PreTrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = BertConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.n_layer,
            num_attention_heads=config.n_head,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )
        return BertForMaskedLM(model_config)

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return MetricEvaluation

    def build_results(
        self,
        pretrainer: ModelPreTraining,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        val_loader: DataLoader,
        config: DictConfig,
        pretraining_seed: int
    ) -> dict:
        """
        Build a results summary dictionary for a pretraining run.
        """
        flops, macs, params = calculate_flop_metrics(model, config)
        cost = measure_inference_cost(pretrainer.unwrapped_model, tokenizer, config, dtype=config.mixed_precision)
        training_compute = compute_training_compute(config, flops)
        activations_list = collect_attention_head_activations(pretrainer, val_loader)
        avg_cka = compute_model_cka(activations_list)
        return {
            **get_run_metadata(pretrainer.wandb_id),
            "pretraining_seed": pretraining_seed,
            "timestamp": datetime.now().isoformat(),
            "model_name": f"{build_model_name(config, pretraining_seed)}.th",
            "attention_mechanism": config.attention_mechanism,
            "dataset": f"{config.dataset_name}_{config.dataset_config_name}" if config.dataset_config_name else config.dataset_name,
            "n_params": params,
            "GFLOPS": f"{flops / 1e9:.3f}",
            "GMACS": f"{macs / 1e9:.3f}",
            "train_tokens": training_compute["train_tokens"],
            "train_flops": f"{training_compute['train_flops']:.3e}",
            "Inf (ms)": f"{cost['median_ms']:.2f}",
            "peak_mem_mb": f"{cost['peak_mem_mb']:.1f}" if cost['peak_mem_mb'] is not None else "n/a",
            "kv": config.kv_compression_dim,
            "q": config.q_compression_dim,
            "o": config.output_compression_dim,
            "pretraining_validation_loss": f"{pretrainer.best_validation_loss:.4f}",
            "pretraining_top1_mlm_accuracy": f"{pretrainer.best_validation_metrics['accuracy']:.4f}",
            "pretraining_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class GPT2PretrainingStrategy(PreTrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = GPT2Config(
            hidden_size=config.hidden_size,
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
        config: DictConfig,
        pretraining_seed: int
    ) -> dict:
        """
        Build a results summary dictionary for a pretraining run.
        """
        flops, macs, params = calculate_flop_metrics(model, config)
        cost = measure_inference_cost(pretrainer.unwrapped_model, tokenizer, config, dtype=config.mixed_precision)
        training_compute = compute_training_compute(config, flops)
        activations_list = collect_attention_head_activations(pretrainer, val_loader)
        avg_cka = compute_model_cka(activations_list)
        return {
            **get_run_metadata(pretrainer.wandb_id),
            "pretraining_seed": pretraining_seed,
            "timestamp": datetime.now().isoformat(),
            "model_name": f"{build_model_name(config, pretraining_seed)}.th",
            "attention_mechanism": config.attention_mechanism,
            "dataset": f"{config.dataset_name}_{config.dataset_config_name}" if config.dataset_config_name else config.dataset_name,
            "n_params": params,
            "GFLOPS": f"{flops / 1e9:.3f}",
            "GMACS": f"{macs / 1e9:.3f}",
            "train_tokens": training_compute["train_tokens"],
            "train_flops": f"{training_compute['train_flops']:.3e}",
            "Inf (ms)": f"{cost['median_ms']:.2f}",
            "peak_mem_mb": f"{cost['peak_mem_mb']:.1f}" if cost['peak_mem_mb'] is not None else "n/a",
            "kv": config.kv_compression_dim,
            "q": config.q_compression_dim,
            "o": config.output_compression_dim,
            "pretraining_validation_loss": f"{pretrainer.best_validation_loss:.4f}",
            "pretraining_validation_perplexity": f"{math.exp(min(pretrainer.best_validation_loss, 20)):.4f}",
            "pretraining_next_token_accuracy": f"{pretrainer.best_validation_metrics['accuracy']:.4f}",
            "pretraining_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class BERTFineTuningStrategy(FineTuningStrategy):
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module:
        bert_model = ModelPreTraining.load_checkpoint(
            model_class=BertForMaskedLM,
            checkpoint_path=paths.pretraining_checkpoint_path,
            config_class=BertConfig,
            config_overrides={"num_labels": config.num_labels},
        )
        bert_classifier = BertForSequenceClassification.from_pretrained_lm(lm_model=bert_model, config=bert_model.config)
        return torch.compile(bert_classifier)

    def metric_cls(self, config) -> Callable[[], MetricEvaluationProtocol]:
        return partial(ClassificationMetricEvaluation, num_labels=config.num_labels)

    def build_results(self, results: dict, config: DictConfig, pretraining_seed: int) -> dict[str, Any]:
        """
        Build a results summary dictionary for a finetuning run.
        """
        return {
            **get_run_metadata(wandb_id=results.get("seed_to_wandb")),
            "pretraining_seed": pretraining_seed,
            "timestamp": datetime.now().isoformat(),
            "model_name": f"{results['pretraining_model_name']}__{config.dataset_config_name}",
            "attention_mechanism": config.attention_mechanism,
            "dataset": config.dataset_config_name,
            "kv": config.kv_compression_dim,
            "q": config.q_compression_dim,
            "o": config.output_compression_dim,
            "avg_finetuning_validation_loss": f"{np.mean(results['loss']):.4f} +/- {np.std(results['loss']):.4f}",
            "avg_finetuning_accuracy": f"{np.mean(results['accuracy']):.4f} +/- {np.std(results['accuracy']):.4f}",
            "avg_finetuning_f1": f"{np.mean(results['f1']):.4f} +/- {np.std(results['f1']):.4f}",
            "avg_finetuning_mcc": f"{np.mean(results['mcc']):.4f} +/- {np.std(results['mcc']):.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class GPT2FineTuningStrategy(FineTuningStrategy):
    def build_model(self, config: DictConfig, paths: Paths) -> nn.Module:
        gpt2_model = ModelPreTraining.load_checkpoint(
            model_class=GPT2LMHeadModel,
            checkpoint_path=paths.pretraining_checkpoint_path,
            config_class=GPT2Config,
            config_overrides={"num_labels": config.num_labels},
        )
        gpt2_classifier = GPT2ForSequenceClassification.from_pretrained_lm(lm_model=gpt2_model, config=gpt2_model.config)
        return torch.compile(gpt2_classifier)

    def metric_cls(self, config) -> Callable[[], MetricEvaluationProtocol]:
        return partial(ClassificationMetricEvaluation, num_labels=config.num_labels)

    def build_results(self, results: dict, config: DictConfig, pretraining_seed: int) -> dict[str, Any]:
        """
        Build a results summary dictionary for a finetuning run.
        """
        return {
            **get_run_metadata(wandb_id=results.get("seed_to_wandb")),
            "pretraining_seed": pretraining_seed,
            "timestamp": datetime.now().isoformat(),
            "model_name": f"{results['pretraining_model_name']}__{config.dataset_config_name}",
            "attention_mechanism": config.attention_mechanism,
            "dataset": config.dataset_config_name,
            "kv": config.kv_compression_dim,
            "q": config.q_compression_dim,
            "o": config.output_compression_dim,
            "avg_finetuning_validation_loss": f"{np.mean(results['loss']):.4f} +/- {np.std(results['loss']):.4f}",
            "avg_finetuning_accuracy": f"{np.mean(results['accuracy']):.4f} +/- {np.std(results['accuracy']):.4f}",
            "avg_finetuning_f1": f"{np.mean(results['f1']):.4f} +/- {np.std(results['f1']):.4f}",
            "avg_finetuning_mcc": f"{np.mean(results['mcc']):.4f} +/- {np.std(results['mcc']):.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }

_PRETRAINING_STRATEGIES: dict[str, PreTrainingStrategy] = {
    "BERT": BERTPretrainingStrategy(),
    "GPT2": GPT2PretrainingStrategy(),
}

_FINETUNING_STRATEGIES: dict[str, FineTuningStrategy] = {
    "BERT": BERTFineTuningStrategy(),
    "GPT2": GPT2FineTuningStrategy(),
}
