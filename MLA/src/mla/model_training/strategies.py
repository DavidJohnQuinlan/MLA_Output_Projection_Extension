from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch import nn
import math

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from mla.model_training.model_training import ModelPreTraining
from mla.utils.model_utils import MetricEvaluationProtocol
from mla.models.BERT.bert_model.bert_config import BertConfig
from mla.models.BERT.bert_model.bert_heads import BertModelForMLM
from mla.models.GPT2.config import GPT2Config
from mla.models.GPT2.heads import GPT2LMHeadModel
from mla.utils.model_utils import MetricEvaluation
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from mla.utils.attention_hooks import collect_attention_head_activations
from mla.utils.attention_utils import compute_model_cka
from mla.utils.model_utils import MetricEvaluation, CausalLMMetricEvaluation
from mla.utils.utils import calculate_flop_metrics, measure_inference_speed



class PretrainingStrategy(ABC):
    @abstractmethod
    def build_model(self, config: DictConfig) -> nn.Module: ...
        
    @abstractmethod
    def metric_cls(self) -> type[MetricEvaluationProtocol]: ...

    @abstractmethod
    def build_results(
        self, 
        pretrainer: ModelPreTraining, 
        model: nn.Module, tokenizer: 
        PreTrainedTokenizerBase, 
        val_loader: DataLoader, 
        config: DictConfig
        ) -> dict: ...


class BERTPretrainingStrategy(PretrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = BertConfig.from_pretrained(
            config.model_config_name,
            loss_type=config.loss_type,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )
        return BertModelForMLM(model_config)

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return MetricEvaluation

    def build_results(
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
            "pre_training_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


class GPT2PretrainingStrategy(PretrainingStrategy):
    def build_model(self, config: DictConfig) -> nn.Module:
        model_config = GPT2Config.from_pretrained(
            config.model_config_name,
            loss_type=config.loss_type,
            attention_mechanism=config.attention_mechanism,
            kv_compression_dim=config.kv_compression_dim,
            q_compression_dim=config.q_compression_dim,
            output_compression_dim=config.output_compression_dim,
        )
        return GPT2LMHeadModel(model_config)

    def metric_cls(self) -> type[MetricEvaluationProtocol]: return CausalLMMetricEvaluation

    def build_results(
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
            "pre_training_validation_perplexity": f"{math.exp(pretrainer.best_eval_loss):.4f}",
            "pre_training_next_token_accuracy": f"{pretrainer.best_eval_metrics['accuracy']:.4f}",
            "pre_training_cka": f"{avg_cka:.4f}",
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        }


_STRATEGIES: dict[str, PretrainingStrategy] = {
    "TinyBERT": BERTPretrainingStrategy(),
    "GPT2": GPT2PretrainingStrategy(),
}





# class FinetuningStrategy(ABC):
#     @abstractmethod
#     def build_model(self, config: DictConfig) -> nn.Module: ...

#     @abstractmethod
#     def metric_cls(self) -> type[MetricEvaluationProtocol]: ...

#     @abstractmethod
#     def build_results(self, results: dict, config: DictConfig) -> dict: ...
