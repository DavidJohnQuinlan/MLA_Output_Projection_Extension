import json
import math
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch
import wandb.integration.torch.wandb_torch as wandb_torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig, get_cosine_schedule_with_warmup

import wandb
from mla.config.paths import Paths
from mla.utils.metrics import LossMeter, MetricEvaluationProtocol
from mla.utils.reporting import get_run_metadata
from mla.utils.profiling import compute_compression_ratio
from mla.utils.setup import get_device, safe_hook_variable_gradient_stats, setup_wandb

wandb_torch.TorchHistory._hook_variable_gradient_stats = safe_hook_variable_gradient_stats


class BaseModelTraining(ABC):
    """
    A class that contains methods to help with pre-training/fine-tuning a model. Accelerator is used to abstract out some of the
    more complex model training/fine-tuning logic. This class also contains methods to define the warm-up scheduler, the logging of
    metrics locally and to WandB.

    Attributes:
        model (nn.Module): Model as defined by the configuration.
        optimizer (Optimizer): Classic optimizer.
        metric_fn (MetricEvaluationProtocol): Class that handles the collection and calculation of both training and validation metrics.
        config (DictConfig): General experiment configuration.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        metric_fn: type[MetricEvaluationProtocol],
        config: DictConfig,
        paths: Paths,
    ):
        self.training_bar = None
        self.validation_bar = None
        self.scheduler = None

        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )
        self.optimizer = self._build_optimizer(model, optimizer)
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)
        self.metric_fn = metric_fn()
        self.best_checkpoint_file_path = paths.best_checkpoint_file_path
        self.recent_checkpoint_prefix = paths.recent_checkpoint_prefix

        self.global_step = 0
        self.last_logged_global_step = 0
        self.epoch = 0
        self.best_validation_loss = float("inf")
        self.best_metric = -float("inf")
        self.best_validation_metrics = None
        self.best_loss_metrics = None
        self._train_step_start_time = None
        self._validation_step_start_time = None

    @property
    def unwrapped_model(self) -> nn.Module:
         model = getattr(self.model, "_orig_mod", self.model)
         return getattr(model, "module", model)

    def _setup_scheduler(self) -> None:
        """
        Configures the learning rate scheduler with a warmup phase and cosine decay.

        The total number of steps and number of warmup steps are calculated based on the total number of training steps.
        """

        # Define warmup steps as a percentage of the total schedule
        warmup_steps = max(1, int(self.config.max_steps * self.config.warmup_rate)) if self.config.warmup else 0

        # Initialize the scheduler: Linear increase followed by Cosine decrease
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.max_steps
        )

        # Prepare for distributed/accelerated training
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _build_optimizer(self, model: nn.Module, optimizer: type[Optimizer]) -> Optimizer:
        """
        Build the optimizer, if weight decay is non-zero apply weight decay to all parameters bar bias's and norm layers.

        Args:
            model (nn.Module): Model whose biases and norm layer where weight decay will not be applied.
            optimizer (Optimizer): The optimization strategy.

        Returns:
            Optimizer: Instantiated optimizer ready for training.
        """
        no_decay = ["bias", "LayerNorm.weight"]

        if self.config.weight_decay != 0.0:

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = optimizer(optimizer_grouped_parameters, lr=self.config.learning_rate)
        else:
            optimizer = optimizer(model.parameters(), lr=self.config.learning_rate)
        return optimizer

    def _init_progress_bars(
        self,
        training_dataloader: DataLoader | None = None,
        validation_dataloader: DataLoader | None = None,
        reset: bool = False,
        mode: str | None = None,
        epoch: int | None = None
    ) -> None:
        """
        Initializes or resets training/validation tqdm progress bars.
        """

        # Updates existing progress bars without creating new objects
        if reset:
            if mode == "train" and training_dataloader is not None:
                self.training_bar.set_description(f"Epoch {epoch}")
                self.training_bar.reset(total=len(training_dataloader))

            elif mode == "validation" and validation_dataloader is not None:
                self.validation_bar.set_description(f"Eval @ Step {self.global_step}")
                self.validation_bar.reset(total=len(validation_dataloader))

            elif mode == "post_training_eval" and validation_dataloader is not None:
                return

        # Create the tqdm progress bars for the first time
        else:
            self.training_bar = tqdm(
                total=len(training_dataloader), position=0, desc="Training - Epoch 1", leave=True,
                disable=not self.accelerator.is_main_process,
            )
            self.validation_bar = tqdm(
                total=len(validation_dataloader), position=1, desc="Validation", leave=True,
                disable=not self.accelerator.is_main_process,
            )

    def _update_progress_bars(self, mode: str, loss_value: float) -> None:
        """
        Update the training or validation tqdm progress bar.
        """
        if mode == "train":
            self.training_bar.update(1)
            self.training_bar.set_postfix(loss=f"{loss_value:.4f}", step=self.global_step)

        elif mode == "validation":
            self.validation_bar.update(1)
            self.validation_bar.set_postfix(loss=f"{loss_value:.4f}", step=self.global_step)

        elif mode == "post_training_eval":
            return

    def _close_progress_bars(self) -> None:
        """
        Close the progress bars once we complete model training.
        """
        if self.training_bar: self.training_bar.close()
        if self.validation_bar: self.validation_bar.close()

    @abstractmethod
    def _is_best_model(
        self,
        validation_loss: LossMeter,
        validation_metrics: dict
    ) -> bool: ...

    @abstractmethod
    def _log_metrics(
        self,
        mode: str,
        loss: LossMeter,
        metrics: dict,
        steps_per_sec: float,
        samples_per_sec: float,
        total_norm: float | None = None,
    ) -> None: ...

    def _run_optimization_loop(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        """
        Main method which details all steps taken during model training. These include defining the warm-up scheduler, preparing
        the training and validation datasets and subsequently iterating through them during model training.

        Args:
            training_dataloader (DataLoader): The pre-batched training dataset.
            validation_dataloader (DataLoader): The pre-batched validation dataset.
        """
        training_dataloader = self.accelerator.prepare(training_dataloader)
        self._setup_scheduler()
        self._resume_from_checkpoint()
        self._init_progress_bars(training_dataloader=training_dataloader, validation_dataloader=validation_dataloader, reset=False)
        self._train_step_start_time = time.time()

        # For each step
        while self.global_step < self.config.max_steps:

            # Prepare for model training
            self.epoch += 1
            self.model.train()
            self.optimizer.zero_grad()
            training_loss = LossMeter()
            self.metric_fn.reset()
            self._init_progress_bars(training_dataloader=training_dataloader, reset=True, mode="train", epoch=self.epoch)

            # For each batch (starting at 1)
            for batch in training_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                # Automatically perform gradient accumulation
                with self.accelerator.accumulate(self.model):

                    # Forward and backward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    # Logging
                    self.metric_fn.update(logits=outputs.logits, labels=batch["labels"], mode="train")
                    n_tokens = (batch["labels"] != -100).sum().item()
                    training_loss.update(loss.item(), n=n_tokens)
                    self._update_progress_bars(mode="train", loss_value=training_loss.avg)

                    # Update the gradients
                    if self.accelerator.sync_gradients:
                        total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm).item()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1

                        # Update logs and metrics every N steps
                        if (self.global_step % self.config.train_eval_steps == 0 and self.global_step > 0):
                            if self.accelerator.is_main_process:

                                # Compute and log the training metrics
                                training_metrics = self.metric_fn.compute()

                                elapsed = time.time() - self._train_step_start_time
                                update_steps = self.global_step - self.last_logged_global_step
                                steps_per_sec = update_steps / elapsed

                                samples_per_sec = (update_steps * self.config.batch_size * self.config.gradient_accumulation_steps) / elapsed

                                self._log_metrics(
                                    mode="train",
                                    loss=training_loss,
                                    metrics=training_metrics,
                                    steps_per_sec=steps_per_sec,
                                    samples_per_sec=samples_per_sec,
                                    total_norm=total_norm,
                                )

                                # Reset the metrics and loss after logging
                                self.metric_fn.reset()
                                training_loss = LossMeter()
                                self.last_logged_global_step = self.global_step
                                self._train_step_start_time = time.time()

                        if (self.global_step % self.config.eval_steps == 0 and self.global_step > 0):
                            validation_loss, validation_metrics = self.eval_model(validation_dataloader=validation_dataloader, training_eval=True)
                            self.model.train()
                            self.metric_fn.reset()
                            self._train_step_start_time = time.time()
                            self._save_checkpoint(checkpoint_type="recent")

                            if self._is_best_model(validation_loss, validation_metrics):
                                self.best_validation_metrics = validation_metrics
                                self.best_loss_metrics = validation_loss
                                if self.accelerator.is_main_process:
                                    self._save_checkpoint(checkpoint_type="best")
        self._close_progress_bars()

    def eval_model(self, validation_dataloader: DataLoader, training_eval: bool=True) -> tuple[LossMeter, dict]:
        """
        Evaluate the model on some pre-batched dataset.

        Calculate the validation metrics and loss for the model, can be invoked either during or after model training.

        Args:
            validation_dataloader (DataLoader): The pre-batched validation dataset.
            training_eval (bool): True if evaluating during training false otherwise. Defaults to True.
        """
        # Prepare for model evaluation
        mode = "validation" if training_eval else "post_training_eval"
        self.model.eval()
        self.metric_fn.reset()
        validation_loss = LossMeter()

        validation_dataloader = self.accelerator.prepare(validation_dataloader)
        self._init_progress_bars(validation_dataloader=validation_dataloader, reset=True, mode=mode)
        self._validation_step_start_time = time.time()

        with torch.no_grad():
            for batch in validation_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss.detach().cpu().item()
                n_tokens = (batch["labels"] != -100).sum().item()
                validation_loss.update(loss, n=n_tokens)
                self._update_progress_bars(mode=mode, loss_value=validation_loss.avg)
                self.metric_fn.update(logits=outputs.logits, labels=batch["labels"], mode=mode)

            validation_loss.reduce(self.accelerator)
            self.metric_fn.reduce(self.accelerator)
            validation_metrics = self.metric_fn.compute()

            if training_eval and self.accelerator.is_main_process:
                elapsed = time.time() - self._validation_step_start_time
                self._log_metrics(
                    mode="validation",
                    loss=validation_loss,
                    metrics=validation_metrics,
                    steps_per_sec=len(validation_dataloader) / elapsed,
                    samples_per_sec=len(validation_dataloader.dataset) / elapsed,
                )
            self.metric_fn.reset()

        return validation_loss, validation_metrics

    def _save_checkpoint(self, checkpoint_type: Literal["best", "recent"]) -> None:
        """
        Unwraps the model from the Accelerator environment and saves its state dict locally. Additionally, creates
        a metadata json file associated with the saved model.
        """
        model_to_save = self.unwrapped_model
        metadata = {
            "config": model_to_save.config.to_dict(),
            "seed": self.seed,
            **get_run_metadata(self.wandb_id)
        }

        if checkpoint_type == "best":
            self.best_checkpoint_file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_th = self.best_checkpoint_file_path.with_suffix(".th.tmp")
            tmp_json = self.best_checkpoint_file_path.with_suffix(".json.tmp")
            self.accelerator.save(model_to_save.state_dict(), tmp_th)
            tmp_json.write_text(json.dumps(metadata, indent=4, default=str))
            tmp_json.replace(self.best_checkpoint_file_path.with_suffix(".json"))
            tmp_th.replace(self.best_checkpoint_file_path)

        elif checkpoint_type == "recent":
            checkpoint_dir = self.recent_checkpoint_prefix / f"step_{self.global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.save_state(str(checkpoint_dir))

            if self.accelerator.is_main_process:
                extra_state = {
                    "step": self.global_step,
                    "epoch": self.epoch,
                    "wandb_id": self.wandb_id,
                    "best_score": self.best_validation_loss,
                    "best_metric": self.best_metric,
                }
                checkpoint_dir_extra_state = checkpoint_dir / "extra_state.json"
                checkpoint_dir_extra_state.write_text(json.dumps(extra_state, indent=4))
                self._prune_recent_checkpoints(keep=self.config.keep_last_n_checkpoints)

    def _prune_recent_checkpoints(self, keep: int) -> None:
        if keep < 1:
            return
        sorted_checkpoints = sorted(
            (p for p in self.recent_checkpoint_prefix.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.removeprefix("step_"))
        )
        for checkpoint in sorted_checkpoints[:-keep]:
            shutil.rmtree(checkpoint)

    def _find_latest_checkpoint(self) -> Path | None:
        checkpoints = [p for p in self.recent_checkpoint_prefix.glob("step_*")
                if p.is_dir() and (p / "extra_state.json").exists()]
        return max(checkpoints, key=lambda p: int(p.name.removeprefix("step_"))) if checkpoints else None

    def _prior_wandb_id(self) -> str | None:
        checkpoint = self._find_latest_checkpoint()
        if checkpoint is None:
            return None
        return json.loads((checkpoint / "extra_state.json").read_text()).get("wandb_id")

    def _resume_from_checkpoint(self) -> None:
        checkpoint = self._find_latest_checkpoint()
        if checkpoint is None:
            return None
        self.accelerator.load_state(str(checkpoint))
        extra_state = json.loads((checkpoint / "extra_state.json").read_text())
        self.global_step = extra_state["step"]
        self.epoch = extra_state.get("epoch")
        self.best_validation_loss = extra_state["best_score"]
        self.best_metric = extra_state["best_metric"]
        self.wandb_id = extra_state["wandb_id"]
        self.last_logged_global_step = self.global_step
        self.accelerator.print(f"Resumed from {checkpoint.name} (step {self.global_step})")

    @classmethod
    def load_checkpoint(
        cls,
        model_class: type[torch.nn.Module],
        checkpoint_path: Path,
        config_class: type[PretrainedConfig],
        config_overrides: dict | None = None,
    ) -> torch.nn.Module:
        """
        Instantiate a model from its checkpoint's own saved config, then load weights.
        """
        device = get_device()
        metadata = json.loads(checkpoint_path.with_suffix(".json").read_text())
        model_config = config_class.from_dict(metadata["config"])
        for k, v in (config_overrides or {}).items():
            setattr(model_config, k, v)

        model = model_class(model_config)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device)


class ModelPreTraining(BaseModelTraining):
    """
    A class that contains methods to help with training a model. Accelerator is used to abstract out some of the more complex model
    training logic. This class also contains methods to define the warm-up scheduler, the logging of metrics locally and to WandB.

    Attributes:
        model (nn.Module): Model we wish to train.
        optimizer (Optimizer): Optimization method.
        metric_fn (MetricEvaluationProtocol): Class that handles the collection and calculation of both training and validation metrics.
        config (DictConfig): General model/experiment configuration.
        paths (Paths): Paths to data/models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        metric_fn: type[MetricEvaluationProtocol],
        config: DictConfig,
        paths: Paths,
        seed: int,
    ):
        super().__init__(model, optimizer, metric_fn, config, paths)
        self.seed = seed
        self.model = torch.compile(self.model)
        self.wandb_mode = setup_wandb()
        wandb_cfg = OmegaConf.to_container(self.config, resolve=True)
        wandb_cfg["kv_ratio"] = compute_compression_ratio(self.config)
        if self.accelerator.is_main_process:
            prior_wandb_id = self._prior_wandb_id()
            resume_kwargs = {"id": prior_wandb_id, "resume": "allow"} if prior_wandb_id else {}
            wandb.init(
                project=self.config.experiment_project,
                group=self.config.experiment_name,
                name=self.best_checkpoint_file_path.stem,
                job_type=self.config.job_type,
                config=wandb_cfg,
                tags=[config.attention_mechanism],
                mode=self.wandb_mode,
                **resume_kwargs,
            )
            self.wandb_id = wandb.run.id if wandb.run is not None else None
            wandb.watch(self.model, log="all", log_freq=self.config.eval_steps)
        else:
            self.wandb_id = None

    def _log_metrics(
        self,
        mode: str,
        loss: LossMeter,
        metrics: dict,
        steps_per_sec: float,
        samples_per_sec: float,
        total_norm: float | None = None
    ) -> None:
        """
        Log the training or validation loss and metrics to wandb UI.

        Args:
            mode (str): The target bar to update; must be "train" or "validation".
            loss (LossMeter): LossMeter object that contains training or validation loss totals and averages.
            metrics (dict): Dictionary containing training or validation metrics.
            steps_per_sec (float): Number of steps taken per second.
            samples_per_sec (float): Number of samples processed per second.
            total_norm (float, optional): The global norm of the gradients. Defaults to None.
        """
        log_dict = {}
        if mode == "train":
            log_dict["train/loss"] = loss.avg
            log_dict["train/perplexity"] = math.exp(min(loss.avg, 20))
            log_dict["train/accuracy"] = metrics["accuracy"]
            log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
            log_dict["train/steps_per_sec"] = steps_per_sec
            log_dict["train/samples_per_sec"] = samples_per_sec
            log_dict["train/grad_norm"] = total_norm

        elif mode == "validation":
            log_dict["validation/loss"] = loss.avg
            log_dict["validation/perplexity"] = math.exp(min(loss.avg, 20))
            log_dict["validation/accuracy"] = metrics["accuracy"]
            log_dict["validation/steps_per_sec"] = steps_per_sec
            log_dict["validation/samples_per_sec"] = samples_per_sec

        wandb.log(log_dict, step=self.global_step)

    def _is_best_model(self, validation_loss: LossMeter, validation_metrics: dict) -> bool:
        """
        Defines whether the current model is better than the best loss value.
        """
        if validation_loss.avg < self.best_validation_loss:
            self.best_validation_loss = validation_loss.avg
            if wandb.run is not None:
                wandb.run.summary["best_validation_loss"] = validation_loss.avg
            return True
        return False

    def model_pretraining(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, validation_dataloader)
        if self.accelerator.is_main_process:
            wandb.unwatch()
            wandb.finish()


class ModelFineTuning(BaseModelTraining):
    """
    A class that contains methods to help with fine-tuning a model. Accelerator is used to abstract out some of the more complex
    model fine-tuning logic. This class also contains methods to define the warm-up scheduler, the logging of metrics locally and to
    WandB.

    Attributes:
        model (nn.Module): Model we wish to train.
        optimizer (Optimizer): Optimization method.
        metric_fn (MetricEvaluationProtocol): Class that handles the collection and calculation of both training and validation metrics.
        config (DictConfig): General model/experiment configuration.
        paths (Paths): Paths to data/models.
        seed (int): The seed value of the experiment.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        metric_fn: type[MetricEvaluationProtocol],
        config: DictConfig,
        paths: Paths,
        seed: int,
    ):
        super().__init__(model, optimizer, metric_fn, config, paths)
        self.seed = seed
        self.wandb_mode = setup_wandb()
        wandb_cfg = OmegaConf.to_container(self.config, resolve=True)
        wandb_cfg["kv_ratio"] = compute_compression_ratio(self.config)
        if self.accelerator.is_main_process:
            prior_wandb_id = self._prior_wandb_id()
            resume_kwargs = {"id": prior_wandb_id, "resume": "allow"} if prior_wandb_id else {}
            wandb.init(
                project=self.config.experiment_project,
                group=self.config.experiment_name,
                name=self.best_checkpoint_file_path.stem,
                job_type=self.config.job_type,
                config=wandb_cfg,
                tags=[config.attention_mechanism, config.dataset_config_name],
                mode=self.wandb_mode,
                **resume_kwargs,
            )
            self.wandb_id = wandb.run.id if wandb.run is not None else None
            wandb.define_metric("validation/accuracy", summary="max")
            wandb.define_metric("validation/loss", summary="min")
            wandb.define_metric("validation/f1", summary="max")
            wandb.define_metric("validation/mcc", summary="max")
            wandb.watch(self.model, log="all", log_freq=self.config.eval_steps)
        else:
            self.wandb_id = None

    def _log_metrics(
        self,
        mode: str,
        loss: LossMeter,
        metrics: dict,
        steps_per_sec: float,
        samples_per_sec: float,
        total_norm: float | None = None
    ) -> None:
        """
        Log the training or validation loss and metrics to wandb UI.

        Args:
            mode (str): The target bar to update; must be "train" or "validation".
            loss (LossMeter): LossMeter object that contains training or validation loss totals and averages.
            metrics (dict): Dictionary containing training or validation metrics.
            steps_per_sec (float): Number of steps taken per second.
            samples_per_sec (float): Number of samples processed per second.
            total_norm (float, optional): The global norm of the gradients. Defaults to None.
        """
        log_dict = {}
        if mode == "train":
            log_dict["train/loss"] = loss.avg
            log_dict["train/accuracy"] = metrics["accuracy"]
            log_dict["train/precision"] = metrics["precision"]
            log_dict["train/recall"] = metrics["recall"]
            log_dict["train/f1"] = metrics["f1"]
            log_dict["train/mcc"] = metrics["mcc"]
            log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
            log_dict["train/steps_per_sec"] = steps_per_sec
            log_dict["train/samples_per_sec"] = samples_per_sec
            log_dict["train/grad_norm"] = total_norm

        elif mode == "validation":
            log_dict["validation/loss"] = loss.avg
            log_dict["validation/accuracy"] = metrics["accuracy"]
            log_dict["validation/precision"] = metrics["precision"]
            log_dict["validation/recall"] = metrics["recall"]
            log_dict["validation/f1"] = metrics["f1"]
            log_dict["validation/mcc"] = metrics["mcc"]
            log_dict["validation/steps_per_sec"] = steps_per_sec
            log_dict["validation/samples_per_sec"] = samples_per_sec

        wandb.log(log_dict, step=self.global_step)

    def _is_best_model(self, validation_loss: LossMeter, validation_metrics: dict) -> bool:
        """
        Defines whether the current model is better than the best metric value.
        """

        eval_metric = self.config.eval_metric
        if validation_metrics[eval_metric] > self.best_metric:
            self.best_metric = validation_metrics[eval_metric]
            if wandb.run is not None:
                wandb.run.summary[f"best_{self.config.eval_metric}"] = validation_metrics[self.config.eval_metric]
                wandb.run.summary["best_validation_loss"] = validation_loss.avg
            return True
        return False

    def model_finetuning(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, validation_dataloader)
        if self.accelerator.is_main_process:
            wandb.unwatch()
            wandb.finish()
