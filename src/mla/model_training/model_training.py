import time
from abc import ABC, abstractmethod
from pathlib import Path

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
from mla.utils.model_utils import LossMeter, MetricEvaluationProtocol
from mla.utils.utils import compute_compression_ratio, get_device, safe_hook_variable_gradient_stats, setup_wandb

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
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
        )
        self.optimizer = self._build_optimizer(model, optimizer)
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)
        self.metric_fn = metric_fn()
        self.model_file_path = paths.model_file_path

        self.global_step = 0
        self.last_logged_global_step = 0
        self.epoch = 0
        self.best_validation_loss = float("inf")
        self._train_step_start_time = None
        self._validation_step_start_time = None
        self.best_validation_metrics = None
        self.best_loss_metrics = None


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
            self.training_bar = tqdm(total=len(training_dataloader), position=0, desc="Training - Epoch 1", leave=True)
            self.validation_bar = tqdm(total=len(validation_dataloader), position=1, desc="Validation", leave=True)

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

    def _is_best_model(self, validation_loss: LossMeter, validation_metrics: dict) -> bool:
        """
        Defines whether the current model is better than the best loss value.
        """
        if validation_loss.avg < self.best_validation_loss:
            self.best_validation_loss = validation_loss.avg
            wandb.run.summary["best_validation_loss"] = validation_loss.avg
            return True
        return False

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
        training_dataloader, validation_dataloader = self.accelerator.prepare(
            training_dataloader, validation_dataloader
        )
        self._setup_scheduler()
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
                    training_loss.update(loss.item(), n=batch["input_ids"].size(0))
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
                            if self.accelerator.is_main_process:

                                # Evaluate the current model on the validation dataset
                                validation_loss, validation_metrics = self.eval_model(validation_dataloader=validation_dataloader, training_eval=True)
                                self.model.train()
                                self.metric_fn.reset()
                                self._train_step_start_time = time.time()

                                # Save the best model
                                if self._is_best_model(validation_loss, validation_metrics):
                                    self.save_model()
                                    self.best_validation_metrics = validation_metrics
                                    self.best_loss_metrics = validation_loss
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

                # Forward pass
                outputs = self.model(**batch)

                # Calculate and update validation loss and metrics
                loss = outputs.loss.detach().cpu().item()
                validation_loss.update(loss, n=batch["input_ids"].size(0))
                self._update_progress_bars(mode=mode, loss_value=validation_loss.avg)
                self.metric_fn.update(logits=outputs.logits, labels=batch["labels"], mode=mode)
                validation_metrics = self.metric_fn.compute()

            elapsed = time.time() - self._validation_step_start_time
            steps_per_sec = len(validation_dataloader) / elapsed
            samples_per_sec = len(validation_dataloader.dataset) / elapsed

            # If evaluating post training run
            if training_eval:
                self._log_metrics(
                    mode="validation",
                    loss=validation_loss,
                    metrics=validation_metrics,
                    steps_per_sec=steps_per_sec,
                    samples_per_sec=samples_per_sec,
                )
            self.metric_fn.reset()

        return validation_loss, validation_metrics

    def save_model(self) -> None:
        """
        Unwraps the model from the Accelerator environment and saves its state dict locally.
        """
        # Ensure the directory to store the model locally exist
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap the accelerator model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_to_save = unwrapped_model._orig_mod if hasattr(unwrapped_model, "_orig_mod") else unwrapped_model

        # Save the unwrapped models state dict locally
        self.accelerator.save(model_to_save.state_dict(), self.model_file_path)

    @classmethod
    def load_model(
        cls,
        model_class: type[torch.nn.Module],
        model_config: PretrainedConfig,
        checkpoint_path: str | Path,
    ) -> torch.nn.Module:
        """
        Factory method to instantiate a model architecture and load its saved weights.
        """
        device = get_device()
        model = model_class(model_config)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
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
    ):
        super().__init__(model, optimizer, metric_fn, config, paths)
        self.model = torch.compile(self.model)
        self.wandb_mode = setup_wandb()
        wandb_cfg = OmegaConf.to_container(self.config, resolve=True)
        wandb_cfg["kv_ratio"] = compute_compression_ratio(self.config)
        wandb.init(
            project=self.config.experiment_project,
            group=self.config.experiment_name,
            name=self.config.pretrained_model_name,
            job_type=self.config.job_type,
            config=wandb_cfg,
            tags=[config.attention_mechanism],
            mode=self.wandb_mode,
        )
        wandb.watch(self.model, log="all", log_freq=self.config.eval_steps)
        self.history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}

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
        Log the training or validation loss and metrics to wandb UI and to local dictionary.

        Args:
            mode (str): The target bar to update; must be "train" or "validation".
            loss (LossMeter): LossMeter object that contains training or validation loss totals and averages.
            metrics (dict): Dictionary containing training or validation metrics.
            steps_per_sec (float): Number of steps taken per second.
            samples_per_sec (float): Number of samples processed per second.
            total_norm (float, optional): The global norm of the gradients. Defaults to None.
        """
        self.history[f"{mode}_loss"].append(loss.avg)
        self.history[f"{mode}_accuracy"].append(metrics["accuracy"])

        log_dict = {}
        if mode == "train":
            log_dict["train/loss"] = loss.avg
            log_dict["train/accuracy"] = metrics["accuracy"]
            log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
            log_dict["train/steps_per_sec"] = steps_per_sec
            log_dict["train/samples_per_sec"] = samples_per_sec
            log_dict["train/grad_norm"] = total_norm

        elif mode == "validation":
            log_dict["validation/loss"] = loss.avg
            log_dict["validation/accuracy"] = metrics["accuracy"]
            log_dict["validation/steps_per_sec"] = steps_per_sec
            log_dict["validation/samples_per_sec"] = samples_per_sec

        wandb.log(log_dict, step=self.global_step)

    def train_model(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, validation_dataloader)
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
        self.wandb_mode = setup_wandb()
        wandb_cfg = OmegaConf.to_container(self.config, resolve=True)
        wandb_cfg["kv_ratio"] = compute_compression_ratio(self.config)
        wandb.init(
            project=self.config.experiment_project,
            group=self.config.experiment_name,
            name=f"{self.config.fine_tuned_model_name}__seed_{seed}",
            job_type=self.config.job_type,
            config=wandb_cfg,
            tags=[config.attention_mechanism, config.dataset_config_name],
            mode=self.wandb_mode,
        )
        wandb.define_metric("validation/accuracy", summary="max")
        wandb.define_metric("validation/loss", summary="min")
        wandb.define_metric("validation/f1", summary="max")
        wandb.define_metric("validation/mcc", summary="max")
        wandb.watch(self.model, log="all", log_freq=self.config.eval_steps)
        self.best_metric = 0.0
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_recall": [],
            "train_precision": [],
            "train_f1": [],
            "train_mcc": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_recall": [],
            "validation_precision": [],
            "validation_f1": [],
            "validation_mcc": [],
        }

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
        Log the training or validation loss and metrics to wandb UI and to local dictionary.

        Args:
            mode (str): The target bar to update; must be "train" or "validation".
            loss (LossMeter): LossMeter object that contains training or validation loss totals and averages.
            metrics (dict): Dictionary containing training or validation metrics.
            steps_per_sec (float): Number of steps taken per second.
            samples_per_sec (float): Number of samples processed per second.
            total_norm (float, optional): The global norm of the gradients. Defaults to None.
        """

        self.history[f"{mode}_loss"].append(loss.avg)
        self.history[f"{mode}_accuracy"].append(metrics["accuracy"])
        self.history[f"{mode}_recall"].append(metrics["recall"])
        self.history[f"{mode}_precision"].append(metrics["precision"])
        self.history[f"{mode}_f1"].append(metrics["f1"])
        self.history[f"{mode}_mcc"].append(metrics["mcc"])

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
            wandb.run.summary[f"best_{self.config.eval_metric}"] = validation_metrics[self.config.eval_metric]
            wandb.run.summary["best_validation_loss"] = validation_loss.avg
            return True
        return False

    def fine_tune_model(self, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, validation_dataloader)
        wandb.unwatch()
        wandb.finish()
