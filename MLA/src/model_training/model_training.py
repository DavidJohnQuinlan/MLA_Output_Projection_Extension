import time
import torch
import os
import wandb
import wandb.integration.torch.wandb_torch as wandb_torch

from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm


from models.BERT.bert_model.bert_heads import BertModelForMLM, BERTModelForClassification
from models.BERT.configuration.configuration import PreTrainConfig, FineTuneConfig
from models.BERT.configuration.hyperparameters import PreTrainingHyperparameters, FineTuneHyperparameters
from utils.model_utils import MetricEvaluation, ClassificationMetricEvaluation, LossMeter
from utils.utils import safe_hook_variable_gradient_stats


class BaseModelTraining:
    """
    A class that contains methods to help with training/fine-tuning a BERT model. Accelerator is used to abstract out some of the 
    more complex model training/fine-tuning logic. This class also contains methods to define the warm-up scheduler, the logging of 
    metrics locally and to WandB. 

    Attributes:
        model (BertModelForMLM): TinyBERT model as defined by the TinyBERT configuration.
        optimizer (AdamW): Classic AdamW optimizer.
        metric_fn (MetricEvaluation): Class that handles the collection and calculation of both training and validation metrics.
        config (Config): General experiment configuration.
        hyperparameters (Hyperparameters): General experiment hyperparameters.
    """

    def __init__(
        self, 
        model: BertModelForMLM | BERTModelForClassification, 
        optimizer: AdamW,
        metric_fn: MetricEvaluation | ClassificationMetricEvaluation, 
        config: PreTrainConfig | FineTuneConfig, 
        hyperparameters: PreTrainingHyperparameters | FineTuneHyperparameters,
    ):
        self.cfg = config
        self.hp = hyperparameters

        wandb_pw = os.getenv("WANDB_PW")
        if wandb_pw:
            wandb.login(key=wandb_pw)
            self.wandb_mode = "online"
        else:
            self.wandb_mode = "offline"

        wandb_torch.TorchHistory._hook_variable_gradient_stats = safe_hook_variable_gradient_stats
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.hp.gradient_accumulation_steps,
            mixed_precision=self.hp.mixed_precision
        )
        self.optimizer = self._apply_weight_decay(model, optimizer)
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)
        self.metric_fn = metric_fn()
        
        self.model_file_path = self.cfg.paths.local.model_file_path
        
        self.global_step = 0
        self.last_logged_global_step = 0
        self.epoch = 0
        self.stop_training = False
        self.best_eval_loss = float("inf")
        
    def _setup_scheduler(self) -> None:
        """
        Configures the learning rate scheduler with a warmup phase and cosine decay.
        
        The total number of steps and number of warmup steps are calculated based on the total number of training steps.
        """

        # Define warmup steps as a percentage of the total schedule
        warmup_steps = max(1, int(self.hp.max_steps * self.hp.warmup_rate)) if self.hp.warmup else 0

        # Initialize the scheduler: Linear increase followed by Cosine decrease
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=self.hp.max_steps
        )

        # Prepare for distributed/accelerated training
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _apply_weight_decay(self, model, optimizer):
        """
        If activated do not apply weight decay to biases and norm layer.

        Args:
            model: Model whose biases and norm layer where weight decay will not be applied.
            hp: Model hyperparmeters defining the weight decay value.
        """
        no_decay = ["bias", "LayerNorm.weight"]

        if self.hp.weight_decay != 0.0:

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hp.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = optimizer(optimizer_grouped_parameters, lr=self.hp.learning_rate)
        return optimizer

    def _init_progress_bars(
        self, 
        training_dataloader: Optional[DataLoader]=None, 
        eval_dataloader: Optional[DataLoader]=None, 
        reset: bool = False, 
        mode: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> None:
        """
        Initializes or resets training/evaluation tqdm progress bars.

        This method ensures that full length training/evaluation bars appear at the very beginning of model training. It also ensures 
        that when entering a new epoch or a new evaluation step that the progress bar is cleared and reset.

        Args:
            training_dataloader (DataLoader, optional): Training set to determine progress bar length. Defaults to None.
            eval_dataloader (DataLoader, optional): Evaluation set to determine progress bar length. Defaults to None.
            reset (bool): Whether to refresh an existing progress bar's state. Defaults to False.
            mode (str, optional): The target progress bar to reset; must be "train", "eval" or "post_training_eval". Defaults to None.
            epoch (int, defaults to None): Current epoch index for labeling. Defaults to None.
        """
        
        # Updates existing progress bars without creating new objects
        if reset:
            if mode == "train" and training_dataloader is not None:
                self.training_bar.set_description(f"Epoch {epoch}")
                self.training_bar.reset(total=len(training_dataloader))
        
            elif mode == "eval" and eval_dataloader is not None:
                self.eval_bar.set_description(f"Eval @ Step {self.global_step}")
                self.eval_bar.reset(total=len(eval_dataloader))

            elif mode == "post_training_eval" and eval_dataloader is not None:
                return

        # Create the tqdm progress bars for the first time
        else:
            self.training_bar = tqdm(total=len(training_dataloader), position=0, desc="Training - Epoch 1", leave=True)
            self.eval_bar = tqdm(total=len(eval_dataloader), position=1, desc="Validation", leave=True)
 
    def _update_progress_bars(self, mode: str, loss_value: float) -> None:
        """
        Update the training or evaluation tqdm progress bar.

        Due to implementation restrictions it was necessary to manually update the training/evaluation progress bars. 
        The global step was also included to help the user understand how far along we are in the model training.

        Args:
            mode (str): The target bar to update; must be "train", "eval" or "post_training_eval".
            loss_value (float): The loss value to include in the bar update.
            
        """
        if mode == "train":
            self.training_bar.update(1)
            self.training_bar.set_postfix(loss=f"{loss_value:.4f}", step=self.global_step)
        
        elif mode == "eval":
            self.eval_bar.update(1)
            self.eval_bar.set_postfix(loss=f"{loss_value:.4f}", step=self.global_step)

        elif mode == "post_training_eval":
            return

    def _close_progress_bars(self) -> None:
        """
        Close the progress bars once we complete model training.
        """
        if self.training_bar: self.training_bar.close()
        if self.eval_bar: self.eval_bar.close()

    def _run_optimization_loop(self, training_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        """
        Main method which details all steps taken during model training. These include defining the warm-up scheduler, preparing 
        the training and evaluation datasets and subsequently iterating through them during model training.

        Args:
            training_dataloader (DataLoader): The pre-batched training dataset.
            eval_dataloader (DataLoader): The pre-batched evaluation dataset.
        """

        training_dataloader, eval_dataloader = self.accelerator.prepare(
            training_dataloader, eval_dataloader
        )
        self._setup_scheduler()
        self._init_progress_bars(training_dataloader=training_dataloader, eval_dataloader=eval_dataloader, reset=False)
        self._train_step_start_time = time.time()

        # For each step
        while self.global_step < self.hp.max_steps:
            
            # Prepare for model training
            self.epoch += 1
            self.model.train()
            self.optimizer.zero_grad()
            training_loss = LossMeter()
            self.metric_fn.reset()
            self._init_progress_bars(training_dataloader=training_dataloader, reset=True, mode="train", epoch=self.epoch)

            # For each batch (starting at 1)
            for i, batch in enumerate(training_dataloader, start=1):
                if self.global_step >= self.hp.max_steps:
                    break

                # Automatically perform gradient accumulation
                with self.accelerator.accumulate(self.model):
                    
                    # Forward and backward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    # Logging
                    self.metric_fn.update(logits=outputs.logits, labels=batch["labels"], mode=None)
                    training_loss.update(loss.item(), n=batch["input_ids"].size(0))
                    self._update_progress_bars(mode="train", loss_value=training_loss.avg)           

                    # Update the gradients
                    if self.accelerator.sync_gradients:
                        total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.hp.max_norm).item()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                    
                        # Update logs and metrics every N steps
                        if (self.global_step % self.hp.train_eval_steps == 0 and self.global_step > 0):
                            if self.accelerator.is_main_process:

                                # Compute and log the training metrics
                                training_metrics = self.metric_fn.compute()
                                
                                elapsed = time.time() - self._train_step_start_time
                                update_steps = self.global_step - self.last_logged_global_step
                                steps_per_sec = update_steps / elapsed
                                
                                samples_per_sec = (update_steps * self.hp.batch_size * self.hp.gradient_accumulation_steps) / elapsed
                                
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
                        print("self.global_step", self.global_step, self.hp.eval_steps)
                        if (self.global_step % self.hp.eval_steps == 0 and self.global_step > 0):
                            if self.accelerator.is_main_process:
                                
                                # Evaluate the current model on the evaluation dataset
                                eval_loss, _ = self.eval_model(eval_dataloader)

                                self.model.train()
                                self.metric_fn.reset()

                                # Save the best model
                                if eval_loss.avg < self.best_eval_loss:
                                    self.best_eval_loss = eval_loss.avg
                                    self.save_model()
                
        # Close off the wandb logging
        self._close_progress_bars()
        
        # Stop recording the models parameters
        wandb.unwatch()
        wandb.finish()
    
    def eval_model(self, eval_dataloader: DataLoader, training_eval: bool=True) -> Dict[str, float]:
        """
        Evaluate the model on some pre-batched dataset.

        Calculate the evaluation metrics and loss for the model, can be envoked either during or after model training.

        Args:
            eval_dataloader (DataLoader): The pre-batched evaluation dataset.
            training_eval (bool): True if evaluating during training false otherwise. Defaults to True.
        """
        # Prepare for model evaluation
        mode = "eval" if training_eval else "post_training_eval"
        self.model.eval()
        self.metric_fn.reset()
        eval_loss = LossMeter()
        
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        self._init_progress_bars(eval_dataloader=eval_dataloader, reset=True, mode=mode)
        self._eval_step_start_time = time.time()
        
        with torch.no_grad():
            for batch in eval_dataloader:

                # Forward pass
                outputs = self.model(**batch)

                # Calculate and update eval loss and metrics
                loss = outputs.loss.detach().cpu().item()
                eval_loss.update(loss, n=batch["input_ids"].size(0))
                self._update_progress_bars(mode=mode, loss_value=eval_loss.avg)
                self.metric_fn.update(logits=outputs.logits, labels=batch["labels"], mode=mode)
                eval_metrics = self.metric_fn.compute()
                
            elapsed = time.time() - self._eval_step_start_time
            steps_per_sec = len(eval_dataloader) / elapsed
            samples_per_sec = len(eval_dataloader.dataset) / elapsed

            # If evaluating post training run 
            if training_eval:
                self._log_metrics(
                    mode="val",
                    loss=eval_loss, 
                    metrics=eval_metrics,
                    steps_per_sec=steps_per_sec,
                    samples_per_sec=samples_per_sec,
                )
            self.metric_fn.reset()
        
        return eval_loss, eval_metrics

    def save_model(self) -> None:
        """
        Unwraps the model from the Accelerator environment and saves its state dict locally.
        """

        # Ensure the directory to store the model locally exist
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap the accelerator model
        unwrapped_model = self.accelerator.unwrap_model(self.model)._orig_mod

        # Save the unwrapped models state dict locally
        print("self.model_file_path", self.model_file_path)
        self.accelerator.save(unwrapped_model.state_dict(), self.model_file_path)

    @classmethod
    def load_model(
        cls, 
        model_class: type[torch.nn.Module], 
        model_config, 
        checkpoint_path: str | Path,
    ) -> torch.nn.Module:
        """
        Factory method to instantiate a BERT model architecture and load its saved weights.
        """
        model = model_class(model_config)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        return model


class ModelPreTraining(BaseModelTraining):
    """
    A class that contains methods to help with training a BERT model. Accelerator is used to abstract out some of the more complex model 
    training logic. This class also contains methods to define the warm-up scheduler, the logging of metrics locally and to WandB.

    Attributes:
        model (BertModelForMLM): TinyBERT model as defined by the TinyBERT configuration.
        optimizer (AdamW): Classic AdamW optimizer.
        metric_fn (MetricEvaluation): Class that handles the collection and calculation of both training and validation metrics.
        config (Config): General experiment configuration.
        hyperparameters (Hyperparameters): General experiment hyperparameters.
    """

    def __init__(
        self, 
        model: BertModelForMLM, 
        optimizer: AdamW, 
        metric_fn: MetricEvaluation, 
        config, 
        hyperparameters,
    ):
        super().__init__(model, optimizer, metric_fn, config, hyperparameters)
        wandb.init(
            project=self.cfg.experiment_project,
            group=self.cfg.experiment_name, 
            name=f"MLA_PT__kv_{self.hp.kv_compression_dim}_q_{self.hp.q_compression_dim}",
            job_type=self.cfg.job_type, 
            config={**vars(self.cfg), **vars(self.hp)},# , **vars(tiny_bert_config)}
            mode=self.wandb_mode,
        )
        wandb.watch(self.model, log="all", log_freq=self.hp.eval_steps)
        self.history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
        
    def _log_metrics(
        self, 
        mode: str, 
        loss: LossMeter, 
        metrics: dict, 
        steps_per_sec: float,
        samples_per_sec: float,
        total_norm: Optional[float]=None
    ) -> None:
        """
        Log the training or evaluation loss and metrics to wandb UI and to local dictionary.

        Args:
            mode (str): The target bar to update; must be "train" or "eval".
            loss (LossMeter): LossMeter object that contains training or evaluation loss totals and averages. 
            metrics (dict): Dictionary containing training or evaluation metrics.
            steps_per_sec (float, optional): Number of steps taken per second.
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
        
        elif mode == "val":
            log_dict["eval/loss"] = loss.avg
            log_dict["eval/accuracy"] = metrics["accuracy"]
            log_dict["eval/steps_per_sec"] = steps_per_sec
            log_dict["eval/samples_per_sec"] = samples_per_sec

        wandb.log(log_dict, step=self.global_step)

    def train_model(self, training_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, eval_dataloader)


class ModelFineTuning(BaseModelTraining):
    """
    A class that contains methods to help with fine-tuning a BERT model. Accelerator is used to abstract out some of the more complex 
    model fine-tuning logic. This class also contains methods to define the warm-up scheduler, the logging of metrics locally and to 
    WandB. 

    Attributes:
        model (BertModelForMLM): TinyBERT model as defined by the TinyBERT configuration.
        optimizer (AdamW): Classic AdamW optimizer.
        metric_fn (MetricEvaluation): Class that handles the collection and calculation of both training and validation metrics.
        config (Config): General experiment configuration.
        hyperparameters (Hyperparameters): General experiment hyperparameters.
    """

    def __init__(
        self, 
        model: BertModelForMLM, 
        optimizer: AdamW, 
        metric_fn: ClassificationMetricEvaluation, 
        config: FineTuneConfig, 
        hyperparameters: FineTuneHyperparameters,
    ):
        super().__init__(model, optimizer, metric_fn, config, hyperparameters)
        wandb.watch(self.model, log="all", log_freq=self.hp.eval_steps)
        self.history = {
            "train_loss": [], 
            "train_accuracy": [], 
            "train_recall": [], 
            "train_precision": [], 
            "train_f1": [], 
            "val_loss": [], 
            "val_accuracy": [],
            "val_recall": [], 
            "val_precision": [], 
            "val_f1": [], 
        }

    def _log_metrics(
        self, 
        mode: str, 
        loss: LossMeter, 
        metrics: dict, 
        steps_per_sec: float,
        samples_per_sec: float,
        total_norm: Optional[float]=None
    ) -> None:
        """
        Log the training or evaluation loss and metrics to wandb UI and to local dictionary.

        Args:
            mode (str): The target bar to update; must be "train" or "eval".
            loss (LossMeter): LossMeter object that contains training or evaluation loss totals and averages. 
            metrics (dict): Dictionary containing training or evaluation metrics.
            steps_per_sec (float, optional): Number of steps taken per second.
            total_norm (float, optional): The global norm of the gradients. Defaults to None.
        """  

        self.history[f"{mode}_loss"].append(loss.avg)
        self.history[f"{mode}_accuracy"].append(metrics["accuracy"])
        self.history[f"{mode}_recall"].append(metrics["recall"])
        self.history[f"{mode}_precision"].append(metrics["precision"])
        self.history[f"{mode}_f1"].append(metrics["f1"])

        log_dict = {}
        if mode == "train":
            log_dict["train/loss"] = loss.avg
            log_dict["train/accuracy"] = metrics["accuracy"]
            log_dict["train/precision"] = metrics["precision"]
            log_dict["train/recall"] = metrics["recall"]
            log_dict["train/f1"] = metrics["f1"]
            log_dict["train/learning_rate"] = self.scheduler.get_last_lr()[0]
            log_dict["train/steps_per_sec"] = steps_per_sec
            log_dict["train/samples_per_sec"] = samples_per_sec
            log_dict["train/grad_norm"] = total_norm
        
        elif mode == "val":
            log_dict["eval/loss"] = loss.avg
            log_dict["eval/accuracy"] = metrics["accuracy"]
            log_dict["eval/precision"] = metrics["precision"]
            log_dict["eval/recall"] = metrics["recall"]
            log_dict["eval/f1"] = metrics["f1"]
            log_dict["eval/steps_per_sec"] = steps_per_sec
            log_dict["eval/samples_per_sec"] = samples_per_sec

        wandb.log(log_dict, step=self.global_step)
        
    def fine_tune_model(self, training_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        self._run_optimization_loop(training_dataloader, eval_dataloader)