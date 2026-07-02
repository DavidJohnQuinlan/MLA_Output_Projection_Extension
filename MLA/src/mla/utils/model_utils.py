import logging
from typing import Protocol

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class LossMeter:
    """
    Computes and stores the current value, running sum, evaluation count,
    and moving mathematical average of metrics or loss values.

    Attributes:
        val (float): The single numeric value registered during the most recent update step.
        avg (float): The true cumulative weighted average across all registered evaluations.
        sum (float): The absolute accumulated metric weight total.
        count (int): The total number of items or steps accounted for.
    """

    def __init__(self):
        """
        Initializes the meter structure and zeroes out all historical tracking states.
        """
        self.reset()

    def reset(self):
        """
        Resets all historical metric tracking metrics back to an initial empty state.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int=1):
        """
        Registers a new metric evaluation and updates cumulative tracking metrics.

        Args:
            val (float): The new scalar value to register (e.g., the step loss float).
            n (int, optional): The structural multiplier or batch size weighting associated
                with the value. Defaults to `1`.
        """
        if not np.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            logger.warning("NaN detected in LossMeter, skipping update.")

    def __str__(self) -> str:
        """
        Generates a scannable string profile displaying current and average progress.

        Returns:
            str: A comma-separated description tracking the current step value and
                overall average, formatted to 4 decimal places. Returns a raw string
                of `val` if no steps have been recorded yet.
        """
        if self.count == 0:
            return str(self.val)
        return f"{self.val:.4f}, {self.avg:.4f}"


class MetricEvaluationProtocol(Protocol):
    def reset(self) -> None: ...
    def update(self, logits, labels, mode) -> None: ...
    def compute(self) -> dict: ...


class MetricEvaluation:
    """
    Computes and aggregates accuracy metrics (Top-1 and Top-5) for Masked Language Modeling.

    This tracker extracts evaluation metrics strictly over active prediction targets,
    safely ignoring unmasked tokens flagged with the standard `-100` cross-entropy index.
    It maintains cumulative counts across multiple evaluation steps to return a stable,
    dataset-wide accuracy profile upon request.

    Attributes:
        total_masked (int): Total number of valid masked tokens evaluated.
        correct_t1 (int): Cumulative count of predictions where the highest-scoring model logit
            matched the ground-truth label.
        correct_t5 (int): Cumulative count of predictions where the ground-truth label appeared
            within the top 5 highest-scoring model logits.
    """
    def __init__(self):
        """Initializes the metric tracker and resets counts to zero."""
        self.reset()

    def reset(self):
        """
        Resizes and zeroes out all historical verification accumulators back to an initial state.
        """
        self.total_masked = 0
        self.correct_t1 = 0
        self.correct_t5 = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, mode: str):
        """
        Processes a raw model output batch, filtering elements and updating tracking totals.

        Args:
            logits (torch.Tensor): Unnormalized raw predictions from the language modeling head.
                Shape can be 3D `(batch_size, seq_len, vocab_size)` or pre-flattened 2D `(total_tokens, vocab_size)`.
            labels (torch.Tensor): Ground-truth target token matrix matching the spatial structure
                of logits. Elements to bypass must be set to `-100`. Shape: `(batch_size, seq_len)` or `(total_tokens,)`.
            mode (str, optional): Determines whether to calculate and store multi-rank
                top-5 matching indexes alongside basic top-1 accuracy.
        """
        # Filter out the -100 ignore indices
        mask = labels != -100
        target = labels[mask]
        preds = logits[mask]

        if target.numel() == 0:
            return

        self.total_masked += target.numel()

        # Top-1 Calculation (argmax is fastest)
        t1_preds = torch.argmax(preds, dim=-1)
        self.correct_t1 += (t1_preds == target).sum().item()

        # Top-5 Calculation
        if mode == "post_training_eval":
            _, t5_indices = preds.topk(5, dim=1)
            # Check if target is in any of the 5 columns by expanding the targets array dimensions
            is_correct_t5 = t5_indices.eq(target.unsqueeze(1)).any(dim=1)
            self.correct_t5 += is_correct_t5.sum().item()

    def compute(self) -> dict[str, float]:
        """
        Calculates final aggregated dataset accuracies based on collected counts.

        Returns:
            dict[str, float]: A dictionary summarizing active scores. Possible keys include:
                - `"accuracy"`: The global percentage score for Top-1 matching.
                - `"top5_accuracy"`: The global percentage score for Top-5 matching (omitted if
                  `include_top5` was never enabled).
                Returns an empty dictionary `{}` if no valid evaluations were registered.
        """
        if self.total_masked == 0:
            return {}

        results = {"accuracy": self.correct_t1 / self.total_masked}

        if self.correct_t5 > 0:
            results["top5_accuracy"] = self.correct_t5 / self.total_masked

        return results


class ClassificationMetricEvaluation:
    """
    Computes and aggregates classification metrics for sequence classification tasks.

    Tracks predictions and ground-truth labels across batches, computing accuracy,
    precision, recall, and F1 score upon request.

    Attributes:
        correct (int): Cumulative count of correctly classified samples.
        total (int): Total number of samples evaluated.
        all_preds (list): Accumulated predicted class indices across all batches.
        all_labels (list): Accumulated ground-truth labels across all batches.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Clears all accumulated results."""
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, logits, labels, mode: str=None):
        """
        Accumulates predictions and labels.
        Args:
            logits: [batch_size, 2] tensor from the model
            labels: [batch_size] tensor of ground truth (0 or 1)
        """
        # Get the predicted class (the index of the highest logit)
        preds = torch.argmax(logits, dim=-1)

        # Accumulate for simple accuracy
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)

        # Store for more complex metrics (F1, etc.)
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())

    def compute(self):
        """Calculates and returns the final metrics."""
        if self.total == 0:
            return {"count": 0, "accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}

        accuracy = self.correct / self.total
        recall = recall_score(self.all_labels, self.all_preds, zero_division=0)
        precision = precision_score(self.all_labels, self.all_preds, zero_division=0)
        f1 = f1_score(self.all_labels, self.all_preds, zero_division=0)

        return {
            "count": self.total,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
