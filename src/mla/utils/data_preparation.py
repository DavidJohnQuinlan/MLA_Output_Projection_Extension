from collections.abc import Callable
from itertools import chain
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

DATA_CREATION_SEED = 42


def load_raw_datasets(config: DictConfig):
    """
    Load the raw dataset from the Hub and normalize to a DatasetDict.

    A sliced load (`train[:N%]`) returns a bare Dataset, so we wrap it so every
    downstream step can assume a DatasetDict with at least a `train` split.
    """
    if config.max_load_pct is not None:
        raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name, split=f"train[:{config.max_load_pct}%]")
    else:
        raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)

    if isinstance(raw_datasets, Dataset):
        raw_datasets = DatasetDict({"train": raw_datasets})
    return raw_datasets


def clean_raw_datasets(raw_datasets, config: DictConfig):
    """Drop rows whose text field(s) are empty or whitespace-only."""
    return raw_datasets.filter(lambda x: all(x[k].strip() != "" for k in config.sentence_keys))


def tokenize_raw_datasets(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, config: DictConfig) -> DatasetDict:
    """Tokenize the text fields.

    Pre-training drops all original columns so only token fields reach `pack`
    (group_texts flattens every key, so stray columns would corrupt it).
    Fine-tuning keeps the label column for the classifier and drops only the text.
    """
    is_fine_tuning = config.task_type == "fine_tuning"
    a_split = next(iter(dataset))
    remove_cols = config.sentence_keys if is_fine_tuning else dataset[a_split].column_names

    return dataset.map(
        lambda x: tokenizer(
            *[x[k] for k in config.sentence_keys],
            truncation=is_fine_tuning,
            max_length=config.max_seq_length if is_fine_tuning else None,
        ),
        batched=True,
        remove_columns=remove_cols,
    )


def group_texts(examples: dict, max_seq_length: int) -> dict:
    """
    Groups and packs tokenized sequences into uniform chunks of a fixed maximum length.

    This function optimizes language modeling training efficiency by concatenating all
    token sequences (e.g., `input_ids`, `attention_mask`) into a single continuous flat list
    per feature key. It then splits this stream into uniform blocks of size `max_length`.
    Any trailing remainder that does not fit into a complete block is safely discarded.

    Args:
        examples (dict[str, list[list[Any]]]): A batch of tokenized data samples. Typical keys
            include `"input_ids"`, `"attention_mask"`, and `"token_type_ids"`, where each value
            is a list of token-integer arrays (one array per sequence).
        max_seq_length (int): The target block size or context window length for the model
            (e.g., `512` or `2048`).

    Returns:
        dict[str, List[List[Any]]]: A packed dictionary containing uniform, block-aligned sequences
            of exact length `max_seq_length`.
    """

    # Concatenate each list of items together (IDs, tokens)
    concatenated_examples = {k: list(chain.from_iterable(examples[k])) for k in examples}

    # Calculate the total length of tokens
    total_length = (len(concatenated_examples["input_ids"]) // max_seq_length) * max_seq_length

    # Split by chunks of block_size
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def pack(dataset: DatasetDict, config: DictConfig) -> DatasetDict:
    """Concatenate and split tokens into fixed `max_seq_length` blocks (LM packing)."""
    return dataset.map(
        lambda x: group_texts(x, max_seq_length=config.max_seq_length),
        batched=True,
        batch_size=10_000,
        num_proc=config.parallel_processes,
    )


def _check_budget_met(name: str, requested_blocks: int | None, available_blocks: int, max_seq_length: int) -> None:
    """Warn if fewer packed blocks are available than the token budget asked for."""
    if requested_blocks is not None and available_blocks < requested_blocks:
        got  = available_blocks * max_seq_length
        want = requested_blocks * max_seq_length
        print(
            f"WARNING: {name} token budget not met — requested {want:,} tokens "
            f"but only {got:,} available ({got / want:.1%} of budget). "
            f"Increase `max_load_pct` (or the source slice) to cover it."
        )


def split_by_token_budget(dataset: DatasetDict, config: DictConfig) -> DatasetDict:
    """
    Create train/validation datasets to exact token budgets via packed-block counts.

    Each packed block is `max_seq_length` tokens, so a token budget maps to
    `budget // max_seq_length` blocks. Datasets with a native validation split
    (e.g. WikiText) keep it; those without (e.g. OpenWebText) get one carved from
    a shuffled copy of the packed train blocks using a fixed seed, so the held-out
    set is identical across all runs.
    """
    n_train = config.train_token_budget // config.max_seq_length if config.train_token_budget is not None else None
    n_val = config.val_token_budget // config.max_seq_length if config.val_token_budget is not None else None

    if "validation" in dataset:
        train_blocks, val_blocks = dataset["train"], dataset["validation"]
    else:
        if n_val is None:
            raise ValueError(
                f"{config.dataset_name} has no validation split; set `val_token_budget`."
            )
        shuffled = dataset["train"].shuffle(seed=DATA_CREATION_SEED)
        n_val_avail = min(n_val, len(shuffled))
        val_blocks = shuffled.select(range(n_val_avail))
        train_blocks = shuffled.select(range(n_val_avail, len(shuffled)))

    _check_budget_met("train", n_train, len(train_blocks), config.max_seq_length)
    _check_budget_met("validation", n_val, len(val_blocks), config.max_seq_length)

    if n_train is not None:
        train_blocks = train_blocks.select(range(min(n_train, len(train_blocks))))
    if n_val is not None:
        val_blocks = val_blocks.select(range(min(n_val, len(val_blocks))))

    return DatasetDict({"train": train_blocks, "validation": val_blocks})


def prepare_finetuning_datasets(dataset: DatasetDict) -> DatasetDict:
    """Rename the classification `label` column to `labels` and drop `idx`."""
    return dataset.rename_column("label", "labels").remove_columns(["idx"])


def import_and_prepare_data(
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    tokenized_dataset_path: str | Path,
) -> DatasetDict | Dataset:
    """
    Loads, tokenizes, packs, and caches a text dataset using a local disk fallback.

    This utility handles the entire end-to-end data preprocessing pipeline for language
    modeling. It first checks if a pre-processed version of the dataset already exists
    at the specified disk path. If found, it skips all processing steps and loads it instantly.
    Otherwise, it downloads the raw dataset from the Hugging Face Hub, tokenizes the text,
    packs the tokens into constant-length blocks using `group_texts`, caches the result to disk
    for future sessions, and returns the clean dataset.

    Args:
        tokenizer (PreTrainedTokenizer): The companion tokenizer instance used to encode the raw string data.
        config (DictConfig): A configuration dataclass instance containing core training settings.
        tokenized_dataset_path (Union[str, Path]): The local directory path where the processed arrow dataset
            should be saved to or loaded from.

    Returns:
        Union[DatasetDict, Dataset]: A processed Hugging Face dataset or split dictionary ready
            to be passed directly into a PyTorch DataLoader or Trainer.
    """
    config_label = config.dataset_config_name or "default"
    if Path.exists(tokenized_dataset_path):
        print(f"Loading {config.dataset_name}/{config_label} dataset from: {tokenized_dataset_path}")
        return load_from_disk(tokenized_dataset_path)

    print(f"Processed dataset not found. Building: {config.dataset_name}/{config_label}")
    raw_datasets = load_raw_datasets(config)
    clean_datasets = clean_raw_datasets(raw_datasets, config)
    tokenize_datasets = tokenize_raw_datasets(clean_datasets, tokenizer, config)

    if config.task_type == "pre_training":
        datasets = split_by_token_budget(pack(tokenize_datasets, config), config)
    else:
        datasets = prepare_finetuning_datasets(tokenize_datasets)

    # Save to disk for future sessions
    print(f"Saving processed dataset to: {tokenized_dataset_path}")
    datasets.save_to_disk(tokenized_dataset_path)

    return datasets


class DeterministicDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # Every time a batch is created, we reset the seed
        # Use a fixed seed or one based on the example IDs if available
        torch.manual_seed(DATA_CREATION_SEED)
        return super().__call__(examples)


class CreateDataloaders:
    """
    Constructs high-performance PyTorch DataLoaders for training and validation.

    This factory class isolates the configuration and preparation of PyTorch
    DataLoader streams, ensuring consistent worker, memory pinning, and batching
    strategies across both execution phases.
    """
    def __init__(self, config: DictConfig):
        """
        Initializes the factory with shared runtime execution properties.

        Args:
            config (DictConfig): A configuration dataclass instance
                containing core training settings (`batch_size`, `pin_memory`,
                `num_workers`, `drop_last`).
        """
        self.config = config

    def create_dataloaders(
        self,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        training_collator: Callable,
        validation_collator: Callable
    ) -> tuple[DataLoader, DataLoader]:
        """
        Generates paired training and validation DataLoader instances.

        Configures training data to shuffle and automatically drop incomplete trailing
        batches if specified. Validation data streaming is optimized for throughput by
        scaling the target batch dimension by 4x to leverage the reduced VRAM
        footprint of inference passes.

        Args:
            training_dataset (Dataset): The tokenized PyTorch or Hugging Face Dataset
                allocated for the training loop.
            validation_dataset (Dataset): The tokenized PyTorch or Hugging Face Dataset
                allocated for the validation/evaluation loop.
            training_collator (Callable): The batch aggregation or dynamic padding
                collation function for training samples.
            validation_collator (Callable): The batch aggregation or dynamic padding
                collation function for validation samples.

        Returns:
            tuple[DataLoader, DataLoader]: A two-element tuple containing:
                - training_loader (DataLoader): The active, shuffled training data stream.
                - validation_loader (DataLoader): The static, sequential validation data stream
                  scaled at 4x training batch capacity.
        """
        training_loader = DataLoader(
            training_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=training_collator,
            drop_last=self.config.drop_last,
        )

        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            collate_fn=validation_collator
        )

        return training_loader, validation_loader


def prepare_dataloaders(dataset: DatasetDict, tokenizer, config: DictConfig, collator_fn=None) -> tuple[DataLoader, DataLoader]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.task_type == "pre_training":

        # Define a seperate training and validation data collator
        training_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=config.mlm,
            mlm_probability=config.mlm_probability
        )

        # This guarantees that the validation metric remains perfectly stable and comparable
        validation_collator = DeterministicDataCollator(
            tokenizer=tokenizer,
            mlm=config.mlm,
            mlm_probability=config.mlm_probability
        )

    elif config.task_type == "fine_tuning":
        training_collator = validation_collator = collator_fn(tokenizer=tokenizer)

    # Prepare the dataloaders
    validation_split = "validation" if "validation" in dataset else "validation_matched"
    training_loader, validation_loader = CreateDataloaders(config).create_dataloaders(
        dataset["train"],
        dataset[validation_split],
        training_collator,
        validation_collator,
    )
    return training_loader, validation_loader
