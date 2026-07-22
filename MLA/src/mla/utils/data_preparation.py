from collections.abc import Callable
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from mla.config.paths import Paths


def group_texts(examples: dict, max_seq_length: int) -> dict:
    """
    Groups and packs tokenized sequences into uniform chunks of a fixed maximum length.

    This function optimizes language modeling training efficiency by concatenating all
    token sequences (e.g., `input_ids`, `attention_mask`) into a single continuous flat list
    per feature key. It then splits this stream into uniform blocks of size `max_length`.
    Any trailing remainder that does not fit into a complete block is safely discarded.
    Finally, it duplicates the packed `input_ids` to serve as the ground-truth training `labels`.

    Args:
        examples (dict[str, list[list[Any]]]): A batch of tokenized data samples. Typical keys
            include `"input_ids"`, `"attention_mask"`, and `"token_type_ids"`, where each value
            is a list of token-integer arrays (one array per sequence).
        max_seq_length (int): The target block size or context window length for the model
            (e.g., `512` or `2048`).

    Returns:
        dict[str, List[List[Any]]]: A packed dictionary containing uniform, block-aligned sequences
            of exact length `max_seq_length`, along with an added `"labels"` key for training.
    """

    # Concatenate each list of items together (IDs, tokens)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Calculate the total length of tokens
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder at the very end
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length

    # Split by chunks of block_size
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }

    # For Language Modeling, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result


def get_processed_dataset(
    tokenized_dataset_path: str | Path,
    dataset_name: str,
    dataset_config_name: str | None,
    tokenizer: PreTrainedTokenizerBase,
    num_proc: int,
    sentence_keys: list[str],
    max_seq_length: int | None = 128,
    task_type: str = "pre_training",
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
        tokenized_dataset_path (Union[str, Path]): The local directory path where the processed
            arrow dataset should be saved to or loaded from.
        dataset_name (str): The name or identifier of the dataset on the Hugging Face Hub (e.g., `"wikitext"`).
        dataset_config_name (str, optional): The specific configuration or subset name of the dataset
            (e.g., `"wikitext-2-raw-v1"`). Pass `None` if the dataset does not use sub-configurations.
        tokenizer (PreTrainedTokenizer): The companion tokenizer instance used to encode the raw string data.
        num_proc (int): The number of CPU worker processes to spin up for parallel map execution.
        sentence_keys (list[str]): The text columns that need to be tokenized.
        max_seq_length (int, optional): The uniform target context window block length. Defaults to `128`.
        task_type (str): Either "training" or "fine-tuning".

    Returns:
        Union[DatasetDict, Dataset]: A processed Hugging Face dataset or split dictionary ready
            to be passed directly into a PyTorch DataLoader or Trainer.
    """

    # Check if the processed dataset already exists on disk
    if Path.exists(tokenized_dataset_path):
        print(f"Loading {dataset_name}/{dataset_config_name} dataset from: {tokenized_dataset_path}")
        return load_from_disk(tokenized_dataset_path)

    # Downloading and loading a dataset from the hub
    print(f"Processed dataset not found. Starting tokenization of: {dataset_name}/{dataset_config_name}")
    raw_datasets = load_dataset(dataset_name, dataset_config_name)

    # Remove any empty texts
    clean_datasets = raw_datasets.filter(lambda x: all(x[k].strip() != "" for k in sentence_keys))

    # Tokenize the dataset
    is_fine_tuning = task_type == "fine_tuning"
    dataset = clean_datasets.map(
        lambda x: tokenizer(
            *[x[k] for k in sentence_keys],
            truncation=is_fine_tuning,
            max_length=max_seq_length if is_fine_tuning else None,
        ),
        batched=True,
        remove_columns=sentence_keys,
    )

    if task_type == "pre_training":

        # Group the datasets
        dataset = dataset.map(
            lambda x: group_texts(x, max_seq_length=max_seq_length),
            batched=True,
            num_proc=num_proc
        )
    elif task_type == "fine_tuning":
        # Rename 'label' to 'labels' and remove idx
        dataset = dataset.rename_column("label", "labels").remove_columns(["idx"])

    # Save to disk for future sessions
    print(f"Saving processed dataset to: {tokenized_dataset_path}")
    dataset.save_to_disk(tokenized_dataset_path)

    return dataset


class DeterministicDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # Every time a batch is created, we reset the seed
        # Use a fixed seed or one based on the example IDs if available
        torch.manual_seed(42)
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


def import_and_prepare_data(tokenizer: PreTrainedTokenizerBase, config: DictConfig, paths: Paths) -> DatasetDict:
    """
    Import and tokenized dataset or load if already tokenized.
    """
    dataset = get_processed_dataset(
        paths.tokenized_data_path,
        config.dataset_name,
        config.dataset_config_name,
        tokenizer,
        num_proc=config.parallel_processes,
        sentence_keys=config.sentence_keys,
        max_seq_length=config.max_seq_length,
        task_type=config.task_type,
    )

    return dataset


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
        dataset["train"].select(range(100)),
        dataset[validation_split].select(range(100)),
        training_collator,
        validation_collator,
    )
    return training_loader, validation_loader
