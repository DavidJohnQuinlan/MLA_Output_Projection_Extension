from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class LocalPaths:
    data_path: Path
    local_training_data_path: Path
    model_file_path: Path


@dataclass
class Paths:
    local: LocalPaths


@dataclass
class BaseExperimentConfig:
    """
    Shared core parameters common to all TinyBERT execution phases.

    Args:
        experiment_name (str): Core name of the experiment used for tracking and filtering.
        tiny_bert_model_name (str): Hugging Face Hub repository string for the base model.
        root_dir (Path): The absolute or relative project root directory.
    """
    kv_compression_dim: int
    q_compression_dim: int
    output_compression_dim: int
    experiment_name: str = "TinyBERT_Baseline_MHA"
    tiny_bert_model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
    root_dir: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class PreTrainConfig(BaseExperimentConfig):
    """
    Configuration strictly for TinyBERT Masked Language Modeling training.

    Args:
        attention_mechanism (str): The type of attention experiment being conducted.
        job_type (str): Operational tag separating training execution from downstream tasks.
        dataset_name (str): Source dataset identifier path on the Hugging Face Hub.
        dataset_config_name (str): Specific version or subset generation profile of the dataset.
        experiment_project (str): The W&B project name.
        model_file_name (str): Local filename assigned to the saved checkpoint binary.
    """
    attention_mechanism: str = "MHA"
    job_type: str = "pre_training"
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-103-raw-v1"
    experiment_project: str = "MLA_TinyBERT_128"

    def __post_init__(self):
        """
        Compiles base strings into structured local workspace file paths.
        """

        if self.kv_compression_dim is None and self.q_compression_dim is None and self.output_compression_dim is None:
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}.th"
        elif self.output_compression_dim is None:
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}__kv{self.kv_compression_dim}_q{self.q_compression_dim}.th"
        else: 
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}__kv{self.kv_compression_dim}_q{self.q_compression_dim}_o{self.output_compression_dim}.th"
        
        self.paths = Paths(
            local=LocalPaths(
                data_path = self.root_dir / "training_data",
                local_training_data_path = self.root_dir / "training_data/pretraining/tokenizied_wikitext-103-raw-v1_128",
                model_file_path = self.root_dir / "training_models/pretraining/" / self.model_file_name,
            )
        )


@dataclass
class FineTuneConfig(BaseExperimentConfig):
    """
    Configuration for fine-tuning TinyBERT on downstream evaluation benchmarks.

    Args:
        job_type (str): Operational tag defining the specific evaluation task environment.
        model_file_name (str): Target filename assigned to the optimized fine-tuned weights.
        eval_benchmark_name (str): Core benchmark umbrella designation (e.g., "glue").
        eval_dataset_name (str): Specific sub-task evaluation slice target (e.g., "sst2").
    """

    attention_mechanism: str = "MHA"
    job_type: str = "fine_tuning"
    dataset_name: str = "glue"
    dataset_config_name: str = "sst2"
    experiment_project: str = "MLA_TinyBERT_128"

    def __post_init__(self):
        """
        Compiles base strings into downstream task validation path locations.
        """
        
        if self.kv_compression_dim is None and self.q_compression_dim is None and self.output_compression_dim is None:
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}.th"
        elif self.output_compression_dim is None:
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}__kv{self.kv_compression_dim}_q{self.q_compression_dim}.th"
        else: 
            self.model_file_name = f"{self.job_type}_{self.dataset_config_name}__TinyBERT_Baseline_{self.attention_mechanism}__kv{self.kv_compression_dim}_q{self.q_compression_dim}_o{self.output_compression_dim}.th"

        self.paths = Paths(
            local=LocalPaths(
                data_path = self.root_dir / "training_data",
                local_training_data_path = self.root_dir / "training_data/fine_tuning/sst2",
                model_file_path = self.root_dir / "training_models/fine_tuning" / self.model_file_name,
            )
        )