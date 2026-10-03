import argparse
import subprocess
import sys

GLUE_TASKS = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]


def get_config_name(config_dir: str, config_root_name: str, architecture: str, task: str) -> str:
    """
    Builds the Hydra config name for a given architecture, task, and compression dimensions.

    For MHA the compression arguments are unused. For MLA and MLAE the config name
    encodes the dimension values, so a matching YAML file must exist under the
    finetuning config directory.

    Args:
        config_dir: GPT2/TinyGPT2, GPT2/GPT2Small, BERT/TinyBERT, or similar.
        config_root_name: Root config file name, tinygpt2, gpt2small, tinybert, or similar.
        architecture: One of "MHA", "MHAE", "MLA", or "MLAE".
        task: GLUE task name in any case (e.g. "sst2", "mrpc").

    Returns:
        Hydra config name string, e.g. "SST2/tinybert_mla_sst2".
    """
    task = task.lower()
    TASK = task.upper()
    architecture = architecture.lower()
    return f"{config_dir}/finetuning/{TASK}/{config_root_name}_{architecture}_{task}"


def run_glue_benchmark(
        config_dir: str,
        config_root_name: str,
        architecture: str,
        pretraining_seed: int | None = None,
        kv: int | None = None,
        q: int | None = None,
        o: int | None = None,
        extra: list[str] | None = None
    ) -> None:
    """
    Runs finetuning across all GLUE tasks for a single architecture.

    Each task is launched as a separate subprocess so that Hydra creates its own
    timestamped output directory and log file per task. A task that exits with a
    non-zero return code is skipped and recorded; remaining tasks continue running.
    A summary of any failures is printed after all tasks complete.

    Args:
        config_dir: Base model and model type path - GPT2/TinyGPT2, GPT2/GPT2Small, BERT/TinyBERT, or similar.
        config_root_name: Root config file name, tinygpt2, gpt2small, tinybert, or similar.
        architecture: Attention mechanism to benchmark — "MHA", "MHAE", "MLA", or "MLAE".
        kv: KV compression dimension used to resolve the config filename (MLA/MLAE only).
        q: Query compression dimension used to resolve the config filename (MLA/MLAE only).
        o: Output compression dimension used to resolve the config filename (MLAE/MHAE only).
    """
    failed = []
    for task in GLUE_TASKS:
        config_name = get_config_name(config_dir, config_root_name, architecture, task)
        seed_label = str(pretraining_seed) if pretraining_seed is not None else "all"
        print(f"\n--- {config_dir} - {architecture} - {task.upper()} - {seed_label} ---")
        cmd = [sys.executable, "-m", "mla.model_training.model_finetuning", f"--config-name={config_name}"]

        if architecture in ["MLA", "MLAE"]:
            cmd += [f"kv_compression_dim={kv}", f"q_compression_dim={q}"]
        if architecture in ["MHAE", "MLAE"]:
            cmd += [f"output_compression_dim={o}"]
        if pretraining_seed is not None:
            cmd.append(f"+pretraining_seed={pretraining_seed}")
        if extra is not None:
            cmd += extra

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"  FAILED — skipping {task.upper()}")
            failed.append(task)

    if failed:
        print(f"\nBenchmark complete. Failed tasks: {', '.join(t.upper() for t in failed)}")
    else:
        print("\nBenchmark complete. All tasks passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", choices=["GPT2/TinyGPT2", "GPT2/GPT2Small", "BERT/TinyBERT"], required=True)
    parser.add_argument("--root_config_name", choices=["tinygpt2", "gpt2small", "tinybert"], required=True)
    parser.add_argument("--architecture", choices=["MHA", "MHAE", "MLA", "MLAE"], required=True)
    parser.add_argument("--pretraining_seed", required=False)
    parser.add_argument("--kv", type=int, default=None, required=False)
    parser.add_argument("--q", type=int, default=None, required=False)
    parser.add_argument("--o", type=int, default=None, required=False)
    args, extra = parser.parse_known_args()
    run_glue_benchmark(args.config_dir, args.root_config_name, args.architecture, args.pretraining_seed, args.kv, args.q, args.o, extra)

if __name__ == "__main__":
    main()
