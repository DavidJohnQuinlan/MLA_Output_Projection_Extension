import argparse
import subprocess
import sys

GLUE_TASKS = ["sst2", "mrpc", "qqp", "rte", "mnli", "qnli", "cola"]


def get_config_name(architecture: str, task: str, kv: int, q: int, o: int) -> str:
    """
    Builds the Hydra config name for a given architecture, task, and compression dimensions.

    For MHA the compression arguments are unused. For MLA and MLAE the config name
    encodes the dimension values, so a matching YAML file must exist under the
    finetuning config directory.

    Args:
        architecture: One of "MHA", "MLA", or "MLAE".
        task: GLUE task name in any case (e.g. "sst2", "MRPC").
        kv: KV compression dimension. Ignored for MHA.
        q: Query compression dimension. Ignored for MHA.
        o: Output compression dimension. Used only for MLAE.

    Returns:
        Hydra config name string, e.g. "SST2/tinybert_mla_sst2_kv128_q128".
    """
    t = task.lower()
    T = task.upper()
    if architecture == "MHA":
        return f"{T}/tinybert_mha_{t}"
    elif architecture == "MLA":
        return f"{T}/tinybert_mla_{t}_kv{kv}_q{q}"
    elif architecture == "MLAE":
        return f"{T}/tinybert_mlae_{t}_kv{kv}_q{q}_o{o}"


def run_glue_benchmark(architecture: str, kv: int, q: int, o: int) -> None:
    """
    Runs fine-tuning across all GLUE tasks for a single architecture.

    Each task is launched as a separate subprocess so that Hydra creates its own
    timestamped output directory and log file per task. A task that exits with a
    non-zero return code is skipped and recorded; remaining tasks continue running.
    A summary of any failures is printed after all tasks complete.

    Args:
        architecture: Attention mechanism to benchmark — "MHA", "MLA", or "MLAE".
        kv: KV compression dimension used to resolve the config filename (MLA/MLAE only).
        q: Query compression dimension used to resolve the config filename (MLA/MLAE only).
        o: Output compression dimension used to resolve the config filename (MLAE only).
    """
    failed = []
    for task in GLUE_TASKS:
        config_name = get_config_name(architecture, task, kv, q, o)
        print(f"\n--- {task.upper()} ({architecture}) ---")
        try:
            subprocess.run(
                [sys.executable, "-m", "mla.model_training.model_finetuning",
                 f"--config-name={config_name}"],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"  FAILED — skipping {task.upper()}")
            failed.append(task)

    if failed:
        print(f"\nBenchmark complete. Failed tasks: {', '.join(t.upper() for t in failed)}")
    else:
        print("\nBenchmark complete. All tasks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["MHA", "MLA", "MLAE"], required=True)
    parser.add_argument("--kv", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--o", type=int, default=None)
    args = parser.parse_args()
    run_glue_benchmark(args.architecture, args.kv, args.q, args.o)