import argparse
import subprocess
import sys

GLUE_TASKS = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]


def get_config_name(base_model: str, architecture: str, task: str, kv: int, q: int, o: int) -> str:
    """
    Builds the Hydra config name for a given architecture, task, and compression dimensions.

    For MHA the compression arguments are unused. For MLA and MLAE the config name
    encodes the dimension values, so a matching YAML file must exist under the
    finetuning config directory.

    Args:
        base_model: One of "BERT" or "GPT2".
        architecture: One of "MHA", "MHAE", "MLA", or "MLAE".
        task: GLUE task name in any case (e.g. "sst2", "MRPC").
        kv: KV compression dimension. Ignored for MHA.
        q: Query compression dimension. Ignored for MHA.
        o: Output compression dimension. Used only for MLAE or MHAE.

    Returns:
        Hydra config name string, e.g. "SST2/tinybert_mla_sst2_kv128_q128".
    """
    task = task.lower()
    TASK = task.upper()
    base_model = base_model.lower()
    BASE_MODEL = base_model.upper()
    if architecture == "MHA":
        return f"{BASE_MODEL}/finetuning/{TASK}/tiny{base_model}_mha_{task}"
    elif architecture == "MHAE":
        return f"{BASE_MODEL}/finetuning/{TASK}/tiny{base_model}_mhae_{task}_o{o}"
    elif architecture == "MLA":
        return f"{BASE_MODEL}/finetuning/{TASK}/tiny{base_model}_mla_{task}_kv{kv}_q{q}"
    elif architecture == "MLAE":
        return f"{BASE_MODEL}/finetuning/{TASK}/tiny{base_model}_mlae_{task}_kv{kv}_q{q}_o{o}"


def run_glue_benchmark(base_model: str, architecture: str, pre_training_seed: int, kv: int, q: int, o: int) -> None:
    """
    Runs fine-tuning across all GLUE tasks for a single architecture.

    Each task is launched as a separate subprocess so that Hydra creates its own
    timestamped output directory and log file per task. A task that exits with a
    non-zero return code is skipped and recorded; remaining tasks continue running.
    A summary of any failures is printed after all tasks complete.

    Args:
        base_model: Base model - "BERT" or "GPT2".
        architecture: Attention mechanism to benchmark — "MHA", "MHAE", "MLA", or "MLAE".
        kv: KV compression dimension used to resolve the config filename (MLA/MLAE only).
        q: Query compression dimension used to resolve the config filename (MLA/MLAE only).
        o: Output compression dimension used to resolve the config filename (MLAE/MHAE only).
    """
    failed = []
    for task in GLUE_TASKS:
        config_name = get_config_name(base_model, architecture, task, kv, q, o)
        print(f"\n--- {base_model} - {architecture} - {task.upper()} - {pre_training_seed} ---")
        try:
            subprocess.run(
                [sys.executable, "-m", "mla.model_training.model_finetuning",
                 f"--config-name={config_name}",
                 f"+pre_training_seed={pre_training_seed}",
                 ],
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
    parser.add_argument("--base_model", choices=["BERT", "GPT2"], required=True)
    parser.add_argument("--architecture", choices=["MHA", "MHAE", "MLA", "MLAE"], required=True)
    parser.add_argument("--pre_training_seed", required=True)
    parser.add_argument("--kv", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--o", type=int, default=None)
    args = parser.parse_args()
    run_glue_benchmark(args.base_model, args.architecture, args.pre_training_seed, args.kv, args.q, args.o)
