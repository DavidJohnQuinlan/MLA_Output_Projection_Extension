import argparse
import subprocess
import sys

GLUE_TASKS = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]


def get_config_name(base_model: str, architecture: str, task: str) -> str:
    """
    Builds the Hydra config name for a given architecture, task, and compression dimensions.

    For MHA the compression arguments are unused. For MLA and MLAE the config name
    encodes the dimension values, so a matching YAML file must exist under the
    finetuning config directory.

    Args:
        base_model: One of "BERT" or "GPT2".
        architecture: One of "MHA", "MHAE", "MLA", or "MLAE".
        task: GLUE task name in any case (e.g. "sst2", "mrpc").

    Returns:
        Hydra config name string, e.g. "SST2/tinybert_mla_sst2".
    """
    task = task.lower()
    TASK = task.upper()
    base_model = base_model.lower()
    BASE_MODEL = base_model.upper()
    architecture = architecture.lower()
    return f"{BASE_MODEL}/finetuning/{TASK}/tiny{base_model}_{architecture}_{task}"


def run_glue_benchmark(
        base_model: str,
        architecture: str,
        pretraining_seed: int | None = None,
        kv: int | None = None,
        q: int | None = None,
        o: int | None = None,
    ) -> None:
    """
    Runs finetuning across all GLUE tasks for a single architecture.

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
        config_name = get_config_name(base_model, architecture, task)
        seed_label = str(pretraining_seed) if pretraining_seed is not None else "all"
        print(f"\n--- {base_model} - {architecture} - {task.upper()} - {seed_label} ---")
        cmd = [sys.executable, "-m", "mla.model_training.model_finetuning", f"--config-name={config_name}"]

        if architecture in ["MLA", "MLAE"]:
            cmd += [f"kv_compression_dim={kv}", f"q_compression_dim={q}"]
        if architecture in ["MHAE", "MLAE"]:
            cmd += [f"output_compression_dim={o}"]

        if pretraining_seed is not None:
            cmd.append(f"+pretraining_seed={pretraining_seed}")
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
    parser.add_argument("--base_model", choices=["BERT", "GPT2"], required=True)
    parser.add_argument("--architecture", choices=["MHA", "MHAE", "MLA", "MLAE"], required=True)
    parser.add_argument("--pretraining_seed", required=False)
    parser.add_argument("--kv", type=int, default=None, required=False)
    parser.add_argument("--q", type=int, default=None, required=False)
    parser.add_argument("--o", type=int, default=None, required=False)
    args = parser.parse_args()
    run_glue_benchmark(args.base_model, args.architecture, args.pretraining_seed, args.kv, args.q, args.o)

if __name__ == "__main__":
    main()
