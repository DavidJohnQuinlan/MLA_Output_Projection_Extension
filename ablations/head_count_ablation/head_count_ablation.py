"""
Head-count ablation study of W^O compression.

I wish to train a set of models, holding model size fixed and varying only
the number of attention heads, n_head. Therefore, head count is isolated 
from scale (width/depth/n_params):

    n_head in {4, 8, 16, 32}       (head_dim {192, 96, 48, 24} at d_model=768)
    per head count: 1 MHA baseline (dense W^O)
                  + 4 MHAE runs    (output_compression_dim in {384, 192, 96, 48})
                                  = d/2, d/4, d/8, d/16

Each of the 20 cells is launched as its own process via the existing Hydra 
pretraining entrypoint.

Usage:
To run all experiments, use the following command:
    uv run python ablations/head_count_ablation/head_count_ablation.py 
To collect all results:
    uv run python ablations/head_count_ablation/head_count_ablation.py --collect True
"""
import argparse, re, subprocess, sys
from pathlib import Path
import pandas as pd


EXPERIMENT_PROJECT = "GPT2_HeadAblation"
ROOT_PATH = Path(__file__).resolve().parents[2]
RESULTS_CSV_PATH = ROOT_PATH / "training_models" / EXPERIMENT_PROJECT / "pretraining" / "pretraining_results.csv"
TIDY_CSV_PATH = ROOT_PATH / "notebooks" / "data" / "ablation_results.csv"
_NAME_RE = re.compile(r"h(?P<d_model>\d+)_l(?P<n_layer>\d+)_a(?P<n_head>\d+)_(?P<attention_mechanism>[A-Z]+)")


config = {
    # Training dataset
    "dataset_name": "Skylion007/openwebtext",
    "dataset_config_name": "null",
    "max_load_pct": 5,
    "train_token_budget": 100_000_000,
    "val_token_budget": 1_000_000,
    "max_steps": 4000,
    "train_eval_steps": 200,
    "eval_steps": 500,
    "seed": 42,

    # Fixed parameters
    "hidden_size": 768,
    "intermediate_size": 3072,
    "n_layer": 6,
    "max_seq_length": 512,
    "max_position_embeddings": 512,

    # Variable parameters 
    "n_head": [4, 8, 16, 32],
    "output_compression_dim": [384, 192, 96, 48],
}


def build_experiment_parameters(config: dict):
    """
    Return the different attention mechanism, n_head and output_compression_dim variations for each 
    experiment.
    """
    for n_head in config["n_head"]:
        yield ("MHA", n_head, None)
        for output_compression_dim in config["output_compression_dim"]:
            yield ("MHAE", n_head, output_compression_dim)


def build_hydra_command(
        config: dict, 
        attention_mechanism: str, 
        n_head: int, 
        output_compression_dim: int
    ) -> str:
    """
    Build the full hydra command related to each ablation experiment.
    """
    cfg = f"GPT2/pretraining/tinygpt2_{attention_mechanism.lower()}"
    hydra_command = [
        "uv", "run", "pretraining", f"--config-name={cfg}",
        f"experiment_project={EXPERIMENT_PROJECT}",
        f"dataset_name={config['dataset_name']}",
        f"dataset_config_name={config['dataset_config_name']}",
        f"max_load_pct={config['max_load_pct']}",
        f"train_token_budget={config['train_token_budget']}",
        f"val_token_budget={config['val_token_budget']}",
        f"n_head={n_head}",
        f"hidden_size={config['hidden_size']}",
        f"intermediate_size={config['intermediate_size']}",
        f"n_layer={config['n_layer']}",
        f"max_seq_length={config['max_seq_length']}",
        f"max_position_embeddings={config['max_position_embeddings']}",
        f"max_steps={config['max_steps']}",
        f"train_eval_steps={config['train_eval_steps']}",
        f"eval_steps={config['eval_steps']}",
        f"pretraining_seeds=[{config['seed']}]",
    ]
    if output_compression_dim is not None:
        hydra_command.append(f"output_compression_dim={output_compression_dim}")
    return hydra_command


def run_experiments(config: dict) -> None:
    """
    Run all ablation experiments, for each pairing of n_head and output_compression_dim.
    """
    experiment_parameters = list(build_experiment_parameters(config))
    print(f"{len(experiment_parameters)} experiments to run\n")

    failures = []
    for i, (attention_mechanism, n_head, output_compression_dim) in enumerate(experiment_parameters, 1):
        hydra_command = build_hydra_command(config, attention_mechanism, n_head, output_compression_dim)
        print(f"[{i}/{len(experiment_parameters)}] {attention_mechanism} n_head={n_head} output_compression_dim={output_compression_dim}")
        print("    " + " ".join(hydra_command))
        
        if subprocess.run(hydra_command, cwd=ROOT_PATH).returncode != 0:
            print(f"    !! FAILED — continuing", file=sys.stderr)
            failures.append((attention_mechanism, n_head, output_compression_dim))

    if failures:
        print(f"\n{len(failures)} cells failed: {failures}", file=sys.stderr)

def collect_ablation_results(config: dict):
    """
    Collect the ablation results and prepare them for review.
    """
    df = pd.read_csv(RESULTS_CSV_PATH)

    # Parse out n_head, d_model, n_layer values out of the experiment name
    meta = df["model_name"].str.extract(_NAME_RE)
    df["d_model"] = pd.to_numeric(meta["d_model"], errors="coerce")
    df["n_head"] = pd.to_numeric(meta["n_head"], errors="coerce")
    df["n_layer"] = pd.to_numeric(meta["n_layer"], errors="coerce")
    df["o"] = pd.to_numeric(df["o"], errors="coerce")
    df["output_compression_dim"] = df["o"].where(df["attention_mechanism"] == "MHAE", df["d_model"])

    keep = ((df["d_model"] == config["hidden_size"]) &
            (df["n_layer"] == config["n_layer"]) &
            (df["n_head"].isin(config["n_head"])))
    df = df[keep].copy()

    df = df.sort_values("timestamp").drop_duplicates(
        subset=["n_head", "attention_mechanism", "output_compression_dim"], keep="last")
    df["r"] = df["output_compression_dim"] / df["d_model"]
    df["ppl"] = df["pretraining_validation_perplexity"]

    # Normalise each experiment to the MHA baseline for the same n_head
    base = df[df["attention_mechanism"] == "MHA"].set_index("n_head")["ppl"]
    df["ppl_norm"] = df.apply(lambda x: x["ppl"] / base.get(x["n_head"], float("nan")), axis=1)

    cols = ["n_head", "attention_mechanism", "d_model", "output_compression_dim", "r", "ppl", "ppl_norm", 
            "pretraining_cka", "pretraining_next_token_accuracy", "n_params", "GFLOPS", "max_steps"]
    output_df = (df[cols].sort_values(["n_head", "r"], ascending=[True, False]).reset_index(drop=True))
    TIDY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(TIDY_CSV_PATH, index=False)
    print(f"wrote {len(output_df)} rows -> {TIDY_CSV_PATH}")
    return output_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="skip training; just collect the results CSV")
    args = ap.parse_args()
    if args.collect:
        collect_ablation_results(config)
    else:
        run_experiments(config)
