import csv
import subprocess
from pathlib import Path

from tabulate import tabulate


def get_run_metadata(wandb_id: str | dict | None = None) -> dict:
    """Generate ID stamps per model training/finetuning. Specifically, we will use git state, seed, wandb id."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
        git = f"{sha[:10]}{'-dirty' if dirty else ''}"
    except Exception:
        git = "unknown"
    return {
        "git_sha": git,
        "wandb_id": wandb_id,
    }


def append_to_results_csv(results: dict, csv_path: Path) -> None:
    """
    Save the results to a central csv file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def print_output_table(title: str, results: dict) -> None:
    """
    Prepare and print the table to screen.
    """
    results = {k: "N/A" if v is None else v for k, v in results.items()}
    table = tabulate(results.items(), tablefmt="rounded_outline")
    width = len(table.splitlines()[0])
    print("\n", title.center(width))
    print(table)
