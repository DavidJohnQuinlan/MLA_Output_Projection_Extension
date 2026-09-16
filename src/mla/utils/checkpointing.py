import json
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import PretrainedConfig

from mla.utils.setup import get_device


def load_checkpoint(
    model_class: type[torch.nn.Module],
    checkpoint_path: Path,
    config_class: type[PretrainedConfig],
    config_overrides: dict | None = None,
) -> torch.nn.Module:
    """
    Instantiate a model from its checkpoint's own saved config, then load weights.
    """
    device = get_device()
    metadata = json.loads(checkpoint_path.with_suffix(".json").read_text())
    model_config = config_class.from_dict(metadata["config"])
    for k, v in (config_overrides or {}).items():
        setattr(model_config, k, v)

    model = model_class(model_config)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


class Checkpointer:
    def __init__(
        self,
        accelerator: Accelerator,
        best_checkpoint_file_path: Path,
        recent_checkpoint_prefix: Path,
        keep_last_n: int,
    ):
        self.accelerator = accelerator
        self.best_checkpoint_file_path = best_checkpoint_file_path
        self.recent_checkpoint_prefix = recent_checkpoint_prefix
        self.keep_last_n = keep_last_n

    def _prune(self) -> None:
        if self.keep_last_n < 1:
            return
        sorted_checkpoints = sorted(
            (p for p in self.recent_checkpoint_prefix.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.removeprefix("step_"))
        )
        for checkpoint in sorted_checkpoints[:-self.keep_last_n]:
            shutil.rmtree(checkpoint)

    def _find_latest(self) -> Path | None:
        checkpoints = [p for p in self.recent_checkpoint_prefix.glob("step_*")
                if p.is_dir() and (p / "extra_state.json").exists()]
        return max(checkpoints, key=lambda p: int(p.name.removeprefix("step_"))) if checkpoints else None

    def save_best(self, model, metadata: dict) -> None:
        """
        Save the best model and associated metadata as a json file.
        """
        self.best_checkpoint_file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_th = self.best_checkpoint_file_path.with_suffix(".th.tmp")
        tmp_json = self.best_checkpoint_file_path.with_suffix(".json.tmp")
        self.accelerator.save(model.state_dict(), tmp_th)
        tmp_json.write_text(json.dumps(metadata, indent=4, default=str))
        tmp_json.replace(self.best_checkpoint_file_path.with_suffix(".json"))
        tmp_th.replace(self.best_checkpoint_file_path)

    def save_recent(self, global_step: int, extra_state: dict) -> None:
        """
        Save the models full state to allow for easy resuming after a failure.
        """
        checkpoint_dir = self.recent_checkpoint_prefix / f"step_{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(checkpoint_dir), safe_serialization=False)

        if self.accelerator.is_main_process:
            checkpoint_dir_extra_state = checkpoint_dir / "extra_state.json"
            checkpoint_dir_extra_state.write_text(json.dumps(extra_state, indent=4))
            self._prune()

    def resume(self) -> dict | None:
        checkpoint = self._find_latest()
        if checkpoint is None:
            return None
        self.accelerator.load_state(str(checkpoint))
        return json.loads((checkpoint / "extra_state.json").read_text())

    def prior_wandb_id(self) -> str | None:
        checkpoint = self._find_latest()
        if checkpoint is None:
            return None
        return json.loads((checkpoint / "extra_state.json").read_text()).get("wandb_id")
