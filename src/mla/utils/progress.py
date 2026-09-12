from tqdm import tqdm


class ProgressBars:
    def __init__(self, disable: bool = False):
        self.disable = disable
        self.train = self.val = None

    def create(self, n_train: int, n_val: int):
        """
        Create the initial tqdm progress bars.
        """
        self.training_bar = tqdm(total=n_train, position=0, desc="Training - Epoch 1", leave=True, disable=self.disable)
        self.validation_bar = tqdm(total=n_val, position=1, desc="Validation", leave=True, disable=self.disable)

    def start_epoch(self, n_train: int, epoch: int | None = None):
        """
        Reset the progress bar at the start of an epoch.
        """
        self.training_bar.set_description(f"Epoch {epoch}")
        self.training_bar.reset(total=n_train)

    def start_eval(self, n_val: int, global_step: int):
        """
        Reset the progress bar for evaluation.
        """
        self.validation_bar.set_description(f"Eval @ Step {global_step}")
        self.validation_bar.reset(total=n_val)

    def update_train(self, loss_value: float, global_step: int) -> None:
        """
        Update the training tqdm progress bar.
        """
        self.training_bar.update(1)
        self.training_bar.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

    def update_eval(self, loss_value: float, global_step: int) -> None:
        """
        Update the validation tqdm progress bar.
        """
        self.validation_bar.update(1)
        self.validation_bar.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

    def close(self) -> None:
        """
        Close the progress bars once model training completes.
        """
        if self.training_bar: self.training_bar.close()
        if self.validation_bar: self.validation_bar.close()
