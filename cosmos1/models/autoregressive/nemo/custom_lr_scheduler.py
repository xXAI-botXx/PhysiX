from typing import Optional
from nemo.lightning.pytorch.optim.base import LRSchedulerModule


class CustomCosineAnnealingScheduler(LRSchedulerModule):
    def __init__(
        self,
        max_steps: int = 10,
        warmup_steps: int = 750,
        constant_steps: int = 80000,
        min_lr: float = 6e-5,
        interval: str = "step",
        frequency: int = 1,
        monitor: str = "val_loss",
        lr_scheduler_state_dict: Optional[dict] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.min_lr = min_lr
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor
        self.lr_scheduler_state_dict = lr_scheduler_state_dict

    def scheduler(self, model, optimizer):
        from nemo.core.optim.lr_scheduler import CosineAnnealing

        lr_scheduler = CosineAnnealing(
            optimizer,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            constant_steps=self.constant_steps,
            min_lr=self.min_lr,
        )
        lr_scheduler.load_state_dict(self.lr_scheduler_state_dict) if self.lr_scheduler_state_dict else None

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": self.frequency,
            },
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor,
        }