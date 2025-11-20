"""Lightning module for BERT masked language modeling."""

from typing import Any

import torch
from lightning import LightningModule


class BERTLitModule(LightningModule):
    """Lightning module for BERT masked language modeling.

    Args:
        net: BERT model
        optimizer: Optimizer partial function (from Hydra with _partial_: true)
        scheduler: Scheduler partial function (from Hydra with _partial_: true)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        super().__init__()

        # Save hyperparameters (excluding net for cleaner logs)
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, loss_weight: torch.Tensor):
        """Forward pass through model."""
        return self.net(input_ids, labels, loss_weight)

    def model_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step (shared by train/val).

        Args:
            batch: Batch dict with keys: input_ids, labels, loss_weight

        Returns:
            Loss tensor
        """
        loss = self.forward(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            loss_weight=batch["loss_weight"],
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
