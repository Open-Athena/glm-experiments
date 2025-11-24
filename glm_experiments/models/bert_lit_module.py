"""Lightning module for BERT masked language modeling."""

from typing import Any

import torch
import torch.nn as nn
from biofoundation.model.scoring import compute_llr_mlm
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from sklearn.metrics import average_precision_score
from torchmetrics.aggregation import CatMetric


class MaskedLMAdapter(nn.Module):
    """Adapter to make BERT compatible with biofoundation's MaskedLM protocol.

    biofoundation's compute_llr_mlm expects a model with forward(input_ids) -> logits,
    but our BERT model has get_logits(input_ids) -> logits.
    """

    def __init__(self, bert: nn.Module):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (MaskedLM protocol)."""
        return self.bert.get_logits(input_ids)


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

        # Adapter for biofoundation's compute_llr_mlm
        self.mlm_adapter = MaskedLMAdapter(net)

        # CatMetrics for TraitGym Mendelian Promoter VEP evaluation
        self.traitgym_mendelian_promoter_labels = CatMetric()
        self.traitgym_mendelian_promoter_scores = CatMetric()

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
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Validation step for MLM and TraitGym Mendelian Promoter dataloaders.

        Args:
            batch: Batch dict (keys depend on dataloader)
            batch_idx: Batch index
            dataloader_idx: 0 for MLM validation, 1 for TraitGym Mendelian Promoter
        """
        if dataloader_idx == 0:
            # MLM validation
            loss = self.model_step(batch)
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
        elif dataloader_idx == 1:
            # TraitGym Mendelian Promoter VEP evaluation
            # Data comes as PyTorch tensors from HuggingFace dataset with set_format("torch")
            scores = -1 * compute_llr_mlm(
                model=self.mlm_adapter,
                input_ids=batch["input_ids"],
                pos=batch["pos"],
                ref=batch["ref"],
                alt=batch["alt"],
            )
            self.traitgym_mendelian_promoter_scores.update(scores)
            self.traitgym_mendelian_promoter_labels.update(batch["label"])

    def on_validation_epoch_end(self) -> None:
        """Compute and log TraitGym Mendelian Promoter AUPRC at end of validation epoch."""
        # Only compute if we have TraitGym Mendelian Promoter data
        if self.traitgym_mendelian_promoter_scores.update_count > 0:
            scores = self.traitgym_mendelian_promoter_scores.compute().cpu().numpy()
            labels = self.traitgym_mendelian_promoter_labels.compute().cpu().numpy()

            auprc = average_precision_score(labels, scores)
            self.log("val/traitgym_mendelian_promoter_auprc", auprc, prog_bar=True)

            # Reset metrics for next epoch
            self.traitgym_mendelian_promoter_scores.reset()
            self.traitgym_mendelian_promoter_labels.reset()

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

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient norm before optimizer step."""
        norms = grad_norm(self, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"])
