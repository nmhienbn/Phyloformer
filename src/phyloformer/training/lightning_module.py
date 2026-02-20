

import lightning
import torch
from transformers import get_linear_schedule_with_warmup

from phyloformer.losses import MRE, MAE

from phyloformer.models import Phyloformer


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Ensure criterion outputs are scalar for Lightning backward/logging."""
    return loss if loss.ndim == 0 else loss.mean()

class LightningAxialTransformer(lightning.LightningModule):
    """Lighnint Object for Phyloformer training"""

    def __init__(
        self,
        nb_blocks: int,
        nb_heads: int,
        embed_dim: int,
        dropout: float,
        learning_rate: float,
        warmup_steps: int,
        total_steps: int,
        batch_size: int,
        optim_func,
        criterion,
    ):
        super().__init__()

        # Initialize model
        self.model = Phyloformer(
            n_blocks=nb_blocks,
            n_heads=nb_heads,
            h_dim=embed_dim,
            dropout=dropout,
        )

        # Optimizer stuff
        self.optim_func = optim_func
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.criterion = criterion
        self.batch_size = batch_size

        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=["criterion"])

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        loss = reduce_loss(self.criterion(y_hat, y.type_as(y_hat).squeeze()))
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        y = y.type_as(y_hat).squeeze()

        # Compute validation metrics and log them
        loss = reduce_loss(self.criterion(y_hat, y))
        d = {"val_mre": MRE(y_hat, y, False), "val_mae": MAE(y_hat, y, False)}
        self.log_dict(dict(val_loss=loss, **d), sync_dist=True)

        return dict(loss=loss, **d)
