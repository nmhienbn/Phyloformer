import inspect
from typing import Dict, Tuple

import lightning
import torch
from transformers import get_linear_schedule_with_warmup

from phyloformer.losses import MRE, MAE

from phyloformer.models import Phyloformer


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """Ensure criterion outputs are scalar for Lightning backward/logging."""
    if not torch.is_tensor(loss):
        loss = torch.tensor(loss)
    return loss if loss.ndim == 0 else loss.mean()


def _criterion_supports_num_leaves(criterion) -> bool:
    forward = getattr(criterion, "forward", None)
    if forward is None:
        return False
    try:
        sig = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    if "num_leaves" in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _normalize_loss_output(loss_output, criterion) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if torch.is_tensor(loss_output):
        return reduce_loss(loss_output), {}

    if isinstance(loss_output, dict):
        if "loss" in loss_output:
            loss = reduce_loss(loss_output["loss"])
            aux_values = {k: v for k, v in loss_output.items() if k != "loss"}
        elif len(loss_output) == 1:
            _, only_val = next(iter(loss_output.items()))
            loss = reduce_loss(only_val)
            aux_values = {}
        else:
            raise ValueError(
                "Criterion dict output must contain key 'loss' or a single item."
            )
        return loss, {k: reduce_loss(v) for k, v in aux_values.items()}

    if isinstance(loss_output, (tuple, list)):
        if len(loss_output) == 0:
            raise ValueError("Criterion returned an empty tuple/list.")
        loss = reduce_loss(loss_output[0])
        names = getattr(criterion, "component_names", ())
        aux = {}
        for idx, value in enumerate(loss_output[1:]):
            name = names[idx] if idx < len(names) else f"aux_{idx + 1}"
            aux[name] = reduce_loss(value)
        return loss, aux

    raise TypeError(
        "Unsupported criterion output type. "
        f"Expected tensor, tuple/list, or dict; got {type(loss_output)}."
    )

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
        n_seqs: int = 50,
    ):
        super().__init__()

        # Initialize model
        self.model = Phyloformer(
            n_blocks=nb_blocks,
            n_heads=nb_heads,
            h_dim=embed_dim,
            dropout=dropout,
            n_seqs=n_seqs,
        )

        # Optimizer stuff
        self.optim_func = optim_func
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.criterion = criterion
        self._criterion_supports_num_leaves = _criterion_supports_num_leaves(criterion)
        self.batch_size = batch_size

        self.save_hyperparameters(ignore=["criterion", "optim_func"])

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor, num_leaves: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        criterion_kwargs = {}
        if self._criterion_supports_num_leaves:
            criterion_kwargs["num_leaves"] = num_leaves
        loss_output = self.criterion(y_hat, y, **criterion_kwargs)
        return _normalize_loss_output(loss_output, self.criterion)

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        y = y.type_as(y_hat).squeeze()
        loss, aux = self._compute_loss(y_hat, y, num_leaves=x.shape[-1])
        for name, value in aux.items():
            self.log(f"train_{name}", value, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        y = y.type_as(y_hat).squeeze()

        # Compute validation metrics and log them
        loss, aux = self._compute_loss(y_hat, y, num_leaves=x.shape[-1])
        d = {"val_mre": MRE(y_hat, y, False), "val_mae": MAE(y_hat, y, False)}
        aux_val = {f"val_{k}": v for k, v in aux.items()}
        self.log_dict(dict(val_loss=reduce_loss(loss), **d, **aux_val), sync_dist=True)

        return dict(loss=loss, **d)
