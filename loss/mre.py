import torch


def MRE(
    input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool = False
) -> torch.Tensor:
    """Computes the Mean Relative Error"""
    if sqrt_preds:
        input = input**2
    return torch.mean(torch.abs(input - target) / target).detach()


class MRELoss(torch.nn.Module):
    """Mean Relative Error loss used for paper fine-tuning."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(input - target) / target)