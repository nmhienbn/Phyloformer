import torch


def MAE(
    input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool = False
) -> torch.Tensor:
    """Computes the Mean Absolute Error"""

    if sqrt_preds:
        input = input**2
    return torch.nn.L1Loss()(input, target).detach()


MAELoss = torch.nn.L1Loss()