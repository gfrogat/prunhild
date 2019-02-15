import torch


def binary_mask_weight(param, prune_mask, cutoff_value):
    return torch.where(
        param.data.abs().le(cutoff_value), torch.zeros_like(prune_mask), prune_mask
    )


def binary_mask_grad(param, prune_mask, cutoff_value):
    return torch.where(
        param.grad.data.abs().le(cutoff_value), torch.zeros_like(prune_mask), prune_mask
    )
