import torch
import torch.nn.functional as F
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization module."""
    def __init__(self, channels, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        x = x.transpose(1, -1)
        return x
