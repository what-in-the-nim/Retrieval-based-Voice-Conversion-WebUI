from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn

from .layer_norm import LayerNorm


class DDSConv(nn.Module):
    """Dilated and Depth-Separable Convolution"""

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        # Build a separable convolutional block
        self.convs_sep = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (dilation * (kernel_size - 1)) // 2
            conv_layer = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                groups=channels,
                dilation=dilation,
                padding=padding,
            )
            self.convs_sep.append(conv_layer)
        # Build a 1x1 convolutional block
        self.convs_1x1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1) for _ in range(n_layers)]
        )
        # Build a normalization block
        self.norms_1 = nn.ModuleList([LayerNorm(channels) for _ in range(n_layers)])
        self.norms_2 = nn.ModuleList([LayerNorm(channels) for _ in range(n_layers)])
        # Build a dropout block
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if g is not None:
            x += g

        for conv_layer, conv_1x1, norm_1, norm_2 in zip(
            self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2
        ):
            y = conv_layer(x * x_mask)
            y = norm_1(y)
            y = F.gelu(y)
            y = conv_1x1(y)
            y = norm_2(y)
            y = F.gelu(y)
            y = self.dropout(y)
            x += y
        x *= x_mask
        return x
