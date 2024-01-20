import torch
import torch.nn as nn

from .layer_norm import LayerNorm


class ConvReluNorm(nn.Module):
    """Convolutional block with ReLU and layer normalization."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        # Check if n_layers is valid
        if n_layers <= 1:
            raise ValueError("Number of layers should be larger than 1.")

        # Build a convolutional block
        self.conv_layers = nn.ModuleList()
        # Add the first convolutional layer
        first_conv_layer = nn.Conv1d(
            in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_layers.append(first_conv_layer)
        # Add the rest convolutional layers
        for _ in range(n_layers - 1):
            hidden_conv_layers = nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
            self.conv_layers.append(hidden_conv_layers)

        # Build a normalization block
        self.norm_layers = nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )
        # Build a relu-dropout block
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        # Build a projection block
        self.projection = nn.Conv1d(hidden_channels, out_channels, 1)
        self.projection.weight.data.zero_()
        self.projection.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_org = x
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x * x_mask)
            x = norm_layer(x)
            x = self.relu_drop(x)
        x = x_org + self.projection(x)
        x *= x_mask
        return x
