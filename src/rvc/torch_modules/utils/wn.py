from typing import Optional

import torch
import torch.nn as nn


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """Apply fused activation."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(nn.Module):
    """WaveNet module."""
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super(WN, self).__init__()
        # Check if kernel_size is odd
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd.")

        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        # Build a cond layer if gin_channels is not 0
        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

        # Build in_layers
        self.in_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

        # Build res_skip_layers
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)
        # Build a dropout block
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.dropout(acts)

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)

    def __prepare_scriptable__(self):
        if self.gin_channels != 0:
            for hook in self.cond_layer._forward_pre_hooks.values():
                if (
                    hook.__module__ == "nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            for hook in layer._forward_pre_hooks.values():
                if (
                    hook.__module__ == "nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            for hook in layer._forward_pre_hooks.values():
                if (
                    hook.__module__ == "nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    nn.utils.remove_weight_norm(layer)
        return self
