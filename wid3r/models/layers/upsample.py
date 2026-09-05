import sys
sys.path.append('.')

import torch
import torch.nn as nn

from functools import wraps

from cam_utils.misc import profile_method


class ResidualConvUnit(nn.Module):  # really slow on CPU...
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        dilation: int = 1,
        layer_scale: float = 1.0,
        use_norm: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.activation = nn.LeakyReLU()
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1))
            if layer_scale > 0.0
            else 1.0
        )
        self.norm1 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()

    @profile_method(verbose=False)
    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.gamma * out + x


class ResUpsampleBil(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim: int = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        use_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim // 2
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    @profile_method(verbose=False)
    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x
