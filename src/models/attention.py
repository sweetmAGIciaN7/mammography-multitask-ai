"""Channel and spatial attention modules."""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """
    Channel attention.

    Learns which feature channels are most informative.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1,
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        avg_pool = torch.mean(
            x,
            dim=(2, 3),
            keepdim=True,
        )

        max_pool = torch.amax(
            x,
            dim=(2, 3),
            keepdim=True,
        )

        attention = self.mlp(avg_pool)
        attention += self.mlp(max_pool)

        attention = self.sigmoid(
            attention
        )

        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention.

    Learns which spatial regions of the mammogram
    are most informative.
    """

    def __init__(
        self,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_map = torch.amax(
            x,
            dim=1,
            keepdim=True,
        )

        combined = torch.cat(
            [avg_map, max_map],
            dim=1,
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


class ChannelSpatialAttention(nn.Module):
    """
    Sequential channel + spatial attention.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x