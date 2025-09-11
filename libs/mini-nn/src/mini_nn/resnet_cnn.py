"""Minimal ResNet CNN model."""

from typing import Literal

import torch
import torch.nn as nn


class PreActResNetBlock(nn.Module):
    """Simple CNN ResNet block where both activations are applied before the residual connection.

    z <- (BN -> ReLU -> Conv -> BN -> ReLU -> Conv)(x)
    out <- z + x

    Args:
        in_channels: Number of input features
        out_channels: Number of output features
        activation_fn: Callable activation function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: nn.Module,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation_fn,
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation_fn,
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.projection = nn.Identity()
        need_projection = (stride != 1) or (in_channels != out_channels)
        if need_projection:
            # kernel_size = 1: only the number of channels changes
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        z = self.net(x)
        out = z + self.projection(x)
        return out


class PostActResNetBlock(nn.Module):
    """Simple CNN ResNet block where the last activation is applied after the residual connection.

    z <- (Conv -> BN -> ReLu -> Conv -> BN)(x) + x
    out <- ReLu(z)

    Args:
        in_channels: Number of input features
        out_channels: Number of output features
        activation_fn: Callable activation function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: nn.Module,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.projection = nn.Identity()
        need_projection = (stride != 1) or (in_channels != out_channels)
        if need_projection:
            # kernel_size = 1: only the number of channels changes
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward methood."""
        z = self.net(x)
        z = z + self.projection(x)
        out = self.activation_fn(z)
        return out


class ResNetGroupBlock(nn.Module):
    """A group of 3 ResNetBlock.

    Args:
        block_type: Type of ResNet block (PreAct or PostAct)
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        block_type: Literal["PreAct", "PostAct"],
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        single_block = (
            PreActResNetBlock if block_type == ["PreAct"] else PostActResNetBlock
        )
        self.block = nn.Sequential(
            single_block(in_channels, out_channels, nn.ReLU(), stride=1),
            single_block(out_channels, out_channels, nn.ReLU(), stride=1),
            single_block(out_channels, out_channels, nn.ReLU(), stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        return self.block(x)


class ResNet(nn.Module):
    """Simple ResNet model.

    Args:
        block_type: Type of ResNet block (PreAct or PostAct)
        channels: List of numbers of channels (includes the number of input channels)
        num_classes: Number of output classes
    """

    def __init__(
        self,
        block_type: Literal["PreAct", "PostAct"],
        channels: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()

        # Input network
        if block_type == "PreAct":
            self.input_model = nn.Sequential(
                nn.Conv2d(
                    channels[0], channels[1], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(),
            )
        else:
            self.input_model = nn.Sequential(
                nn.Conv2d(
                    channels[0], channels[1], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(),
            )

        # ResNet blocks
        blocks = [
            ResNetGroupBlock(block_type, in_c, out_c)
            for (in_c, out_c) in zip(channels[1:-1], channels[2:])
        ]
        self.blocks = nn.Sequential(*blocks)

        # Output network
        self.output_model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.input_model(x)
        x = self.blocks(x)
        x = self.output_model(x)
        return x
