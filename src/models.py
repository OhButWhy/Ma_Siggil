"""Neural network models for road segmentation."""

from typing import Optional

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary semantic segmentation.

    Compact version suitable for CPU training with configurable depth and base channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,  # 1 for binary (road)
        base_channels: int = 16,
        depth: int = 4,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for binary segmentation)
            base_channels: Base number of channels (multiplied at each depth level)
            depth: Number of downsampling levels (4 = 5 resolution levels)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth

        # Input convolution
        self.inc = DoubleConv(in_channels, base_channels)

        # Downsampling path
        self.down_layers = nn.ModuleList()
        in_ch = base_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch))
            in_ch = out_ch

        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(depth, 0, -1):
            in_ch = base_channels * (2**i)
            out_ch = base_channels * (2 ** (i - 1)) if i > 1 else base_channels
            # Concat channels = decoder feature channels + skip feature channels
            # skip channels at this stage equal out_ch
            self.up_layers.append(Up(in_ch + out_ch, out_ch))

        # Output layer
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        # Input convolution
        x1 = self.inc(x)

        # Downsampling path: store intermediate features
        features = [x1]
        x = x1
        for down in self.down_layers:
            x = down(x)
            features.append(x)

        # Upsampling path: use skip connections
        for i, up in enumerate(self.up_layers):
            # Take the feature map from downsampling path (skip connection)
            skip = features[-(i + 2)]
            x = up(x, skip)

        # Output layer
        logits = self.outc(x)

        return logits


def create_model(
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 16,
    depth: int = 4,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Create and return a U-Net segmentation model.

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_channels: Base number of channels
        depth: Model depth
        device: Device to put model on ('cpu' or 'cuda')

    Returns:
        Model instance (U-Net)
    """
    model = UNet(in_channels, num_classes, base_channels, depth)
    if device:
        model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
