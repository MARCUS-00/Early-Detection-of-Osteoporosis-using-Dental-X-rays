"""
U-Net architecture — single canonical copy, deduplicated from Unet_Extraction.ipynb
which contains approximately 4 copies of the same definition.

Architecture (ported verbatim):
  Encoder: conv_blocks + MaxPool2d(2) at each level.
  Bottleneck: features[-1] * 2 channels.
  Decoder: ConvTranspose2d up-sampling; F.interpolate to match skip-connection size.
  Output: Conv2d(features[0], out_channels, 1) -> torch.sigmoid.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm -> ReLU) blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net with configurable feature sizes.

    Default: in_channels=3, out_channels=1, features=[64, 128, 256, 512].
    Output is passed through torch.sigmoid so values are in [0, 1].

    Source: Unet_Extraction.ipynb — ported verbatim; deduplicated.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(_DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(_DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]

            # F.interpolate to match skip-connection spatial size (verbatim)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final_conv(x))
