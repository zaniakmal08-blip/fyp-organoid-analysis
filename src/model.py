"""TinyUNet: the small U-Net used for binary organoid segmentation.

Architecture (29,321 trainable parameters)::

    input 1ch ─ e1 (8ch) ──────────────────────────── skip ─┐
                 └ pool ─ e2 (16ch) ───────── skip ─┐       │
                           └ pool ─ middle (32ch)   │       │
                                     └ u2 ↑ ─ cat ─ d2 (16ch)
                                                └ u1 ↑ ─ cat ─ d1 (8ch) ─ out (1ch logit)

The network returns *logits*. Apply ``sigmoid`` to get the organoid probability
and threshold at 0.5 for the mask.

IMPORTANT: the attribute names (``e1``, ``e2``, ``middle``, ``pool``, ``u2``,
``d2``, ``u1``, ``d1``, ``out``) are the keys stored in ``models/tiny_unet.pt``.
Renaming them, or changing channel counts, makes the saved checkpoint unloadable.
A different architecture should be a new class with its own checkpoint.
"""

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by ReLU. Spatial size is unchanged."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class TinyUNet(nn.Module):
    """Two-level U-Net. Input ``(N, 1, H, W)`` with H and W divisible by 4."""

    def __init__(self):
        super().__init__()
        # Encoder: features get deeper while the image gets smaller.
        self.e1 = conv_block(1, 8)
        self.e2 = conv_block(8, 16)
        self.middle = conv_block(16, 32)
        self.pool = nn.MaxPool2d(2)
        # Decoder: upsample, then merge with the matching encoder features.
        self.u2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.d2 = conv_block(32, 16)
        self.u1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.d1 = conv_block(16, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.e1(x)
        skip2 = self.e2(self.pool(skip1))
        bottom = self.middle(self.pool(skip2))
        up2 = self.d2(torch.cat([self.u2(bottom), skip2], dim=1))
        up1 = self.d1(torch.cat([self.u1(up2), skip1], dim=1))
        return self.out(up1)
