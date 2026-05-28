# experiment_3_gnn/parameterizations/cnn_reference.py
# ─────────────────────────────────────────────────────────────────────────────
# CNN reference condition (R0) for Experiment 3.
#
# R0 replicates Experiment 2 Condition P7 — the best overall result from Exp 2:
#   3-level U-Net, 32 base channels, skip connections, BatchNorm, AdamW.
#
# This file is self-contained (does not import from experiment_2_cnn_deep_dive)
# so Experiment 3 can run standalone without the Exp 2 subdirectory.
# The architecture is identical to StandardUNet in experiment_2's cnn_variants.py.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS  (identical to Experiment 2)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → [BatchNorm2d] → ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def _match_and_concat(upsampled, skip, use_skip):
    """Align spatial dims then concatenate skip features."""
    if upsampled.shape[2:] != skip.shape[2:]:
        upsampled = F.interpolate(
            upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
    if use_skip:
        return torch.cat([upsampled, skip], dim=1)
    return upsampled


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD U-NET  (3 encoder levels, identical to Exp 2 StandardUNet)
# ─────────────────────────────────────────────────────────────────────────────

class StandardUNet(nn.Module):
    """
    3-level U-Net: encoder → bottleneck → decoder with skip connections.

    Channel progression (base_channels = c):
      Encoder:    1 → c → 2c → 4c (with MaxPool between levels)
      Bottleneck: 4c → 8c
      Decoder:    8c → 4c → 2c → c (with ConvTranspose + skip concat)
      Output:     c → 1 (sigmoid)
    """

    def __init__(self, base_channels=32, use_skip=True, use_batchnorm=True):
        super().__init__()
        c  = base_channels
        bn = use_batchnorm
        self.use_skip = use_skip

        # Encoder
        self.enc1 = ConvBlock(1,   c,   use_batchnorm=bn)
        self.enc2 = ConvBlock(c,   c*2, use_batchnorm=bn)
        self.enc3 = ConvBlock(c*2, c*4, use_batchnorm=bn)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(c*4, c*8, use_batchnorm=bn)

        # Decoder
        s3 = c*4 if use_skip else 0
        s2 = c*2 if use_skip else 0
        s1 = c   if use_skip else 0

        self.up3  = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = ConvBlock(c*4 + s3, c*4, use_batchnorm=bn)

        self.up2  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = ConvBlock(c*2 + s2, c*2, use_batchnorm=bn)

        self.up1  = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = ConvBlock(c + s1, c, use_batchnorm=bn)

        self.final = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = _match_and_concat(self.up3(b), e3, self.use_skip)
        d3 = self.dec3(d3)

        d2 = _match_and_concat(self.up2(d3), e2, self.use_skip)
        d2 = self.dec2(d2)

        d1 = _match_and_concat(self.up1(d2), e1, self.use_skip)
        d1 = self.dec1(d1)

        return self.final(d1)


# ─────────────────────────────────────────────────────────────────────────────
# CNN REFERENCE PARAMETERIZATION  (R0)
# ─────────────────────────────────────────────────────────────────────────────

class CNNReferenceParameterization:
    """
    Parameterization wrapper for R0: the best CNN from Experiment 2.

    Convention (identical to Experiments 1 & 2):
      - Fixed random latent z (1×1×H×W) is the CNN input, initialized from seed 42.
      - CNN weights W are the optimization variables.
      - z stays fixed throughout optimization.

    Interface compatible with run_gradient_optimizer() in optimizers/.
    """

    def __init__(self, args, arch="standard", base_channels=32,
                 use_skip=True, use_batchnorm=True):
        self.args          = args
        self.nely          = args.nely
        self.nelx          = args.nelx
        self.base_channels = base_channels
        self.use_skip      = use_skip
        self.use_batchnorm = use_batchnorm

        self.model = StandardUNet(
            base_channels = base_channels,
            use_skip      = use_skip,
            use_batchnorm = use_batchnorm,
        )

        # Fixed random latent — same seed as Experiments 1 & 2 for comparability
        rng       = np.random.RandomState(42)
        z_np      = rng.randn(1, 1, args.nely, args.nelx).astype(np.float32)
        self.z    = torch.from_numpy(z_np)

        self._n_params = sum(p.numel() for p in self.model.parameters())

    # ── Interface expected by run_gradient_optimizer ──────────────────────

    def initial_params(self):
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def to_density(self, params_vec):
        """Forward pass (no grad): flat params → (nely, nelx) density array."""
        self._load_params(params_vec)
        with torch.no_grad():
            density = self.model(self.z)
        density = F.interpolate(
            density, size=(self.nely, self.nelx),
            mode="bilinear", align_corners=False,
        )
        return density.squeeze().numpy()

    def to_density_with_grad(self, params_vec):
        """Forward pass retaining autograd graph for backpropagation."""
        self._load_params(params_vec)
        density = self.model(self.z)
        density = F.interpolate(
            density, size=(self.nely, self.nelx),
            mode="bilinear", align_corners=False,
        )
        return density.squeeze()

    def description(self):
        n = self._n_params
        return (
            f"CNN Reference (R0): StandardUNet 3L, {self.base_channels}ch, "
            f"skip={'yes' if self.use_skip else 'no'}, "
            f"BN={'yes' if self.use_batchnorm else 'no'}. "
            f"Parameters: {n:,}."
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_flat_params(self):
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.model.parameters()
        ])

    def _load_params(self, flat_params):
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(
                    flat_params[offset:offset + n].reshape(p.shape).astype(np.float32)
                )
            )
            offset += n
