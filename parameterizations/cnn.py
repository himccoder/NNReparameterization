# parameterizations/cnn.py
# ─────────────────────────────────────────────────────────────────────────────
# CNN (U-Net style) reparameterization — Conditions I, J, K, L.
#
# This is the Hoyer et al. (2019) approach. A CNN maps a fixed random latent
# tensor z to a density field:
#
#   z (latent noise) → CNN(W) → density field (nely × nelx)
#
# Key architectural features:
#   - Convolutional layers: weight sharing across spatial locations
#   - U-Net skip connections: preserve multi-scale spatial information
#   - The input z is a fixed random tensor (only weights W are optimized)
#
# The spatial inductive bias of convolution means:
#   - Nearby elements share information (local correlation)
#   - The same feature detector applies everywhere (translation equivariance)
#   - Skip connections allow coarse-to-fine structure
#
# Conditions:
#   I: Full CNN (U-Net with skip connections) + Adam  ← Hoyer et al. method
#   J: Full CNN + MMA  ← tests if CNN prior helps with classical optimizer
#   K: CNN without skip connections + Adam  ← ablates multi-scale structure
#   L: CNN with FROZEN weights + Adam  ← THE KILLER EXPERIMENT
#
# Condition L is the most important: if frozen random CNN weights still produce
# better results than direct parameterization, it means the spatial prior is
# purely architectural — the CNN is acting as a fixed spatial basis, not a
# learned representation. This would strongly support H6.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetCNN(nn.Module):
    """
    U-Net style CNN for density field generation.

    Architecture:
      Encoder: progressively downsamples the spatial resolution
      Decoder: progressively upsamples back to full resolution
      Skip connections: concatenate encoder features into decoder at each scale

    The multi-scale structure means the CNN can model both:
      - Coarse global structure (at downsampled scales)
      - Fine-grained local detail (at full resolution)

    This is the key property that Hoyer et al. argue makes CNNs effective
    for structural optimization: the mesh-dependency problem disappears because
    optimization happens across all scales simultaneously.

    Input:  (1, nely, nelx) fixed random latent tensor
    Output: (1, nely, nelx) density field ∈ (0, 1)
    """

    def __init__(self, use_skip_connections=True, base_channels=32):
        """
        Args:
            use_skip_connections: if False, becomes a plain encoder-decoder CNN
                                  (Condition K: ablates multi-scale structure)
            base_channels:        number of channels in the first conv layer
        """
        super().__init__()
        self.use_skip = use_skip_connections
        c = base_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each encoder level halves the spatial resolution
        self.enc1 = ConvBlock(1,   c)       # full resolution
        self.enc2 = ConvBlock(c,   c*2)     # half resolution
        self.enc3 = ConvBlock(c*2, c*4)     # quarter resolution

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(c*4, c*8)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Upsampling + optional skip connections
        skip3_in = c*4 if use_skip_connections else 0 # If the skip connections are used then the input channels for the third decoder block are the number of channels in the fourth encoder block
        skip2_in = c*2 if use_skip_connections else 0 # If the skip connections are used then the input channels for the second decoder block are the number of channels in the third encoder block
        skip1_in = c   if use_skip_connections else 0 # If the skip connections are used then the input channels for the first decoder block are the number of channels in the second encoder block

        self.up3  = nn.ConvTranspose2d(c*8, c*4, 2, stride=2) # Transpose convolution to upsample the third decoder block
        self.dec3 = ConvBlock(c*4 + skip3_in, c*4) # Convolution block to decode the third decoder block

        self.up2  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2) # Transpose convolution to upsample the second decoder block
        self.dec2 = ConvBlock(c*2 + skip2_in, c*2) # Convolution block to decode the second decoder block

        self.up1  = nn.ConvTranspose2d(c*2, c, 2, stride=2) # Transpose convolution to upsample the first decoder block
        self.dec1 = ConvBlock(c + skip1_in, c) # Convolution block to decode the first decoder block

        # Final 1×1 conv to produce single-channel density, sigmoid to [0,1]
        self.final = nn.Sequential(
            nn.Conv2d(c, 1, 1), # Convolution to produce the final density field
            nn.Sigmoid(), # Sigmoid activation function to ensure the density field is in the range [0, 1]
        )

    def forward(self, z):
        """
        Args:
            z: (1, 1, H, W) latent tensor (fixed random input)

        Returns:
            density: (1, 1, H, W) density field ∈ (0, 1)
        """
        # Encoder
        e1 = self.enc1(z)               # full resolution
        e2 = self.enc2(self.pool(e1))   # half resolution
        e3 = self.enc3(self.pool(e2))   # quarter resolution

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with optional skip connections
        d3 = self.up3(b)
        d3 = self._match_and_concat(d3, e3, self.use_skip)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2, self.use_skip)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1, self.use_skip)
        d1 = self.dec1(d1)

        return self.final(d1)

    @staticmethod
    def _match_and_concat(upsampled, skip, use_skip):
        """
        Match spatial dimensions (handle odd sizes) and optionally concatenate skip.
        """
        # Crop or pad to match dimensions (handles cases where input isn't power of 2)
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        if use_skip:
            return torch.cat([upsampled, skip], dim=1)
        return upsampled


# ─────────────────────────────────────────────────────────────────────────────
# CNN PARAMETERIZATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class CNNParameterization:
    """
    CNN-based reparameterization of the density field.

    A fixed random latent tensor z is fed through a CNN to produce the density.
    The optimization variables are the CNN weights (except in Condition L,
    where the weights are frozen and only z is optimized).

    Key design choices matching Hoyer et al. (2019):
      - z is fixed random (not optimized) in Conditions I/J/K
      - Only the CNN weights are updated
      - For Condition L: z is optimized, weights are frozen
    """

    def __init__(self, args, use_skip_connections=True, frozen=False, base_channels=32):
        """
        Args:
            args:                problem ObjectView
            use_skip_connections: whether to use U-Net skip connections (Conditions I/J vs K)
            frozen:              if True, freeze all CNN weights and optimize z instead (Condition L)
            base_channels:       channels in first conv layer (controls model capacity)
        """
        self.args   = args
        self.nely   = args.nely
        self.nelx   = args.nelx
        self.frozen = frozen

        self.model = UNetCNN(
            use_skip_connections=use_skip_connections,
            base_channels=base_channels,
        )

        # Fixed random latent input (seeded for reproducibility)
        rng = np.random.RandomState(42)
        self.z_np = rng.randn(1, 1, args.nely, args.nelx).astype(np.float32)
        self.z    = torch.from_numpy(self.z_np)

        if frozen:
            # CONDITION L: Freeze all CNN weights.
            # Only the latent tensor z is optimized.
            # This tests whether the spatial prior alone (without any weight learning)
            # is sufficient to explain the CNN's advantage.
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.z = nn.Parameter(torch.from_numpy(self.z_np))
            self._n_params = self.z.numel()  # only z is optimized
        else:
            # Conditions I/J/K: optimize CNN weights, z is fixed
            self._n_params = sum(p.numel() for p in self.model.parameters())

    def to_density(self, params_vec):
        """
        Forward pass: params → density field.

        Args:
            params_vec: flat numpy array (CNN weights, or z if frozen)

        Returns:
            density: (nely, nelx) numpy array
        """
        self._load_params(params_vec)

        with torch.no_grad():
            z_in = self.z if not self.frozen else self.z.data
            density = self.model(z_in)   # (1, 1, nely, nelx)

        # Resize to exact target dimensions (handles any rounding from pooling)
        density = F.interpolate(
            density,
            size=(self.nely, self.nelx),
            mode="bilinear",
            align_corners=False,
        )
        return density.squeeze().numpy()

    def to_density_with_grad(self, params_vec):
        """
        Forward pass retaining gradients for PyTorch-based optimizers.
        """
        self._load_params(params_vec)
        z_in = self.z if not self.frozen else self.z
        density = self.model(z_in)
        density = F.interpolate(
            density,
            size=(self.nely, self.nelx),
            mode="bilinear",
            align_corners=False,
        )
        return density.squeeze()

    def initial_params(self):
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def _get_flat_params(self):
        if self.frozen:
            # Return the latent z as the parameter vector
            return self.z.data.cpu().numpy().ravel()
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.model.parameters()
        ])

    def _load_params(self, flat_params):
        if self.frozen:
            # Load into z
            z_tensor = torch.from_numpy(
                flat_params.reshape(self.z.shape).astype(np.float32)
            )
            if isinstance(self.z, nn.Parameter):
                self.z.data.copy_(z_tensor)
            else:
                self.z = z_tensor
        else:
            # Load into CNN weights
            offset = 0
            for p in self.model.parameters():
                n = p.numel()
                p.data.copy_(
                    torch.from_numpy(
                        flat_params[offset:offset + n].reshape(p.shape).astype(np.float32)
                    )
                )
                offset += n

    def description(self):
        mode = "frozen weights (pure prior, z optimized)" if self.frozen else "trainable weights"
        skip = "with skip connections" if self.model.use_skip else "without skip connections"
        return (
            f"CNN reparameterization ({mode}, {skip}). "
            f"U-Net architecture with {self.model.enc1.block[0].out_channels} base channels. "
            f"Parameters: {self._n_params}."
        )
