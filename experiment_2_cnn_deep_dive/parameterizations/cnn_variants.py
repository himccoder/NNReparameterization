# experiment_2_cnn_deep_dive/parameterizations/cnn_variants.py
# ─────────────────────────────────────────────────────────────────────────────
# CNN architecture variants for Experiment 2.
#
# All conditions use the same optimization loop and the same latent-input
# convention as Experiment 1 (z → CNN(W) → density). Only the architecture
# of the CNN changes across conditions.
#
# Architectures provided:
#   standard  — 3-level U-Net, configurable channels/BN/skip  (P1/P4/P5/P6/P7/P8/P9)
#   shallow   — 2-level U-Net  (P2)
#   deep      — 4-level U-Net  (P3)
#
# All architectures share the same CNNVariantParameterization wrapper so the
# runner code does not need to know which architecture is in use.
#
# Key architectural axis compared to Experiment 1 reference (Condition I):
#   Depth:      2 levels vs 3 (P2) vs 4 (P3)
#   Capacity:   16 ch (P4) vs 32 ch (P1 ref) vs 64 ch (P5)
#   Norm:       with BatchNorm (P1) vs without (P6)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → [BatchNorm] → ReLU block. BatchNorm is optional."""

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
    """
    Align spatial dimensions (handles grids that are not exact powers of 2)
    and concatenate skip-connection features when use_skip is True.
    """
    if upsampled.shape[2:] != skip.shape[2:]:
        upsampled = F.interpolate(
            upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
    if use_skip:
        return torch.cat([upsampled, skip], dim=1)
    return upsampled


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD U-NET  (3 encoder levels)  — conditions P1, P4, P5, P6, P7, P8, P9
# ─────────────────────────────────────────────────────────────────────────────

class StandardUNet(nn.Module):
    """
    3-level U-Net: the reference architecture matching Experiment 1 Condition I.

    Encoder: full → half → quarter → eighth resolution
    Decoder: eighth → quarter → half → full resolution
    Skip connections: optional, concatenate encoder maps into decoder.
    BatchNorm:        optional (ablated in P6).

    Varying base_channels controls model capacity (P4: 16, P1: 32, P5: 64).
    """

    def __init__(self, base_channels=32, use_skip=True, use_batchnorm=True):
        super().__init__()
        c   = base_channels
        bn  = use_batchnorm
        self.use_skip = use_skip

        # Encoder
        self.enc1 = ConvBlock(1,   c,   use_batchnorm=bn)   # full res
        self.enc2 = ConvBlock(c,   c*2, use_batchnorm=bn)   # half res
        self.enc3 = ConvBlock(c*2, c*4, use_batchnorm=bn)   # quarter res
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

        d3 = self.up3(b)
        d3 = _match_and_concat(d3, e3, self.use_skip)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _match_and_concat(d2, e2, self.use_skip)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_and_concat(d1, e1, self.use_skip)
        d1 = self.dec1(d1)

        return self.final(d1)

    @property
    def n_levels(self):
        return 3


# ─────────────────────────────────────────────────────────────────────────────
# SHALLOW U-NET  (2 encoder levels)  — condition P2
# ─────────────────────────────────────────────────────────────────────────────

class ShallowUNet(nn.Module):
    """
    2-level U-Net: one fewer pooling step than the reference.

    Encoder: full → half → quarter resolution
    Decoder: quarter → half → full resolution

    Research question: does reducing multi-scale depth hurt performance?
    The bottleneck only sees quarter-resolution context (vs eighth in P1),
    meaning the CNN has less capacity to model global structure.
    """

    def __init__(self, base_channels=32, use_skip=True, use_batchnorm=True):
        super().__init__()
        c  = base_channels
        bn = use_batchnorm
        self.use_skip = use_skip

        # Encoder (2 levels)
        self.enc1 = ConvBlock(1, c,   use_batchnorm=bn)    # full res
        self.enc2 = ConvBlock(c, c*2, use_batchnorm=bn)    # half res
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck at quarter resolution
        self.bottleneck = ConvBlock(c*2, c*4, use_batchnorm=bn)

        # Decoder (2 levels)
        s2 = c*2 if use_skip else 0
        s1 = c   if use_skip else 0

        self.up2  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = ConvBlock(c*2 + s2, c*2, use_batchnorm=bn)

        self.up1  = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = ConvBlock(c + s1, c, use_batchnorm=bn)

        self.final = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = _match_and_concat(d2, e2, self.use_skip)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_and_concat(d1, e1, self.use_skip)
        d1 = self.dec1(d1)

        return self.final(d1)

    @property
    def n_levels(self):
        return 2


# ─────────────────────────────────────────────────────────────────────────────
# DEEP U-NET  (4 encoder levels)  — condition P3
# ─────────────────────────────────────────────────────────────────────────────

class DeepUNet(nn.Module):
    """
    4-level U-Net: one extra pooling step beyond the reference.

    Encoder: full → half → quarter → eighth → sixteenth resolution
    Decoder: sixteenth → eighth → quarter → half → full resolution

    Research question: does adding a deeper bottleneck (sixteenth resolution)
    improve the ability to model global structural layouts?

    Note: the sixteenth-resolution feature map becomes very small for some
    grids (e.g. 25-high MBB beam → 1-2 rows at bottom level). The bilinear
    interpolation in _match_and_concat handles this gracefully.
    """

    def __init__(self, base_channels=32, use_skip=True, use_batchnorm=True):
        super().__init__()
        c  = base_channels
        bn = use_batchnorm
        self.use_skip = use_skip

        # Encoder (4 levels)
        self.enc1 = ConvBlock(1,   c,    use_batchnorm=bn)   # full res
        self.enc2 = ConvBlock(c,   c*2,  use_batchnorm=bn)   # half res
        self.enc3 = ConvBlock(c*2, c*4,  use_batchnorm=bn)   # quarter res
        self.enc4 = ConvBlock(c*4, c*8,  use_batchnorm=bn)   # eighth res
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck at sixteenth resolution
        self.bottleneck = ConvBlock(c*8, c*16, use_batchnorm=bn)

        # Decoder (4 levels)
        s4 = c*8 if use_skip else 0
        s3 = c*4 if use_skip else 0
        s2 = c*2 if use_skip else 0
        s1 = c   if use_skip else 0

        self.up4  = nn.ConvTranspose2d(c*16, c*8, 2, stride=2)
        self.dec4 = ConvBlock(c*8 + s4, c*8, use_batchnorm=bn)

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
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = _match_and_concat(d4, e4, self.use_skip)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = _match_and_concat(d3, e3, self.use_skip)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _match_and_concat(d2, e2, self.use_skip)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_and_concat(d1, e1, self.use_skip)
        d1 = self.dec1(d1)

        return self.final(d1)

    @property
    def n_levels(self):
        return 4


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

_ARCH_MAP = {
    "standard": StandardUNet,
    "shallow":  ShallowUNet,
    "deep":     DeepUNet,
}


def build_cnn(arch, base_channels=32, use_skip=True, use_batchnorm=True):
    """
    Instantiate a CNN architecture by name.

    Args:
        arch:          one of "standard", "shallow", "deep"
        base_channels: channels in the first encoder block
        use_skip:      whether to use skip connections
        use_batchnorm: whether to include BatchNorm layers

    Returns:
        nn.Module
    """
    if arch not in _ARCH_MAP:
        raise ValueError(
            f"Unknown CNN architecture '{arch}'. "
            f"Available: {list(_ARCH_MAP.keys())}"
        )
    return _ARCH_MAP[arch](
        base_channels=base_channels,
        use_skip=use_skip,
        use_batchnorm=use_batchnorm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERIZATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class CNNVariantParameterization:
    """
    Wraps any CNN architecture variant for use by the Experiment 2 runner.

    Convention (identical to Experiment 1):
      - A fixed random latent tensor z (shape 1×1×H×W) is the CNN input.
      - The CNN weights W are the optimization variables.
      - z is fixed at a reproducible random seed.

    This wrapper is compatible with run_gradient_optimizer() from
    optimizers/gradient_optimizer.py.
    """

    def __init__(self, args, arch="standard", base_channels=32,
                 use_skip=True, use_batchnorm=True):
        """
        Args:
            args:          problem ObjectView (contains nelx, nely, density)
            arch:          CNN architecture name ("standard", "shallow", "deep")
            base_channels: number of channels in the first encoder conv
            use_skip:      whether to use U-Net skip connections
            use_batchnorm: whether to include BatchNorm layers
        """
        self.args          = args
        self.nely          = args.nely
        self.nelx          = args.nelx
        self.arch_name     = arch
        self.base_channels = base_channels
        self.use_skip      = use_skip
        self.use_batchnorm = use_batchnorm
        self.frozen        = False  # CNN weights are always trained in Exp 2

        self.model = build_cnn(
            arch          = arch,
            base_channels = base_channels,
            use_skip      = use_skip,
            use_batchnorm = use_batchnorm,
        )

        # Fixed random latent input — same seed as Experiment 1 for comparability
        rng       = np.random.RandomState(42)
        self.z_np = rng.randn(1, 1, args.nely, args.nelx).astype(np.float32)
        self.z    = torch.from_numpy(self.z_np)

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
        arch_display = {
            "standard": f"U-Net {self.model.n_levels}L",
            "shallow":  f"Shallow U-Net {self.model.n_levels}L",
            "deep":     f"Deep U-Net {self.model.n_levels}L",
        }.get(self.arch_name, self.arch_name)

        bn_str   = "BN"  if self.use_batchnorm else "no-BN"
        skip_str = "skip" if self.use_skip else "no-skip"
        return (
            f"CNN variant: {arch_display}, {self.base_channels}ch, "
            f"{skip_str}, {bn_str}. "
            f"Parameters: {self._n_params:,}."
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
