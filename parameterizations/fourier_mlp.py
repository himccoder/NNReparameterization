# parameterizations/fourier_mlp.py
# ─────────────────────────────────────────────────────────────────────────────
# Frequency-biased parameterizations — Conditions G (Fourier MLP) and H (SIREN).
#
# Both architectures introduce a strong spatial frequency prior without using
# any convolution. This lets us test H5: "Is frequency bias, achievable without
# convolution, enough to explain the CNN's advantage?"
#
# ── CONDITION G: Fourier MLP ─────────────────────────────────────────────────
# Maps (x, y) coordinates through a Fourier feature encoding before the MLP.
# This expands the 2D coordinate into a high-dimensional sinusoidal embedding,
# giving the network a head start at representing spatial frequency content.
#
#   coord → [sin(2π·B·coord), cos(2π·B·coord)] → MLP → density
#
# where B is a fixed random frequency matrix. (Tancik et al. 2020)
#
# ── CONDITION H: SIREN ──────────────────────────────────────────────────────
# Sinusoidal Representation Network. All activations are sin(ω₀ · Wx + b).
# The entire network is a composition of sinusoidal functions, giving it a
# strong prior toward smooth, periodic spatial patterns. (Sitzmann et al. 2020)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION G: FOURIER FEATURE MLP
# ─────────────────────────────────────────────────────────────────────────────

class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier feature embedding for 2D coordinates.

    Projects (x, y) to a 2*num_frequencies dimensional vector:
        φ(v) = [sin(2π B v), cos(2π B v)]

    where B is a (num_frequencies, 2) matrix sampled from N(0, σ²).
    The scale σ controls the frequency bandwidth: larger σ = higher frequencies.

    This is the 'positional encoding' trick popularized by NeRF.
    """

    def __init__(self, num_frequencies=32, scale=1.0):
        super().__init__()
        # B is fixed (not learned) — it defines the frequency basis
        B = torch.randn(num_frequencies, 2) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num_frequencies

    def forward(self, coords):
        """
        Args:
            coords: (N, 2) coordinate tensor

        Returns:
            embedding: (N, 2*num_frequencies) Fourier features
        """
        proj = 2 * np.pi * coords @ self.B.T   # (N, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


class FourierMLP(nn.Module):
    """
    MLP with Fourier positional encoding as input layer.

    The Fourier embedding gives the network access to a rich multi-frequency
    representation of the coordinate, without any convolution or weight sharing.
    """

    def __init__(self, num_frequencies=32, hidden_layers=4, hidden_dim=64, scale=1.0):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(num_frequencies, scale)
        in_dim = self.embedding.out_dim

        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        features = self.embedding(coords)
        return self.net(features)


class FourierMLPParameterization:
    """Fourier-feature MLP reparameterization (Condition G)."""

    def __init__(self, args, hidden_layers=4, hidden_dim=64, num_frequencies=32, scale=1.0):
        self.args           = args
        self.nely           = args.nely
        self.nelx           = args.nelx
        self.num_frequencies = num_frequencies

        self.model = FourierMLP(
            num_frequencies=num_frequencies,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            scale=scale,
        )

        # Fixed coordinate grid
        ys = np.linspace(0, 1, args.nely)
        xs = np.linspace(0, 1, args.nelx)
        xg, yg = np.meshgrid(xs, ys)
        coords = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        self.coords = torch.from_numpy(coords)

        self._n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def to_density(self, params_vec):
        self._load_params(params_vec)
        with torch.no_grad():
            density = self.model(self.coords).squeeze(1).reshape(self.nely, self.nelx)
        return density.numpy()

    def to_density_with_grad(self, params_vec):
        self._load_params(params_vec)
        density = self.model(self.coords).squeeze(1).reshape(self.nely, self.nelx)
        return density

    def initial_params(self):
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def _get_flat_params(self):
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.model.parameters()
            if p.requires_grad
        ])

    def _load_params(self, flat_params):
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(
                    flat_params[offset:offset + n].reshape(p.shape).astype(np.float32)
                )
            )
            offset += n

    def description(self):
        return (
            f"Fourier MLP reparameterization. "
            f"Encoding: {self.num_frequencies} random Fourier features. "
            f"Frequency bias without convolution. "
            f"Parameters: {self._n_params}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION H: SIREN (Sinusoidal Representation Network)
# ─────────────────────────────────────────────────────────────────────────────

class SineLayer(nn.Module):
    """
    Single SIREN layer: Linear → sin(ω₀ · ·).

    The sine activation makes the network naturally represent smooth, periodic
    functions. The ω₀ parameter controls the frequency of the oscillations.
    """

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        """
        Args:
            omega_0:   frequency scale (30.0 recommended by Sitzmann et al.)
            is_first:  use different initialization for first layer
        """
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_features, out_features)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features, is_first):
        """
        SIREN-specific weight initialization.
        Ensures the distribution of activations is uniform and stable.
        """
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network for 2D coordinate → density mapping.

    All hidden layers use sine activations. The output layer uses sigmoid
    to clamp the density to [0, 1].
    """

    def __init__(self, hidden_layers=4, hidden_dim=64, omega_0=30.0):
        super().__init__()
        layers = [SineLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.hidden = nn.Sequential(*layers)

        # Final linear + sigmoid for density output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, coords):
        return self.output(self.hidden(coords))


class SIRENParameterization:
    """SIREN reparameterization (Condition H)."""

    def __init__(self, args, hidden_layers=4, hidden_dim=64, omega_0=30.0):
        self.args         = args
        self.nely         = args.nely
        self.nelx         = args.nelx
        self.omega_0      = omega_0
        self.hidden_layers = hidden_layers

        self.model = SIREN(hidden_layers=hidden_layers, hidden_dim=hidden_dim, omega_0=omega_0)

        # Fixed coordinate grid
        ys = np.linspace(0, 1, args.nely)
        xs = np.linspace(0, 1, args.nelx)
        xg, yg = np.meshgrid(xs, ys)
        coords = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        self.coords = torch.from_numpy(coords)

        self._n_params = sum(p.numel() for p in self.model.parameters())

    def to_density(self, params_vec):
        self._load_params(params_vec)
        with torch.no_grad():
            density = self.model(self.coords).squeeze(1).reshape(self.nely, self.nelx)
        return density.numpy()

    def to_density_with_grad(self, params_vec):
        self._load_params(params_vec)
        density = self.model(self.coords).squeeze(1).reshape(self.nely, self.nelx)
        return density

    def initial_params(self):
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def _get_flat_params(self):
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.model.parameters()])

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

    def description(self):
        return (
            f"SIREN reparameterization. "
            f"Sine activations with ω₀={self.omega_0}. "
            f"Strong frequency prior without convolution. "
            f"Layers: {self.hidden_layers}. Parameters: {self._n_params}."
        )
