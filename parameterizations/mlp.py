# parameterizations/mlp.py
# ─────────────────────────────────────────────────────────────────────────────
# MLP (Multi-Layer Perceptron) reparameterization — Conditions E and F.
#
# Instead of optimizing densities x[i,j] directly, we optimize the weights W
# of a fully-connected neural network that maps a fixed coordinate grid to
# densities:
#
#   (x_coord, y_coord) → MLP(W) → density[i,j]
#
# The MLP has NO spatial inductive bias: it treats each grid position
# independently and does not share weights across spatial locations.
#
# Conditions:
#   E: Shallow MLP (3 hidden layers) — tests "is any NN better?"
#   F: Deep MLP (8 hidden layers)    — tests "does depth without spatial bias help?"
#
# If E or F match the CNN (Condition I), it means the NN parameterization
# helps in general — not because of CNN-specific spatial priors.
# If they don't, it supports H1: the CNN's spatial structure is essential.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected MLP mapping 2D grid coordinates to density values.

    Input: (N, 2) tensor of normalized (x, y) coordinates ∈ [0, 1]
    Output: (nely, nelx) density field ∈ [0, 1]

    Architecture: Linear → [ReLU → Linear] × n_hidden → Sigmoid
    """

    def __init__(self, hidden_layers=3, hidden_dim=64):
        """
        Args:
            hidden_layers: number of hidden layers (3 = shallow, 8 = deep)
            hidden_dim:    number of neurons per hidden layer
        """
        super().__init__()

        layers = []
        in_dim = 2  # input: (x_coord, y_coord)

        # Build hidden layers
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer: single density value per coordinate
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # clamp output to [0, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """
        Args:
            coords: (N, 2) normalized coordinates

        Returns:
            densities: (N, 1) density values
        """
        return self.net(coords)


class MLPParameterization:
    """
    MLP-based reparameterization of the density field.

    The optimization variables are the MLP weights. At each step, we:
      1. Run a forward pass through the MLP using a fixed coordinate grid
      2. Reshape the output to the (nely, nelx) density grid
      3. Pass this density grid to the physics solver

    The coordinate grid is fixed throughout optimization — only the weights change.
    """

    def __init__(self, args, hidden_layers=3, hidden_dim=64):
        """
        Args:
            args:          problem ObjectView
            hidden_layers: number of hidden layers (3 for shallow, 8 for deep)
            hidden_dim:    neurons per hidden layer
        """
        self.args         = args
        self.nely         = args.nely
        self.nelx         = args.nelx
        self.hidden_layers = hidden_layers
        self.hidden_dim    = hidden_dim

        # Build the network
        self.model = MLP(hidden_layers=hidden_layers, hidden_dim=hidden_dim)

        # Build fixed coordinate grid: each element's center in [0,1]^2
        # Shape: (nely * nelx, 2)
        ys = np.linspace(0, 1, args.nely)
        xs = np.linspace(0, 1, args.nelx)
        xg, yg = np.meshgrid(xs, ys)
        coords = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        self.coords = torch.from_numpy(coords)  # (nely*nelx, 2)

        # Count parameters
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def to_density(self, params_vec):
        """
        Convert flat weight vector to density field via MLP forward pass.

        Args:
            params_vec: (n_params,) numpy array of MLP weights

        Returns:
            density: (nely, nelx) numpy array
        """
        # Load flat weight vector into model parameters
        self._load_params(params_vec)

        with torch.no_grad():
            density = self.model(self.coords)       # (nely*nelx, 1)
            density = density.squeeze(1)            # (nely*nelx,)
            density = density.reshape(self.nely, self.nelx)

        return density.numpy()

    def to_density_with_grad(self, params_vec):
        """
        Forward pass that retains gradients (used by gradient-based optimizers).

        Returns:
            density: (nely, nelx) torch tensor with grad_fn attached
        """
        self._load_params(params_vec)
        density = self.model(self.coords)
        density = density.squeeze(1).reshape(self.nely, self.nelx)
        return density

    def initial_params(self):
        """
        Return flat initial weight vector.
        Weights are initialized by PyTorch default (Xavier uniform).
        """
        return self._get_flat_params()

    def param_count(self):
        return self._n_params

    def _get_flat_params(self):
        """Flatten all model parameters into a single 1D numpy array."""
        return np.concatenate([
            p.data.cpu().numpy().ravel()
            for p in self.model.parameters()
        ])

    def _load_params(self, flat_params):
        """Load a flat parameter vector back into the model."""
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
            f"MLP reparameterization. "
            f"Architecture: 2 → [{self.hidden_dim}×ReLU]×{self.hidden_layers} → 1→Sigmoid. "
            f"Parameters: {self._n_params}. "
            f"No spatial inductive bias."
        )
