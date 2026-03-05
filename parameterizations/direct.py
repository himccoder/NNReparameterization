# parameterizations/direct.py
# ─────────────────────────────────────────────────────────────────────────────
# Direct density parameterization — Conditions A, B, C, D.
#
# This is the classical approach: the optimization variables ARE the material
# densities x[i,j] ∈ [0,1] for each finite element.
#
# There is no neural network involved. The parameters are directly updated
# by the optimizer (MMA, Adam, or L-BFGS) using gradient information from
# the physics simulation.
#
# Condition D uses a heavier Gaussian filter (filter_width > 1) as a control:
# if Condition D matches Condition I (CNN), it means smoothing alone explains
# the CNN's advantage — not any architectural inductive bias.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import autograd.numpy as anp


class DirectParameterization:
    """
    Direct density parameterization.

    The optimization variables are the raw element densities. They are clipped
    to [xmin, xmax] during initialization and clamped by the optimizer's bounds.

    This is the baseline against which all other parameterizations are compared.
    """

    def __init__(self, args):
        """
        Args:
            args: problem ObjectView (needs nely, nelx, density, xmin, xmax)
        """
        self.args = args
        self.nely = args.nely
        self.nelx = args.nelx
        self.n_params = args.nely * args.nelx  # one density per finite element

        # Initialize uniformly at the target density
        self.x0 = np.ones(self.n_params) * args.density

    def to_density(self, params):
        """
        Convert optimization parameters directly to density field.

        For direct parameterization, this is just a reshape.
        No neural network forward pass required.

        Args:
            params: (n_params,) flat array of density values

        Returns:
            x: (nely, nelx) density grid
        """
        return anp.clip(
            params.reshape(self.nely, self.nelx),
            self.args.xmin,
            self.args.xmax
        )

    def initial_params(self):
        """Return the initial parameter vector for the optimizer."""
        return self.x0.copy()

    def param_count(self):
        """Number of optimization parameters."""
        return self.n_params

    def description(self):
        return (
            f"Direct density parameterization. "
            f"Grid: {self.nelx}×{self.nely} = {self.n_params} parameters. "
            f"Filter width: {self.args.filter_width}."
        )
