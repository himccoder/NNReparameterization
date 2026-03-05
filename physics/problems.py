# physics/problems.py
# ─────────────────────────────────────────────────────────────────────────────
# Problem setup functions.
#
# Each function defines a structural optimization problem by specifying:
#   - normals: where boundary conditions (fixed supports) are applied
#   - forces:  where external loads are applied
#   - density: target material volume fraction
#
# The design space is a rectangular grid of (width × height) finite elements.
# Nodes are indexed columnwise from left to right.
#
# normals and forces are arrays of shape (width+1, height+1, 2), where:
#   - axis 0: x-position of nodes (0 to width)
#   - axis 1: y-position of nodes (0 to height)
#   - axis 2: [horizontal DOF, vertical DOF]
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np


class ObjectView:
    """Simple namespace object. Converts a dict to attribute access."""
    def __init__(self, d):
        self.__dict__ = d


def get_args(normals, forces, density=0.4, filter_width=1, extra_filter_width=None):
    """
    Build the full argument object for a structural optimization problem.

    This collects all problem parameters into a single namespace that gets
    passed through the physics and optimization code.

    Args:
        normals:            (width+1, height+1, 2) boundary condition array
        forces:             (width+1, height+1, 2) applied force array
        density:            target material volume fraction (0 to 1)
        filter_width:       Gaussian smoothing radius for density field
        extra_filter_width: override filter_width (used for Condition D)

    Returns:
        ObjectView with all problem parameters as attributes
    """
    width  = normals.shape[0] - 1   # number of elements in x-direction
    height = normals.shape[1] - 1   # number of elements in y-direction

    # Degrees of freedom: each node has 2 (horizontal + vertical displacement)
    fixdofs  = np.flatnonzero(normals.ravel())  # fixed (constrained) DOFs
    alldofs  = np.arange(2 * (width + 1) * (height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))  # free DOFs

    fw = extra_filter_width if extra_filter_width is not None else filter_width

    params = {
        # Material properties (standard values from Andreassen 2010)
        "young":     1.0,
        "young_min": 1e-9,
        "poisson":   0.3,
        "g":         0,

        # Optimization constraints
        "density":   density,
        "xmin":      0.001,
        "xmax":      1.0,

        # Problem geometry
        "nelx":      width,
        "nely":      height,
        "mask":      1,          # 1 = no masking; can be (nelx, nely) array

        # Physics parameters
        "penal":     3.0,        # SIMP penalization (drives binary designs)
        "filter_width": fw,      # Gaussian smoothing radius

        # Boundary conditions
        "freedofs":  freedofs,
        "fixdofs":   fixdofs,
        "forces":    forces.ravel(),

        # Logging
        "opt_steps":   80,
        "print_every": 10,
    }
    return ObjectView(params)


# ─────────────────────────────────────────────────────────────────────────────
# MBB BEAM (Messerschmitt-Bölkow-Blohm beam)
# ─────────────────────────────────────────────────────────────────────────────

def mbb_beam(width=80, height=25, density=0.4, **kwargs):
    """
    Classic symmetric cantilever beam benchmark.

    Setup (half-beam, mirrored for visualization):
      - Force applied downward at top-left corner (beam center)
      - Horizontal fixity along left edge (symmetry condition)
      - Vertical fixity at bottom-right corner (wall support)

    This is the most commonly used benchmark in structural optimization.
    Small grid makes it fast to run; well-studied so results are easy to compare.

    Grid: width × height finite elements
    """
    normals = np.zeros((width + 1, height + 1, 2))
    normals[-1, -1, 1] = 1   # vertical fixity: bottom-right corner
    normals[0, :, 0]   = 1   # horizontal fixity: entire left edge (symmetry)

    forces = np.zeros((width + 1, height + 1, 2))
    forces[0, 0, 1] = -1     # downward unit force at top-left (beam center)

    return normals, forces, density


# ─────────────────────────────────────────────────────────────────────────────
# MULTISTORY BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def multistory_building(width=64, height=128, density=0.2, interval=32, **kwargs):
    """
    Tall building with uniform floor loads at regular vertical intervals.

    Setup:
      - Vertical fixity along the entire bottom edge (ground)
      - Horizontal fixity along the right edge (lateral stability)
      - Downward forces uniformly distributed across all floor levels

    This is where Hoyer et al. claim the largest improvement from NN
    reparameterization, because the problem has strong multi-scale structure:
    large load-bearing columns at macro scale + intricate bracing at micro scale.

    Grid: width × height finite elements
    """
    normals = np.zeros((width + 1, height + 1, 2))
    normals[:, -1, 1] = 1    # vertical fixity: entire bottom edge (ground)
    normals[-1, :, 0] = 1    # horizontal fixity: right edge (lateral support)

    forces = np.zeros((width + 1, height + 1, 2))
    # Apply downward unit force at each floor level (uniformly distributed)
    forces[:, ::interval, 1] = -1 / width

    return normals, forces, density


# ─────────────────────────────────────────────────────────────────────────────
# CAUSEWAY BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def causeway_bridge(width=96, height=96, density=0.08, deck_level=0.2, **kwargs):
    """
    Bridge supported at the sides with a deck load at a fixed height.

    Setup:
      - Vertical fixity at bottom-right corner
      - Horizontal fixity along both left and right edges
      - Uniform downward load along the deck level

    This problem encourages arch-like solutions: material is concentrated into
    efficient load-bearing arches, which is a topologically complex solution
    that tests whether the optimizer/parameterization can find the right topology.

    Grid: width × height finite elements
    """
    normals = np.zeros((width + 1, height + 1, 2))
    normals[-1, -1, 1] = 1   # vertical fixity: bottom-right corner
    normals[-1, :, 0]  = 1   # horizontal fixity: right edge
    normals[0, :, 0]   = 1   # horizontal fixity: left edge

    forces = np.zeros((width + 1, height + 1, 2))
    deck_row = round(height * (1 - deck_level))
    # Uniform downward load along deck row
    forces[:, deck_row, 1] = -1 / width

    return normals, forces, density


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM_BUILDERS = {
    "mbb_beam":            mbb_beam,
    "multistory_building": multistory_building,
    "causeway_bridge":     causeway_bridge,
}


def build_problem(name, config_overrides=None):
    """
    Build a problem by name with optional config overrides.

    Args:
        name:             one of the keys in PROBLEM_BUILDERS
        config_overrides: dict of kwargs to override problem defaults

    Returns:
        args: ObjectView with all problem parameters
    """
    from config import PROBLEMS

    if name not in PROBLEM_BUILDERS:
        raise ValueError(f"Unknown problem: '{name}'. Choose from {list(PROBLEM_BUILDERS.keys())}")

    # Start from default config, then apply any overrides
    problem_config = dict(PROBLEMS[name])
    if config_overrides:
        problem_config.update(config_overrides)

    # Extract filter_width if present (not passed to the builder)
    filter_width       = problem_config.pop("filter_width", 1)
    extra_filter_width = problem_config.pop("extra_filter_width", None)
    problem_config.pop("description", None)

    normals, forces, density = PROBLEM_BUILDERS[name](**problem_config)

    return get_args(
        normals, forces, density,
        filter_width=filter_width,
        extra_filter_width=extra_filter_width,
    )
