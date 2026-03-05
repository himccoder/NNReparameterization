# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the ablation experiment.
# All hyperparameters, condition definitions, and problem settings live here.
# Edit this file to add/remove conditions or change the experiment scope.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS / MATERIAL CONSTANTS
# These are standard values from Andreassen et al. (2010).
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS = dict(
    young       = 1.0,      # Young's modulus (material stiffness)
    young_min   = 1e-9,     # Minimum stiffness (prevents singular stiffness matrix)
    poisson     = 0.3,      # Poisson's ratio (lateral contraction under tension)
    penal       = 3.0,      # SIMP penalization exponent (drives densities toward 0/1)
    filter_width= 1,        # Gaussian filter width (spatial smoothing of density)
)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZER_SETTINGS = dict(
    # MMA (Method of Moving Asymptotes) — classical structural optimization solver
    mma = dict(
        opt_steps = 80,
    ),
    # Adam — standard deep learning optimizer
    adam = dict(
        opt_steps = 200,    # More steps since each step is cheaper than MMA
        lr        = 1e-2,
    ),
    # L-BFGS — quasi-Newton method, good for smooth objectives
    lbfgs = dict(
        opt_steps = 200,
        lr        = 1.0,
        max_iter  = 20,     # Inner iterations per step
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM DEFINITIONS
# Each problem specifies its grid size and target material density.
# ─────────────────────────────────────────────────────────────────────────────

PROBLEMS = {
    "mbb_beam": dict(
        width   = 80,
        height  = 25,
        density = 0.4,
        description = "Classic MBB cantilever beam. Small grid, fast to run, standard benchmark."
    ),
    "multistory_building": dict(
        width   = 64,       # Reduced from 128 for tractability; scale up if time allows
        height  = 128,
        density = 0.2,
        description = "Tall structure with floor-level loads. Hoyer et al. claim largest NN gains here."
    ),
    "causeway_bridge": dict(
        width   = 96,
        height  = 96,
        density = 0.08,
        description = "Arch-type bridge with distributed deck load and side supports."
    ),
}

# For quick smoke tests, use a tiny version of the MBB beam
SMOKE_TEST_PROBLEM = dict(
    width   = 40,
    height  = 12,
    density = 0.4,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION DEFINITIONS
# Each condition specifies:
#   - parameterization: which density generator to use
#   - optimizer: which update rule to apply
#   - param_kwargs: hyperparameters passed to the parameterization
#   - hypothesis: which hypothesis this condition tests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    label:          str
    parameterization: str       # One of: direct, mlp, fourier_mlp, siren, cnn
    optimizer:      str         # One of: mma, adam, lbfgs
    param_kwargs:   dict = field(default_factory=dict)
    description:    str  = ""
    hypothesis:     str  = ""


CONDITIONS = [

    # ── BASELINES ──────────────────────────────────────────────────────────

    Condition(
        label            = "A",
        parameterization = "direct",
        optimizer        = "mma",
        description      = "Classical baseline: direct density parameterization with MMA.",
        hypothesis       = "Baseline — all other conditions compared against this.",
    ),
    Condition(
        label            = "B",
        parameterization = "direct",
        optimizer        = "adam",
        description      = "Direct density with Adam optimizer.",
        hypothesis       = "H3: Tests whether Adam alone (without NN) explains the gains.",
    ),
    Condition(
        label            = "C",
        parameterization = "direct",
        optimizer        = "lbfgs",
        description      = "Direct density with L-BFGS optimizer.",
        hypothesis       = "H3: Tests whether a stronger optimizer alone explains the gains.",
    ),

    # ── SMOOTHING ABLATION ─────────────────────────────────────────────────

    Condition(
        label            = "D",
        parameterization = "direct",
        optimizer        = "mma",
        param_kwargs     = dict(filter_width=3),  # Much heavier smoothing than default
        description      = "Direct density with aggressive Gaussian smoothing (filter_width=3).",
        hypothesis       = "H2: Tests whether smoothing alone (without NN) explains the gains.",
    ),

    # ── MLP VARIANTS ───────────────────────────────────────────────────────

    Condition(
        label            = "E",
        parameterization = "mlp",
        optimizer        = "adam",
        param_kwargs     = dict(hidden_layers=3, hidden_dim=64),
        description      = "Shallow MLP (3 layers, ReLU). No spatial inductive bias.",
        hypothesis       = "H1: Is any NN better, or specifically a spatially-biased CNN?",
    ),
    Condition(
        label            = "F",
        parameterization = "mlp",
        optimizer        = "adam",
        param_kwargs     = dict(hidden_layers=8, hidden_dim=64),
        description      = "Deep MLP (8 layers, ReLU). Depth without spatial/conv bias.",
        hypothesis       = "H1: Does depth alone (without spatial structure) help?",
    ),

    # ── FREQUENCY PRIOR (no convolution) ───────────────────────────────────

    Condition(
        label            = "G",
        parameterization = "fourier_mlp",
        optimizer        = "adam",
        param_kwargs     = dict(hidden_layers=4, hidden_dim=64, num_frequencies=32),
        description      = "MLP with Fourier positional encoding. Spatial frequency bias without conv.",
        hypothesis       = "H5: Can frequency bias alone (without conv) match CNN performance?",
    ),
    Condition(
        label            = "H",
        parameterization = "siren",
        optimizer        = "adam",
        param_kwargs     = dict(hidden_layers=4, hidden_dim=64, omega_0=30.0),
        description      = "SIREN: sinusoidal implicit neural representation. Strong frequency prior.",
        hypothesis       = "H5: Alternative frequency-biased architecture without convolution.",
    ),

    # ── CNN VARIANTS ────────────────────────────────────────────────────────

    Condition(
        label            = "I",
        parameterization = "cnn",
        optimizer        = "adam",
        param_kwargs     = dict(use_skip_connections=True, frozen=False),
        description      = "CNN (U-Net style with skip connections). The Hoyer et al. method.",
        hypothesis       = "Reference: the paper's claimed best method.",
    ),
    Condition(
        label            = "J",
        parameterization = "cnn",
        optimizer        = "mma",
        param_kwargs     = dict(use_skip_connections=True, frozen=False),
        description      = "CNN reparameterization but optimized with MMA (not Adam).",
        hypothesis       = "H1/H3: Does the CNN prior help even with the classical optimizer?",
    ),
    Condition(
        label            = "K",
        parameterization = "cnn",
        optimizer        = "adam",
        param_kwargs     = dict(use_skip_connections=False, frozen=False),
        description      = "CNN without skip connections (no multi-scale feature fusion).",
        hypothesis       = "H1: Is multi-scale structure within the CNN necessary?",
    ),

    # ── THE KILLER EXPERIMENT ───────────────────────────────────────────────

    Condition(
        label            = "L",
        parameterization = "cnn",
        optimizer        = "adam",
        param_kwargs     = dict(use_skip_connections=True, frozen=True),
        description      = "CNN with completely FROZEN random weights. Only the latent input is optimized.",
        hypothesis       = (
            "H6 (killer): If frozen CNN beats direct, the spatial PRIOR alone explains "
            "the gains — not any learning inside the network. This is the most important condition."
        ),
    ),
]

# Convenient lookup by label
CONDITIONS_BY_LABEL = {c.label: c for c in CONDITIONS}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING / OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = "results"
LOGS_DIR     = "results/logs"
PLOTS_DIR    = "results/plots"
SUMMARY_CSV  = "results/summary.csv"

# Log compliance at these step checkpoints (in addition to final)
CHECKPOINT_STEPS = [20, 40, 80]

# Print progress every N steps
PRINT_EVERY = 10
