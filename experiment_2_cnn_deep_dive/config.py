# experiment_2_cnn_deep_dive/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Configuration for Experiment 2: CNN Architecture & Optimizer Deep Dive.
#
# Experiment 1 established that CNN + Adam converges fastest across all
# structural problems. Experiment 2 asks the follow-up questions:
#   1. Which CNN architecture variant works best?
#   2. Which optimizer pairs best with the CNN?
#   3. Does the answer depend on the structural problem context?
#
# Conditions P1–P9 are organised into two groups:
#   Architecture variants (P1–P6): all use Adam, vary the CNN structure.
#   Optimizer variants   (P7–P9): all use the reference CNN, vary optimizer.
#
# P1 is the reference condition — it replicates Experiment 1 Condition I
# (U-Net 3 levels, 32 base channels, skip connections, BatchNorm, Adam) so
# results are directly comparable across experiments.
#
# Usage:
#   python run_experiment.py                        # all 9 × 3 = 27 runs
#   python run_experiment.py --condition P3         # single condition
#   python run_experiment.py --problem mbb_beam     # single problem
#   python run_experiment.py --smoke_test           # quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS / MATERIAL CONSTANTS  (identical to Experiment 1 for comparability)
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS = dict(
    young       = 1.0,
    young_min   = 1e-9,
    poisson     = 0.3,
    penal       = 3.0,
    filter_width= 1,
)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZER_SETTINGS = dict(
    adam = dict(
        opt_steps = 200,
        lr        = 1e-2,
    ),
    adamw = dict(
        opt_steps    = 200,
        lr           = 1e-2,
        weight_decay = 0.01,    # L2 regularisation on CNN weights
    ),
    sgd = dict(
        opt_steps = 200,
        lr        = 1e-2,
        momentum  = 0.9,        # Nesterov-style momentum
    ),
    rmsprop = dict(
        opt_steps = 200,
        lr        = 1e-3,       # RMSprop is sensitive to lr; 1e-3 is safer
        alpha     = 0.99,       # smoothing constant for the squared gradient average
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM DEFINITIONS  (identical to Experiment 1 for comparability)
# ─────────────────────────────────────────────────────────────────────────────

PROBLEMS = {
    "mbb_beam": dict(
        width   = 80,
        height  = 25,
        density = 0.4,
        description = "Classic MBB cantilever beam. Small grid, fast to run, standard benchmark."
    ),
    "multistory_building": dict(
        width   = 64,
        height  = 128,
        density = 0.2,
        description = "Tall structure with floor-level loads."
    ),
    "causeway_bridge": dict(
        width   = 96,
        height  = 96,
        density = 0.08,
        description = "Arch-type bridge with distributed deck load and side supports."
    ),
}

SMOKE_TEST_PROBLEM = dict(
    width   = 40,
    height  = 12,
    density = 0.4,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    label:          str
    parameterization: str   # "cnn_variant" for all exp-2 conditions
    optimizer:      str     # one of: adam, adamw, sgd, rmsprop
    param_kwargs:   dict = field(default_factory=dict)
    description:    str  = ""
    question:       str  = ""   # the research question this condition addresses


CONDITIONS = [

    # ── ARCHITECTURE VARIANTS (all Adam) ──────────────────────────────────

    Condition(
        label            = "P1",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "standard",   # 3-level U-Net, 32 channels, BN, skip
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Reference: U-Net 3 levels, 32ch, skip+BN, Adam. Replicates Exp 1 Condition I.",
        question         = "Reference point — all other conditions compared to this.",
    ),
    Condition(
        label            = "P2",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "shallow",    # 2-level U-Net
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Shallow U-Net (2 encoder levels), 32ch, skip+BN, Adam.",
        question         = "Does reducing multi-scale depth hurt? How important is the 3rd pooling level?",
    ),
    Condition(
        label            = "P3",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "deep",       # 4-level U-Net
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Deep U-Net (4 encoder levels), 32ch, skip+BN, Adam.",
        question         = "Does adding more multi-scale levels beyond 3 provide further gains?",
    ),
    Condition(
        label            = "P4",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 16,           # half the capacity of P1
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "U-Net 3 levels, 16ch (small), skip+BN, Adam.",
        question         = "Does reducing channel capacity degrade performance? Is 32ch necessary?",
    ),
    Condition(
        label            = "P5",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 64,           # double the capacity of P1
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "U-Net 3 levels, 64ch (large), skip+BN, Adam.",
        question         = "Does doubling channel capacity improve performance or just slow it down?",
    ),
    Condition(
        label            = "P6",
        parameterization = "cnn_variant",
        optimizer        = "adam",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= False,        # remove BatchNorm
        ),
        description      = "U-Net 3 levels, 32ch, skip, NO BatchNorm, Adam.",
        question         = "How important is BatchNorm? Does normalisation help in structural opt?",
    ),

    # ── OPTIMIZER VARIANTS (all reference architecture P1) ─────────────────

    Condition(
        label            = "P7",
        parameterization = "cnn_variant",
        optimizer        = "adamw",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Reference CNN + AdamW (weight_decay=0.01).",
        question         = "Does L2 regularisation on CNN weights help find sparser, cleaner designs?",
    ),
    Condition(
        label            = "P8",
        parameterization = "cnn_variant",
        optimizer        = "sgd",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Reference CNN + SGD with momentum (0.9).",
        question         = "Can classical momentum-SGD match Adam? Is adaptive lr necessary for CNN?",
    ),
    Condition(
        label            = "P9",
        parameterization = "cnn_variant",
        optimizer        = "rmsprop",
        param_kwargs     = dict(
            arch         = "standard",
            base_channels= 32,
            use_skip     = True,
            use_batchnorm= True,
        ),
        description      = "Reference CNN + RMSprop (lr=1e-3, alpha=0.99).",
        question         = "Does RMSprop's adaptive scaling (without Adam's momentum) match Adam?",
    ),
]

CONDITIONS_BY_LABEL = {c.label: c for c in CONDITIONS}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────

import os

_EXP_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_EXP_DIR, "results")
LOGS_DIR    = os.path.join(_EXP_DIR, "results", "logs")
PLOTS_DIR   = os.path.join(_EXP_DIR, "results", "plots")
SUMMARY_CSV = os.path.join(_EXP_DIR, "results", "summary.csv")

CHECKPOINT_STEPS = [20, 40, 80, 120]
PRINT_EVERY      = 10
