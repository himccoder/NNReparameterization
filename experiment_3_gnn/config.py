# experiment_3_gnn/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Configuration for Experiment 3: CNN vs Graph Neural Networks.
#
# Experiment 2 established the optimal CNN: 3-level U-Net, 32 base channels,
# skip connections, BatchNorm, AdamW (weight_decay=0.01). Experiment 3 asks:
#   Can GNNs match or beat this CNN, given that they share the same inductive
#   bias properties (local connectivity, weight sharing) but implement them
#   through graph message passing rather than spatial convolution?
#
# The four properties that made CNNs win (from Exp 1 & 2):
#   1. Local spatial connectivity — each output depends on a local neighborhood
#   2. Weight sharing — same transformation applied at every position
#   3. Multi-scale hierarchy — U-Net encoder-decoder captures global + local
#   4. BatchNorm — normalises activations for stable optimisation
#
# Each GNN condition isolates which of these properties is sufficient:
#   G1 (Flat GCN):        properties 1, 2, 4 — no hierarchy
#   G2 (Directional GCN): properties 1, 2, 4 — adds direction sensitivity
#   G3 (Hierarchical GNN): properties 1, 2, 3, 4 — full structural analogue
#
# Conditions:
#   R0 — CNN reference (best from Exp 2: P7, 3L U-Net 32ch + AdamW)
#   G1 — Flat GCN (6 layers, 64 hidden channels, BatchNorm, AdamW)
#   G2 — Directional GCN (6 layers, 64 hidden channels, BatchNorm, AdamW)
#   G3 — Hierarchical GNN (3-level U-Net-style, 32 base channels, BatchNorm, AdamW)
#
# 4 conditions × 3 problems = 12 runs.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS / MATERIAL CONSTANTS  (identical to Experiments 1 & 2)
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
# All conditions use AdamW (best from Exp 2, stable across all densities).
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZER_SETTINGS = dict(
    adamw = dict(
        opt_steps    = 200,
        lr           = 1e-2,
        weight_decay = 0.01,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM DEFINITIONS  (identical to Experiments 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

PROBLEMS = {
    "mbb_beam": dict(
        width   = 80,
        height  = 25,
        density = 0.4,
        description = "Classic MBB cantilever beam. 80×25, 40% density.",
    ),
    "multistory_building": dict(
        width   = 64,
        height  = 128,
        density = 0.2,
        description = "Tall building with floor loads. 64×128, 20% density.",
    ),
    "causeway_bridge": dict(
        width   = 96,
        height  = 96,
        density = 0.08,
        description = "Arch bridge with deck load. 96×96, 8% density.",
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
    label:            str
    parameterization: str   # "cnn_reference" | "gnn_variant"
    optimizer:        str   # "adamw" for all conditions in Exp 3
    param_kwargs:     dict  = field(default_factory=dict)
    description:      str  = ""
    question:         str  = ""


CONDITIONS = [

    # ── CNN REFERENCE ─────────────────────────────────────────────────────────
    # Replicates Experiment 2 Condition P7 (the best overall from Exp 2):
    # 3-level U-Net, 32 base channels, skip connections, BatchNorm, AdamW.

    Condition(
        label            = "R0",
        parameterization = "cnn_reference",
        optimizer        = "adamw",
        param_kwargs     = dict(
            arch          = "standard",
            base_channels = 32,
            use_skip      = True,
            use_batchnorm = True,
        ),
        description = "CNN reference: U-Net 3L 32ch, skip+BN, AdamW. Replicates Exp 2 best (P7).",
        question    = "Reference point — all GNN conditions compared to this.",
    ),

    # ── GNN VARIANTS ──────────────────────────────────────────────────────────

    Condition(
        label            = "G1",
        parameterization = "gnn_variant",
        optimizer        = "adamw",
        param_kwargs     = dict(
            gnn_type      = "flat_gcn",
            hidden        = 64,
            n_layers      = 6,
            use_batchnorm = True,
        ),
        description = "Flat GCN: 6-layer standard GCN on 4-connected grid, 64 hidden ch, BN, AdamW.",
        question    = (
            "Does locality + weight sharing alone (no hierarchy) suffice? "
            "G1 vs R0 isolates whether multi-scale structure is necessary."
        ),
    ),

    Condition(
        label            = "G2",
        parameterization = "gnn_variant",
        optimizer        = "adamw",
        param_kwargs     = dict(
            gnn_type      = "directional_gcn",
            hidden        = 64,
            n_layers      = 6,
            use_batchnorm = True,
        ),
        description = (
            "Directional GCN: 6-layer GCN with per-direction weight matrices "
            "(up/down/left/right/self), 64 hidden ch, BN, AdamW."
        ),
        question    = (
            "Does adding direction sensitivity (separate W per spatial direction) "
            "close the gap with CNN? G2 vs G1 isolates whether directionality matters."
        ),
    ),

    Condition(
        label            = "G3",
        parameterization = "gnn_variant",
        optimizer        = "adamw",
        param_kwargs     = dict(
            gnn_type      = "hierarchical_gnn",
            base_channels = 32,
            use_batchnorm = True,
        ),
        description = (
            "Hierarchical GNN: 3-level encoder-decoder with GCN layers, "
            "grid pooling/unpooling, skip connections, 32 base ch, BN, AdamW."
        ),
        question    = (
            "Does replicating the CNN's U-Net structure via GCN layers match the CNN? "
            "G3 vs R0 isolates whether the convolution mechanism vs graph aggregation is key."
        ),
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
