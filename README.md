# Structural Optimization: Why Do Neural Networks Work?

A systematic ablation study that isolates *which part* of neural reparameterization
actually drives improvement in structural optimization — the optimizer, the smoothing,
or the CNN's spatial inductive bias.

---

## Table of Contents

1. [Background](#background)
2. [The central question](#the-central-question)
3. [Key concepts](#key-concepts)
4. [How we evaluate](#how-we-evaluate)
5. [Hypotheses](#hypotheses)
6. [Experiment conditions](#experiment-conditions)
7. [Problem types](#problem-types)
8. [Project structure](#project-structure)
9. [How to run](#how-to-run)
10. [Reading the results](#reading-the-results)
11. [References](#references)

---

## Background

**Structural optimization** (specifically topology optimization) is the problem of
deciding where to place material in a 2D grid to build the stiffest possible structure
under a fixed material budget. Every cell in the grid gets a density value between 0
(empty) and 1 (solid). The algorithm's job is to find the density assignment that
minimizes **compliance** — the elastic potential energy stored in the structure when
forces are applied. Lower compliance = stiffer structure = better design.

The classical pipeline:

```
x (density grid)  →  FEM solver  →  compliance  →  MMA update  →  repeat
```

`x` is optimized directly. **MMA** (Method of Moving Asymptotes, Svanberg 1987) is a
specialized gradient-based solver designed for this class of problems. It is the
industry standard.

Hoyer et al. (2019) proposed replacing direct density optimization with a CNN
**reparameterization**:

```
z (latent code)  →  CNN(z)  →  x (density grid)  →  FEM solver  →  compliance  →  Adam update
```

Instead of optimizing the density grid `x` directly, you optimize a compact latent
input `z` and pass it through a CNN to produce `x`. The CNN acts as a learned or
structural prior over designs. They showed this significantly reduces compliance on
several benchmark problems — but did not isolate *why*.

---

## The central question

Three competing explanations could account for the improvement:

1. **Optimizer effect** — Adam may just be a better optimizer than MMA for this
   problem, independently of the neural network.
2. **Smoothing effect** — The CNN implicitly smooths the density field, preventing
   noisy intermediate designs. A simple Gaussian filter on direct density might do
   the same thing.
3. **Architectural prior** — The CNN's convolutional structure encodes a spatial
   inductive bias that steers optimization toward better regions of the design space,
   and this bias is the actual driver.

These have different implications. If it's (1), you don't need a neural network at
all. If it's (2), a filter is enough. Only if it's (3) does the architecture itself
deserve credit. This experiment disentangles them.

---

## Key concepts

**Compliance** — The scalar objective being minimized. Formally, `C = u^T K u` where
`u` is the displacement vector and `K` is the global stiffness matrix assembled by
FEM. Lower is better.

**MMA (Method of Moving Asymptotes)** — A classical constrained optimizer tailored
for topology optimization. Uses first-order gradient information but restricts each
update via per-variable asymptotes to ensure stable convergence. The baseline solver.

**Adam** — Standard deep learning optimizer. No structural-optimization-specific
knowledge. Used here both as a standalone optimizer (on direct density) and to train
the neural parameterizations.

**L-BFGS** — A quasi-Newton method that builds an approximation to the Hessian from
recent gradients. More powerful than first-order methods; included to test whether any
strong general optimizer matches CNN performance.

**Parameterization** — The choice of what variables to optimize and how they map to
the density field. Direct parameterization optimizes `x` directly. Neural
parameterizations optimize a latent `z` and use a network `f: z → x`.

**Spatial inductive bias** — A CNN's convolutional layers enforce local spatial
correlations: each output value depends only on a local receptive field of the input.
This makes it structurally easier for the network to produce spatially smooth, locally
coherent designs — which happen to be the kind that structural optimizers converge to.

**Multi-scale prior (U-Net / skip connections)** — The CNN used here is U-Net style:
an encoder compresses the latent code down to a small bottleneck, then a decoder
expands it back to full resolution. Skip connections between encoder and decoder layers
pass fine-scale spatial detail across at every resolution. The result is a network
with simultaneous awareness of coarse global structure and fine local detail. Condition
K ablates this by removing the skip connections.

**SIREN (Sinusoidal Representation Network)** — An implicit neural network where every
activation function is `sin(ωx)` instead of ReLU. The periodic activations give SIREN
a strong spectral bias toward smooth, band-limited signals. Used here as an alternative
frequency-prior architecture without any convolution.

**Fourier MLP** — A standard MLP preceded by a positional encoding layer that maps
input coordinates `(x, y)` to a stack of `sin` and `cos` features at many frequencies
(Tancik et al., 2020). Like SIREN, this imposes a frequency prior on the design, but
without sinusoidal activations throughout.

**Frozen weights (Condition L)** — The CNN's weights are randomly initialized and
never updated. Only the latent input `z` is optimized. This tests whether the CNN
architecture's structural prior is sufficient on its own — i.e., whether it is an
untrained, fixed spatial filter that still shapes the optimization landscape favorably.
This is the most important condition in the experiment.

---

## How we evaluate

Every run records:

| Metric | Definition | Goal |
|--------|-----------|------|
| **Final compliance** | `C = u^T K u` at the last optimization step | Minimize |
| **Compliance @ step 20/40/80** | Convergence speed profile | Lower, faster |
| **Topology sparsity** | Fraction of cells with density > 0.5 | Higher = crisper design |
| **Wall-clock time (s)** | Total runtime of the run | Lower |

All conditions are run with the same material volume fraction constraint and the same
number of FEM evaluations (normalized across optimizers). Comparisons are made on
final compliance as the primary metric.

---

## Hypotheses

| ID | Claim | Confirmed if |
|----|-------|-------------|
| **H1** | It's the CNN's spatial/multi-scale prior | CNN (I) beats MLP (E/F) of equal capacity; CNN + MMA (J) also beats direct + MMA (A) |
| **H2** | It's implicit smoothing | Heavily filtered direct density (D) matches CNN (I) |
| **H3** | It's the optimizer | Adam on direct density (B) matches Adam + CNN (I) |
| **H4** | It's implicit regularization / early stopping | Compliance curves show CNN's lead shrinks as MMA runs longer |
| **H5** | It's frequency bias, not convolution | Fourier MLP (G) or SIREN (H) matches CNN (I) |
| **H6** | The architectural prior alone is sufficient | Frozen-weight CNN (L) beats classical baseline (A) |

**H6 is the "killer experiment."** If a randomly initialized, weight-frozen CNN
outperforms direct density + MMA, then no learning is needed — the architectural
prior itself restructures the optimization landscape. This connects directly to the
"deep image prior" phenomenon (Ulyanov et al., 2018), where random CNNs act as
surprisingly effective image priors.

---

## Experiment conditions

12 conditions formed by crossing parameterizations with optimizers:

| Label | Parameterization | Architecture | Optimizer | Tests |
|-------|-----------------|-------------|-----------|-------|
| A | Direct density | — | MMA | Classical baseline |
| B | Direct density | — | Adam | H3: optimizer effect |
| C | Direct density | — | L-BFGS | H3: stronger optimizer |
| D | Direct + Gaussian filter (width=3) | — | MMA | H2: smoothing effect |
| E | MLP | 3 layers, 64 hidden, ReLU | Adam | H1: any NN vs CNN |
| F | MLP | 8 layers, 64 hidden, ReLU | Adam | H1: depth without spatial bias |
| G | MLP + Fourier encoding | 4 layers, ReLU, 32 frequencies | Adam | H5: frequency prior without conv |
| H | SIREN | 4 layers, sin activations, ω=30 | Adam | H5: periodic prior without conv |
| I | CNN (U-Net) | Encoder-decoder + skip connections | Adam | Reference: Hoyer et al. method |
| J | CNN (U-Net) | Encoder-decoder + skip connections | MMA | H1/H3: CNN prior + classical optimizer |
| K | CNN (no skip) | Encoder-decoder, no shortcuts | Adam | H1: is multi-scale structure necessary? |
| L | CNN (frozen) | U-Net, weights fixed at init | Adam | **H6: architectural prior only** |

---

## Problem types

Three benchmark problems of increasing difficulty, each run with all 12 conditions:

| Problem | Grid | Vol. fraction | Description |
|---------|------|--------------|-------------|
| **MBB Beam** | 80 × 25 | 0.40 | Standard cantilever beam benchmark. Fixed at both ends, point load at centre. Fast convergence, easy to compare visually. |
| **Multistory Building** | 64 × 128 | 0.20 | Tall structure with distributed floor loads. Hoyer et al. report the largest gains here. Tall aspect ratio makes multi-scale structure more important. |
| **Causeway Bridge** | 96 × 96 | 0.08 | Arch-type bridge with distributed deck load. Low volume fraction makes the problem harder — less material, more complex load paths. |

---

## Project structure

```
NNReparametrizationResearch/
├── run_experiment.py           ← Main entry point
├── config.py                   ← All hyperparameters and condition definitions
│
├── physics/
│   ├── fem.py                  ← FEM solver: stiffness matrix assembly, displacement solve
│   ├── objective.py            ← Compliance + autograd-compatible gradient computation
│   └── problems.py             ← Problem setup: boundary conditions, loads, grids
│
├── parameterizations/
│   ├── direct.py               ← Conditions A/B/C/D
│   ├── mlp.py                  ← Conditions E/F
│   ├── fourier_mlp.py          ← Condition G (Fourier MLP) and H (SIREN)
│   └── cnn.py                  ← Conditions I/J/K/L
│
├── optimizers/
│   ├── mma_optimizer.py        ← MMA via NLopt
│   └── gradient_optimizer.py   ← Adam and L-BFGS via PyTorch
│
├── analysis/
│   ├── metrics.py              ← Metric collection, JSON logging, CSV aggregation
│   └── visualize.py            ← Convergence curves, design grids, hypothesis matrix
│
└── results/
    ├── logs/                   ← Per-run JSON: all metrics and compliance history
    ├── plots/                  ← Convergence curves, design images, hypothesis matrix
    └── summary.csv             ← Aggregated results table
```

---

## How to run

```bash
# Install dependencies
pip install numpy scipy autograd nlopt torch torchvision matplotlib pandas scikit-image

# Full experiment (all 36 runs)
python run_experiment.py

# Single condition + problem
python run_experiment.py --condition I --problem mbb_beam

# Quick smoke test (tiny grid, 5 steps — just verifies nothing is broken)
python run_experiment.py --smoke_test       # Linux/macOS
$env:PYTHONUTF8=1; python run_experiment.py --smoke_test   # Windows

# Resume: skip runs that already have saved results
python run_experiment.py --skip_existing

# Regenerate plots from existing logs
python analysis/visualize.py --results_dir results/
```

---

## Reading the results

### Key comparisons (priority order)

| Comparison | Hypothesis | What a large gap means |
|------------|-----------|------------------------|
| L vs A | H6 | Architectural prior alone is the driver |
| I vs E, F | H1 | CNN specifically (not just any NN) is needed |
| B vs A | H3 | Optimizer is the driver |
| D vs A | H2 | Smoothing is the driver |
| G/H vs I | H5 | Frequency prior replicates CNN without convolution |
| J vs A | H1 | CNN prior helps even with MMA |
| K vs I | H1 | Multi-scale skip connections are necessary within CNN |

### Output files

- `results/summary.csv` — full results table
- `results/plots/hypothesis_matrix.png` — the key comparison figure
- `results/plots/convergence_*.png` — compliance curves per problem
- `results/plots/designs_*.png` — final density maps for all conditions

---

## References

- Hoyer et al. (2019). *Neural Reparameterization Improves Structural Optimization.*
- Andreassen et al. (2010). *Efficient topology optimization in MATLAB using 88 lines of code.*
- Svanberg (1987). *The method of moving asymptotes.*
- Ulyanov et al. (2018). *Deep Image Prior.*
- Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions.*
- Sitzmann et al. (2020). *Implicit Neural Representations with Periodic Activation Functions (SIREN).*
