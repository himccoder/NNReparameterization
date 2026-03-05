# Structural Optimization: Why Do Neural Networks Work?

A research experiment that systematically pulls apart *why* neural networks improve
structural optimization — isolating the optimizer, the architecture, and the network's
built-in spatial biases one piece at a time.

---

## Table of Contents

1. [What is structural optimization?](#what-is-structural-optimization)
2. [The central question](#the-central-question)
3. [Glossary of technical terms](#glossary-of-technical-terms)
4. [How we measure success](#how-we-measure-success)
5. [Background and motivation](#background-and-motivation)
6. [The 12 experiment conditions](#the-12-experiment-conditions)
7. [The 6 hypotheses being tested](#the-6-hypotheses-being-tested)
8. [Problem types](#problem-types)
9. [Project structure](#project-structure)
10. [How to run](#how-to-run)
11. [Reading the results](#reading-the-results)
12. [References](#references)

---

## What is structural optimization?

Imagine you are an engineer who must design a bridge, and you have a fixed amount of
steel. Your goal is to place that steel as efficiently as possible so the bridge is as
stiff and strong as it can be, while using only the material budget you have.

Structural optimization is the computer science version of that problem. The computer
is given:

- a rectangular grid of cells (think of it as a blank canvas),
- boundary conditions (where the structure is fixed to the ground),
- applied loads (where forces push or pull on the structure), and
- a material budget (what fraction of cells can be filled).

The algorithm then decides for every cell: *should this cell be solid material or empty
space?* The best answer is the one where the finished structure bends the least when
forces are applied.

**Compliance** is the number that measures how much the structure bends. A lower
compliance means a stiffer, more efficient structure. It is the single most important
number in this experiment — lower is always better.

---

## The central question

In 2019, researchers (Hoyer et al.) showed that replacing the classical structural
optimization solver with a neural network produced significantly better designs. The
neural network acts as a filter: instead of directly deciding material placement, the
algorithm optimizes a compact internal code and feeds it through the network, which
produces the final design.

This raised an obvious but unanswered question:

> **What part of the neural network setup is actually responsible for the improvement?**
> Is it the network's internal structure? The optimizer used to train it? The way it
> smooths the design? Something else entirely?

This codebase runs 12 carefully designed experiments across 3 different structural
problems to find out.

---

## Glossary of technical terms

### Compliance
A number measuring how much a structure deforms under load. Think of it as
"structural sloppiness" — lower compliance means a stiffer, better design. Every
experiment is trying to minimize compliance.

### MMA — Method of Moving Asymptotes
The classical, industry-standard algorithm for structural optimization. It was
designed specifically for this type of problem and has been the gold standard since
1987. It works by carefully restricting how much the design can change at each step,
using mathematical bounds (the "asymptotes"). Conditions A, C, D, and J use MMA.

### Adam
A popular optimizer from the deep learning world. Instead of using specialized
structural knowledge like MMA does, Adam simply follows the gradient (the direction
of improvement) while automatically adjusting its step size for each parameter. It was
not designed for structural optimization — it is a general-purpose learning algorithm.
Most neural-network conditions (B, E, F, G, H, I, K, L) use Adam.

### L-BFGS
A sophisticated mathematical optimizer that builds up a memory of past gradients to
make better update steps. It is more powerful than plain gradient descent but requires
more memory. Condition C uses L-BFGS to test whether a high-quality classical
optimizer alone can match neural network performance.

### MLP — Multi-Layer Perceptron
The simplest kind of neural network: a stack of layers where each neuron connects to
every neuron in the next layer. An MLP with 3 layers and 64 neurons per layer takes
in a position (x, y coordinates of a cell) and outputs a density value for that cell.
MLPs have no built-in sense of space — they treat each position independently.
Conditions E and F use MLPs.

### CNN — Convolutional Neural Network
A neural network with a fundamentally different structure. Instead of each neuron
connecting to every other neuron, a CNN applies small sliding filters (convolutions)
across the entire grid. This gives the CNN a built-in awareness of local spatial
patterns — neighbouring cells naturally influence each other. This is called a
**spatial inductive bias**: the architecture is hardwired to pay attention to nearby
structure. Conditions I, J, K, and L use CNNs.

### U-Net (skip connections / multi-scale prior)
A specific CNN architecture shaped like the letter U. It first compresses the design
down to a small, abstract representation (the encoder), then expands it back up to
full size (the decoder). Crucially, it adds "skip connections" — direct shortcuts
that pass fine-grained detail from the encoder directly to the decoder.

The result is that the network sees the design at multiple scales simultaneously:
broad, structural patterns *and* fine local details at the same time. This is called
a **multi-scale prior** or **spatial/multi-scale prior** — the architecture is
pre-wired to combine coarse and fine spatial information. Condition K removes the skip
connections to test whether this multi-scale capability is the key ingredient.

### SIREN — Sinusoidal Representation Network
An MLP where every neuron uses a sine wave activation function instead of the typical
ReLU. Sine activations give the network a strong preference for representing smooth,
wave-like patterns at many frequencies. In other words, SIREN has a natural "frequency
prior" — it encodes designs in terms of overlapping waves rather than step functions.
Condition H uses SIREN.

### Fourier MLP
An MLP that first transforms the input coordinates using a bank of sine and cosine
functions at many different frequencies (a Fourier feature encoding). This pre-bakes a
frequency bias into the inputs before the MLP even processes them, mimicking one
aspect of SIREN without changing the network internals. Condition G uses a Fourier MLP.

### Spatial prior / inductive bias
A prior is anything that constrains the search before the optimization begins.
A **spatial prior** is a constraint that comes from the shape or structure of the
network itself: it makes certain types of solutions (e.g. smooth, locally consistent
designs) much easier for the network to express than others (e.g. random, noisy
designs). The CNN's convolutional structure is its spatial prior. The experiment asks:
is this prior the key reason CNNs work better, or is something else going on?

### Frozen weights
In condition L, the CNN's internal parameters (the filter weights) are randomly
initialised and then *never updated*. Only the input code is optimized. The CNN is
used purely as a fixed, random mathematical transformation. If this still beats the
classical method, it means the CNN's architecture alone — not anything it learned —
is responsible for the improvement.

### Parameterization
A mathematical choice about *what you optimize*. In the direct approach you optimize
the density of every cell directly. In the neural network approach you optimize a
compact latent code and use the network to translate it into densities. The network is
the parameterization. A good parameterization can make the optimization landscape
smoother and easier to navigate.

### Latent code
A small vector of numbers that acts as the "seed" fed into the neural network. The
optimization algorithm adjusts this seed, and the network converts it into a full
design. It is called "latent" because the actual design is hidden inside the network's
transformation of this code.

---

## How we measure success

Every experiment produces the same core measurements:

| Metric | What it means | Better = |
|--------|---------------|----------|
| **Final compliance** | Stiffness of the finished design | Lower |
| **Compliance at step 20/40/80** | How fast the design improves | Lower, faster drop |
| **Topology sparsity** | Fraction of cells that are clearly solid or clearly empty | Higher (crisp design) |
| **Wall-clock time** | How long the run took in seconds | Lower |

The **hypothesis matrix plot** (`results/plots/hypothesis_matrix.png`) is the key
figure. It shows at a glance which conditions beat the baseline and by how much.

---

## Background and motivation

### Why would a neural network help here?

Classical structural optimization (MMA) works directly in the space of all possible
density grids. This space is enormous and full of local traps — small basins where
the algorithm gets stuck. The classical approach uses mathematical constraints to
navigate this landscape carefully.

Neural networks change the search space. Instead of searching over raw density grids,
you search over the network's internal codes. Because the network maps these codes to
designs in a smooth, structured way, the resulting optimization landscape can be much
easier to navigate. This is the "reparameterization" idea.

### Why is this experiment important?

Hoyer et al. showed the approach works but did not rigorously isolate *which part*
drives the improvement. Three competing explanations are plausible:

1. **The optimizer**: Adam may simply be better than MMA for this task.
2. **Smoothing**: The network may act as an implicit smoother, preventing noisy designs.
3. **The architecture**: The CNN's spatial structure may encode useful priors about
   what good structural designs look like.

These explanations have different implications. If it is just the optimizer, you do not
need neural networks at all — just swap MMA for Adam. If it is smoothing, a simple
Gaussian filter might be enough. Only if it is the architecture does the neural network
itself deserve credit.

Understanding which explanation is correct matters for:
- deciding when to use neural methods in new engineering problems,
- designing better architectures for structural optimization,
- understanding why "deep image priors" (random CNNs used as image generators) work at
  all — a broader open question in machine learning.

---

## The 12 experiment conditions

Each condition is one specific combination of *how the design is represented* (the
parameterization) and *how it is optimized* (the optimizer). Think of each condition
as one competing approach to the same problem.

| Label | In plain English | Parameterization | Optimizer | Tests |
|-------|-----------------|-----------------|-----------|-------|
| **A** | Classical approach. No neural network. | Direct density | MMA | Baseline |
| **B** | Classical design, modern optimizer. | Direct density | Adam | Is Adam alone better than MMA? |
| **C** | Classical design, powerful math optimizer. | Direct density | L-BFGS | Is any good optimizer sufficient? |
| **D** | Classical design with heavy blurring. | Direct + strong blur | MMA | Does smoothing alone explain gains? |
| **E** | Simplest neural network. | Small MLP (3 layers) | Adam | Does any NN help, or only CNNs? |
| **F** | Deeper neural network, still no spatial awareness. | Deep MLP (8 layers) | Adam | Does more depth (without spatial bias) help? |
| **G** | MLP that sees the design as waves. | MLP + Fourier encoding | Adam | Can wave-based encoding replace convolution? |
| **H** | Wave-based network throughout. | SIREN | Adam | Strong frequency prior without convolution. |
| **I** | Full CNN — the method from the paper. | U-Net CNN | Adam | The paper's claimed best approach. |
| **J** | CNN with classical optimizer. | U-Net CNN | MMA | Does CNN prior help even without Adam? |
| **K** | CNN without the multi-scale shortcuts. | CNN, no skip connections | Adam | Are skip connections necessary inside the CNN? |
| **L** | CNN with completely frozen random weights. | Frozen U-Net CNN | Adam | Does the CNN's *shape alone* explain gains? |

**Condition L is the most important experiment.** The CNN's weights are never
updated — they stay at their random initial values. If this still beats condition A,
the CNN architecture's spatial structure alone is responsible for the improvement,
completely independent of any learning.

---

## The 6 hypotheses being tested

### H1 — It is specifically the CNN's spatial structure
*"CNNs outperform MLPs because of their built-in spatial awareness, not just because
they are neural networks."*

Relevant comparisons: I vs E and F (CNN vs plain MLP), J vs A (CNN + MMA vs direct
+ MMA).

**Why it matters:** If H1 is confirmed, architecture design is what counts. Engineers
should specifically choose spatially-aware architectures for structural tasks.

---

### H2 — It is just smoothing
*"The CNN is acting as a spatial smoother. A simple Gaussian blur applied to direct
density achieves the same effect."*

Relevant comparison: D vs A (blurred direct vs classical baseline).

**Why it matters:** If H2 is confirmed, you do not need neural networks at all — just
add a filter. This would mean the entire neural reparameterization literature is
solving a much simpler problem than it appears.

---

### H3 — It is the optimizer (Adam vs MMA)
*"Adam is simply a better optimizer than MMA for this type of problem. The neural
network is incidental."*

Relevant comparisons: B vs A (Adam on direct density vs MMA on direct density).

**Why it matters:** If H3 is confirmed, the research community should focus on
optimizer improvements rather than network architecture. The neural network is just
a vehicle for using Adam.

---

### H4 — It is implicit regularization / early stopping
*"Neural networks implicitly regularize the solution. The improvement comes from
stopping the optimization before the design overfits to noise."*

Relevant comparison: compliance curves over time — does the CNN's advantage disappear
if MMA is run for many more steps?

**Why it matters:** If H4 is confirmed, classical methods can match neural ones simply
by tuning their convergence criteria.

---

### H5 — It is frequency bias, not convolution
*"The CNN works because it produces designs with the right frequency content (smooth
at large scales, detailed at small scales). You can achieve the same with SIREN or
Fourier MLPs without any convolution."*

Relevant comparisons: G vs I (Fourier MLP vs CNN), H vs I (SIREN vs CNN).

**Why it matters:** If H5 is confirmed, convolution is not the key ingredient — any
architecture with the right frequency bias would work. This has broad implications for
implicit neural representation research.

---

### H6 — The structural prior alone explains everything (the "killer experiment")
*"The CNN with completely frozen random weights still outperforms direct
parameterization. No learning is needed — the architecture's spatial shape is a
sufficient prior for finding good designs."*

Relevant comparison: L vs A (frozen CNN vs classical baseline).

**Why it matters:** This is the most striking possible outcome. It would mean that
the convolution operation itself, applied to any reasonable random weights, imposes a
bias on the design space that happens to align with what good structural designs look
like. This would explain why "deep image prior" methods work across many domains —
not just structural optimization.

---

## Problem types

Each condition is run on three engineering problems of increasing complexity:

### MBB Beam (80 × 25 grid)
The standard benchmark in structural optimization. A horizontal beam is fixed at both
ends and a downward force is applied at the centre. The algorithm must find the
most efficient truss-like structure to carry this load. Fast to run and easy to
interpret visually.

### Multistory Building (64 × 128 grid)
A tall vertical structure with loads applied at each floor level. This is the problem
where Hoyer et al. claimed the largest improvement from neural reparameterization. The
tall, narrow aspect ratio and multiple load points make it structurally more complex
than the beam.

### Causeway Bridge (96 × 96 grid)
A bridge structure with distributed loads along a deck and supports at the sides.
The arch-like geometry and unusual support conditions make this a harder optimization
problem with more potential for local traps.

Running all 12 conditions on all 3 problems gives **36 total runs** (or fewer if using
`--skip_existing`).

---

## Project structure

```
NNReparametrizationResearch/
├── README.md                   ← This file
├── run_experiment.py           ← Main entry point: runs all conditions
├── config.py                   ← All hyperparameters and condition definitions
│
├── physics/                    ← Physical simulation
│   ├── fem.py                  ← Finite element solver (stiffness matrix, displacements)
│   ├── objective.py            ← Compliance calculation + gradients
│   └── problems.py             ← Problem setup: beam, bridge, building
│
├── parameterizations/          ← How the design is represented
│   ├── direct.py               ← Conditions A/B/C/D: raw density grid
│   ├── mlp.py                  ← Conditions E/F: plain neural networks
│   ├── fourier_mlp.py          ← Condition G: MLP with wave-encoded inputs; Condition H: SIREN
│   └── cnn.py                  ← Conditions I/J/K/L: convolutional networks
│
├── optimizers/                 ← How the design is improved step by step
│   ├── mma_optimizer.py        ← Classical structural optimizer (MMA)
│   └── gradient_optimizer.py   ← Deep learning optimizers (Adam, L-BFGS)
│
├── analysis/                   ← Measuring and visualizing results
│   ├── metrics.py              ← Compliance, sparsity, timing, CSV output
│   └── visualize.py            ← Convergence curves, design images, hypothesis matrix
│
└── results/                    ← Created when you run the experiment
    ├── logs/                   ← One JSON file per run
    ├── plots/                  ← Convergence curves and final design images
    └── summary.csv             ← All results in one table
```

---

## How to run

### 1. Install dependencies
```bash
pip install numpy scipy autograd nlopt torch torchvision matplotlib pandas scikit-image
```

### 2. Run the full experiment (all 36 conditions × problems)
```bash
python run_experiment.py
```

### 3. Run a specific condition on a specific problem
```bash
python run_experiment.py --condition I --problem mbb_beam
```
Valid condition labels: A B C D E F G H I J K L  
Valid problem names: `mbb_beam`, `multistory_building`, `causeway_bridge`

### 4. Quick smoke test (small grid, few steps — good for checking everything works)
```bash
$env:PYTHONUTF8=1; python run_experiment.py --smoke_test
```
(The `PYTHONUTF8=1` is only needed on Windows to handle Unicode characters in the
terminal output.)

### 5. Skip runs that already have saved results
```bash
python run_experiment.py --skip_existing
```

### 6. Regenerate plots from existing results
```bash
python analysis/visualize.py --results_dir results/
```

---

## Reading the results

After running, the key outputs are:

| File | What to look for |
|------|-----------------|
| `results/summary.csv` | Full table of all conditions × problems × metrics |
| `results/plots/hypothesis_matrix.png` | The main figure — which conditions beat baseline and by how much |
| `results/plots/convergence_*.png` | How quickly each condition improves over optimization steps |
| `results/plots/designs_*.png` | The final material layout for every condition |

### Key comparisons to read first

| Compare | The question it answers |
|---------|------------------------|
| **L vs A** | Does the CNN's shape alone explain gains? *(H6 — the killer experiment)* |
| **I vs E / F** | Is CNN specifically better, or does any neural network help? *(H1)* |
| **B vs A** | Is Adam alone (no neural network) sufficient? *(H3)* |
| **D vs A** | Is smoothing alone sufficient? *(H2)* |
| **G vs I** | Can frequency encoding replace convolution? *(H5)* |
| **I vs J** | Does the CNN need Adam, or does it work with MMA too? *(H1/H3)* |
| **K vs I** | Do skip connections (multi-scale structure) matter inside the CNN? *(H1)* |

---

## References

- Hoyer et al. (2019). *Neural Reparameterization Improves Structural Optimization.*
- Andreassen et al. (2010). *Efficient topology optimization in MATLAB using 88 lines of code.*
- Svanberg (1987). *The method of moving asymptotes — a new class of globally convergent approximation schemes.*
- Ulyanov et al. (2018). *Deep Image Prior.*
- Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions.*
- Sitzmann et al. (2020). *Implicit Neural Representations with Periodic Activation Functions (SIREN).*
