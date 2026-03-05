# Structural Optimization: Why Do Neural Networks Work?

## Overview

This experiment is a systematic ablation study designed to answer a deceptively simple question:

> **When neural networks improve structural optimization, what is actually responsible — the optimizer, the architecture, or the inductive biases baked into the network's structure?**

This directly interrogates the central claim of Hoyer et al. (2019), *"Neural Reparameterization Improves Structural Optimization"*, which argues that CNN reparameterization — not the optimizer — is the key driver of improvement.

---

## Background

Structural optimization places material in a 2D design space to resist applied forces as efficiently as possible. The quality of a design is measured by its **compliance** (elastic potential energy) — lower is better.

The classical pipeline:
```
x (density grid) → physics simulation → compliance → MMA update
```

The neural reparameterization pipeline (Hoyer et al.):
```
z (latent code) → CNN(z) → x (density grid) → physics simulation → compliance → Adam update
```

The question is: **which part of this change drives the improvement?**

---

## Hypotheses Being Tested

| ID | Hypothesis | What would confirm it |
|----|------------|----------------------|
| H1 | It's the CNN's spatial/multi-scale prior | CNN beats MLP of equal capacity regardless of optimizer |
| H2 | It's just implicit smoothing | Direct params + strong Gaussian filter matches CNN |
| H3 | It's the optimizer (Adam vs MMA) | Adam on direct params matches CNN+Adam |
| H4 | It's implicit regularization / early stopping | CNN advantage disappears with more MMA steps |
| H5 | It's frequency bias, not convolution | Fourier-featured MLP matches CNN |
| H6 | The prior is structural (not learned) | CNN with **frozen random weights** still beats direct |

---

## Experiment Conditions (12 total)

| Label | Parameterization       | Architecture          | Optimizer | Primary Purpose                        |
|-------|------------------------|-----------------------|-----------|----------------------------------------|
| A     | Direct density         | —                     | MMA       | Classical baseline                     |
| B     | Direct density         | —                     | Adam      | Isolate optimizer effect               |
| C     | Direct density         | —                     | L-BFGS    | Isolate optimizer effect               |
| D     | Direct + heavy filter  | —                     | MMA       | Isolate smoothing as mechanism         |
| E     | MLP shallow            | 3 layers, ReLU        | Adam      | Is any NN better, or specifically CNN? |
| F     | MLP deep               | 8 layers, ReLU        | Adam      | Depth without spatial bias             |
| G     | MLP + Fourier features | 4 layers, ReLU        | Adam      | Spatial frequency prior without conv   |
| H     | SIREN                  | 4 layers, Sine        | Adam      | Strong frequency prior, no conv        |
| I     | CNN (Hoyer et al.)     | U-Net style           | Adam      | The paper's claimed winner             |
| J     | CNN                    | U-Net style           | MMA       | Best prior + best classical optimizer  |
| K     | CNN no skip connections| Plain conv            | Adam      | Ablate multi-scale within CNN          |
| L     | CNN frozen weights     | U-Net frozen          | Adam      | **Pure architectural prior** (killer experiment) |

**Condition L is the most important.** If a CNN with completely frozen random weights still
outperforms direct parameterization, the spatial inductive bias alone explains the gains —
not any learning that happens inside the network.

---

## Problem Types

Each condition is run on three problem types of increasing complexity:

1. **MBB Beam** — Simple cantilever beam. Fast, standard benchmark. Small grid (80×25).
2. **Multistory Building** — Tall structure, complex load distribution. Where Hoyer et al. claim the largest gains. Medium grid (128×256).
3. **Causeway Bridge** — Arch-type structure, irregular loads. Medium grid (128×128).

---

## Metrics Collected

For each condition × problem:
- **Final compliance** — primary quality metric (lower = better)
- **Compliance at steps 20, 40, 80** — convergence speed profile
- **Topology sparsity** — fraction of elements with density > 0.5 (structural clarity)
- **Wall-clock time** — computational cost

---

## Project Structure

```
stopt_experiment/
├── README.md                   ← You are here
├── run_experiment.py           ← Main entry point: runs all conditions
├── config.py                   ← All hyperparameters and condition definitions
│
├── physics/
│   ├── __init__.py
│   ├── fem.py                  ← Finite element method (stiffness matrix, displacements)
│   ├── objective.py            ← Compliance objective + autograd-compatible physics
│   └── problems.py             ← Problem setup: MBB beam, bridge, building
│
├── parameterizations/
│   ├── __init__.py
│   ├── direct.py               ← Condition A/B/C/D: direct density parameterization
│   ├── mlp.py                  ← Conditions E/F: shallow and deep MLPs
│   ├── fourier_mlp.py          ← Condition G: MLP with Fourier positional encoding
│   ├── siren.py                ← Condition H: sinusoidal representation network
│   └── cnn.py                  ← Conditions I/J/K/L: CNN (U-Net style, variants)
│
├── optimizers/
│   ├── __init__.py
│   ├── mma_optimizer.py        ← MMA via NLopt wrapper
│   └── gradient_optimizer.py   ← Adam and L-BFGS via PyTorch
│
├── analysis/
│   ├── __init__.py
│   ├── metrics.py              ← Metric collection and storage
│   └── visualize.py            ← Plot results, convergence curves, design images
│
└── results/                    ← Output directory (created at runtime)
    ├── logs/                   ← Per-run JSON logs
    ├── plots/                  ← Convergence curves and design visualizations
    └── summary.csv             ← Aggregated results table
```

---

## How to Run

### Install dependencies
```bash
pip install numpy scipy autograd nlopt torch torchvision matplotlib pandas scikit-image
```

### Run the full experiment
```bash
python run_experiment.py
```

### Run a specific condition on a specific problem
```bash
python run_experiment.py --condition I --problem mbb_beam
```

### Run a quick smoke test (fewer steps, small grid)
```bash
python run_experiment.py --smoke_test
```

### Generate plots from existing results
```bash
python analysis/visualize.py --results_dir results/
```

---

## Reading the Results

After running, check:
- `results/summary.csv` — full table of all conditions × problems × metrics
- `results/plots/convergence_*.png` — compliance over steps per problem
- `results/plots/designs_*.png` — final material density maps
- `results/plots/hypothesis_matrix.png` — the key comparison figure

### Interpreting the key comparisons

| Compare | Tells you |
|---------|-----------|
| L vs A  | Does the CNN spatial prior alone explain gains? |
| I vs E/F | Is it specifically CNN, or any NN? |
| D vs A  | Is smoothing alone sufficient? |
| B vs A  | Is Adam alone sufficient? |
| I vs J  | Does CNN benefit from Adam specifically? |
| G vs I  | Is frequency bias achievable without convolution? |

---

## References

- Hoyer et al. (2019). *Neural Reparameterization Improves Structural Optimization.*
- Andreassen et al. (2010). *Efficient topology optimization in MATLAB using 88 lines of code.*
- Svanberg (1987). *The method of moving asymptotes.*
- Ulyanov et al. (2018). *Deep Image Prior.*
- Mordvintsev et al. (2018). *Differentiable Image Parameterizations.*
