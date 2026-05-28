# Experiment 3: CNN vs Graph Neural Networks

## Background and Motivation

Experiments 1 and 2 established that the best architecture for neural reparameterization
of structural optimization is a **3-level U-Net with 32 base channels, skip connections,
BatchNorm, and AdamW** (Exp 2 Condition P7, here called R0).

Four properties were identified as driving the CNN's advantage:
1. **Local spatial connectivity** — each output element depends on a local neighbourhood
2. **Weight sharing** — the same transformation is applied at every position
3. **Multi-scale hierarchy** — U-Net encoder-decoder captures global layout and fine detail simultaneously
4. **BatchNorm** — normalises internal activations for stable optimisation

The key question for Experiment 3:

> **Can Graph Neural Networks match or beat the CNN, given that they share the same
> inductive bias properties but implement them through graph message passing
> rather than spatial convolution?**

This is well-motivated because FEM — the physics solver underlying structural optimization —
natively operates on a graph: each finite element is a node, and shared element boundaries
are edges. The CNN approximates this graph with a regular pixel grid. GNNs can express the
physical mesh structure directly.

---

## Experiment Design

**4 conditions × 3 problems = 12 runs.** All conditions use AdamW (weight_decay=0.01,
established as the most stable optimizer in Experiment 2).

### Conditions

| Label | Architecture | Properties | Research Question |
|-------|-------------|-----------|------------------|
| **R0** | CNN: U-Net 3L, 32ch, skip, BN + AdamW | All 4 | Reference (replicates Exp 2 P7) |
| **G1** | Flat GCN: 6 layers, 64ch, BN + AdamW | 1, 2, 4 (no hierarchy) | Does locality + weight sharing alone suffice without multi-scale structure? |
| **G2** | Directional GCN: 6 layers, 64ch, BN + AdamW | 1, 2, 4 + direction | Does adding direction sensitivity (separate W per spatial direction) close the gap? |
| **G3** | Hierarchical GNN: 3L encoder-decoder, 32ch, skip, BN + AdamW | All 4 | Does replicating the full U-Net structure via GCN layers match the CNN? |

### Key Comparisons

| Comparison | What it tests |
|------------|--------------|
| **G1 vs R0** | Is multi-scale hierarchy necessary, or does flat local message passing suffice? |
| **G2 vs G1** | Does direction sensitivity in message passing help (analogous to asymmetric conv kernels)? |
| **G3 vs R0** | Does graph aggregation (mean-pool from neighbours) match convolutional weighting? |
| **G3 vs G1** | Does adding hierarchy to GCN improve performance as much as it does for CNN? |

### Problems

| Problem | Grid | Density | Character |
|---------|------|---------|-----------|
| MBB Beam | 80×25 | 40% | Standard benchmark, generous material budget |
| Multistory Building | 64×128 | 20% | Tall structure, multi-scale load paths |
| Causeway Bridge | 96×96 | 8% | Sparse arch geometry, tight material budget |

---

## Architecture Details

### R0 — CNN Reference (identical to Exp 2 Condition P7)

Standard U-Net: encoder (1→32→64→128→256 channels with MaxPool2d) and decoder
(ConvTranspose2d + skip concat). 3×3 conv filters at every level. ~949k parameters.

### G1 — Flat GCN

```
z (N, 1)  →  GCNLayer(1→64)  →  GCNLayer(64→64) × 5  →  Linear(64→1)  →  Sigmoid  →  (H, W)
```

Each `GCNLayer` computes:
```
h_i = ReLU( BN( W · mean_{j ∈ N(i) ∪ {i}} h_j ) )
```

where N(i) is the set of 4-connected neighbours of node i on the FEM grid.
The weight matrix W is shared across all nodes (translation equivariance).
No spatial pooling: all 6 layers operate at full grid resolution.
**~21k parameters.** Receptive field: 6 hops in each direction.

### G2 — Directional GCN

Same depth and width as G1, but each layer uses **5 separate weight matrices** —
one for each direction (self, up, down, left, right):

```
h_i = ReLU( BN(
    W_self  · h_i
  + W_up    · h_{up(i)}
  + W_down  · h_{down(i)}
  + W_left  · h_{left(i)}
  + W_right · h_{right(i)}
))
```

Boundary nodes receive zero contribution from missing directions.
This is the closest GNN analogue to a CNN conv kernel: a 3×3 CNN kernel encodes 9
directional patterns; the directional GCN encodes 5 (cardinal directions + self).
**~103k parameters** (5× more per layer than G1).

### G3 — Hierarchical GNN (Graph U-Net)

Mirror image of the CNN U-Net, replacing `ConvBlock` with `GCNLayer` at each resolution:

```
Encoder:
  (N0=H×W, 1)   → GCNLayer(1→32)  → avg_pool2d ↓2 → (N1, 32)
  (N1, 32)      → GCNLayer(32→64) → avg_pool2d ↓2 → (N2, 64)
  (N2, 64)      → GCNLayer(64→128)→ avg_pool2d ↓2 → (N3, 128)

Bottleneck:
  (N3, 128)     → GCNLayer(128→256)

Decoder:
  (N3, 256) → bilinear ↑2 → cat skip(128) → GCNLayer(384→128) → (N2, 128)
  (N2, 128) → bilinear ↑2 → cat skip(64)  → GCNLayer(192→64)  → (N1, 64)
  (N1, 64)  → bilinear ↑2 → cat skip(32)  → GCNLayer(96→32)   → (N0, 32)
  (N0, 32)  → Linear(32→1) + Sigmoid → (H, W)
```

At each level, the GCN adjacency is built for that resolution's grid.
**~109k parameters** — identical channel structure to the CNN but 9× fewer params
because GCN uses `Linear(F_in, F_out)` where CNN uses `Conv2d(F_in, F_out, 3×3)`.

### Why GNNs Have Fewer Parameters

The parameter count difference stems from a fundamental structural property:
- **CNN `ConvBlock(A, B)`**: `A × B × 3 × 3 = 9·A·B` kernel parameters per layer
- **GNN `GCNLayer(A, B)`**: `A × B = A·B` linear parameters per layer

A CNN conv layer encodes **9 distinct spatial filter patterns** (one per position in the 3×3 kernel).
A GCN layer encodes **1 aggregation transform** applied uniformly to the mean-pooled neighbourhood.

This is precisely the inductive bias under test. If G3 (109k params, 1 pattern per layer)
matches R0 (949k params, 9 patterns per layer), it suggests that the _mean neighbourhood
aggregation_ is as effective as _spatially-weighted convolution_, and the CNN's extra capacity
is spent encoding spatial weight patterns that the optimizer doesn't actually need.

---

## Parameter Summary

| Condition | Architecture | Parameters | vs R0 |
|-----------|-------------|-----------|-------|
| R0 (CNN) | U-Net 3L, 32ch | ~949k | — |
| G1 (Flat GCN) | 6-layer GCN, 64ch | ~21k | 45× fewer |
| G2 (Dir. GCN) | 6-layer dir. GCN, 64ch | ~103k | 9× fewer |
| G3 (Hier. GNN) | U-Net style GCN, 32ch | ~109k | 9× fewer |

---

## Hypotheses

| ID | Claim | Confirmed if |
|----|-------|-------------|
| **H-hier** | Multi-scale hierarchy is necessary | G1 (flat, no hierarchy) performs much worse than R0 |
| **H-dir** | Direction sensitivity matters | G2 significantly better than G1 |
| **H-gnn** | Graph aggregation can match convolution | G3 ≈ R0 (within ~10%) |
| **H-graph** | GNNs have the right inductive bias | Any GNN matches or beats R0 on any problem |
| **H-sparse** | GNNs are better on sparse/complex problems | GNN advantage grows for causeway bridge (8% density) |

---

## Output Files

```
results/
├── logs/           — one JSON per run (loss history, final density, metrics)
├── plots/
│   ├── convergence_mbb_beam.png
│   ├── convergence_multistory_building.png
│   ├── convergence_causeway_bridge.png
│   ├── designs_mbb_beam.png
│   ├── designs_multistory_building.png
│   ├── designs_causeway_bridge.png
│   ├── comparison_matrix.png     ← bar chart, % vs R0 reference
│   └── property_heatmap.png      ← compliance ratio across conditions × problems
└── summary.csv     — all metrics in one table
```

---

## How to Run

```powershell
# From the repo root — all 12 runs
python experiment_3_gnn/run_experiment.py

# Single condition
python experiment_3_gnn/run_experiment.py --condition G3

# Single problem
python experiment_3_gnn/run_experiment.py --problem causeway_bridge

# Quick smoke test (40×12 grid, 5 steps — checks imports and forward passes)
python experiment_3_gnn/run_experiment.py --smoke_test

# Resume after interruption
python experiment_3_gnn/run_experiment.py --skip_existing

# Regenerate plots from existing results
python experiment_3_gnn/run_experiment.py --plots_only
```

**No additional dependencies required.** GNNs are implemented from scratch using
only PyTorch's built-in operations (`scatter_add`, `avg_pool2d`, `interpolate`).
The same packages as Experiments 1 & 2 are sufficient:
```
numpy scipy autograd nlopt torch torchvision matplotlib pandas scikit-image
```

---

## References

- Hoyer et al. (2019). *Neural Reparameterization Improves Structural Optimization.*
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
- Gao & Ji (2019). *Graph U-Nets.*
- Andreassen et al. (2010). *Efficient topology optimization in MATLAB using 88 lines of code.*
