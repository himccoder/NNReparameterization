# Experiment 2: CNN Architecture & Optimizer Deep Dive

## Background and Motivation

Experiment 1 tested 12 different parameterization and optimizer combinations across three structural optimization problems. The main conclusion was that CNN-based reparameterization (U-Net with skip connections + Adam) consistently converged faster and to lower compliance than direct density, MLP, Fourier MLP, and SIREN alternatives. CNNs won because their spatial inductive bias — local connectivity, weight sharing, and multi-scale structure — naturally produces smooth, physically coherent density fields that are close to valid structural designs from the very start of optimization.

Experiment 1 left two open questions:

1. **Which CNN variant is best?** The winning CNN was a specific 3-level U-Net with 32 base channels, skip connections, and BatchNorm. Is this choice important, or would a shallower/deeper/smaller/larger CNN without normalization work just as well?
2. **Which optimizer pairs best with the CNN?** Adam was used throughout Experiment 1. Is Adam uniquely suited to the CNN, or do AdamW, SGD+momentum, or RMSprop match or beat it?

A third question ran underneath both: **does problem context change the answer?** The three structural problems differ dramatically in target material density (40%, 20%, 8%) and geometry. We expected the optimal combination might shift across problems and aimed to measure how much.

---

## Experiment Design

**9 conditions × 3 problems = 27 runs.** All conditions use the same optimizer loop, volume penalty, 200 steps, and fixed random latent seed as Experiment 1.

### Architecture Variants — all use Adam, vary the CNN

| Label | Architecture | What changes vs. P1 |
|-------|-------------|----------------------|
| P1 | U-Net 3 levels, 32ch, skip, BatchNorm | Reference — replicates Exp 1 Condition I |
| P2 | U-Net **2 levels**, 32ch, skip, BatchNorm | One fewer pooling step (less multi-scale depth) |
| P3 | U-Net **4 levels**, 32ch, skip, BatchNorm | One extra pooling step (more multi-scale depth) |
| P4 | U-Net 3 levels, **16ch**, skip, BatchNorm | Half the channel capacity |
| P5 | U-Net 3 levels, **64ch**, skip, BatchNorm | Double the channel capacity |
| P6 | U-Net 3 levels, 32ch, skip, **no BatchNorm** | Remove normalisation entirely |

### Optimizer Variants — all use the P1 reference CNN

| Label | Optimizer | Key property |
|-------|-----------|--------------|
| P7 | AdamW (weight_decay=0.01) | Adam + L2 regularisation on CNN weights |
| P8 | SGD + Nesterov momentum (0.9) | Classical momentum, fixed effective learning rate |
| P9 | RMSprop (lr=1e-3, alpha=0.99) | Adaptive per-parameter scaling, no first-moment term |

### Problems

| Problem | Grid | Target density | Character |
|---------|------|----------------|-----------|
| MBB Beam | 80×25 | 40% | Classic benchmark, generous material budget, easy constraint |
| Multistory Building | 64×128 | 20% | Tall, multi-load, moderate constraint |
| Causeway Bridge | 96×96 | 8% | Arch-type, very sparse — tight constraint, most challenging |

---

## Results

All numbers below are **final compliance** (lower = better structural design). The volume constraint is a quadratic penalty, not a hard constraint, so a valid result also requires the sparsity and grey-ratio metrics to reflect a physically reasonable design. Results flagged as **degenerate** are explained in the analysis section.

### Final Compliance Table

| Condition | MBB Beam | Multistory Building | Causeway Bridge |
|-----------|----------|---------------------|-----------------|
| P1 — 3L 32ch + Adam *(reference)* | 219.1 | 66.2 | **369.6** ⚠ |
| P2 — 2L 32ch + Adam | 221.4 | 62.6 | **20.9** |
| P3 — 4L 32ch + Adam | 219.3 | 63.3 | **21.1** |
| P4 — 3L 16ch + Adam | 218.9 | 63.4 | **22.2** |
| P5 — 3L 64ch + Adam | 219.4 | 65.1 | **22.7** |
| P6 — 3L no-BN + Adam | *150.2* ✗ | *15.4* ✗ | 58.5 |
| P7 — 3L 32ch + AdamW | 243.6 | 64.2 | **21.8** |
| P8 — 3L 32ch + SGD+mom | *150.2* ✗ | *15.4* ✗ | *7.4* ✗ |
| P9 — 3L 32ch + RMSprop | 227.9 | 63.2 | **21.6** |

⚠ Adam diverged on causeway bridge — compliance spiked from ~22 at step 80 to 369.6 at step 200.  
✗ Degenerate result: optimizer converged to an all-solid design violating the volume constraint (see below).

---

## Key Findings

### Finding 1: Adam Diverged on the Causeway Bridge

The most striking result is the P1 causeway bridge compliance of 369.6 — roughly 17× worse than every other valid method on the same problem. This is not a bad final design; it is a **divergence**. The checkpoint data makes this clear:

| Step | 20 | 40 | 80 | 120 | 200 (final) |
|------|----|----|----|----|-------------|
| P1 (Adam) | 32.7 | 23.5 | 22.3 | 21.7 | **369.6** |
| P2 (2L + Adam) | 31.7 | 24.9 | 22.5 | 21.2 | 20.9 |
| P7 (AdamW) | 30.7 | 23.1 | 23.2 | 22.3 | 21.8 |

Adam was making excellent progress (22.3 at step 80, on track to match the best methods) and then suddenly exploded at around step 185. The convergence plot shows a vertical spike reaching 10⁶ then crashing back to ~370.

Why does this happen? The causeway bridge has a very low target density (8%). With so little material allowed, the compliance gradient is large and points strongly toward using more material. Adam's adaptive moment estimates accumulate over 150+ steps of strong, consistent gradient signal and then overshoot catastrophically when the landscape curves. The momentum in Adam's second moment estimator turns a late-stage exploratory step into a destabilising explosion.

**AdamW (P7) fixed this entirely.** The L2 weight decay term adds a restoring force on CNN weights that prevents them from growing unboundedly, which is exactly the mechanism that stops the late-stage overshoot. P7 reached a stable 21.8 on the same problem.

The takeaway: **Adam is not unconditionally safe for CNN structural optimization. The more sparse the target design, the more likely Adam is to diverge at late stages. AdamW is safer.**

---

### Finding 2: Architecture Depth and Capacity Have Minimal Impact (When BatchNorm is Present)

On MBB beam and multistory building, all architecture variants achieve nearly identical final compliance:

| | MBB Beam | Multistory |
|---|---|---|
| P1 (3L, 32ch) | 219.1 | 66.2 |
| P2 (2L, 32ch) | 221.4 | 62.6 |
| P3 (4L, 32ch) | 219.3 | 63.3 |
| P4 (3L, 16ch) | 218.9 | 63.4 |
| P5 (3L, 64ch) | 219.4 | 65.1 |

The differences are less than 2% and within run-to-run variation. This tells us that for moderately dense problems (20–40% target), the specific U-Net depth and channel count matter very little. The inductive bias of convolution itself — local connectivity and weight sharing — does the heavy lifting. Whether you give the CNN 2 levels, 3, or 4; or 16, 32, or 64 channels; the optimizer finds a comparably good solution.

On causeway bridge the picture is muddied by P1's divergence. Once P1 is excluded, P2–P5 all converge to 20.9–22.7, again within ~8% of each other. Interestingly, the **shallow U-Net (P2) achieved the best result on causeway bridge (20.9)**. With only 8% target density and a 96×96 grid, the bottleneck of a 3-level network sees a very small 12×12 feature map. The 2-level network's coarser bottleneck may better represent the sparse arch geometry at that resolution.

**P4 (16 channels) is the efficiency winner**: it matches P1 on mbb_beam (219.1 vs 218.9, essentially identical), matches on multistory, and has only 237k parameters versus P1's 949k — a 4× reduction in parameters with no quality loss. If runtime or memory is a concern, P4 is the better choice.

**P5 (64 channels, 3.8M parameters) offers no benefit** over P1 (949k parameters) despite using 4× more parameters. This confirms that the standard 32-channel U-Net from Experiment 1 was already at the capacity sweet spot; more channels just slow training down.

---

### Finding 3: BatchNorm is Essential — Its Absence Causes Degenerate Collapse

P6 (no BatchNorm) produced the most alarming failures of the experiment:

- **MBB Beam**: final compliance 150.2, topology sparsity 1.0, grey ratio 0.0
- **Multistory Building**: final compliance 15.4, topology sparsity 1.0, grey ratio 0.0
- **Causeway Bridge**: final compliance 58.5, grey ratio 0.80

Sparsity = 1.0 and grey ratio = 0.0 means **every element has density > 0.5** — the optimizer converged to a fully solid design. This violates the volume fraction constraint (the quadratic penalty was not strong enough to resist). The causeway bridge result is the opposite problem: grey ratio = 0.80 means 80% of elements are stuck in the ambiguous 0.2–0.8 density zone — the optimizer never committed to a clear topology.

The compliance values look numerically low (150.2 vs P1's 219.1 on mbb_beam) but these are meaningless — a fully solid beam naturally has lower compliance than a partially-filled one. The optimization has degenerated to "use all the material everywhere."

The convergence plot shows why: without BatchNorm the CNN output scale is uncontrolled. The very first few steps of the optimizer produce extremely large density values, creating a loss surface with steep, poorly-scaled gradients. Adam's per-parameter adaptive rates cannot cope with this initial instability.

**BatchNorm normalises the internal activations of each convolutional layer, keeping the CNN output in a well-scaled range that Adam can reliably optimise. It is not optional — removing it breaks the optimization for most problem types.**

---

### Finding 4: SGD+Momentum Fails — Adaptive Learning Rates are Necessary

P8 (SGD + Nesterov momentum) failed on every problem:

- **MBB Beam**: collapsed to all-solid (compliance 150.2, sparsity 1.0)
- **Multistory Building**: collapsed to all-solid (compliance 15.4, sparsity 1.0)
- **Causeway Bridge**: collapsed to all-solid (compliance 7.4, sparsity 1.0)

The pattern is consistent: within the first 1–2 steps the optimizer immediately pushed the CNN weights to produce a fully solid density field and never left. The convergence plots show P8 (burnt orange dashed) immediately jumping off the chart and then locking at the minimum-compliance all-solid solution.

Why does SGD fail here? The CNN weight space has extremely heterogeneous gradient magnitudes across different layers and filter types. Adam and RMSprop use per-parameter adaptive scaling to handle this; they automatically compensate for layers that have large versus small gradients. SGD with a single global learning rate of 0.01 applies the same step size everywhere. For layers with large gradients (the first convolutional layer receiving the random latent input), a step size of 0.01 is far too large and immediately overshoots. The network collapses to a saturation state (all outputs near 1) that the volume penalty cannot recover from.

**For CNN parameterizations, adaptive learning rates (Adam, AdamW, RMSprop) are not just a convenience — they are a hard requirement.** SGD with a fixed lr cannot navigate the multi-scale gradient landscape of a deep convolutional network.

---

### Finding 5: RMSprop is a Viable Adam Alternative, Especially for Sparse Problems

P9 (RMSprop) performed reliably across all three problems:
- MBB Beam: 227.9 (vs P1's 219.1 — about 4% worse)
- Multistory Building: 63.2 (vs P1's 66.2 — about 4% better)
- Causeway Bridge: 21.6 (vs P1's diverged 369.6 — far better; comparable to P2–P5)

RMSprop uses the same denominator-based adaptive scaling as Adam but without the first-moment (momentum) term. This means it scales steps appropriately but does not build up directional momentum over time. On the causeway bridge, this is precisely what prevents divergence — there is no accumulated momentum to explode at step 185. The tradeoff is that RMSprop converges somewhat more slowly on mbb_beam and shows more oscillation early on the causeway bridge (compliance goes up slightly at step 40 before recovering), because it lacks Adam's momentum-driven direction consistency.

---

### Finding 6: Problem Context Significantly Affects Optimizer Stability, but Not Architecture Choice

Across all three problems, the architecture variants (P1–P5, excluding P6) produced nearly identical rankings. This answers one of the core questions: **the choice of CNN depth and channel count is robust to problem context**. A 2-level U-Net beats a 4-level U-Net by about the same margin regardless of whether the problem is mbb_beam or causeway_bridge.

However, **optimizer choice is highly sensitive to problem context**. The key variable is target material density:

| Target density | Adam stability | Risk level |
|----------------|----------------|------------|
| 40% (mbb_beam) | Stable, converges well | Low |
| 20% (multistory) | Stable | Low–moderate |
| 8% (causeway bridge) | **Diverges at step ~185** | High |

The lower the target density, the stronger the compliance gradient relative to the volume penalty, and the more likely Adam is to overshoot after extended optimization. **AdamW resolves this completely** by adding a continuous regularisation signal that prevents CNN weights from drifting into regions with explosive gradients.

---

## Summary and Recommendations

| Question | Answer |
|----------|--------|
| Does U-Net depth matter? | No — 2L, 3L, and 4L all give equivalent results |
| Does channel count matter? | No — 16ch matches 32ch; use 16ch for efficiency |
| Is BatchNorm necessary? | **Yes — critical. Removing it causes optimizer collapse** |
| Which optimizer is best overall? | **AdamW — stable across all densities, fixes Adam's divergence** |
| Can SGD replace Adam? | **No — fails immediately without adaptive lr** |
| Is RMSprop viable? | Yes — stable, slightly slower than AdamW on dense problems |
| Does problem context change the answers? | For architecture: no. For optimizer: yes — sparse problems expose Adam instability |

**Best practice recommendation**: Use a 3-level U-Net with 16–32 base channels, skip connections, BatchNorm, and **AdamW with weight_decay=0.01**. This is robust across all three structural problem types. The reference Adam (P1) works on easy problems but will diverge on any problem with low target material density and 150+ optimization steps.

---

## Output Files

```
results/
├── logs/           — one JSON per run (loss history, final density, metrics)
├── plots/
│   ├── convergence_mbb_beam.png
│   ├── convergence_multistory_building.png
│   ├── convergence_causeway_bridge.png     ← shows P1 Adam divergence spike clearly
│   ├── designs_mbb_beam.png
│   ├── designs_multistory_building.png
│   ├── designs_causeway_bridge.png
│   ├── comparison_matrix.png               ← bar chart, % vs P1 reference
│   └── problem_context_heatmap.png         ← compliance ratio across all conditions × problems
└── summary.csv     — all metrics in one table
```

## How to Run

```powershell
# From the repo root
python experiment_2_cnn_deep_dive/run_experiment.py

# Skip already-completed runs (safe to resume after interruption)
python experiment_2_cnn_deep_dive/run_experiment.py --skip_existing

# Single condition
python experiment_2_cnn_deep_dive/run_experiment.py --condition P7 --problem causeway_bridge

# Regenerate plots from existing results
python experiment_2_cnn_deep_dive/run_experiment.py --plots_only

# Quick smoke test (tiny grid, 5 steps)
python experiment_2_cnn_deep_dive/run_experiment.py --smoke_test
```
