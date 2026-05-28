# experiment_3_gnn/visualize.py
# ─────────────────────────────────────────────────────────────────────────────
# Visualization for Experiment 3: CNN vs Graph Neural Networks.
#
# Generates four plot types:
#   1. Convergence curves   — compliance vs. step per problem, R0 vs G1/G2/G3
#   2. Design grids         — final density maps for all 4 conditions
#   3. Comparison matrix    — bar chart of final compliance (% vs R0)
#   4. Property heatmap     — compliance ratio matrix (conditions × problems)
#      with a row added for Exp 2 P7 (same as R0) and Exp 1 Cond I if available
#
# Colour scheme:
#   R0 (CNN)  — teal     (matching Exp 2's P7 colour)
#   G1 (Flat GCN)        — coral red
#   G2 (Directional GCN) — amber orange
#   G3 (Hierarchical)    — purple
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.metrics import load_all_results


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / LABEL TABLES
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_COLORS = {
    "R0": "#2a9d8f",   # teal    — CNN reference
    "G1": "#e63946",   # red     — Flat GCN
    "G2": "#f4a261",   # amber   — Directional GCN
    "G3": "#7b2d8b",   # purple  — Hierarchical GNN
}

CONDITION_LABELS = {
    "R0": "R0: CNN U-Net 3L 32ch + AdamW  (reference)",
    "G1": "G1: Flat GCN 6L 64ch + AdamW  (no hierarchy)",
    "G2": "G2: Directional GCN 6L 64ch + AdamW",
    "G3": "G3: Hierarchical GNN 3L 32ch + AdamW",
}

CNN_CONDITIONS = {"R0"}
GNN_CONDITIONS = {"G1", "G2", "G3"}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — CONVERGENCE CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_curves(results, problem_name, output_path):
    """Compliance vs. step for all 4 conditions on one problem."""
    problem_results = [r for r in results if r["problem"] == problem_name]
    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for r in sorted(problem_results, key=lambda r: r["condition"]):
        cond   = r["condition"]
        color  = CONDITION_COLORS.get(cond, "#888888")
        label  = CONDITION_LABELS.get(cond, f"Condition {cond}")
        losses = r.get("loss_history", [])
        if not losses:
            continue

        linestyle = "-" if cond in CNN_CONDITIONS else "--"
        lw        = 2.5 if cond == "R0" else 1.8
        ax.semilogy(
            range(1, len(losses) + 1), losses,
            color=color, linewidth=lw, linestyle=linestyle,
            label=label, alpha=0.9,
        )

    ax.set_xlabel("Optimization Step", fontsize=13)
    ax.set_ylabel("Compliance (log scale)", fontsize=13)
    ax.set_title(
        f"Exp 3 — Convergence: {problem_name.replace('_', ' ').title()}\n"
        "Solid = CNN reference (R0)  |  Dashed = GNN variants (G1–G3)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — DESIGN GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def plot_design_grid(results, problem_name, output_path):
    """Final density maps for all 4 conditions."""
    problem_results = sorted(
        [r for r in results if r["problem"] == problem_name],
        key=lambda r: r["condition"],
    )
    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    n     = len(problem_results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3))
    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [list(axes)]
    else:
        axes = [list(row) for row in axes]
    axes_flat = [ax for row in axes for ax in row]

    for ax, r in zip(axes_flat, problem_results):
        cond       = r["condition"]
        density    = np.array(r["final_density"])
        compliance = r["metrics"].get("final_compliance", float("nan"))
        sparsity   = r["metrics"].get("topology_sparsity", float("nan"))
        n_params   = r["metrics"].get("n_params", 0)

        display = (
            np.concatenate([density[:, ::-1], density], axis=1)
            if problem_name == "mbb_beam" else density
        )

        ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"{cond}\nc={compliance:.2e}  sp={sparsity:.2f}\n{n_params:,} params",
            fontsize=8, pad=4,
        )
        ax.axis("off")

        color = CONDITION_COLORS.get(cond, "#888888")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for ax in axes_flat[len(problem_results):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Exp 3 — Final Designs: {problem_name.replace('_', ' ').title()}\n"
        "(c = compliance, sp = sparsity)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved design grid: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — COMPARISON MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_matrix(results, output_path, exp2_summary_csv=None):
    """
    Bar chart of final compliance for each condition, grouped by problem.

    Reference lines:
      - R0 (Exp 3 CNN reference) — teal dashed
      - Exp 2 P7 result — grey dotted (if exp2_summary_csv available)
    """
    data = {}
    for r in results:
        problem = r["problem"]
        cond    = r["condition"]
        val     = r["metrics"].get("final_compliance", float("nan"))
        data.setdefault(problem, {})[cond] = val

    if not data:
        print("No results to plot comparison matrix.")
        return

    exp2_ref = {}
    if exp2_summary_csv and os.path.exists(exp2_summary_csv):
        try:
            import pandas as pd
            df2 = pd.read_csv(exp2_summary_csv)
            for _, row in df2[df2["condition"] == "P7"].iterrows():
                exp2_ref[row["problem"]] = float(row["final_compliance"])
        except Exception as e:
            print(f"Could not load Exp 2 reference: {e}")

    problems   = sorted(data.keys())
    conditions = ["R0", "G1", "G2", "G3"]

    fig, axes = plt.subplots(1, len(problems), figsize=(5.5 * len(problems), 7))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        cond_data = data[problem]
        r0_val    = cond_data.get("R0")

        bars_x, bars_h, bars_c = [], [], []
        for cond in conditions:
            if cond not in cond_data:
                continue
            bars_x.append(cond)
            bars_h.append(cond_data[cond])
            bars_c.append(CONDITION_COLORS.get(cond, "#aaaaaa"))

        x_pos = np.arange(len(bars_x))
        rects = ax.bar(x_pos, bars_h, color=bars_c, edgecolor="white", linewidth=0.8, width=0.6)

        if r0_val is not None:
            ax.axhline(r0_val, color=CONDITION_COLORS["R0"], linestyle="--",
                       linewidth=1.5, alpha=0.8, label=f"R0 (ref): {r0_val:.2e}")
        if problem in exp2_ref:
            ax.axhline(exp2_ref[problem], color="#adb5bd", linestyle=":",
                       linewidth=1.5, alpha=0.8,
                       label=f"Exp2 P7: {exp2_ref[problem]:.2e}")

        if r0_val is not None:
            for rect, val in zip(rects, bars_h):
                if np.isfinite(val):
                    pct = (val - r0_val) / r0_val * 100
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        val * 1.01, f"{pct:+.0f}%",
                        ha="center", va="bottom", fontsize=9,
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bars_x, fontsize=12, fontweight="bold")
        ax.set_ylabel("Final Compliance", fontsize=11)
        ax.set_title(
            f"{problem.replace('_', ' ').title()}\n(lower = better)", fontsize=12
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Exp 3 — Final Compliance: CNN vs GNN Variants\n"
        "% labels = change vs R0 (CNN reference)\n"
        "Teal = CNN  |  Red/Amber/Purple = GNN variants",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison matrix: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — PROPERTY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_property_heatmap(results, output_path):
    """
    Heatmap: rows = conditions (R0, G1, G2, G3), columns = problems.
    Cell = final compliance normalised by R0 value for that problem.
    Green (< 1) = better than R0.  Red (> 1) = worse.

    This shows at a glance which GNN property (no-hierarchy / directional /
    full-hierarchy) is most effective across the three structural problem types.
    """
    data = {}
    for r in results:
        problem = r["problem"]
        cond    = r["condition"]
        val     = r["metrics"].get("final_compliance", float("nan"))
        data.setdefault(problem, {})[cond] = val

    if not data:
        return

    problems   = sorted(data.keys())
    conditions = ["R0", "G1", "G2", "G3"]

    matrix = np.full((len(conditions), len(problems)), np.nan)
    for j, problem in enumerate(problems):
        r0_val = data[problem].get("R0", np.nan)
        for i, cond in enumerate(conditions):
            val = data[problem].get(cond, np.nan)
            if not np.isnan(r0_val) and r0_val > 0:
                matrix[i, j] = val / r0_val

    fig, ax = plt.subplots(figsize=(max(6, len(problems) * 2.5), 4))

    valid = matrix[~np.isnan(matrix)]
    vmin  = min(0.85, float(np.min(valid))) if len(valid) else 0.85
    vmax  = max(1.15, float(np.max(valid))) if len(valid) else 1.15

    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([p.replace("_", "\n") for p in problems], fontsize=11)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions], fontsize=9
    )

    for i in range(len(conditions)):
        for j in range(len(problems)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(
        "Compliance / R0\n(< 1.0 = better than CNN reference)", fontsize=9
    )
    ax.set_title(
        "Exp 3 — Problem Context Effect\n"
        "Compliance normalised to R0 (CNN).  Green < 1 = GNN beats CNN  |  Red > 1 = worse",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved property heatmap: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(results_dir, plots_dir, exp2_summary_csv=None):
    """Generate all Experiment 3 plots from saved JSON logs."""
    from config import PROBLEMS

    results = load_all_results(os.path.join(results_dir, "logs"))
    if not results:
        print("No results found. Run the experiment first.")
        return

    problems = list(PROBLEMS.keys())

    for problem in problems:
        plot_convergence_curves(
            results, problem,
            output_path=os.path.join(plots_dir, f"convergence_{problem}.png"),
        )
        plot_design_grid(
            results, problem,
            output_path=os.path.join(plots_dir, f"designs_{problem}.png"),
        )

    plot_comparison_matrix(
        results,
        output_path=os.path.join(plots_dir, "comparison_matrix.png"),
        exp2_summary_csv=exp2_summary_csv,
    )
    plot_property_heatmap(
        results,
        output_path=os.path.join(plots_dir, "property_heatmap.png"),
    )
