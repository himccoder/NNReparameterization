# experiment_2_cnn_deep_dive/visualize.py
# ─────────────────────────────────────────────────────────────────────────────
# Visualization for Experiment 2.
#
# Generates four plot types:
#   1. Convergence curves   — compliance vs. step, one plot per problem
#   2. Design grids         — final density map for each condition
#   3. Comparison matrix    — bar chart of final compliance by condition/problem
#   4. Problem context heatmap — compliance ratio matrix (conditions × problems)
#
# The comparison matrix draws a cross-experiment reference line at the
# Experiment 1 Condition I result if exp1_summary_csv is provided.
#
# This file lives at the experiment root (not inside analysis/) to avoid
# shadowing the parent's analysis package when resolving imports.
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.metrics import load_all_results  # resolves to parent's analysis/


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / LABEL TABLES  (P1–P9)
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_COLORS = {
    "P1": "#2a9d8f",   # teal         — reference (matches Exp 1 Condition I)
    "P2": "#4cc9f0",   # sky blue     — shallow U-Net
    "P3": "#0077b6",   # deep blue    — deep U-Net
    "P4": "#90e0ef",   # pale cyan    — small capacity
    "P5": "#023e8a",   # navy         — large capacity
    "P6": "#adb5bd",   # grey         — no BatchNorm
    "P7": "#f4a261",   # amber        — AdamW
    "P8": "#e76f51",   # burnt orange — SGD + momentum
    "P9": "#e63946",   # red          — RMSprop
}

CONDITION_LABELS = {
    "P1": "P1: U-Net 3L 32ch + Adam  (reference)",
    "P2": "P2: Shallow U-Net 2L + Adam",
    "P3": "P3: Deep U-Net 4L + Adam",
    "P4": "P4: U-Net 3L 16ch (small) + Adam",
    "P5": "P5: U-Net 3L 64ch (large) + Adam",
    "P6": "P6: U-Net 3L no-BN + Adam",
    "P7": "P7: U-Net 3L + AdamW",
    "P8": "P8: U-Net 3L + SGD+mom",
    "P9": "P9: U-Net 3L + RMSprop",
}

ARCH_CONDITIONS = {"P1", "P2", "P3", "P4", "P5", "P6"}
OPT_CONDITIONS  = {"P7", "P8", "P9"}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — CONVERGENCE CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_curves(results, problem_name, output_path):
    """Compliance vs. step for all conditions on one problem."""
    problem_results = [r for r in results if r["problem"] == problem_name]
    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    fig, ax = plt.subplots(figsize=(13, 7))

    for r in sorted(problem_results, key=lambda r: r["condition"]):
        cond   = r["condition"]
        color  = CONDITION_COLORS.get(cond, "#888888")
        label  = CONDITION_LABELS.get(cond, f"Condition {cond}")
        losses = r.get("loss_history", [])
        if not losses:
            continue

        linestyle = "--" if cond in OPT_CONDITIONS else "-"
        ax.semilogy(
            range(1, len(losses) + 1), losses,
            color=color, linewidth=2.0, linestyle=linestyle,
            label=label, alpha=0.9,
        )

    ax.set_xlabel("Optimization Step", fontsize=13)
    ax.set_ylabel("Compliance (log scale)", fontsize=13)
    ax.set_title(
        f"Exp 2 — Convergence Curves: {problem_name.replace('_', ' ').title()}\n"
        "Solid = architecture variants (P1–P6)  |  Dashed = optimizer variants (P7–P9)",
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
    """Final density maps for all conditions side by side."""
    problem_results = sorted(
        [r for r in results if r["problem"] == problem_name],
        key=lambda r: r["condition"],
    )
    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    n     = len(problem_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for ax, r in zip(axes_flat, problem_results):
        cond       = r["condition"]
        density    = np.array(r["final_density"])
        compliance = r["metrics"].get("final_compliance", float("nan"))
        sparsity   = r["metrics"].get("topology_sparsity", float("nan"))

        display = (
            np.concatenate([density[:, ::-1], density], axis=1)
            if problem_name == "mbb_beam" else density
        )

        ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"{cond}\nc={compliance:.2e}  sp={sparsity:.2f}",
            fontsize=9, pad=4,
        )
        ax.axis("off")

        if cond == "P1":
            for spine in ax.spines.values():
                spine.set_edgecolor(CONDITION_COLORS["P1"])
                spine.set_linewidth(3)

    for ax in axes_flat[len(problem_results):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Exp 2 — Final Designs: {problem_name.replace('_', ' ').title()}\n"
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

def plot_comparison_matrix(results, output_path, exp1_summary_csv=None):
    """
    Bar chart of final compliance for every condition, grouped by problem.

    Reference lines:
      - P1 (Experiment 2 reference) — teal dashed
      - Experiment 1 Condition I result — purple dotted (if available)
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

    exp1_ref = {}
    if exp1_summary_csv and os.path.exists(exp1_summary_csv):
        try:
            import pandas as pd
            df1 = pd.read_csv(exp1_summary_csv)
            for _, row in df1[df1["condition"] == "I"].iterrows():
                exp1_ref[row["problem"]] = float(row["final_compliance"])
        except Exception as e:
            print(f"Could not load Exp 1 reference: {e}")

    problems   = sorted(data.keys())
    conditions = sorted({c for d in data.values() for c in d.keys()})

    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 7))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        cond_data = data[problem]
        p1_val    = cond_data.get("P1")

        bars_x, bars_h, bars_c = [], [], []
        for cond in conditions:
            if cond not in cond_data:
                continue
            bars_x.append(cond)
            bars_h.append(cond_data[cond])
            bars_c.append(CONDITION_COLORS.get(cond, "#aaaaaa"))

        x_pos = np.arange(len(bars_x))
        rects = ax.bar(x_pos, bars_h, color=bars_c, edgecolor="white", linewidth=0.8)

        if p1_val is not None:
            ax.axhline(p1_val, color=CONDITION_COLORS["P1"], linestyle="--",
                       linewidth=1.5, alpha=0.8, label=f"P1 (ref): {p1_val:.2e}")
        if problem in exp1_ref:
            ax.axhline(exp1_ref[problem], color="#c77dff", linestyle=":",
                       linewidth=1.5, alpha=0.8,
                       label=f"Exp1 Cond I: {exp1_ref[problem]:.2e}")

        if p1_val is not None:
            for rect, val in zip(rects, bars_h):
                pct = (val - p1_val) / p1_val * 100
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    val * 1.01, f"{pct:+.0f}%",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bars_x, fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Compliance", fontsize=11)
        ax.set_title(
            f"{problem.replace('_', ' ').title()}\n(lower = better)", fontsize=12
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Exp 2 — Final Compliance by Condition\n"
        "% labels = change vs P1 (reference CNN + Adam)\n"
        "Architecture variants: P1–P6  |  Optimizer variants: P7–P9",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison matrix: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — PROBLEM CONTEXT HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_problem_context_heatmap(results, output_path):
    """
    Heatmap: rows = conditions, columns = problems.
    Cell value = final compliance normalised by the P1 value for that problem.

    Green (< 1) = better than the P1 reference for that problem.
    Red   (> 1) = worse.

    This directly answers: does the winning combination change by structural
    problem context?
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
    conditions = sorted({c for d in data.values() for c in d.keys()})

    matrix = np.full((len(conditions), len(problems)), np.nan)
    for j, problem in enumerate(problems):
        p1_val = data[problem].get("P1", np.nan)
        for i, cond in enumerate(conditions):
            val = data[problem].get(cond, np.nan)
            if not np.isnan(p1_val) and p1_val > 0:
                matrix[i, j] = val / p1_val

    fig, ax = plt.subplots(
        figsize=(max(6, len(problems) * 2.5), max(5, len(conditions) * 0.7))
    )

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
                        fontsize=9, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(
        "Compliance / P1 compliance\n(< 1 = better than reference)", fontsize=9
    )
    ax.set_title(
        "Exp 2 — Problem Context Effect\n"
        "Does the best combination change by structural problem?  "
        "Green < 1 = beats P1  |  Red > 1 = worse",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved problem context heatmap: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(results_dir, plots_dir, exp1_summary_csv=None):
    """
    Generate all Experiment 2 plots from saved JSON logs.

    Args:
        results_dir:       path to experiment_2/results/
        plots_dir:         path to experiment_2/results/plots/
        exp1_summary_csv:  optional path to Experiment 1 results/summary.csv
    """
    from config import PROBLEMS  # resolves to experiment_2/config.py

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
        exp1_summary_csv=exp1_summary_csv,
    )
    plot_problem_context_heatmap(
        results,
        output_path=os.path.join(plots_dir, "problem_context_heatmap.png"),
    )
