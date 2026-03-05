# analysis/visualize.py
# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities.
#
# Generates three types of plots:
#   1. Convergence curves:  compliance vs. step for all conditions on one problem
#   2. Design images:       final density maps for all conditions
#   3. Hypothesis matrix:   bar chart summarizing the key comparisons
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from analysis.metrics import load_all_results


# Colour palette for conditions (consistent across all plots)
CONDITION_COLORS = {
    "A": "#2d3142",  # dark navy    — classical baseline
    "B": "#4f5d75",  # steel blue   — Adam on direct
    "C": "#8d99ae",  # grey blue    — L-BFGS on direct
    "D": "#bfc0c0",  # light grey   — smoothing ablation
    "E": "#ef8354",  # orange       — shallow MLP
    "F": "#e63946",  # red          — deep MLP
    "G": "#f4a261",  # amber        — Fourier MLP
    "H": "#e76f51",  # burnt orange — SIREN
    "I": "#2a9d8f",  # teal         — CNN (Hoyer et al.)
    "J": "#264653",  # dark teal    — CNN + MMA
    "K": "#57cc99",  # mint green   — CNN no skip
    "L": "#c77dff",  # purple       — frozen CNN (killer)
}

CONDITION_LABELS = {
    "A": "A: Direct + MMA (baseline)",
    "B": "B: Direct + Adam",
    "C": "C: Direct + L-BFGS",
    "D": "D: Direct + heavy filter + MMA",
    "E": "E: MLP shallow + Adam",
    "F": "F: MLP deep + Adam",
    "G": "G: Fourier MLP + Adam",
    "H": "H: SIREN + Adam",
    "I": "I: CNN (Hoyer et al.) + Adam",
    "J": "J: CNN + MMA",
    "K": "K: CNN no skip + Adam",
    "L": "L: CNN frozen + Adam (killer)",
}


def plot_convergence_curves(results, problem_name, output_path):
    """
    Plot compliance vs. optimization step for all conditions on one problem.

    Args:
        results:      list of run result dicts (from load_all_results)
        problem_name: filter to this problem
        output_path:  save plot to this path
    """
    problem_results = [r for r in results if r["problem"] == problem_name]
    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for r in sorted(problem_results, key=lambda r: r["condition"]):
        cond  = r["condition"]
        color = CONDITION_COLORS.get(cond, "#888888")
        label = CONDITION_LABELS.get(cond, f"Condition {cond}")
        losses = r.get("loss_history", [])

        if not losses:
            continue

        ax.semilogy(
            range(1, len(losses) + 1),
            losses,
            color=color,
            linewidth=2.0,
            label=label,
            alpha=0.9,
        )

    ax.set_xlabel("Optimization Step", fontsize=13)
    ax.set_ylabel("Compliance (log scale)", fontsize=13)
    ax.set_title(
        f"Convergence Curves — {problem_name.replace('_', ' ').title()}\n"
        f"Lower = better structural design",
        fontsize=14,
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


def plot_design_grid(results, problem_name, output_path):
    """
    Show the final density map for every condition side by side.

    Args:
        results:      list of run result dicts
        problem_name: filter to this problem
        output_path:  save plot to this path
    """
    problem_results = [r for r in results if r["problem"] == problem_name]
    problem_results = sorted(problem_results, key=lambda r: r["condition"])

    if not problem_results:
        print(f"No results for problem: {problem_name}")
        return

    n = len(problem_results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for ax, r in zip(axes, problem_results):
        cond = r["condition"]
        density = np.array(r["final_density"])
        compliance = r["metrics"].get("final_compliance", float("nan"))
        sparsity   = r["metrics"].get("topology_sparsity", float("nan"))

        # Mirror horizontally for MBB beam (it's a half-beam)
        if problem_name == "mbb_beam":
            display = np.concatenate([density[:, ::-1], density], axis=1)
        else:
            display = density

        ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"Cond {cond}\nc={compliance:.2e}  |  sp={sparsity:.2f}",
            fontsize=9,
            pad=4,
        )
        ax.axis("off")

        # Highlight Condition I (reference) and L (killer) with colored borders
        if cond == "I":
            for spine in ax.spines.values():
                spine.set_edgecolor(CONDITION_COLORS["I"])
                spine.set_linewidth(3)
        elif cond == "L":
            for spine in ax.spines.values():
                spine.set_edgecolor(CONDITION_COLORS["L"])
                spine.set_linewidth(3)

    # Hide empty subplots
    for ax in axes[len(problem_results):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Final Designs — {problem_name.replace('_', ' ').title()}\n"
        f"(c = final compliance, sp = topology sparsity)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved design grid: {output_path}")


def plot_hypothesis_matrix(results, output_path):
    """
    Bar chart comparing final compliance across conditions, grouped by problem.

    Visually highlights the key comparisons:
      - Baseline (A) vs. CNN (I) shows total NN benefit
      - Other conditions show which components contribute

    Args:
        results:     list of run result dicts
        output_path: save plot to this path
    """
    # Build data: {problem: {condition: final_compliance}}
    data = {}
    for r in results:
        problem = r["problem"]
        cond    = r["condition"]
        val     = r["metrics"].get("final_compliance", float("nan"))
        if problem not in data:
            data[problem] = {}
        data[problem][cond] = val

    if not data:
        print("No results to plot hypothesis matrix.")
        return

    problems = sorted(data.keys())
    conditions = sorted({c for d in data.values() for c in d.keys()})

    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 7))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        cond_data = data[problem]

        # Normalize by the baseline (Condition A)
        baseline = cond_data.get("A", None)

        bars_x = []
        bars_h = []
        bars_c = []
        bars_l = []

        for cond in conditions:
            if cond not in cond_data:
                continue
            val = cond_data[cond]
            bars_x.append(cond)
            bars_h.append(val)
            bars_c.append(CONDITION_COLORS.get(cond, "#aaaaaa"))
            bars_l.append(CONDITION_LABELS.get(cond, cond))

        x_pos = np.arange(len(bars_x))
        rects = ax.bar(x_pos, bars_h, color=bars_c, edgecolor="white", linewidth=0.8)

        # Draw a horizontal dashed line at the baseline
        if baseline is not None:
            ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1.5,
                       alpha=0.6, label=f"Baseline A: {baseline:.2e}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bars_x, fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Compliance", fontsize=11)
        ax.set_title(
            f"{problem.replace('_', ' ').title()}\n(lower = better)",
            fontsize=12,
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if baseline:
            ax.legend(fontsize=9)

        # Annotate bars with % relative to baseline
        if baseline is not None:
            for rect, val in zip(rects, bars_h):
                pct = (val - baseline) / baseline * 100
                label = f"{pct:+.0f}%"
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    val * 1.01,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    fig.suptitle(
        "Final Compliance by Condition\n"
        "% labels = change relative to Condition A (classical baseline)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hypothesis matrix: {output_path}")


def generate_all_plots(results_dir, plots_dir):
    """
    Generate all plots from saved results.

    Args:
        results_dir: directory containing JSON log files
        plots_dir:   directory to save plots
    """
    from config import PROBLEMS

    results = load_all_results(os.path.join(results_dir, "logs"))
    if not results:
        print("No results found. Run the experiment first.")
        return

    problems = list(PROBLEMS.keys())

    # Convergence curves per problem
    for problem in problems:
        plot_convergence_curves(
            results, problem,
            output_path=os.path.join(plots_dir, f"convergence_{problem}.png")
        )

    # Design grids per problem
    for problem in problems:
        plot_design_grid(
            results, problem,
            output_path=os.path.join(plots_dir, f"designs_{problem}.png")
        )

    # Overall hypothesis comparison
    plot_hypothesis_matrix(
        results,
        output_path=os.path.join(plots_dir, "hypothesis_matrix.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from experiment results.")
    parser.add_argument("--results_dir", default="results", help="Path to results directory")
    parser.add_argument("--plots_dir",   default="results/plots", help="Where to save plots")
    args = parser.parse_args()

    generate_all_plots(args.results_dir, args.plots_dir)
