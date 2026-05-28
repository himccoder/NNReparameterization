# experiment_3_gnn/run_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: CNN vs Graph Neural Networks.
#
# 4 conditions × 3 problems = 12 runs.
#
# Conditions:
#   R0 — CNN reference (best from Exp 2: U-Net 3L 32ch + AdamW)
#   G1 — Flat GCN (6 layers, 64ch, no hierarchy, AdamW)
#   G2 — Directional GCN (6 layers, 64ch, per-direction weights, AdamW)
#   G3 — Hierarchical GNN (3-level Graph U-Net, 32ch, skip, AdamW)
#
# Usage:
#   # All 12 runs from repo root:
#   python experiment_3_gnn/run_experiment.py
#
#   # Single condition:
#   python experiment_3_gnn/run_experiment.py --condition G3
#
#   # Single problem:
#   python experiment_3_gnn/run_experiment.py --problem causeway_bridge
#
#   # Quick sanity check (tiny grid, 5 steps):
#   python experiment_3_gnn/run_experiment.py --smoke_test
#
#   # Regenerate plots only:
#   python experiment_3_gnn/run_experiment.py --plots_only
#
#   # Skip already-completed runs:
#   python experiment_3_gnn/run_experiment.py --skip_existing
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ── sys.path setup ────────────────────────────────────────────────────────────
# _THIS_DIR first  → "import config" resolves to experiment_3_gnn/config.py
# _PARENT_DIR next → physics/, analysis/metrics, etc. resolve to repo root
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)

for _p in [_PARENT_DIR, _THIS_DIR]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
# After loop: [THIS_DIR, PARENT_DIR, ...] — THIS_DIR always first.
# ─────────────────────────────────────────────────────────────────────────────

import time
import argparse
import traceback
import numpy as np

from config import (
    CONDITIONS, CONDITIONS_BY_LABEL, PROBLEMS, SMOKE_TEST_PROBLEM,
    OPTIMIZER_SETTINGS, CHECKPOINT_STEPS, PRINT_EVERY,
    RESULTS_DIR, LOGS_DIR, PLOTS_DIR, SUMMARY_CSV,
)
from physics.problems import build_problem
from analysis.metrics import compute_metrics, save_run_result, build_summary_csv
from visualize import generate_all_plots


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERIZATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_parameterization(condition, args):
    """Instantiate the correct parameterization for a condition."""

    if condition.parameterization == "cnn_reference":
        from parameterizations.cnn_reference import CNNReferenceParameterization
        return CNNReferenceParameterization(args, **condition.param_kwargs)

    elif condition.parameterization == "gnn_variant":
        from parameterizations.gnn_variants import GNNParameterization
        return GNNParameterization(args, **condition.param_kwargs)

    else:
        raise ValueError(
            f"Unknown parameterization '{condition.parameterization}' "
            f"for condition {condition.label}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(condition, parameterization, args, opt_steps_override=None):
    """Run AdamW (or any configured optimizer) for a condition."""
    from optimizers.gradient_optimizer import run_gradient_optimizer

    opt_type = condition.optimizer
    opt_cfg  = dict(OPTIMIZER_SETTINGS[opt_type])

    if opt_steps_override is not None:
        opt_cfg["opt_steps"] = opt_steps_override

    return run_gradient_optimizer(
        parameterization,
        args,
        optimizer_type = opt_type,
        opt_steps      = opt_cfg["opt_steps"],
        lr             = opt_cfg.get("lr", 1e-2),
        print_every    = PRINT_EVERY,
        checkpoints    = CHECKPOINT_STEPS,
        weight_decay   = opt_cfg.get("weight_decay", 0.0),
        momentum       = opt_cfg.get("momentum", 0.9),
        nesterov       = opt_cfg.get("nesterov", True),
        alpha          = opt_cfg.get("alpha", 0.99),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single(condition_label, problem_name, smoke_test=False, verbose=True):
    """Execute one full run: condition × problem. Returns metrics dict or None."""
    condition = CONDITIONS_BY_LABEL[condition_label]

    print(f"\n{'-'*60}")
    print(f"  Condition {condition_label}: {condition.description}")
    print(f"  Problem:   {problem_name}")
    print(f"  Question:  {condition.question}")
    print(f"{'-'*60}")

    try:
        # ── Build problem ──────────────────────────────────────────────────
        if smoke_test:
            from physics.problems import get_args, mbb_beam
            normals, forces, density = mbb_beam(
                width   = SMOKE_TEST_PROBLEM["width"],
                height  = SMOKE_TEST_PROBLEM["height"],
                density = SMOKE_TEST_PROBLEM["density"],
            )
            args = get_args(normals, forces, density)
        else:
            args = build_problem(problem_name)

        if verbose:
            print(f"  Grid: {args.nelx}×{args.nely}  |  "
                  f"Free DOFs: {len(args.freedofs)}")

        # ── Build parameterization ─────────────────────────────────────────
        param = build_parameterization(condition, args)
        if verbose:
            print(f"  Parameterization: {param.description()}")

        # ── Run optimizer ──────────────────────────────────────────────────
        opt_steps_override = 5 if smoke_test else None
        t_start = time.time()

        losses, frames, checkpoint_losses = run_optimizer(
            condition, param, args,
            opt_steps_override=opt_steps_override,
        )

        wall_time = time.time() - t_start

        if verbose:
            if len(losses) > 0:
                print(f"\n  Finished in {wall_time:.1f}s  |  "
                      f"Final compliance: {losses[-1]:.4e}")
            else:
                print(f"\n  Finished in {wall_time:.1f}s (no losses recorded)")

        # ── Compute and save metrics ───────────────────────────────────────
        final_density = (
            frames[-1] if len(frames) > 0
            else np.zeros((args.nely, args.nelx))
        )

        metrics = compute_metrics(
            losses, final_density, wall_time, checkpoint_losses, param.param_count()
        )

        if verbose:
            print(f"  Sparsity: {metrics['topology_sparsity']:.3f}  |  "
                  f"Grey ratio: {metrics['grey_ratio']:.3f}")

        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = save_run_result(
            condition_label, problem_name,
            metrics, losses, final_density,
            logs_dir=LOGS_DIR,
        )
        if verbose:
            print(f"  Saved: {log_path}")

        return metrics

    except Exception as e:
        print(f"\n  Run FAILED: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FULL EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_full_experiment(
    condition_filter = None,
    problem_filter   = None,
    smoke_test       = False,
    skip_existing    = False,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    conditions_to_run = (
        [CONDITIONS_BY_LABEL[condition_filter]] if condition_filter
        else CONDITIONS
    )
    problems_to_run = (
        [problem_filter] if problem_filter
        else list(PROBLEMS.keys())
    )

    total     = len(conditions_to_run) * len(problems_to_run)
    completed = 0
    failed    = 0

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3 — CNN vs Graph Neural Networks")
    print(f"  {len(conditions_to_run)} conditions × {len(problems_to_run)} problems = {total} runs")
    if smoke_test:
        print(f"  MODE: SMOKE TEST (tiny grid, 5 steps)")
    print(f"{'='*60}")

    for condition in conditions_to_run:
        for problem_name in problems_to_run:

            if skip_existing and _run_exists(condition.label, problem_name):
                print(f"\n  Skipping {condition.label} × {problem_name} (already exists)")
                completed += 1
                continue

            result = run_single(condition.label, problem_name, smoke_test=smoke_test)

            if result is not None:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3 COMPLETE")
    print(f"  Completed: {completed}/{total}  |  Failed: {failed}")
    print(f"{'='*60}")

    df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)

    if not df.empty:
        _print_exp3_summary(df)

    print("\nGenerating plots...")
    try:
        exp2_csv = os.path.join(_PARENT_DIR, "experiment_2_cnn_deep_dive", "results", "summary.csv")
        generate_all_plots(
            RESULTS_DIR, PLOTS_DIR,
            exp2_summary_csv=exp2_csv if os.path.exists(exp2_csv) else None,
        )
    except Exception as e:
        print(f"Plot generation failed: {e}")
        traceback.print_exc()

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  Summary: {SUMMARY_CSV}")
    print(f"  Plots:   {PLOTS_DIR}/")


def _run_exists(condition_label, problem_name):
    if not os.path.exists(LOGS_DIR):
        return False
    for fname in os.listdir(LOGS_DIR):
        if (fname.startswith(f"run_{condition_label}_{problem_name}_")
                and fname.endswith(".json")):
            return True
    return False


def _print_exp3_summary(df):
    """Print condensed results comparing GNN conditions to R0 reference."""
    print("\n" + "="*70)
    print("EXPERIMENT 3 — RESULTS SUMMARY")
    print("  Compliance values relative to R0 (CNN reference)")
    print("="*70)

    problems = df["problem"].unique()

    for problem in sorted(problems):
        print(f"\n--- {problem} ---")
        prob_df = df[df["problem"] == problem].copy()

        r0_rows = prob_df[prob_df["condition"] == "R0"]
        if r0_rows.empty:
            print("  R0 (CNN reference) not yet run.")
            continue
        r0_val = float(r0_rows["final_compliance"].iloc[-1])
        print(f"  R0 (CNN reference): {r0_val:.4e}")

        for cond in ["G1", "G2", "G3"]:
            rows = prob_df[prob_df["condition"] == cond]
            if rows.empty:
                print(f"  {cond}: not yet run")
                continue
            val  = float(rows["final_compliance"].iloc[-1])
            pct  = (val - r0_val) / r0_val * 100
            sign = "better" if pct < 0 else "worse"
            npar = int(rows["n_params"].iloc[-1]) if "n_params" in rows.columns else "?"
            print(f"  {cond}: {val:.4e}  ({abs(pct):.1f}% {sign} than R0)  [{npar:,} params]")

    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: CNN vs Graph Neural Networks.\n"
            "4 conditions × 3 problems = 12 runs.\n\n"
            "R0 — CNN reference (best from Exp 2)\n"
            "G1 — Flat GCN (locality + weight sharing, no hierarchy)\n"
            "G2 — Directional GCN (adds direction sensitivity to G1)\n"
            "G3 — Hierarchical GNN (full Graph U-Net analogue of CNN)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--condition",     type=str, default=None,
                        help="Run only this condition (e.g. --condition G3). Default: all 4.")
    parser.add_argument("--problem",       type=str, default=None,
                        help="Run only this problem (e.g. --problem mbb_beam). Default: all 3.")
    parser.add_argument("--smoke_test",    action="store_true",
                        help="Tiny grid (40×12), 5 steps. Checks imports and forward passes.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs that already have a saved log file.")
    parser.add_argument("--plots_only",    action="store_true",
                        help="Regenerate plots from existing results without re-running.")

    args = parser.parse_args()

    if args.plots_only:
        exp2_csv = os.path.join(_PARENT_DIR, "experiment_2_cnn_deep_dive", "results", "summary.csv")
        generate_all_plots(
            RESULTS_DIR, PLOTS_DIR,
            exp2_summary_csv=exp2_csv if os.path.exists(exp2_csv) else None,
        )
        df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)
        if not df.empty:
            _print_exp3_summary(df)
    else:
        run_full_experiment(
            condition_filter = args.condition,
            problem_filter   = args.problem,
            smoke_test       = args.smoke_test,
            skip_existing    = args.skip_existing,
        )
