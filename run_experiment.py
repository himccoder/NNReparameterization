# run_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner.
#
# This script runs all 12 conditions × 3 problems = 36 experimental runs
# and saves the results. Each run:
#   1. Builds the structural optimization problem (forces, boundary conditions)
#   2. Instantiates the parameterization (direct, MLP, CNN, etc.)
#   3. Runs the optimizer (MMA, Adam, or L-BFGS)
#   4. Collects metrics and saves results
#
# Usage:
#   python run_experiment.py                              # full experiment
#   python run_experiment.py --condition I --problem mbb_beam  # single run
#   python run_experiment.py --smoke_test                 # quick test
#   python run_experiment.py --skip_existing              # skip completed runs
#
# Results are saved in:
#   results/logs/      — one JSON file per run
#   results/plots/     — convergence curves and design images
#   results/summary.csv — aggregated results table
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import traceback
import numpy as np

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CONDITIONS, CONDITIONS_BY_LABEL, PROBLEMS, SMOKE_TEST_PROBLEM,
    OPTIMIZER_SETTINGS, CHECKPOINT_STEPS, PRINT_EVERY,
    RESULTS_DIR, LOGS_DIR, PLOTS_DIR, SUMMARY_CSV,
)
from physics.problems import build_problem
from analysis.metrics import compute_metrics, save_run_result, build_summary_csv, print_hypothesis_summary
from analysis.visualize import generate_all_plots


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERIZATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_parameterization(condition, args):
    """
    Instantiate the correct parameterization object for a given condition.

    Args:
        condition: Condition dataclass from config.py
        args:      problem ObjectView

    Returns:
        parameterization object with .to_density(), .initial_params(), etc.
    """
    ptype  = condition.parameterization
    kwargs = condition.param_kwargs

    if ptype == "direct":
        from parameterizations.direct import DirectParameterization
        return DirectParameterization(args)

    elif ptype == "mlp":
        from parameterizations.mlp import MLPParameterization
        return MLPParameterization(args, **kwargs)

    elif ptype == "fourier_mlp":
        from parameterizations.fourier_mlp import FourierMLPParameterization
        return FourierMLPParameterization(args, **kwargs)

    elif ptype == "siren":
        from parameterizations.fourier_mlp import SIRENParameterization
        return SIRENParameterization(args, **kwargs)

    elif ptype == "cnn":
        from parameterizations.cnn import CNNParameterization
        return CNNParameterization(args, **kwargs)

    else:
        raise ValueError(f"Unknown parameterization type: '{ptype}'")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(condition, parameterization, args, opt_steps_override=None):
    """
    Run the appropriate optimizer for a given condition.

    Args:
        condition:            Condition dataclass
        parameterization:     instantiated parameterization object
        args:                 problem ObjectView
        opt_steps_override:   override opt_steps (e.g. for smoke test)

    Returns:
        losses:            array of compliance values per step
        frames:            array of density fields per step
        checkpoint_losses: dict of {step: compliance}
    """
    opt_type = condition.optimizer
    opt_cfg  = dict(OPTIMIZER_SETTINGS[opt_type])  # copy to avoid mutation

    if opt_steps_override is not None:
        opt_cfg["opt_steps"] = opt_steps_override

    if opt_type == "mma":
        from optimizers.mma_optimizer import run_mma
        return run_mma(
            parameterization, args,
            opt_steps   = opt_cfg["opt_steps"],
            print_every = PRINT_EVERY,
            checkpoints = CHECKPOINT_STEPS,
        )

    elif opt_type in ("adam", "lbfgs"):
        from optimizers.gradient_optimizer import run_gradient_optimizer
        return run_gradient_optimizer(
            parameterization, args,
            optimizer_type = opt_type,
            opt_steps      = opt_cfg["opt_steps"],
            lr             = opt_cfg.get("lr", 1e-2),
            print_every    = PRINT_EVERY,
            checkpoints    = CHECKPOINT_STEPS,
            lbfgs_max_iter = opt_cfg.get("max_iter", 20),
        )

    else:
        raise ValueError(f"Unknown optimizer type: '{opt_type}'")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single(condition_label, problem_name, smoke_test=False, verbose=True):
    """
    Execute one full experimental run: condition × problem.

    Args:
        condition_label: e.g., "I"
        problem_name:    e.g., "mbb_beam"
        smoke_test:      if True, use tiny grid and few steps
        verbose:         if True, print progress

    Returns:
        metrics dict, or None if the run failed
    """
    condition = CONDITIONS_BY_LABEL[condition_label]

    # ── Print run header ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Condition {condition_label}: {condition.description}")
    print(f"  Problem:   {problem_name}")
    print(f"  Hypothesis: {condition.hypothesis}")
    print(f"{'─'*60}")

    try:
        # ── Build problem ─────────────────────────────────────────────────────
        if smoke_test:
            # Tiny problem for fast smoke testing
            from physics.problems import get_args, mbb_beam
            normals, forces, density = mbb_beam(
                width=SMOKE_TEST_PROBLEM["width"],
                height=SMOKE_TEST_PROBLEM["height"],
                density=SMOKE_TEST_PROBLEM["density"],
            )
            args = get_args(normals, forces, density)
        else:
            # Check if condition D needs extra filter width
            config_overrides = {}
            if condition_label == "D":
                config_overrides["extra_filter_width"] = condition.param_kwargs.get(
                    "filter_width", 3
                )
            args = build_problem(problem_name, config_overrides)

        if verbose:
            print(f"  Grid: {args.nelx}×{args.nely}  |  "
                  f"Nodes: {len(args.forces)}  |  "
                  f"Free DOFs: {len(args.freedofs)}")

        # ── Build parameterization ────────────────────────────────────────────
        param = build_parameterization(condition, args)
        if verbose:
            print(f"  Parameterization: {param.description()}")
            print(f"  Parameters to optimize: {param.param_count():,}")

        # ── Run optimizer ─────────────────────────────────────────────────────
        opt_steps_override = 5 if smoke_test else None
        t_start = time.time()

        losses, frames, checkpoint_losses = run_optimizer(
            condition, param, args,
            opt_steps_override=opt_steps_override,
        )

        wall_time = time.time() - t_start

        if verbose:
            if len(losses) > 0:
                print(f"\n  ✓ Finished in {wall_time:.1f}s  |  "
                      f"Final compliance: {losses[-1]:.4e}")
            else:
                print(f"\n  ✓ Finished in {wall_time:.1f}s (no losses recorded)")

        # ── Compute metrics ───────────────────────────────────────────────────
        final_density = frames[-1] if len(frames) > 0 else np.zeros((args.nely, args.nelx))

        metrics = compute_metrics(
            losses, final_density, wall_time, checkpoint_losses, param.param_count()
        )

        if verbose:
            print(f"  Topology sparsity: {metrics['topology_sparsity']:.3f}  |  "
                  f"Grey ratio: {metrics['grey_ratio']:.3f}")

        # ── Save results ──────────────────────────────────────────────────────
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
        print(f"\n  ✗ Run FAILED: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FULL EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_full_experiment(
    condition_filter=None,
    problem_filter=None,
    smoke_test=False,
    skip_existing=False,
):
    """
    Run all conditions × problems (or a filtered subset).

    Args:
        condition_filter: if set, only run this condition label (e.g. "I")
        problem_filter:   if set, only run this problem (e.g. "mbb_beam")
        smoke_test:       use tiny grids and few steps for quick testing
        skip_existing:    skip runs that already have a saved log
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Determine which conditions and problems to run
    conditions_to_run = (
        [CONDITIONS_BY_LABEL[condition_filter]] if condition_filter
        else CONDITIONS
    )
    problems_to_run = (
        [problem_filter] if problem_filter
        else list(PROBLEMS.keys())
    )

    total = len(conditions_to_run) * len(problems_to_run)
    completed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  STRUCTURAL OPTIMIZATION ABLATION STUDY")
    print(f"  {len(conditions_to_run)} conditions × {len(problems_to_run)} problems = {total} runs")
    if smoke_test:
        print(f"  MODE: SMOKE TEST (tiny grid, few steps)")
    print(f"{'='*60}")

    for condition in conditions_to_run:
        for problem_name in problems_to_run:

            # Check if this run already exists (for --skip_existing)
            if skip_existing and _run_exists(condition.label, problem_name, LOGS_DIR):
                print(f"\n  Skipping {condition.label} × {problem_name} (already exists)")
                completed += 1
                continue

            result = run_single(condition.label, problem_name, smoke_test=smoke_test)

            if result is not None:
                completed += 1
            else:
                failed += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Completed: {completed}/{total}  |  Failed: {failed}")
    print(f"{'='*60}")

    # Build summary CSV
    df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)

    # Print hypothesis analysis
    if not df.empty:
        print_hypothesis_summary(df)

    # Generate all plots
    print("\nGenerating plots...")
    try:
        generate_all_plots(RESULTS_DIR, PLOTS_DIR)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  Summary table: {SUMMARY_CSV}")
    print(f"  Plots:         {PLOTS_DIR}/")


def _run_exists(condition_label, problem_name, logs_dir):
    """Check if a log file already exists for this condition × problem."""
    if not os.path.exists(logs_dir):
        return False
    for fname in os.listdir(logs_dir):
        if (fname.startswith(f"run_{condition_label}_{problem_name}_")
                and fname.endswith(".json")):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study: Why do neural networks improve structural optimization?\n"
            "Runs 12 conditions × 3 problems and saves results + plots."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Run only this condition (e.g. --condition I). Default: all 12."
    )
    parser.add_argument(
        "--problem", type=str, default=None,
        help="Run only this problem (e.g. --problem mbb_beam). Default: all 3."
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Use tiny grid and 5 steps for fast sanity checking."
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip runs that already have saved log files."
    )
    parser.add_argument(
        "--plots_only", action="store_true",
        help="Skip optimization, just regenerate plots from existing results."
    )

    args = parser.parse_args()

    if args.plots_only:
        generate_all_plots(RESULTS_DIR, PLOTS_DIR)
        df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)
        if not df.empty:
            print_hypothesis_summary(df)
    else:
        run_full_experiment(
            condition_filter = args.condition,
            problem_filter   = args.problem,
            smoke_test       = args.smoke_test,
            skip_existing    = args.skip_existing,
        )
