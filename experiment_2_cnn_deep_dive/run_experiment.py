# experiment_2_cnn_deep_dive/run_experiment.py
# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: CNN Architecture & Optimizer Deep Dive.
#
# 9 conditions × 3 problems = 27 runs.
#
# Conditions split into two groups:
#   Architecture variants (P1–P6): Adam optimizer, vary CNN depth/capacity/BN.
#   Optimizer variants   (P7–P9): reference CNN, vary optimizer.
#
# Research questions answered by this experiment:
#   1. Which CNN architecture variant converges fastest / reaches lowest
#      compliance? (P1–P6, fixed Adam)
#   2. Which optimizer pairs best with the reference CNN? (P1, P7–P9)
#   3. Does the winning combination change by structural problem context?
#      (all conditions run on mbb_beam, causeway_bridge, multistory_building)
#
# Usage:
#   # Run all 27 conditions from the repo root:
#   python experiment_2_cnn_deep_dive/run_experiment.py
#
#   # Run from inside the subfolder:
#   cd experiment_2_cnn_deep_dive
#   python run_experiment.py
#
#   # Single condition:
#   python experiment_2_cnn_deep_dive/run_experiment.py --condition P3
#
#   # Single problem:
#   python experiment_2_cnn_deep_dive/run_experiment.py --problem mbb_beam
#
#   # Quick sanity check (tiny grid, 5 steps):
#   python experiment_2_cnn_deep_dive/run_experiment.py --smoke_test
#
#   # Regenerate plots only (no re-running optimization):
#   python experiment_2_cnn_deep_dive/run_experiment.py --plots_only
#
#   # Skip already-completed runs:
#   python experiment_2_cnn_deep_dive/run_experiment.py --skip_existing
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ── sys.path setup ────────────────────────────────────────────────────────────
# This file may be run from the repo root OR from inside the subfolder.
# We need:
#   1. This experiment's directory FIRST   → so "config" resolves to local config.py
#   2. The repo root SECOND                → so physics/, optimizers/ (parent),
#                                             parameterizations/cnn.py, analysis/metrics.py
#                                             are all importable.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)

# Insert unconditionally so the final order is guaranteed to be:
#   sys.path[0] = _THIS_DIR   → "import config" gets experiment_2/config.py
#   sys.path[1] = _PARENT_DIR → physics/, analysis/metrics, etc. resolve to parent
# We remove first to avoid duplicates, then re-insert in the correct order.
for _p in [_PARENT_DIR, _THIS_DIR]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
# After the loop: [THIS_DIR, PARENT_DIR, ...] — THIS_DIR is always first.
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
from visualize import generate_all_plots   # experiment_2_cnn_deep_dive/visualize.py


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERIZATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_parameterization(condition, args):
    """
    Instantiate the CNN variant parameterization for a given condition.

    All Experiment 2 conditions use CNNVariantParameterization.
    The `arch`, `base_channels`, `use_skip`, and `use_batchnorm` kwargs
    in condition.param_kwargs control which architecture is built.
    """
    if condition.parameterization != "cnn_variant":
        raise ValueError(
            f"Experiment 2 only supports parameterization='cnn_variant', "
            f"got '{condition.parameterization}' for condition {condition.label}."
        )

    from parameterizations.cnn_variants import CNNVariantParameterization
    return CNNVariantParameterization(args, **condition.param_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def run_optimizer(condition, parameterization, args, opt_steps_override=None):
    """
    Run the appropriate gradient optimizer for a condition.

    Uses the local optimizers/gradient_optimizer.py which supports
    adam, adamw, sgd, and rmsprop.
    """
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
    """
    Execute one full experimental run: condition × problem.

    Returns:
        metrics dict, or None if the run failed.
    """
    condition = CONDITIONS_BY_LABEL[condition_label]

    print(f"\n{'-'*60}")
    print(f"  Condition {condition_label}: {condition.description}")
    print(f"  Problem:   {problem_name}")
    print(f"  Question:  {condition.question}")
    print(f"{'-'*60}")

    try:
        # ── Build problem ─────────────────────────────────────────────────
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

        # ── Build parameterization ────────────────────────────────────────
        param = build_parameterization(condition, args)
        if verbose:
            print(f"  Parameterization: {param.description()}")

        # ── Run optimizer ─────────────────────────────────────────────────
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

        # ── Compute and save metrics ──────────────────────────────────────
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
    print(f"  EXPERIMENT 2 — CNN Architecture & Optimizer Deep Dive")
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
    print(f"  EXPERIMENT 2 COMPLETE")
    print(f"  Completed: {completed}/{total}  |  Failed: {failed}")
    print(f"{'='*60}")

    df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)

    if not df.empty:
        _print_exp2_summary(df)

    print("\nGenerating plots...")
    try:
        # Pass Exp 1 summary CSV path for cross-experiment reference lines
        exp1_csv = os.path.join(_PARENT_DIR, "results", "summary.csv")
        generate_all_plots(
            RESULTS_DIR, PLOTS_DIR,
            exp1_summary_csv=exp1_csv if os.path.exists(exp1_csv) else None,
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


def _print_exp2_summary(df):
    """Print a condensed summary comparing all conditions to P1."""
    print("\n" + "="*70)
    print("EXPERIMENT 2 — RESULTS SUMMARY")
    print("  All compliance values relative to P1 (reference CNN + Adam)")
    print("="*70)

    problems = df["problem"].unique()

    for problem in sorted(problems):
        print(f"\n--- {problem} ---")
        prob_df = df[df["problem"] == problem].copy()

        p1_rows = prob_df[prob_df["condition"] == "P1"]
        if p1_rows.empty:
            print("  P1 (reference) not yet run.")
            continue
        p1_val = float(p1_rows["final_compliance"].iloc[-1])
        print(f"  P1 (reference): {p1_val:.4e}")

        for _, row in prob_df.sort_values("condition").iterrows():
            cond = row["condition"]
            if cond == "P1":
                continue
            val  = float(row["final_compliance"])
            pct  = (val - p1_val) / p1_val * 100
            sign = "better" if pct < 0 else "worse"
            print(f"  {cond}: {val:.4e}  ({abs(pct):.1f}% {sign} than P1)")

    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2: CNN Architecture & Optimizer Deep Dive.\n"
            "9 conditions × 3 problems = 27 runs.\n\n"
            "Architecture variants (P1–P6): fixed Adam, vary CNN depth/capacity/BN.\n"
            "Optimizer variants   (P7–P9): fixed reference CNN, vary optimizer."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Run only this condition (e.g. --condition P3). Default: all 9."
    )
    parser.add_argument(
        "--problem", type=str, default=None,
        help="Run only this problem (e.g. --problem mbb_beam). Default: all 3."
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Tiny grid, 5 steps. Use for quick import/runtime checks."
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip conditions that already have a saved log file."
    )
    parser.add_argument(
        "--plots_only", action="store_true",
        help="Regenerate plots from existing results without re-running optimization."
    )

    args = parser.parse_args()

    if args.plots_only:
        exp1_csv = os.path.join(_PARENT_DIR, "results", "summary.csv")
        generate_all_plots(
            RESULTS_DIR, PLOTS_DIR,
            exp1_summary_csv=exp1_csv if os.path.exists(exp1_csv) else None,
        )
        df = build_summary_csv(LOGS_DIR, SUMMARY_CSV)
        if not df.empty:
            _print_exp2_summary(df)
    else:
        run_full_experiment(
            condition_filter = args.condition,
            problem_filter   = args.problem,
            smoke_test       = args.smoke_test,
            skip_existing    = args.skip_existing,
        )
