# analysis/metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Metrics collection and result storage.
#
# For each experimental condition × problem, we collect:
#   - final_compliance:     the primary quality metric (lower = better)
#   - checkpoint_losses:    compliance at steps 20, 40, 80 (convergence speed)
#   - topology_sparsity:    fraction of elements with density > 0.5 (design clarity)
#   - wall_time:            total optimization time in seconds
#   - n_params:             number of optimization parameters
#
# Results are stored as JSON (one file per run) and aggregated into a CSV.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime


def compute_metrics(losses, final_density, wall_time, checkpoint_losses, n_params):
    """
    Compute all metrics for a single experimental run.

    Args:
        losses:            array of compliance values at each step
        final_density:     (nely, nelx) density array at end of optimization
        wall_time:         total time in seconds
        checkpoint_losses: dict of {step: compliance} at checkpoint steps
        n_params:          number of optimization parameters

    Returns:
        dict of metric name → value
    """
    final_compliance = float(losses[-1]) if len(losses) > 0 else float("inf")

    # Topology sparsity: fraction of elements that are clearly solid (density > 0.5)
    # A high sparsity ratio indicates a clean binary design (good for fabrication)
    sparsity = float(np.mean(final_density > 0.5))

    # Grey ratio: fraction of elements in the "grey zone" (0.2 < density < 0.8)
    # Lower is better — grey elements indicate optimizer hasn't converged to binary
    grey_ratio = float(np.mean((final_density > 0.2) & (final_density < 0.8)))

    # Convergence speed: how many steps to reach within 10% of final compliance
    convergence_step = _compute_convergence_step(losses, threshold=0.10)

    metrics = {
        "final_compliance":  final_compliance,
        "topology_sparsity": sparsity,
        "grey_ratio":        grey_ratio,
        "convergence_step":  convergence_step,
        "wall_time_s":       wall_time,
        "n_params":          n_params,
        "n_steps":           len(losses),
    }

    # Add checkpoint losses
    for step, val in checkpoint_losses.items():
        metrics[f"compliance_at_step_{step}"] = float(val)

    return metrics


def _compute_convergence_step(losses, threshold=0.10):
    """
    Find the first step where compliance is within `threshold` of the final value.

    Args:
        losses:    array of compliance values
        threshold: fractional tolerance (0.10 = within 10% of final)

    Returns:
        step index, or n_steps if never converged
    """
    if len(losses) == 0:
        return 0
    final = losses[-1]
    target = final * (1 + threshold)  # within 10% above final
    for i, l in enumerate(losses):
        if l <= target:
            return i + 1
    return len(losses)


def save_run_result(condition_label, problem_name, metrics, losses,
                    final_density, logs_dir, run_id=None):
    """
    Save the full result of one experimental run to disk as JSON.

    Args:
        condition_label: e.g., "I"
        problem_name:    e.g., "mbb_beam"
        metrics:         dict of computed metrics
        losses:          full compliance history array
        final_density:   (nely, nelx) final density array
        logs_dir:        directory to write logs
        run_id:          optional run identifier string

    Returns:
        path to the saved log file
    """
    os.makedirs(logs_dir, exist_ok=True)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "run_id":          run_id,
        "condition":       condition_label,
        "problem":         problem_name,
        "timestamp":       datetime.now().isoformat(),
        "metrics":         metrics,
        "loss_history":    losses.tolist() if hasattr(losses, "tolist") else list(losses),
        "final_density":   final_density.tolist(),
    }

    fname = f"run_{condition_label}_{problem_name}_{run_id}.json"
    fpath = os.path.join(logs_dir, fname)

    with open(fpath, "w") as f:
        json.dump(result, f, indent=2)

    return fpath


def load_all_results(logs_dir):
    """
    Load all saved run results from the logs directory.

    Returns:
        list of result dicts
    """
    results = []
    if not os.path.exists(logs_dir):
        return results

    for fname in sorted(os.listdir(logs_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(logs_dir, fname)
            try:
                with open(fpath) as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Warning: could not load {fpath}: {e}")

    return results


def build_summary_csv(logs_dir, output_path):
    """
    Aggregate all run results into a single CSV summary table.

    Each row is one (condition, problem) pair.
    Columns include all metrics plus run metadata.

    Args:
        logs_dir:    directory containing JSON log files
        output_path: path to write the CSV

    Returns:
        pandas DataFrame of results
    """
    results = load_all_results(logs_dir)
    if not results:
        print("No results found.")
        return pd.DataFrame()

    rows = []
    for r in results:
        row = {
            "condition":  r["condition"],
            "problem":    r["problem"],
            "timestamp":  r.get("timestamp", ""),
        }
        row.update(r.get("metrics", {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by condition label and problem name
    df = df.sort_values(["condition", "problem"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")

    return df


def print_hypothesis_summary(df):
    """
    Print a formatted summary of the key hypothesis comparisons.

    Reads the summary DataFrame and prints which hypotheses are supported
    or refuted based on the experimental results.

    Args:
        df: summary DataFrame from build_summary_csv
    """
    print("\n" + "="*70)
    print("HYPOTHESIS TEST RESULTS")
    print("="*70)

    problems = df["problem"].unique()

    comparisons = [
        ("H2: Smoothing explains it", "D", "A",
         "If D ≈ A, smoothing alone is NOT the answer. If D ≈ I, it IS."),
        ("H3: Optimizer explains it", "B", "A",
         "If B >> A, Adam alone matters. If B ≈ A, optimizer is not the key."),
        ("H1: Any NN helps",          "E", "I",
         "If E ≈ I, any NN helps. If E << I, CNN spatial prior is needed."),
        ("H5: Frequency bias",        "G", "I",
         "If G ≈ I, frequency encoding replaces conv. If G << I, conv is key."),
        ("H6: Structural prior only", "L", "A",
         "If L >> A, the ARCHITECTURE ALONE explains the gains (killer result)."),
        ("H1: Skip connections",      "K", "I",
         "If K ≈ I, skip connections don't matter. If K << I, multi-scale is key."),
    ]

    for problem in problems:
        print(f"\n── Problem: {problem} ──")
        prob_df = df[df["problem"] == problem].set_index("condition")

        if "final_compliance" not in prob_df.columns:
            print("  (no results yet)")
            continue

        def _scalar(val):
            """Return a plain float from a scalar or a Series (last run wins)."""
            import pandas as pd
            if isinstance(val, pd.Series):
                val = val.iloc[-1]
            return float(val)

        baseline = _scalar(prob_df.loc["A", "final_compliance"]) if "A" in prob_df.index else None
        cnn      = _scalar(prob_df.loc["I", "final_compliance"]) if "I" in prob_df.index else None

        if baseline is not None:
            print(f"  Baseline (A, MMA direct): {baseline:.4e}")
        if cnn is not None:
            print(f"  CNN reference (I):        {cnn:.4e}")
            if baseline is not None:
                improvement = (baseline - cnn) / baseline * 100
                print(f"  CNN improvement over A:   {improvement:.1f}%")

        print()
        for name, cond1, cond2, interpretation in comparisons:
            if cond1 not in prob_df.index:
                print(f"  {name}: {cond1} not yet run")
                continue

            v1 = _scalar(prob_df.loc[cond1, "final_compliance"])
            v2 = _scalar(prob_df.loc[cond2, "final_compliance"]) if cond2 in prob_df.index else None

            if v2 is not None:
                delta_pct = (v2 - v1) / v2 * 100
                direction = "better" if delta_pct > 0 else "worse"
                print(f"  {name}:")
                print(f"    {cond1}: {v1:.4e}  vs  {cond2}: {v2:.4e}  "
                      f"({abs(delta_pct):.1f}% {direction})")
                print(f"    → {interpretation}")
            else:
                print(f"  {name}: {cond1}: {v1:.4e} (reference {cond2} not run)")

    print("\n" + "="*70)
