# optimizers/mma_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
# MMA (Method of Moving Asymptotes) optimizer wrapper.
#
# MMA is the classical solver for structural optimization. It is a gradient-
# based optimizer that handles nonlinear inequality constraints natively and
# is specifically designed for topology optimization problems.
#
# Reference: Svanberg (1987), "The method of moving asymptotes — a new method
# for structural optimization."
#
# We use the NLopt implementation via the Python nlopt package.
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import autograd

from physics.objective import objective, mean_density, physical_density


def run_mma(parameterization, args, opt_steps=80, print_every=10, checkpoints=None):
    """
    Optimize using MMA (Method of Moving Asymptotes) via NLopt.

    MMA is used as the classical baseline optimizer. It accepts:
      - A differentiable objective (compliance)
      - A nonlinear inequality constraint (volume constraint)
      - Bounds on the parameter space [0, 1]

    Gradients are computed via Autograd automatic differentiation, which
    differentiates through the full physics chain:
        params → density → FEM solve → compliance

    Args:
        parameterization: a parameterization object (DirectParameterization, etc.)
        args:             problem ObjectView
        opt_steps:        maximum number of optimizer steps
        print_every:      print progress every N steps
        checkpoints:      list of step numbers at which to record compliance

    Returns:
        losses: list of compliance values at each step
        frames: list of density arrays at each step
        checkpoint_losses: dict mapping step number → compliance
    """
    try:
        import nlopt
    except ImportError:
        raise ImportError(
            "NLopt is required for MMA optimizer. Install with: pip install nlopt"
        )

    x0 = parameterization.initial_params()
    n_params = len(x0)

    losses, frames = [], []
    checkpoint_losses = {}
    dt = time.time()

    def objective_fn(params):
        """Compute compliance from current parameters."""
        # Get density field from parameterization
        density = parameterization.to_density(params)
        # Compute compliance via autograd-traced physics
        return objective(density.ravel(), args, use_filter=True)

    def constraint_fn(params):
        """Volume constraint: mean_density - target_density ≤ 0."""
        density = parameterization.to_density(params)
        return mean_density(density.ravel(), args) - args.density

    def wrap_autograd(func, losses_list=None, frames_list=None):
        """
        Wrap a function to: (1) compute gradient via Autograd, (2) log progress.

        NLopt requires that the objective function accepts (x, grad) and fills
        grad in-place with the gradient.
        """
        def wrapper(x, grad):
            if grad.size > 0:
                # Autograd computes both value and gradient in one pass
                value, gradient = autograd.value_and_grad(func)(x)
                grad[:] = gradient
            else:
                value = func(x)

            if losses_list is not None:
                losses_list.append(float(value))
            if frames_list is not None:
                density = parameterization.to_density(x)
                frames_list.append(density.copy())

                step = len(frames_list)

                # Record checkpoint compliance
                if checkpoints and step in checkpoints:
                    checkpoint_losses[step] = float(value)

                if step % print_every == 0:
                    print(f"  Step {step:4d}  |  compliance: {value:.4e}  |  "
                          f"t = {time.time() - dt:.1f}s")

            return float(value)
        return wrapper

    # ── Set up NLopt ──────────────────────────────────────────────────────────
    opt = nlopt.opt(nlopt.LD_MMA, n_params)

    # Bounds: densities must stay in [0, 1]
    opt.set_lower_bounds(np.full(n_params, 0.0))
    opt.set_upper_bounds(np.full(n_params, 1.0))

    # Minimize compliance (objective function)
    opt.set_min_objective(wrap_autograd(objective_fn, losses, frames))

    # Volume inequality constraint (with tolerance 1e-8)
    opt.add_inequality_constraint(wrap_autograd(constraint_fn), 1e-8)

    opt.set_maxeval(opt_steps + 1)

    # ── Run optimization ──────────────────────────────────────────────────────
    print(f"  Running MMA for {opt_steps} steps on {n_params} parameters...")
    opt.optimize(x0.flatten())

    # Record final checkpoint
    if checkpoints:
        for step in checkpoints:
            if step not in checkpoint_losses and losses:
                # Use closest available loss
                idx = min(step, len(losses)) - 1
                checkpoint_losses[step] = losses[idx] if losses else float("inf")

    return (
        np.array(losses),
        np.array(frames) if frames else np.array([]),
        checkpoint_losses,
    )
