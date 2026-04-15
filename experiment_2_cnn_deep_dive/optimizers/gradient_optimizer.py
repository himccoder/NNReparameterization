# experiment_2_cnn_deep_dive/optimizers/gradient_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Extended gradient-based optimizer for Experiment 2.
#
# Extends the Experiment 1 gradient optimizer with three additional optimizers:
#   adamw   — Adam with L2 weight decay (tests whether regularisation helps)
#   sgd     — SGD with Nesterov momentum (classical alternative to adaptive lr)
#   rmsprop — RMSprop (adaptive lr without momentum correction)
#
# The core optimization loop and numpy↔torch gradient bridge are unchanged
# from Experiment 1. Only the optimizer construction block is extended.
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import torch
import torch.optim as optim

from physics.objective import objective, mean_density


def _compliance_val(density_np, args):
    """Compliance as a float for logging (not differentiation)."""
    return float(objective(density_np.ravel(), args, use_filter=True))


def run_gradient_optimizer(
    parameterization,
    args,
    optimizer_type = "adam",
    opt_steps      = 200,
    lr             = 1e-2,
    print_every    = 10,
    checkpoints    = None,
    volume_penalty = 1e3,
    # Adam / AdamW
    lbfgs_max_iter = 20,    # unused in Exp 2; kept for API compatibility
    weight_decay   = 0.0,   # AdamW weight decay coefficient
    # SGD
    momentum       = 0.9,
    nesterov       = True,
    # RMSprop
    alpha          = 0.99,
):
    """
    Optimize using Adam, AdamW, SGD+momentum, or RMSprop.

    All four optimizers follow the same loop:
      1. Forward: parameterization → density (numpy)
      2. Physics gradient: autograd compliance → density gradient (numpy)
      3. Volume penalty gradient (numpy)
      4. Bridge: density gradient → parameter gradient via torch VJP
      5. Optimizer step

    Args:
        parameterization: CNNVariantParameterization instance
        args:             problem ObjectView
        optimizer_type:   "adam" | "adamw" | "sgd" | "rmsprop"
        opt_steps:        total optimizer steps
        lr:               learning rate
        print_every:      log every N steps
        checkpoints:      list of steps to record compliance
        volume_penalty:   λ for quadratic volume-violation penalty
        weight_decay:     L2 coefficient for AdamW
        momentum:         SGD momentum factor
        nesterov:         whether to use Nesterov momentum for SGD
        alpha:            RMSprop smoothing constant

    Returns:
        losses:            np.array of compliance per step
        frames:            np.array of density fields per step
        checkpoint_losses: dict {step: compliance}
    """
    x0     = parameterization.initial_params()
    params = torch.tensor(x0, dtype=torch.float32, requires_grad=True)

    # ── Build optimizer ───────────────────────────────────────────────────────
    if optimizer_type == "adam":
        optimizer = optim.Adam([params], lr=lr)

    elif optimizer_type == "adamw":
        optimizer = optim.AdamW([params], lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            [params], lr=lr, momentum=momentum, nesterov=nesterov
        )

    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop([params], lr=lr, alpha=alpha)

    elif optimizer_type == "lbfgs":
        # L-BFGS kept for completeness but not used in Experiment 2 conditions
        optimizer = optim.LBFGS(
            [params],
            lr             = lr,
            max_iter       = lbfgs_max_iter,
            history_size   = 10,
            line_search_fn = "strong_wolfe",
        )

    else:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Choose from: adam, adamw, sgd, rmsprop."
        )

    losses, frames = [], []
    checkpoint_losses = {}
    t0 = time.time()

    print(f"  Running {optimizer_type.upper()} for {opt_steps} steps "
          f"on {len(x0):,} parameters...")

    for step in range(1, opt_steps + 1):

        def closure():
            optimizer.zero_grad()

            params_np  = params.detach().cpu().numpy()
            density_np = parameterization.to_density(params_np)

            # ── Compliance gradient (autograd, numpy side) ────────────────
            import autograd
            compliance_val, compliance_grad = autograd.value_and_grad(
                lambda x: objective(x, args, use_filter=True)
            )(density_np.ravel())

            # ── Volume penalty ────────────────────────────────────────────
            mean_d           = float(mean_density(density_np.ravel(), args))
            volume_violation = max(0.0, mean_d - args.density)
            penalty_val      = volume_penalty * volume_violation ** 2

            if volume_violation > 0:
                penalty_grad = (
                    2 * volume_penalty * volume_violation
                    / density_np.size
                    * np.ones(density_np.size)
                )
            else:
                penalty_grad = np.zeros(density_np.size)

            total_density_grad = compliance_grad + penalty_grad

            # ── Bridge density gradient → parameter gradient via torch VJP ─
            params_grad = _compute_param_grad(
                params_np, total_density_grad, parameterization
            )
            params.grad = torch.tensor(params_grad, dtype=torch.float32)

            return torch.tensor(
                float(compliance_val) + float(penalty_val), dtype=torch.float32
            )

        optimizer.step(closure)

        # ── Logging ──────────────────────────────────────────────────────
        params_np      = params.detach().cpu().numpy()
        density_np     = parameterization.to_density(params_np)
        compliance_val = _compliance_val(density_np, args)

        losses.append(compliance_val)
        frames.append(density_np.copy())

        if checkpoints and step in checkpoints:
            checkpoint_losses[step] = compliance_val

        if step % print_every == 0:
            mean_d = float(mean_density(density_np.ravel(), args))
            print(f"  Step {step:4d}  |  compliance: {compliance_val:.4e}  |  "
                  f"density: {mean_d:.3f}  |  t = {time.time() - t0:.1f}s")

    return np.array(losses), np.array(frames), checkpoint_losses


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT BRIDGE  (unchanged from Experiment 1)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_param_grad(params_np, density_grad_np, parameterization):
    """
    Compute d(loss)/d(params) by backpropagating the density gradient through
    the CNN via torch autograd (vector-Jacobian product).

    Falls back to finite differences if torch backprop is unavailable.
    """
    # Primary path: torch VJP through CNN weights
    if hasattr(parameterization, "model"):
        try:
            trainable = list(parameterization.model.parameters())
            for p in trainable:
                if p.grad is not None:
                    p.grad.zero_()

            density = parameterization.to_density_with_grad(params_np)
            if isinstance(density, torch.Tensor):
                grad_out = torch.tensor(
                    density_grad_np.reshape(density.shape), dtype=torch.float32
                )
                density.backward(grad_out)

                grad_list = []
                for p in trainable:
                    if p.grad is not None:
                        grad_list.append(p.grad.cpu().numpy().ravel())
                    else:
                        grad_list.append(np.zeros(p.numel()))
                return np.concatenate(grad_list)
        except Exception:
            pass

    # Fallback: random-projection finite differences
    eps      = 1e-5
    grad     = np.zeros_like(params_np)
    n        = len(params_np)
    n_probes = min(n, 100)
    indices  = np.random.choice(n, size=n_probes, replace=False)
    density_flat = density_grad_np.ravel()

    for i in indices:
        params_plus = params_np.copy()
        params_plus[i] += eps
        jacobian_col = (
            parameterization.to_density(params_plus).ravel()
            - parameterization.to_density(params_np).ravel()
        ) / eps
        grad[i] = np.dot(density_flat, jacobian_col)

    return grad
