# experiment_3_gnn/optimizers/gradient_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Gradient-based optimizer for Experiment 3.
#
# All conditions in Experiment 3 use AdamW (established as the most stable
# optimizer in Experiment 2: fixes Adam's late-stage divergence on sparse
# problems like causeway bridge).
#
# The optimization loop and numpy↔torch gradient bridge are unchanged from
# Experiments 1 & 2. Both CNN and GNN parameterizations expose the same
# interface (initial_params, to_density, to_density_with_grad, model attribute),
# so this file works identically for both.
#
# Gradient bridge:
#   1. Forward: parameterization → density (numpy, H×W)
#   2. Physics gradient: autograd compliance w.r.t. density (numpy)
#   3. Volume penalty gradient (numpy)
#   4. Combined density gradient → parameter gradient via torch VJP
#      (backpropagate through CNN/GNN using density.backward(grad_out))
#   5. AdamW step
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import torch
import torch.optim as optim

from physics.objective import objective, mean_density


def _compliance_val(density_np, args):
    return float(objective(density_np.ravel(), args, use_filter=True))


def run_gradient_optimizer(
    parameterization,
    args,
    optimizer_type = "adamw",
    opt_steps      = 200,
    lr             = 1e-2,
    print_every    = 10,
    checkpoints    = None,
    volume_penalty = 1e3,
    weight_decay   = 0.01,
    # Unused kwargs kept for API compatibility with Exp 2 runner signatures
    lbfgs_max_iter = 20,
    momentum       = 0.9,
    nesterov       = True,
    alpha          = 0.99,
):
    """
    Optimize structural design using AdamW (or Adam/SGD/RMSprop for compatibility).

    Args:
        parameterization: CNNReferenceParameterization or GNNParameterization
        args:             problem ObjectView
        optimizer_type:   "adamw" (recommended) | "adam" | "sgd" | "rmsprop"
        opt_steps:        total optimizer steps
        lr:               learning rate
        print_every:      log compliance every N steps
        checkpoints:      list of steps to record compliance in dict
        volume_penalty:   λ for quadratic volume-violation penalty
        weight_decay:     L2 coefficient (used by AdamW)

    Returns:
        losses:            np.array of compliance per step
        frames:            np.array of density fields per step
        checkpoint_losses: dict {step: compliance}
    """
    x0     = parameterization.initial_params()
    params = torch.tensor(x0, dtype=torch.float32, requires_grad=True)

    # ── Build optimizer ───────────────────────────────────────────────────────
    if optimizer_type == "adamw":
        optimizer = optim.AdamW([params], lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam([params], lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD([params], lr=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop([params], lr=lr, alpha=alpha)
    else:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            "Choose from: adamw, adam, sgd, rmsprop."
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

            # ── Compliance gradient (autograd, numpy side) ─────────────────
            import autograd
            compliance_val, compliance_grad = autograd.value_and_grad(
                lambda x: objective(x, args, use_filter=True)
            )(density_np.ravel())

            # ── Volume penalty ─────────────────────────────────────────────
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

            # ── Bridge density gradient → parameter gradient via torch VJP ──
            params_grad = _compute_param_grad(
                params_np, total_density_grad, parameterization
            )
            params.grad = torch.tensor(params_grad, dtype=torch.float32)

            return torch.tensor(
                float(compliance_val) + float(penalty_val), dtype=torch.float32
            )

        optimizer.step(closure)

        # ── Logging ───────────────────────────────────────────────────────
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
# GRADIENT BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_param_grad(params_np, density_grad_np, parameterization):
    """
    Compute d(loss)/d(params) by backpropagating the density gradient
    through the CNN or GNN via torch autograd (vector-Jacobian product).

    Both CNNReferenceParameterization and GNNParameterization expose:
        - parameterization.model   (nn.Module with nn.Parameters)
        - parameterization.to_density_with_grad(params_np)  → (H, W) tensor

    Falls back to random-projection finite differences if torch backprop fails.
    """
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
                    g = p.grad.cpu().numpy().ravel() if p.grad is not None else np.zeros(p.numel())
                    grad_list.append(g)
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
