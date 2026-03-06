# optimizers/gradient_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Gradient-based optimizers (Adam, L-BFGS) via PyTorch.
#
# These optimizers work directly with the parameterization's internal
# PyTorch parameters (for NN-based parameterizations) or via a numpy-to-torch
# bridge (for direct parameterization).
#
# The key difference from MMA:
#   - MMA uses a specialized structural optimization algorithm with explicit
#     inequality constraints and bounds
#   - Adam/L-BFGS are general-purpose gradient descent variants, without
#     native constraint handling (we enforce volume via penalty instead)
#
# Using the same Adam optimizer for both NN and direct parameterizations
# lets us isolate the effect of the parameterization itself.
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np
import torch
import torch.optim as optim

from physics.objective import objective, mean_density


def _compliance_torch(density_np, args):
    """
    Compute compliance as a float (for logging, not differentiation).
    Calls the numpy/autograd-based objective.
    """
    return float(objective(density_np.ravel(), args, use_filter=True)) 


def run_gradient_optimizer( # CONDITION I/J/K: optimize CNN weights, z is fixed
    parameterization,
    args,
    optimizer_type="adam",
    opt_steps=200,
    lr=1e-2,
    print_every=10,
    checkpoints=None,
    volume_penalty=1e3,
    lbfgs_max_iter=20,
):
    """
    Optimize using Adam or L-BFGS.

    Strategy: treat the parameterization's parameter vector as a torch.Tensor
    with requires_grad=True. At each step:
      1. Forward pass through parameterization → density
      2. Forward pass through physics (numpy/autograd) → compliance
      3. Compute gradient via torch.autograd (bridges to numpy via custom func)
      4. Apply optimizer update

    Volume constraint is enforced as a quadratic penalty added to the objective:
        total_loss = compliance + λ * max(0, mean_density - target_density)²

    Args:
        parameterization: a parameterization object
        args:             problem ObjectView
        optimizer_type:   "adam" or "lbfgs"
        opt_steps:        number of optimizer steps
        lr:               learning rate
        print_every:      logging frequency
        checkpoints:      list of steps at which to record compliance
        volume_penalty:   coefficient λ for the volume penalty term
        lbfgs_max_iter:   L-BFGS inner loop iterations per step

    Returns:
        losses: list of compliance values (not including penalty)
        frames: list of density arrays at each step
        checkpoint_losses: dict mapping step → compliance
    """
    x0 = parameterization.initial_params() # Get the initial parameters

    # Optimization variable: flat parameter vector as a torch tensor
    params = torch.tensor(x0, dtype=torch.float32, requires_grad=True) # Create a torch tensor from the initial parameters

    # ── Optimizer setup ───────────────────────────────────────────────────────
    if optimizer_type == "adam":
        optimizer = optim.Adam([params], lr=lr) # Create an Adam optimizer
    elif optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(
            [params],
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}") # If the optimizer type is not known, raise an error

    losses, frames = [], []
    checkpoint_losses = {}
    dt = time.time()

    print(f"  Running {optimizer_type.upper()} for {opt_steps} steps " 
          f"on {len(x0)} parameters...") # Print the optimizer type and the number of parameters

    for step in range(1, opt_steps + 1): # Loop through the optimizer steps

        def closure():
            """
            Closure required by L-BFGS (also works for Adam).
            Computes loss and populates param.grad.
            """
            optimizer.zero_grad() # Zero out the gradients of the parameters

            # Get density from parameterization (numpy array)
            params_np = params.detach().cpu().numpy()
            density_np = parameterization.to_density(params_np)

            # ── Compute compliance gradient via autograd (numpy side) ────────
            import autograd
            import autograd.numpy as anp

            def compliance_from_density(x): 
                return objective(x, args, use_filter=True) # Compute the compliance from the density    

            compliance_val, compliance_grad = autograd.value_and_grad( # Compute the compliance and the gradient of the compliance with respect to the density
                compliance_from_density
            )(density_np.ravel())

            # ── Volume penalty ────────────────────────────────────────────────
            mean_d = float(mean_density(density_np.ravel(), args)) # Compute the mean density
            volume_violation = max(0.0, mean_d - args.density) # Compute the volume violation
            penalty_val = volume_penalty * volume_violation ** 2 # Compute the penalty value

            # Gradient of penalty w.r.t. density (subgradient)
            if volume_violation > 0: # If the volume violation is greater than 0 then compute the gradient of the penalty with respect to the density
                # d(penalty)/d(mean_density) * d(mean_density)/d(density[i])
                # = 2λ * violation * (1/n_elements)
                penalty_grad_density = (
                    2 * volume_penalty * volume_violation
                    / density_np.size
                    * np.ones_like(density_np.ravel())
                )
            else: # If the volume violation is less than or equal to 0 then set the gradient of the penalty with respect to the density to 0
                penalty_grad_density = np.zeros(density_np.size) # Set the gradient of the penalty with respect to the density to 0

            total_grad_density = compliance_grad + penalty_grad_density # Compute the total gradient of the density

            # ── Bridge: density gradient → parameter gradient ─────────────────
            # We use a custom torch Function to pass the numpy gradient back
            # into the torch autograd graph.
            density_torch = _NumpyDensityFunction.apply( # Apply the numpy density function to the parameters, the parameterization, and the total gradient of the density
                params,
                parameterization,
                torch.tensor(total_grad_density.reshape(density_np.shape), dtype=torch.float32),
            )

            # Total loss = compliance + penalty (as torch scalars for .backward())
            total_loss = torch.tensor(
                compliance_val + penalty_val, dtype=torch.float32, requires_grad=True
            )

            # Manually set grad on params from the density gradient chain
            # (since we bridged through numpy, we handle this manually)
            if params.grad is None:
                params.grad = torch.zeros_like(params)

            # Compute parameter gradient from density gradient via finite differences
            # approximation — or use the jacobian if the parameterization supports it.
            # For simplicity, we use a direct gradient passthrough pattern:
            params_grad = _compute_param_grad(
                params_np, total_grad_density, parameterization
            )
            params.grad = torch.tensor(params_grad, dtype=torch.float32)

            return torch.tensor(compliance_val + penalty_val, dtype=torch.float32)

        loss = optimizer.step(closure)

        # ── Logging ──────────────────────────────────────────────────────────
        params_np  = params.detach().cpu().numpy()
        density_np = parameterization.to_density(params_np)
        compliance_val = _compliance_torch(density_np, args)

        losses.append(compliance_val)
        frames.append(density_np.copy())

        if checkpoints and step in checkpoints:
            checkpoint_losses[step] = compliance_val

        if step % print_every == 0:
            mean_d = float(mean_density(density_np.ravel(), args))
            print(f"  Step {step:4d}  |  compliance: {compliance_val:.4e}  |  "
                  f"density: {mean_d:.3f}  |  t = {time.time() - dt:.1f}s")

    return np.array(losses), np.array(frames), checkpoint_losses


def _compute_param_grad(params_np, density_grad_np, parameterization):
    """
    Compute gradient of the loss w.r.t. the parameter vector.

    Uses finite differences to estimate d(density)/d(params), then chains
    with the density gradient: d(loss)/d(params) = d(loss)/d(density) · d(density)/d(params)

    For small parameter counts this is exact; for large NN parameter counts
    we use a vector-Jacobian product approximation via torch autograd.
    """
    # For NN parameterizations with torch support, use torch autograd
    if hasattr(parameterization, "to_density_with_grad"): # If the parameterization has a to_density_with_grad method
        try:
            params_tensor = torch.tensor(
                params_np, dtype=torch.float32, requires_grad=True # Create a torch tensor from the parameters
            )
            density = parameterization.to_density_with_grad(params_np) # Forward pass through the network to get the density

            if isinstance(density, torch.Tensor) and density.requires_grad: # If the density is a torch tensor and requires grad
                density_grad_tensor = torch.tensor(
                    density_grad_np.reshape(density.shape), dtype=torch.float32 # Reshape the density gradient to the shape of the density
                )
                density.backward(density_grad_tensor) # Backward pass through the network to get the gradients of the trainable parameters
                if params_tensor.grad is not None: # If the gradients of the trainable parameters are not None
                    return params_tensor.grad.cpu().numpy() # Return the gradients of the trainable parameters
        except Exception: # If an error occurs, pass
            pass

    # For torch-based parameterizations, compute VJP through the network, VJP = Jacobian-Vector Product
    if hasattr(parameterization, "model"): # If the parameterization has a model
        try:
            import torch.nn as nn
            # Collect all trainable parameters
            if parameterization.frozen: # CONDITION L: Freeze all CNN weights.
                trainable = [parameterization.z] # Collect the frozen parameters
            else: # Conditions I/J/K: optimize CNN weights, z is fixed
                trainable = list(parameterization.model.parameters()) # Collect the trainable parameters

            for p in trainable: # Zero out the gradients of the trainable parameters
                if p.grad is not None: # If the gradients of the trainable parameters are not None
                    p.grad.zero_()

            density = parameterization.to_density_with_grad(params_np) # Forward pass through the network to get the density
            if isinstance(density, torch.Tensor):
                grad_out = torch.tensor(
                    density_grad_np.reshape(density.shape), dtype=torch.float32 # Reshape the density gradient to the shape of the density
                )
                density.backward(grad_out) # Backward pass through the network to get the gradients of the trainable parameters

                grad_list = []
                for p in trainable: # Collect the gradients of the trainable parameters
                    if p.grad is not None:
                        grad_list.append(p.grad.cpu().numpy().ravel()) # Collect the gradients of the trainable parameters
                    else:
                        grad_list.append(np.zeros(p.numel()))
                return np.concatenate(grad_list) # Concatenate the gradients of the trainable parameters
        except Exception as e:
            pass # If an error occurs, pass 

    # Fallback: finite differences (slow but always correct)
    eps = 1e-5
    grad = np.zeros_like(params_np)
    density_flat = density_grad_np.ravel()

    # Estimate via directional derivative (random projection to reduce cost)
    n = len(params_np)
    n_probes = min(n, 100)  # limit finite difference probes
    indices = np.random.choice(n, size=n_probes, replace=False)

    for i in indices: 
        params_plus = params_np.copy()
        params_plus[i] += eps # Add epsilon to the parameter at the index
        density_plus = parameterization.to_density(params_plus).ravel() # Forward pass through the network to get the density
        density_base = parameterization.to_density(params_np).ravel() # Forward pass through the network to get the density. .ravel() is used to flatten the density to a 1D array
        jacobian_col = (density_plus - density_base) / eps # Compute the Jacobian column
        grad[i] = np.dot(density_flat, jacobian_col) # Compute the gradient

    return grad


class _NumpyDensityFunction(torch.autograd.Function):
    """
    Torch autograd Function that bridges the numpy density gradient
    back into the torch computational graph.
    """
    @staticmethod
    def forward(ctx, params, parameterization, density_grad): # Forward pass through the network to get the density
        ctx.save_for_backward(density_grad) # Save the density gradient for the backward pass
        return params  # passthrough

    @staticmethod
    def backward(ctx, grad_output): # Backward pass through the network to get the gradients of the trainable parameters
        density_grad, = ctx.saved_tensors
        return density_grad, None, None # Return the density gradient and None for the backward pass
