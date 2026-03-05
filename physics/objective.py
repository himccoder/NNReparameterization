# physics/objective.py
# ─────────────────────────────────────────────────────────────────────────────
# The compliance objective function and supporting density utilities.
#
# Compliance = U^T * K * U = total elastic potential energy of the structure.
# Minimizing compliance = maximizing stiffness = best structural design.
#
# The key pipeline is:
#   raw params (x_raw)
#   → physical density (x_phys)  [via Gaussian filter + SIMP mapping]
#   → compliance (scalar)         [via FEM physics]
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.ndimage
import autograd
import autograd.numpy as anp

from physics.fem import get_stiffness_matrix, displace


# ─────────────────────────────────────────────────────────────────────────────
# SIMP MATERIAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def young_modulus(x, e_0, e_min, p=3):
    """
    Modified SIMP interpolation: maps density [0,1] to Young's modulus.

    E(x) = E_min + x^p * (E_0 - E_min)

    The penalization exponent p > 1 pushes densities toward binary (0 or 1),
    because intermediate densities are disproportionately penalized in stiffness.
    This gives clean black-and-white structural designs rather than grey blobs.

    Args:
        x:     density values in [0, 1]
        e_0:   Young's modulus of solid material
        e_min: minimum Young's modulus (avoids singular stiffness matrix)
        p:     penalization exponent (typically 3)
    """
    return e_min + x ** p * (e_0 - e_min)


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN FILTER (with Autograd-compatible custom gradient)
# ─────────────────────────────────────────────────────────────────────────────

@autograd.extend.primitive
def gaussian_filter(x, width):
    """
    2D Gaussian blur applied to the density field.

    This enforces spatial smoothness and prevents checkerboard artifacts
    (a common failure mode in discrete structural optimization). The filter
    width controls the minimum feature size in the optimized structure.

    Wrapped as an Autograd primitive so we can differentiate through it.
    """
    return scipy.ndimage.gaussian_filter(x, width, mode="reflect")


def _gaussian_filter_vjp(ans, x, width):
    """
    Reverse-mode gradient for Gaussian filter.

    The Gaussian filter is self-adjoint (symmetric), so its VJP is just
    another Gaussian filter applied to the upstream gradient.
    """
    del ans, x  # unused
    return lambda g: gaussian_filter(g, width)


autograd.extend.defvjp(gaussian_filter, _gaussian_filter_vjp)


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def physical_density(x, args, use_filter=True):
    """
    Convert raw optimization parameters to physical material densities.

    Steps:
      1. Reshape flat vector to 2D grid
      2. Apply mask (e.g., to exclude non-design regions)
      3. Optionally apply Gaussian smoothing

    Args:
        x:          1D or 2D array of raw density values
        args:       problem arguments (shape, mask, filter_width)
        use_filter: whether to apply Gaussian smoothing

    Returns:
        x_phys: (nely, nelx) physical density field
    """
    x = args.mask * x.reshape(args.nely, args.nelx)
    if use_filter:
        return gaussian_filter(x, args.filter_width)
    return x


def mean_density(x, args, use_filter=True):
    """
    Compute the mean (normalized) material density.
    Used to enforce the volume constraint: mean_density <= target_density.
    """
    return anp.mean(physical_density(x, args, use_filter)) / anp.mean(args.mask)


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
    """
    Compute structural compliance: c = U^T * K * U = sum_e E(x_e) * u_e^T * ke * u_e

    Instead of computing the full matrix product U^T * K * U (expensive),
    we decompose it element-wise. Each element contributes:
        c_e = E(x_e) * u_e^T * ke * u_e
    where u_e is the 8-DOF displacement vector of that element's nodes.

    This vectorized form avoids explicit loops over elements.

    Args:
        x_phys: (nely, nelx) physical density field
        u:      (2N,) nodal displacement vector
        ke:     (8, 8) element stiffness matrix
        penal:  SIMP exponent
        e_min:  minimum Young's modulus
        e_0:    baseline Young's modulus

    Returns:
        c: scalar compliance (lower = stiffer = better design)
    """
    nely, nelx = x_phys.shape
    ely, elx = anp.meshgrid(range(nely), range(nelx))

    # Compute node indices for each element (same logic as in fem.py)
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    all_ixs = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

    # Gather element displacement vectors: shape (8, nelx, nely)
    u_selected = u[all_ixs]

    # Compute ke @ u_e for each element (einsum over 8x8 matrix × 8-vec)
    ke_u = anp.einsum("ij,jkl->ikl", ke, u_selected)

    # Compute u_e^T @ ke @ u_e for each element
    ce = anp.einsum("ijk,ijk->jk", u_selected, ke_u)

    # Scale by local Young's modulus and sum
    C = young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
    return anp.sum(C)


# ─────────────────────────────────────────────────────────────────────────────
# FULL OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def objective(x, args, use_filter=True):
    """
    Full compliance objective: x_raw → x_phys → U → compliance.

    This is the function we minimize. Autograd differentiates through the
    entire chain: density processing → FEM solve → compliance calculation.

    Args:
        x:          raw density parameters (1D flat array)
        args:       problem arguments
        use_filter: whether to apply Gaussian smoothing in density processing

    Returns:
        c: scalar compliance value
    """
    kwargs = dict(penal=args.penal, e_min=args.young_min, e_0=args.young)

    # Step 1: Convert raw params to physical densities
    x_phys = physical_density(x, args, use_filter=use_filter)

    # Step 2: Compute element stiffness matrix (constant for given material)
    ke = get_stiffness_matrix(args.young, args.poisson)

    # Step 3: Solve for nodal displacements via FEM (most expensive step)
    u = displace(x_phys, ke, args.forces, args.freedofs, args.fixdofs, **kwargs)

    # Step 4: Compute compliance from displacements
    c = compliance(x_phys, u, ke, **kwargs)

    return c
