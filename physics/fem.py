# physics/fem.py
# ─────────────────────────────────────────────────────────────────────────────
# Finite Element Method (FEM) physics engine.
#
# This module implements the core mechanics of 2D elastic structural analysis:
#   1. Building the element stiffness matrix (ke)
#   2. Assembling the global sparse stiffness matrix (K)
#   3. Solving the linear system K*U = F to get nodal displacements (U)
#
# The design space is a grid of square finite elements. Each element has 4
# corner nodes, each with 2 degrees of freedom (horizontal + vertical).
# Adjacent elements share nodes, so forces propagate through the mesh.
#
# Reference: Andreassen et al. (2010), "Efficient topology optimization in
# MATLAB using 88 lines of code."
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import autograd
import autograd.numpy as anp


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT STIFFNESS MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def get_stiffness_matrix(e, nu):
    """
    Compute the 8x8 element stiffness matrix for a unit square finite element.

    This matrix encodes how the 4 corner nodes of a square element resist
    deformation. It's the 2D analogue of a spring constant k in F=kx, but
    generalized to 8 DOFs (2 per corner node: horizontal + vertical).

    The potential energy of one element is: PE = 0.5 * u.T @ ke @ u
    The full structure's compliance is the sum of all elements' PE.

    Args:
        e:  Young's modulus (material stiffness)
        nu: Poisson's ratio (lateral contraction ratio)

    Returns:
        ke: (8, 8) element stiffness matrix
    """
    k = anp.array([
        1/2 - nu/6,   1/8 + nu/8,  -1/4 - nu/12, -1/8 + 3*nu/8,
       -1/4 + nu/12, -1/8 - nu/8,   nu/6,          1/8 - 3*nu/8
    ])
    return e / (1 - nu**2) * anp.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STIFFNESS MATRIX ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def get_k(stiffness, ke):
    """
    Assemble the global stiffness matrix K in sparse COO format.

    Instead of building a dense 2N×2N matrix (too large for memory), we store
    only the nonzero entries as (value, row, col) triplets. Each finite element
    contributes 64 entries (8 DOFs × 8 DOFs) to K.

    Args:
        stiffness: (nely, nelx) array of per-element Young's moduli
        ke:        (8, 8) element stiffness matrix

    Returns:
        value_list: flat array of matrix values
        y_list:     flat array of row indices
        x_list:     flat array of column indices
    """
    nely, nelx = stiffness.shape

    # Build index grid — (elx[i,j], ely[i,j]) gives the col/row of element (i,j)
    ely, elx = anp.meshgrid(range(nely), range(nelx))
    ely = ely.reshape(-1, 1)
    elx = elx.reshape(-1, 1)

    # Compute the 4 node indices for each element (column-major ordering)
    # Node numbering: n1=top-left, n2=top-right, n3=bottom-right, n4=bottom-left
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)

    # Each node has 2 DOFs: 2n = horizontal, 2n+1 = vertical
    edof = anp.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
    edof = edof.T[0]  # shape: (nelx*nely, 8)

    # Build COO index lists by repeating/tiling the DOF indices
    x_list = anp.repeat(edof, 8)        # row indices, repeated 8 times per element
    y_list = anp.tile(edof, 8).flatten()  # col indices, tiled 8 times per element

    # Scale each element's ke by its local Young's modulus
    kd = stiffness.T.reshape(nelx * nely, 1, 1)
    value_list = (kd * anp.tile(ke, kd.shape)).flatten()

    return value_list, y_list, x_list


# ─────────────────────────────────────────────────────────────────────────────
# DEGREE OF FREEDOM (DOF) INDEX MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def inverse_permutation(indices):
    """
    Compute the inverse of an index permutation.

    If 'indices' permutes array A, returns inv such that A[indices][inv] == A.
    Used to map between free/fixed DOF ordering and original DOF ordering.
    """
    inverse_perm = np.zeros(len(indices), dtype=anp.int64)
    inverse_perm[indices] = np.arange(len(indices), dtype=anp.int64)
    return inverse_perm


def get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
    """
    Filter the global stiffness matrix to retain only free DOFs.

    Fixed DOFs (where boundary conditions are applied) don't need to be solved
    for — we already know their displacements are zero. Removing them reduces
    the solve from 2N equations to len(freedofs) equations.

    Args:
        freedofs: indices of unconstrained DOFs
        fixdofs:  indices of constrained (fixed) DOFs
        k_xlist:  COO row indices of K
        k_ylist:  COO col indices of K

    Returns:
        index_map: permutation to recover original DOF ordering after solve
        keep:      boolean mask selecting entries where both DOFs are free
        indices:   stacked [row, col] indices for the reduced sparse system
    """
    index_map = inverse_permutation(anp.concatenate([freedofs, fixdofs]))
    keep = anp.isin(k_xlist, freedofs) & anp.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    return index_map, keep, anp.stack([i, j])


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE LINEAR SYSTEM SOLVE (with custom Autograd gradient)
# ─────────────────────────────────────────────────────────────────────────────

def get_solver(a_entries, a_indices, size):
    """Build and factorize the sparse stiffness matrix using SuperLU."""
    a = scipy.sparse.coo_matrix(
        (a_entries, a_indices), shape=(size, size)
    ).tocsc()
    return scipy.sparse.linalg.splu(a).solve


@autograd.primitive
def solve_coo(a_entries, a_indices, b, sym_pos=False):
    """
    Solve the sparse linear system A*x = b.

    This is the computational bottleneck — it's where most runtime is spent.
    We register it as an Autograd primitive so we can differentiate through it.

    Args:
        a_entries: nonzero values of A (COO format)
        a_indices: [row, col] indices of nonzero values
        b:         right-hand side vector (applied forces at free DOFs)
        sym_pos:   whether A is symmetric positive definite

    Returns:
        x: solution vector (nodal displacements at free DOFs)
    """
    solver = get_solver(a_entries, a_indices, b.size)
    return solver(b)


def grad_solve_coo_entries(ans, a_entries, a_indices, b, sym_pos=False):
    """
    Custom reverse-mode gradient for the sparse solve.

    Uses the identity: d(A^{-1}b)/dA = -A^{-T} * grad * x^T
    See Giles (2008), "An extended collection of matrix derivative results."

    This allows Autograd to differentiate through the physics simulation
    with respect to the material densities x.
    """
    def jvp(grad_ans):
        lambda_ = solve_coo(
            a_entries,
            a_indices if sym_pos else a_indices[::-1],
            grad_ans,
            sym_pos
        )
        i, j = a_indices
        return -lambda_[i] * ans[j]
    return jvp


# Register the custom gradient with Autograd
autograd.extend.defvjp(
    solve_coo,
    grad_solve_coo_entries,
    lambda: print("err: gradient w.r.t. indices undefined"),
    lambda: print("err: gradient w.r.t. b not implemented"),
)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLACEMENT SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1):
    """
    Compute nodal displacements U by solving K(x)*U = F.

    This is the forward physics pass. Given a material density distribution x,
    it:
      1. Computes per-element Young's moduli via the SIMP rule
      2. Assembles the global stiffness matrix K
      3. Solves K*U = F for the nodal displacement vector U

    Args:
        x_phys:   (nely, nelx) physical density field (0=void, 1=solid)
        ke:       (8, 8) element stiffness matrix
        forces:   (2N,) vector of applied forces at each DOF
        freedofs: indices of unconstrained DOFs
        fixdofs:  indices of constrained DOFs (displacement = 0)
        penal:    SIMP penalization exponent
        e_min:    minimum Young's modulus (prevents singular K)
        e_0:      baseline Young's modulus

    Returns:
        u: (2N,) vector of nodal displacements (zero at fixdofs)
    """
    from physics.objective import young_modulus

    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)

    index_map, keep, indices = get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist)

    # Solve the reduced system (only free DOFs)
    u_nonzero = solve_coo(k_entries[keep], indices, forces[freedofs], sym_pos=True)

    # Reconstruct full displacement vector (fixed DOFs have zero displacement)
    u_values = anp.concatenate([u_nonzero, anp.zeros(len(fixdofs))])
    return u_values[index_map]
