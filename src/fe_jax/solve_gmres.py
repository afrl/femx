"""JAX GMRES with a true-residual restart-loop stopping test.

Uses JAX's existing left-preconditioned Arnoldi/least-squares kernels unchanged.
Only the restart driver and returned status are maintained locally. This is
not stock jax.scipy GMRES: benchmark reports must disclose the stopping fix.
Private kernel signatures below match JAX 0.10.1; no installed code is patched.
"""
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_structure
from jax._src.scipy.sparse.linalg import (
    _gmres_batched, _gmres_incremental, _normalize_matvec,
    _safe_normalize, _norm,
)


def gmres_true_residual(A, b, x0=None, *, tol=1e-5, atol=0.0,
                        restart=20, maxiter=None, M=None, solve_method="batched"):
    """Match JAX GMRES's left-preconditioned solve, but test ||b-Ax||.

    maxiter counts restart cycles, as in JAX. Status: 0 means true residual
    accepted, 1 means finite but not converged, -1 means nonfinite output.
    The inner kernel retains JAX's preconditioned tolerance calculation.
    Transpose preconditioning follows JAX's existing convention (same M).
    """
    kernels = {"batched": _gmres_batched, "incremental": _gmres_incremental}
    if solve_method not in kernels:
        raise ValueError("Expected solve_method='batched' or 'incremental'; no fallback is used.")
    if restart < 1 or (maxiter is not None and maxiter < 1):
        raise ValueError("restart and maxiter must be positive; maxiter counts restart cycles.")
    operator = _normalize_matvec(A)
    precondition = (lambda x: x) if M is None else _normalize_matvec(M)
    if x0 is None:
        x0 = tree_map(jnp.zeros_like, b)
    b, x0 = jax.device_put((b, x0))
    if tree_structure(b) != tree_structure(x0) or [x.shape for x in tree_leaves(b)] != [x.shape for x in tree_leaves(x0)]:
        raise ValueError("GMRES expects matching RHS/initial-guess shapes and pytree structures.")
    size = sum(x.size for x in tree_leaves(b))
    restart = min(restart, size)
    if maxiter is None:
        maxiter = 10 * size
    kernel = kernels[solve_method]

    def subtract(x, y):
        return tree_map(lambda a, c: a - c, x, y)

    def solve(active_operator, rhs):
        # Recompute the bound for the actual RHS, including implicit AD solves.
        bnorm = _norm(rhs)
        bound = jnp.maximum(tol * bnorm, atol)
        denominator = jnp.where(bnorm > 0, bnorm, 1.)
        ptol = _norm(precondition(rhs)) * jnp.minimum(1., bound / denominator)
        r = subtract(rhs, active_operator(x0))
        unit, pc_norm = _safe_normalize(precondition(r))

        def condition(carry):
            _, count, _, _, true_norm = carry
            return (count < maxiter) & jnp.isfinite(true_norm) & (true_norm > bound)

        def step(carry):
            x, count, unit, pc_norm, _ = carry
            x, unit, pc_norm = kernel(
                active_operator, rhs, x, unit, pc_norm, ptol, restart, precondition,
            )
            true_norm = _norm(subtract(rhs, active_operator(x)))
            return x, count + 1, unit, pc_norm, true_norm

        result, *_ = jax.lax.while_loop(
            condition, step, (x0, 0, unit, pc_norm, _norm(r)),
        )
        return result

    x = jax.lax.custom_linear_solve(operator, b, solve=solve, transpose_solve=solve)
    residual_norm = _norm(subtract(b, operator(x)))
    bound = jnp.maximum(tol * _norm(b), atol)
    finite = jnp.isfinite(_norm(x)) & jnp.isfinite(residual_norm)
    status = jnp.where(~finite, -1, jnp.where(residual_norm <= bound, 0, 1))
    return x, status
