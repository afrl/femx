"""Small conversion/AD checks for the explicit assembled-CSR FEA option.

Run on the configured Linux test environment; no PETSc solve is needed here.
"""

from helper import *
from fe_jax.sparse_matrix import coo_to_csr
from fe_jax.sparse_linear_solve import (
    Residual, Jacobian, JacobianDiagonl, linear_solve, init_solver_info,
)
from fe_jax.constraint_system import ConstraintSystem
from fe_jax.solve_gmres import gmres_true_residual
import jax.experimental.sparse as jsparse
import pytest
import importlib


def test_coo_to_csr_unsorted_empty_row_and_jvp():
    jax.config.update("jax_enable_x64", True)
    rows, cols = jnp.array([2, 0, 3, 0]), jnp.array([2, 1, 3, 0])
    values = jnp.array([5., -1., 0., 4.])
    vector = jnp.array([.2, -.3, .4, 1.])

    @jax.jit
    def multiply(data):
        return coo_to_csr(jsparse.COO((data, rows, cols), shape=(4, 4))) @ vector

    coo = jsparse.COO((values, rows, cols), shape=(4, 4))
    np.testing.assert_allclose(multiply(values), coo @ vector)
    tangent = jnp.array([.1, .2, -.1, .3])
    _, actual = jax.jvp(multiply, (values,), (tangent,))
    np.testing.assert_allclose(actual, jsparse.COO((tangent, rows, cols), shape=(4, 4)) @ vector)
    with pytest.raises(TypeError, match="expects a JAX COO"):
        coo_to_csr(jnp.eye(2))


@pytest.mark.parametrize("kwargs", [
    {"gmres_solve_method": "invalid"}, {"gmres_restart": 0},
])
def test_gmres_diagnostic_options_reject_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SolverOptions(**kwargs)


@pytest.mark.parametrize("solve_method", ["batched", "incremental"])
def test_gmres_true_residual_with_small_left_preconditioned_residual(solve_method):
    jax.config.update("jax_enable_x64", True)
    # Typical constrained stiffness scaling: unit BC row, large free entries.
    a = jnp.array([[1., 0., 0.], [0., 4e12, 1e12], [0., .2e12, 3e12]])
    rows, cols = jnp.repeat(jnp.arange(3), 3), jnp.tile(jnp.arange(3), 3)
    csr = coo_to_csr(jsparse.COO((a.ravel(), rows, cols), shape=(3, 3)))
    pc = lambda r: r / jnp.diag(a)

    def solve(p):
        b = jnp.concatenate((jnp.zeros(1), p))
        return gmres_true_residual(
            csr, b, M=pc, tol=1e-10, atol=1e-14,
            restart=3, maxiter=5, solve_method=solve_method,
        )

    p, direction = jnp.array([1., -2.]), jnp.array([.2, .3])
    b = jnp.concatenate((jnp.zeros(1), p))
    # Establish exactly the premature-stop condition seen in the BVP log.
    assert float(jnp.linalg.norm(pc(b))) < 1e-10 * float(jnp.linalg.norm(b))
    x, status = jax.jit(solve)(p)
    assert int(status) == 0
    assert float(jnp.linalg.norm(b - csr @ x)) <= 1e-10 * float(jnp.linalg.norm(b))
    reference = lambda q: jnp.linalg.solve(a, jnp.concatenate((jnp.zeros(1), q)))
    np.testing.assert_allclose(x, reference(p), rtol=1e-9, atol=1e-23)
    actual_jvp = jax.jit(lambda q, d: jax.jvp(lambda z: solve(z)[0], (q,), (d,)))(p, direction)
    for actual, expected in zip(actual_jvp, jax.jvp(reference, (p,), (direction,)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-23)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda q: 1e12 * jnp.sum(solve(q)[0])))(p),
        jax.grad(lambda q: 1e12 * jnp.sum(reference(q)))(p), rtol=1e-9, atol=1e-12,
    )


def test_gmres_true_residual_reports_exhaustion_and_accepts_zero_rhs():
    jax.config.update("jax_enable_x64", True)
    a = jnp.array([[3., 2.], [0., 1.]])
    x, status = gmres_true_residual(
        a, jnp.array([1., -1.]), restart=1, maxiter=1, tol=1e-12,
    )
    assert np.isfinite(np.asarray(x)).all()
    assert int(status) == 1  # Finite is not synonymous with converged.
    zero, status = gmres_true_residual(a, jnp.zeros(2), tol=1e-12)
    np.testing.assert_array_equal(zero, jnp.zeros(2))
    assert int(status) == 0


@pytest.mark.parametrize("use_jax_stream", [False, True])
def test_csr_ilu_synchronized_jvp_resource_lifetime(use_jax_stream):
    linear = importlib.import_module("fe_jax.sparse_linear_solve")
    if not linear.CUPY_AVAILABLE or jax.default_backend() != "gpu":
        pytest.skip("Requires the Linux CUDA/CuPy environment for synchronized ILU JVP")
    jax.config.update("jax_enable_x64", True)
    constraints = ConstraintSystem(
        dep_dofs=jnp.zeros(0, dtype=jnp.int32), g=jnp.zeros(0),
        P=jsparse.BCOO((jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, 2)),
    )
    settings = SolverOptions(
        linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY_CSR,
        linear_precond_type=PreconditionerType.ILU_CUPY,
        gmres_solve_method="incremental", ilu_synchronize=not use_jax_stream,
        ilu_use_jax_stream=use_jax_stream,
        linear_relative_tol=1e-12, linear_absolute_tol=1e-14, linear_max_iter=30,
    )
    rows, cols = jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1])

    def matrix(p):
        return jnp.array([[3. + p[0], .2], [.2, 2. + p[0]]])

    def solve(p, owner):
        a, b = matrix(p), p[1:]
        return linear_solve(
            residual=Residual(jax.tree_util.Partial(lambda a, b, x: a @ x - b, a, b), True),
            jacobian=Jacobian(jax.tree_util.Partial(
                lambda a, x: jsparse.COO((a.ravel(), rows, cols), shape=(2, 2)), a), True),
            jacobian_diagonal=None, constraints=constraints, solver_options=settings,
            solver_info_0=init_solver_info(settings), check_consistency=False,
            x_0=jnp.zeros(2), f_ext=None, resource_owner=owner,
        )[0]

    compiled = jax.jit(lambda p, d, owner: jax.jvp(lambda q: solve(q, owner), (p,), (d,)))
    p, direction = jnp.array([.5, 1., -.3]), jnp.array([.2, -.1, .4])
    expected = jax.jvp(lambda q: jnp.linalg.solve(matrix(q), q[1:]), (p,), (direction,))
    before = len(linear._OBJECT_STORE)
    for _ in range(2):
        with linear.cupy_factor_scope() as owner:
            result = jax.tree_util.tree_map(lambda x: x.block_until_ready(), compiled(p, direction, owner))
            for actual, reference in zip(result, expected, strict=True):
                np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-11)
        assert len(linear._OBJECT_STORE) == before


@pytest.mark.parametrize("provide_diagonal", [False, True])
@pytest.mark.parametrize("method, solve_method", [
    (LinearSolverType.CG_JAX_SCIPY_CSR, "batched"),
    (LinearSolverType.GMRES_JAX_SCIPY_CSR, "batched"),
    (LinearSolverType.GMRES_JAX_SCIPY_CSR, "incremental"),
])
def test_csr_linear_solve_parameter_jvp_and_reverse(provide_diagonal, method, solve_method):
    jax.config.update("jax_enable_x64", True)
    constraints = ConstraintSystem(
        dep_dofs=jnp.zeros(0, dtype=jnp.int32), g=jnp.zeros(0),
        P=jsparse.BCOO((jnp.zeros(0), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, 2)),
    )
    settings = SolverOptions(
        linear_solve_type=method,
        linear_precond_type=PreconditionerType.JACOBI,
        linear_relative_tol=1e-12, linear_absolute_tol=1e-14, linear_max_iter=50,
        gmres_solve_method=solve_method,
    )

    def matrix(p):
        return jnp.array([[3. + p[0], .2], [.2, 2. + p[0]]])

    def jacobian(a, b, x):
        return jsparse.COO((a.reshape(-1), jnp.array([0, 0, 1, 1]),
                            jnp.array([0, 1, 0, 1])), shape=(2, 2))

    def unused_diagonal(x):
        raise AssertionError("CSR Jacobi must reuse the assembled matrix, not the diagonal callback")

    def solve(p):
        a, b = matrix(p), p[1:]
        return linear_solve(
            residual=Residual(jax.tree_util.Partial(lambda a, b, x: a @ x - b, a, b), True),
            jacobian=Jacobian(jax.tree_util.Partial(jacobian, a, b), True),
            jacobian_diagonal=(JacobianDiagonl(jax.tree_util.Partial(unused_diagonal), True)
                               if provide_diagonal else None),
            constraints=constraints, solver_options=settings, solver_info_0=init_solver_info(settings),
            check_consistency=False, x_0=jnp.zeros(2), f_ext=None,
        )[0]

    expected = lambda p: jnp.linalg.solve(matrix(p), p[1:])
    p, direction = jnp.array([.5, 1., -.3]), jnp.array([.2, -.1, .4])
    np.testing.assert_allclose(jax.jit(solve)(p), expected(p), atol=1e-11)
    actual_jvp = jax.jit(lambda p, d: jax.jvp(solve, (p,), (d,)))(p, direction)
    for actual, reference in zip(actual_jvp, jax.jvp(expected, (p,), (direction,)), strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda p: jnp.sum(solve(p)**2)))(p),
                               jax.grad(lambda p: jnp.sum(expected(p)**2))(p), atol=1e-11)
