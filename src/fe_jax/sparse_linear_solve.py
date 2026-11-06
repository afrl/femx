import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.extend as jextend
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental import checkify
from jax.dlpack import from_dlpack

# For CPU solver
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from flax import struct
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Optional
from functools import partial, wraps
import inspect
import warnings
from contextlib import contextmanager
import logging

from .utils import debug_print
from .sparse_matrix import *
from .constraint_system import ConstraintSystem
from .solve_cg import cg as cg_w_info
from .solve_gmres import gmres_true_residual

_logger = logging.getLogger(__name__)

try:
    import jaxopt.linear_solve

    JAXOPT_AVAILABLE = True
    _logger.info("'jaxopt' imported, adding related solvers.")
except ImportError:
    JAXOPT_AVAILABLE = False
    _logger.info("'jaxopt' was not imported successfully, skipping related solvers.")

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg

    CUPY_AVAILABLE = True
    _logger.info("'cupy' imported, adding related solvers/preconditioners.")
except ImportError:
    CUPY_AVAILABLE = False
    _logger.info(
        "'cupy' was not imported successfully, skipping related solvers/preconditioners."
    )


try:
    import pyamgx

    assert (
        CUPY_AVAILABLE
    ), "The interface to `pyamgx` requires `cupy`, which was not available. Please install `cupy`."
    PYAMX_AVAILABLE = True
    _logger.info("'pyamgx' imported, adding related solvers/preconditioners.")
except ImportError:
    PYAMX_AVAILABLE = False
    _logger.info(
        "'pyamgx' was not imported successfully, skipping related solvers/preconditioners."
    )

try:
    import pypardiso

    if not CUPY_AVAILABLE:
        _logger.warning(
            "Warning: 'pypardiso' was imported successfully, but 'cupy' is not available. 'pypardiso' will not be available."
        )
    assert CUPY_AVAILABLE
    PYPARDISO_AVAILABLE = True
    _logger.info("'pypardiso' imported, adding related solvers.")
except ImportError:
    PYPARDISO_AVAILABLE = False
    _logger.info("'pypardiso' was not imported successfully, skipping related solvers.")


class PreconditionerType(Enum):
    NONE = auto()
    JACOBI = auto()
    ILU_CUPY = auto()


class LinearSolverType(Enum):
    DENSE_INVERSE_JNP = auto()
    CG_JAX_SCIPY = auto()
    CG_JAX_SCIPY_W_INFO = auto()
    GMRES_JAX_SCIPY = auto()
    BICGSTAB_JAX_SCIPY = auto()
    DENSE_INVERSE_JAXOPT = auto()
    LU_JAXOPT = auto()
    CHOLESKY_JAXOPT = auto()
    CG_JAXOPT = auto()
    GMRES_JAXOPT = auto()
    BICGSTAB_JAXOPT = auto()
    SPSOLVE_CUPY = auto()
    LU_CUPY = auto()
    # TODO GMRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.gmres.html
    # TODO CGS_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.cgs.html
    # TODO MINRES_CUPY : https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.sparse.linalg.minres.html
    AMGX = auto()
    SPSOLVE_PYPARDISO = auto()
    # Explicit assembled CSR alternative; existing CG options stay matrix-free.
    CG_JAX_SCIPY_CSR = auto()
    GMRES_JAX_SCIPY_CSR = auto()


@dataclass(eq=True, frozen=True)
class SolverOptions:
    linear_precond_type: PreconditionerType = PreconditionerType.NONE
    linear_solve_type: LinearSolverType = LinearSolverType.CG_JAX_SCIPY_W_INFO
    linear_max_iter: int = 1000
    linear_relative_tol: float = 1e-14
    linear_absolute_tol: float = 1e-10
    nonlinear_max_iter: int = 10
    nonlinear_relative_tol: float = 1e-10
    nonlinear_absolute_tol: float = 1e-8
    gmres_solve_method: str = "batched"
    gmres_restart: int = 30
    gmres_stopping_test: str = "true_residual"
    linear_diagnostics: bool = False
    ilu_application_diagnostics: bool = False
    # Diagnostic only: device-wide fences around each CuPy ILU application.
    # Disabled by default; not a performance-oriented stream handoff.
    ilu_synchronize: bool = False
    ilu_use_jax_stream: bool = False

    def __post_init__(self):
        if self.ilu_use_jax_stream and self.ilu_synchronize:
            raise ValueError("Select either ilu_use_jax_stream=True or ilu_synchronize=True, not both; the latter is the device-fenced reference mode.")
        if self.gmres_stopping_test not in ("true_residual", "jax"):
            raise ValueError("gmres_stopping_test must be 'true_residual' (local stopping fix) or 'jax' (unmodified JAX stopping behavior).")
        if self.gmres_solve_method not in ("batched", "incremental"):
            raise ValueError("gmres_solve_method must be 'batched' or 'incremental'; no solver fallback is performed.")
        if not isinstance(self.gmres_restart, int) or self.gmres_restart < 1:
            raise ValueError("gmres_restart must be a positive integer Krylov subspace size.")
        # Validate that the selected preconditioner is available
        if "CUPY" in self.linear_precond_type.name:
            assert (
                CUPY_AVAILABLE
            ), f"The selected preconditioner ({self.linear_precond_type.name}) requires cupy, which is not available."
        # Validate that the selected solver is available
        if "JAXOPT" in self.linear_solve_type.name:
            assert (
                JAXOPT_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires jaxopt, which is not available."
        if "CUPY" in self.linear_solve_type.name:
            assert (
                CUPY_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires cupy, which is not available."
        if "AMGX" in self.linear_solve_type.name:
            assert (
                PYAMX_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires pyamgx, which is not available."
        if "PYPARDISO" in self.linear_solve_type.name:
            assert (
                PYPARDISO_AVAILABLE
            ), f"The selected solver ({self.linear_solve_type.name}) requires pypardiso, which is not available."


@struct.dataclass
class SolverResultInfo:
    nonlinear_iterations: int
    cumulative_linear_iterations: int
    linear_iterations_per_nonlinear_iteration: jnp.ndarray
    # NOTE length will be nonlinear_iterations + cumulative_linear_iterations because the residual
    # norm history for each nonlinear iteration begins with the starting residual norm before a
    # linear solve
    cumulative_residual_norm_history: jnp.ndarray

    def increment_nl_iteration(self):
        """
        Call at the end of each nonlinear iteration to create a copy of the struct that carries
        the solver history forward.
        """
        return SolverResultInfo(
            nonlinear_iterations=self.nonlinear_iterations + 1,
            cumulative_linear_iterations=self.cumulative_linear_iterations,
            linear_iterations_per_nonlinear_iteration=self.linear_iterations_per_nonlinear_iteration,
            cumulative_residual_norm_history=self.cumulative_residual_norm_history,
        )


def init_solver_info(opts: SolverOptions):
    """
    Initialize a SolverResultInfo object to begin tracking solves.
    """
    return SolverResultInfo(
        nonlinear_iterations=0,
        cumulative_linear_iterations=0,
        linear_iterations_per_nonlinear_iteration=jnp.zeros((opts.nonlinear_max_iter,)),
        cumulative_residual_norm_history=jnp.zeros(
            (opts.linear_max_iter * opts.nonlinear_max_iter + 1,)
        ),
    )


def finite_residual_accepted(residual, bound):
    """Reject nonfinite entries, norm overflow, and invalid stopping bounds."""
    norm = jnp.linalg.norm(residual)
    return (jnp.all(jnp.isfinite(residual)) & jnp.isfinite(norm)
            & jnp.isfinite(bound) & (norm <= bound))


@struct.dataclass
class Residual:
    # Function that produces the residual vector (jnp.ndarray).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jax.Array]
    # Indicates whether Dirichlet boundary conditions are built into the residual.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class Jacobian:
    # Function that produces the sparse matrix (jsparse.COO).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jsparse.COO]
    # Indicates whether Dirichlet boundary conditions are built into the Jacobian.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@struct.dataclass
class JacobianDiagonl:
    # Function that produces the diagonal of the Jacobian matrix (jnp.ndarray).
    # NOTE the solution must be the first argument, though additional args can follow.
    function: Callable[[jax.Array], jax.Array]
    # Indicates whether Dirichlet boundary conditions are built into the Jacobian.
    dirichlet_bcs_builtin: bool = struct.field(pytree_node=False)


@partial(jax.jit, static_argnames=["solver_options", "check_consistency"])
def linear_solve(
    residual: Residual,
    jacobian: Optional[Jacobian],
    jacobian_diagonal: Optional[JacobianDiagonl],
    constraints: ConstraintSystem,
    solver_options: SolverOptions,
    solver_info_0: SolverResultInfo,
    check_consistency: bool,
    x_0: jnp.ndarray,
    f_ext,
    *args,
    resource_owner=None,
    **kwargs,
) -> tuple[jnp.ndarray, SolverResultInfo]:
    """
    Solve a linear system of equations emerging from Newton's method: J(x) * x = -R(x)

    TODO finish documentation
    """

    if residual.dirichlet_bcs_builtin:
        R_w_dirichlet = lambda x: residual.function(x, *args, **kwargs)
    else:
        raise Exception("TODO (straightforward) implementation needed")

    if jacobian is not None:
        if jacobian.dirichlet_bcs_builtin:
            J_w_dirichlet = lambda x: jacobian.function(x, *args, **kwargs)
        else:
            J_w_dirichlet = lambda x: apply_dirichlet_bcs_lhs(
                jacobian.function(x, *args, **kwargs), constraints.dep_dofs
            )
    else:
        J_w_dirichlet = None

    if jacobian_diagonal is not None:
        if jacobian_diagonal.dirichlet_bcs_builtin:
            diag_J_w_dirichlet = lambda x: jacobian_diagonal.function(
                x, *args, **kwargs
            )
        else:
            diag_J_w_dirichlet = (
                lambda x: jacobian_diagonal.function(x, *args, **kwargs)
                .at[constraints.dep_dofs]
                .set(1.0)
            )
    else:
        diag_J_w_dirichlet = None

    J_vp = jax.tree_util.Partial(
        lambda x, z: jax.jvp(
            R_w_dirichlet,
            (x,),
            (z,),
        )[1],
        x_0,
    )

    if check_consistency:
        v = jax.random.uniform(jax.random.key(0), x_0.shape, x_0.dtype)
        J_dense = jax.jacfwd(R_w_dirichlet)(x_0)

        jax.debug.print(
            "Jacobian-vector product via autodiff matches product via dense Jacobian from jacfwd: {}",
            jnp.isclose(J_vp(v), J_dense @ v).all(),
        )

        if J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian inferred from residual (via jacfwd) matches given Jacobian function: {}",
                jnp.isclose(J_dense, J_w_dirichlet(x_0).todense()).all(),
            )

        if diag_J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian diagonal inferred from residual (via diag(jacfwd)) matches Jacobian diagonal function: {}",
                jnp.isclose(jnp.diag(J_dense), diag_J_w_dirichlet(x_0)).all(),
            )

        if J_w_dirichlet is not None:
            jax.debug.print(
                "Jacobian-vector product via autodiff matches product via the given Jacobian function: {}",
                jnp.isclose(J_vp(v), J_w_dirichlet(x_0) @ v).all(),
            )

    # breakpoint()
    # R_0 = f_ext.apply_to_residual(R_w_dirichlet(x_0))
    R_0 = R_w_dirichlet(x_0)
    delta_x = constraints.apply_to_solution(jnp.zeros_like(R_0))
    info = solver_info_0

    use_csr = solver_options.linear_solve_type in (
        LinearSolverType.CG_JAX_SCIPY_CSR, LinearSolverType.GMRES_JAX_SCIPY_CSR,
    )
    if use_csr:
        if J_w_dirichlet is None:
            raise ValueError("CSR solves require an explicit sparse Jacobian function; provide Jacobian(...) or choose a matrix-free solver. No dense Jacobian will be inferred.")
        if constraints.P.nse:
            raise ValueError("CSR solves currently support fixed Dirichlet constraints only; assembled multipoint-constraint Jacobians are not implemented. Use a matrix-free solver for MPCs.")
        J_coo = J_w_dirichlet(x_0)
        J_csr = coo_to_csr(J_coo)

    match solver_options.linear_precond_type:
        ##########################################################################################
        # jax native preconditioners
        case PreconditionerType.NONE:
            preconditioner = None

        case PreconditionerType.JACOBI:
            if use_csr:
                # Reuse the constrained matrix assembled for CG. Scatter-add
                # also handles duplicate diagonal coordinates without densifying.
                diagonal = jnp.zeros_like(R_0).at[J_coo.row].add(
                    jnp.where(J_coo.row == J_coo.col, J_coo.data, 0.0)
                )
                preconditioner = lambda x: x / diagonal
            else:
                assert (
                    diag_J_w_dirichlet is not None
                ), f"{solver_options.linear_precond_type} requires the `jacobian_diagonal` argument to be provided."

                preconditioner = lambda x: x / diag_J_w_dirichlet(x_0)

        ##########################################################################################
        # cupy preconditioners

        case PreconditionerType.ILU_CUPY:
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            # The CSR branch has already assembled its constrained operator.
            J_sparse = J_coo if use_csr else J_w_dirichlet(x_0)
            ilu_ctx = __cupy_spilu_init(J_sparse, diagnostics=solver_options.linear_diagnostics,
                                      resource_owner=resource_owner,
                                      complete_setup=solver_options.ilu_use_jax_stream)
            preconditioner = lambda x: __cupy_solve(
                ilu_ctx, x, synchronize=solver_options.ilu_synchronize,
                use_jax_stream=solver_options.ilu_use_jax_stream,
            )
            if solver_options.ilu_application_diagnostics:
                # Diagnostic host copies only: compare the normal GPU callback
                # with direct CuPy using the exact same live factor object.
                audit_rhs = -R_0
                audit_x = jax.random.normal(jax.random.key(71), audit_rhs.shape, dtype=audit_rhs.dtype)
                audit_y = jax.random.normal(jax.random.key(72), audit_rhs.shape, dtype=audit_rhs.dtype)
                audit_x = audit_x.at[constraints.dep_dofs].set(0.)
                audit_y = audit_y.at[constraints.dep_dofs].set(0.)
                audit_csr = J_csr if use_csr else coo_to_csr(J_sparse)
                jax.debug.callback(
                    partial(__cupy_ilu_application_audit,
                            maxiter=solver_options.linear_max_iter,
                            tol=solver_options.linear_relative_tol,
                            atol=solver_options.linear_absolute_tol),
                    ilu_ctx.handle, info.nonlinear_iterations,
                    audit_rhs, preconditioner(audit_rhs), preconditioner(audit_rhs),
                    audit_x, audit_y, preconditioner(audit_x), preconditioner(audit_y),
                    audit_csr.data, audit_csr.indices, audit_csr.indptr,
                    delta_x, constraints.dep_dofs,
                    ordered=True,
                )

        case _:
            raise Exception(
                f"Preconditioner type {solver_options.linear_precond_type} is not implemented"
            )

    diagnose_cg = solver_options.linear_diagnostics and solver_options.linear_solve_type in (
        LinearSolverType.CG_JAX_SCIPY, LinearSolverType.CG_JAX_SCIPY_CSR,
        LinearSolverType.CG_JAX_SCIPY_W_INFO,
    )
    if diagnose_cg:
        diagnostic_operator = (lambda v: J_csr @ v) if use_csr else J_vp
        diagnostic_pc = preconditioner if preconditioner is not None else lambda v: v
        cg_rhs = -R_0
        cg_initial_r = cg_rhs - diagnostic_operator(delta_x)
        cg_pc_rhs = diagnostic_pc(cg_rhs)
        cg_pc_probe = diagnostic_pc(jax.random.normal(jax.random.key(42), cg_rhs.shape, dtype=cg_rhs.dtype))
        cg_bound = jnp.maximum(solver_options.linear_absolute_tol,
                               solver_options.linear_relative_tol * jnp.linalg.norm(cg_rhs))
        jax.debug.print(
            "BVP_JAX_CG_INPUT step={step} rhs_norm={bn} initial_true_residual={rn} "
            "true_bound={bound} rhs_finite={bf} pc_rhs_finite={mf} pc_probe_finite={pf}",
            step=info.nonlinear_iterations, bn=jnp.linalg.norm(cg_rhs),
            rn=jnp.linalg.norm(cg_initial_r), bound=cg_bound,
            bf=jnp.all(jnp.isfinite(cg_rhs)), mf=jnp.all(jnp.isfinite(cg_pc_rhs)),
            pf=jnp.all(jnp.isfinite(cg_pc_probe)), ordered=True,
        )
        if solver_options.ilu_application_diagnostics:
            # Separate short replay, not a replacement for native JAX CG.
            def trace_first_steps(_):
                r = cg_rhs - diagnostic_operator(delta_x)
                z = diagnostic_pc(r)
                gamma = jnp.vdot(r, z).real

                def condition(carry):
                    k, x, r, p, gamma = carry
                    rn = jnp.linalg.norm(r)
                    return (k < 8) & jnp.isfinite(rn) & (rn > cg_bound) & jnp.isfinite(gamma)

                def step(carry):
                    k, x, r, p, gamma = carry
                    ap = diagnostic_operator(p)
                    pap = jnp.vdot(p, ap).real
                    alpha = gamma / pap
                    x = x + alpha * p
                    r = r - alpha * ap
                    z = diagnostic_pc(r)
                    next_gamma = jnp.vdot(r, z).real
                    beta = next_gamma / gamma
                    true_r = cg_rhs - diagnostic_operator(x)
                    jax.debug.print(
                        "BVP_JAX_PCG_TRACE iteration={i} rMz={gamma} pAp={pap} "
                        "alpha={alpha} beta={beta} recursive_residual={rn} "
                        "true_residual={tn} constrained_true_residual={cn}",
                        i=k + 1, gamma=gamma, pap=pap, alpha=alpha, beta=beta,
                        rn=jnp.linalg.norm(r), tn=jnp.linalg.norm(true_r),
                        cn=jnp.linalg.norm(true_r[constraints.dep_dofs]), ordered=True,
                    )
                    return k + 1, x, r, z + beta * p, next_gamma

                jax.lax.while_loop(condition, step, (0, delta_x, r, z, gamma))
                return jnp.int32(0)

            jax.lax.cond(info.nonlinear_iterations == 0, trace_first_steps,
                         lambda _: jnp.int32(0), operand=None)

    match solver_options.linear_solve_type:
        ##########################################################################################
        # jax native solvers

        case LinearSolverType.DENSE_INVERSE_JNP:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            # NOTE jacfwd of R_w_dirichlet will automatically include in-place elimination
            #      of Dirichlet BCs.
            J_dense = jax.jacfwd(R_w_dirichlet)(x_0)
            delta_x = jnp.array(jnp.dot(jnp.linalg.inv(J_dense), -R_0))

        case LinearSolverType.CG_JAX_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )

        case LinearSolverType.CG_JAX_SCIPY_CSR:
            delta_x, _ = jax.scipy.sparse.linalg.cg(
                A=J_csr, b=-R_0, x0=delta_x, M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )

        case LinearSolverType.CG_JAX_SCIPY_W_INFO:
            delta_x, cg_info = cg_w_info(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                maxiter=solver_options.linear_max_iter,
            )
            info = SolverResultInfo(
                nonlinear_iterations=solver_info_0.nonlinear_iterations,
                cumulative_linear_iterations=solver_info_0.cumulative_linear_iterations
                + cg_info["iterations"],
                linear_iterations_per_nonlinear_iteration=solver_info_0.linear_iterations_per_nonlinear_iteration.at[
                    solver_info_0.nonlinear_iterations
                ].set(
                    cg_info["iterations"]
                ),
                cumulative_residual_norm_history=jax.lax.dynamic_update_slice(
                    operand=solver_info_0.cumulative_residual_norm_history,
                    update=cg_info["residual_norm_history"],
                    start_indices=[
                        solver_info_0.cumulative_linear_iterations
                        + solver_info_0.nonlinear_iterations
                    ],
                ),
            )

        case LinearSolverType.GMRES_JAX_SCIPY | LinearSolverType.GMRES_JAX_SCIPY_CSR:
            # JAX maxiter counts restart cycles, not inner iterations. Use
            # restart 30 (PETSc's default); round the inner budget upward.
            restart = min(solver_options.gmres_restart, R_0.shape[0])
            operator = J_csr if use_csr else J_vp
            rhs = -R_0
            apply_pc = preconditioner if preconditioner is not None else lambda v: v
            if solver_options.linear_diagnostics:
                initial_r = rhs - operator @ delta_x if use_csr else rhs - operator(delta_x)
                pc_rhs, pc_initial = apply_pc(rhs), apply_pc(initial_r)
                probe = jax.random.normal(jax.random.key(42), rhs.shape, dtype=rhs.dtype)
                pc_probe = apply_pc(probe)
                rhs_norm = jnp.linalg.norm(rhs)
                true_bound = jnp.maximum(solver_options.linear_absolute_tol,
                                         solver_options.linear_relative_tol * rhs_norm)
                # Mirror the threshold formulas in JAX 0.10.1, without
                # changing its stopping test or the preconditioner scaling.
                inner_bound = jnp.linalg.norm(pc_rhs) * jnp.minimum(1., true_bound / rhs_norm)
                jax.debug.print(
                    "BVP_JAX_GMRES_INPUT step={step} restart={restart} cycles={cycles} "
                    "rhs_norm={bn} initial_true_residual={rn} true_bound={bound} "
                    "pc_rhs_norm={mbn} initial_pc_residual={mrn} inner_pc_bound={inner} "
                    "rhs_finite={bf} pc_rhs_finite={mf} pc_probe_finite={pf} "
                    "initial_pc_below_true_bound={early}",
                    step=info.nonlinear_iterations, restart=restart,
                    cycles=(solver_options.linear_max_iter + restart - 1) // restart,
                    bn=rhs_norm, rn=jnp.linalg.norm(initial_r), bound=true_bound,
                    mbn=jnp.linalg.norm(pc_rhs), mrn=jnp.linalg.norm(pc_initial), inner=inner_bound,
                    bf=jnp.all(jnp.isfinite(rhs)), mf=jnp.all(jnp.isfinite(pc_rhs)),
                    pf=jnp.all(jnp.isfinite(pc_probe)),
                    early=jnp.linalg.norm(pc_initial) <= true_bound, ordered=True,
                )
                if use_csr and solver_options.linear_precond_type is PreconditionerType.JACOBI:
                    jax.debug.print(
                        "BVP_JAX_JACOBI step={step} min={lo} max={hi} negative={neg} zero={zero} nonfinite={bad}",
                        step=info.nonlinear_iterations, lo=jnp.min(diagonal), hi=jnp.max(diagonal),
                        neg=jnp.sum(diagonal < 0), zero=jnp.sum(diagonal == 0),
                        bad=jnp.sum(~jnp.isfinite(diagonal)), ordered=True,
                    )
            gmres_solve = (gmres_true_residual if solver_options.gmres_stopping_test == "true_residual"
                           else jax.scipy.sparse.linalg.gmres)
            delta_x, gmres_status = gmres_solve(
                A=operator,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
                restart=restart,
                maxiter=(solver_options.linear_max_iter + restart - 1) // restart,
                solve_method=solver_options.gmres_solve_method,
            )
            if solver_options.linear_diagnostics:
                final_r = rhs - operator @ delta_x if use_csr else rhs - operator(delta_x)
                pc_final = apply_pc(final_r)
                jax.debug.print(
                    "BVP_JAX_GMRES_OUTPUT step={step} status={status} solution_finite={finite} "
                    "true_residual={rn} pc_residual={mrn} true_bound={bound} accepted={ok}",
                    step=info.nonlinear_iterations, status=gmres_status,
                    finite=jnp.all(jnp.isfinite(delta_x)), rn=jnp.linalg.norm(final_r),
                    mrn=jnp.linalg.norm(pc_final), bound=true_bound,
                    ok=finite_residual_accepted(final_r, true_bound),
                    ordered=True,
                )

        case LinearSolverType.BICGSTAB_JAX_SCIPY:
            delta_x, _ = jax.scipy.sparse.linalg.bicgstab(
                A=J_vp,
                b=-R_0,
                x0=delta_x,
                M=preconditioner,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        ##########################################################################################
        # jaxopt solvers

        case LinearSolverType.DENSE_INVERSE_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_inv(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.LU_JAXOPT:
            delta_x = jaxopt.linear_solve.solve_lu(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.CHOLESKY_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_cholesky(matvec=J_vp, b=-R_0)  # type: ignore

        case LinearSolverType.CG_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_cg(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.GMRES_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_gmres(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        case LinearSolverType.BICGSTAB_JAXOPT:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            delta_x = jaxopt.linear_solve.solve_bicgstab(  # type: ignore
                matvec=J_vp,
                b=-R_0,
                init=delta_x,
                tol=solver_options.linear_relative_tol,
                atol=solver_options.linear_absolute_tol,
            )

        ##########################################################################################
        # cupy solvers

        case LinearSolverType.SPSOLVE_CUPY:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            delta_x = __spsolve(J_sparse, -R_0)

        case LinearSolverType.LU_CUPY:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)

            ilu_ctx = __cupy_splu_init(J_sparse)
            delta_x = __cupy_solve(ilu_ctx, -R_0)

        ##########################################################################################
        # amgx solvers

        case LinearSolverType.AMGX:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            ctx = __amgx_init(J_sparse)
            delta_x = __amgx_solve(ctx, -R_0, delta_x)
            __amgx_finalize(ctx)

        ##########################################################################################
        # pypardiso solvers

        case LinearSolverType.SPSOLVE_PYPARDISO:
            if preconditioner is not None:
                print(
                    f"WARNING: a preconditioner was specifed but unused by {solver_options.linear_solve_type}"
                )
            assert (
                J_w_dirichlet is not None
            ), f"{solver_options.linear_solve_type} requires the `jacobian` argument to be provided."

            J_sparse = J_w_dirichlet(x_0)
            delta_x = __pypardiso_solve(J_sparse, -R_0)

        case _:
            raise Exception(
                f"Linear solver type {solver_options.linear_solve_type} is not implemented"
            )

    if diagnose_cg:
        cg_final_r = cg_rhs - diagnostic_operator(delta_x)
        jax.debug.print(
            "BVP_JAX_CG_OUTPUT step={step} solution_finite={finite} true_residual={rn} "
            "pc_residual={mrn} true_bound={bound} accepted={ok}",
            step=info.nonlinear_iterations, finite=jnp.all(jnp.isfinite(delta_x)),
            rn=jnp.linalg.norm(cg_final_r), mrn=jnp.linalg.norm(diagnostic_pc(cg_final_r)),
            bound=cg_bound, ok=finite_residual_accepted(cg_final_r, cg_bound), ordered=True,
        )

    # jax.scipy solvers will not arrive at the right values for the constraints for any size of
    # problem but even the jaxopt solvers will only get close for large problems.
    # Consequently, overwrite the values directly to ensure the BCs are right, even though the
    # residual may increase.
    delta_x = constraints.apply_to_delta_solution(delta_x,x_0)

    return delta_x, info


def plot_solver_info(opts: SolverOptions, info: SolverResultInfo):
    """
    TODO document
    """
    import matplotlib.pyplot as plt

    cum_iters = np.concat(
        [
            [0],
            np.cumsum(np.asarray(info.linear_iterations_per_nonlinear_iteration)),
        ]
    )
    x_iter = np.hstack(
        [
            np.linspace(
                cum_iters[i],
                cum_iters[i + 1],
                int(info.linear_iterations_per_nonlinear_iteration[i]) + 1,
            )
            for i in range(info.nonlinear_iterations)
        ]
    )

    y_r_norm = info.cumulative_residual_norm_history[
        0 : info.cumulative_linear_iterations + info.nonlinear_iterations
    ]

    plt.plot(x_iter, y_r_norm)
    plt.title(f"Residual History During Iteration\nUsing {opts.linear_solve_type}")
    plt.xlabel("iteration")
    plt.ylabel("|R|")
    plt.yscale("log")

    for i in range(info.nonlinear_iterations):
        plt.axvline(
            x=cum_iters[i],
            color="r",
            linestyle="--",
            label=f"Start of nonlinear iter {i}",
        )
    plt.legend()

    plt.show()
    plt.savefig("solver_convergence.png")


##################################################################################################
# Object store for external library context information

# Global registry to hold generic Python objects
_OBJECT_STORE = {}
_NEXT_ID = 0
_RESOURCE_GROUPS = {}
_NEXT_RESOURCE_GROUP = 0


def __store_object(obj, resource_owner=-1):
    global _NEXT_ID
    uid = _NEXT_ID
    _OBJECT_STORE[uid] = obj
    owner = int(resource_owner)
    if owner >= 0:
        if owner not in _RESOURCE_GROUPS:
            _OBJECT_STORE.pop(uid)
            raise RuntimeError("CuPy factor setup outlived its solve_bvp resource scope; synchronize callbacks before cleanup.")
        _RESOURCE_GROUPS[owner].add(uid)
    _NEXT_ID += 1
    return np.int64(uid)  # Return as a JAX-compatible type


def __retrieve_object(uid):
    return _OBJECT_STORE[int(uid)]


def _release_resource_group(owner):
    """Release only this forward solve's factors, after device work finishes."""
    handles = _RESOURCE_GROUPS[owner]
    # Buffer callbacks may submit CuPy work on a separate stream. Complete it
    # before dropping factor ownership; never clear unrelated objects/pools.
    devices = {_OBJECT_STORE[h].L.data.device.id for h in handles}
    for device in devices:
        with cp.cuda.Device(device):
            cp.cuda.runtime.deviceSynchronize()
    for handle in handles:
        del _OBJECT_STORE[handle]
    del _RESOURCE_GROUPS[owner]


@contextmanager
def cupy_factor_scope():
    """Own factors for a complete eager invocation of a compiled solve/JVP.

    Pass the yielded owner as a dynamic operand. All primal/tangent outputs
    must be blocked inside this scope; do not return a deferred AD pullback
    that can reference these factors after scope exit.
    """
    global _NEXT_RESOURCE_GROUP
    owner = _NEXT_RESOURCE_GROUP
    _NEXT_RESOURCE_GROUP += 1
    _RESOURCE_GROUPS[owner] = set()
    try:
        yield jnp.asarray(owner, dtype=jnp.int64)
    finally:
        try:
            jax.effects_barrier()
        finally:
            _release_resource_group(owner)


def with_forward_solver_resources(function):
    """Own ILU factors for an eager forward BVP, not an AD/traced lifetime.

    Pass the owner as a dynamic operand (not a static SolverOptions field),
    so repeated benchmark calls do not trigger new compilation per owner.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments["solver_options"]
        if options.linear_precond_type is not PreconditionerType.ILU_CUPY:
            return function(*args, **kwargs)
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves((args, kwargs))):
            warnings.warn(
                "Traced/AD solve_bvp with CuPy ILU retains its factors: automatic "
                "forward-only cleanup cannot safely release an AD callback's factors. "
                "This resource fix covers eager forward calls, not traced histories.",
                RuntimeWarning, stacklevel=2,
            )
            return function(*args, **kwargs)
        global _NEXT_RESOURCE_GROUP
        owner = _NEXT_RESOURCE_GROUP
        _NEXT_RESOURCE_GROUP += 1
        _RESOURCE_GROUPS[owner] = set()
        bound.arguments["_resource_owner"] = jnp.asarray(owner, dtype=jnp.int64)
        before = len(_OBJECT_STORE)
        try:
            result = function(*bound.args, **bound.kwargs)
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x, result,
            )
            return result
        finally:
            try:
                jax.effects_barrier()
            finally:
                created = len(_RESOURCE_GROUPS[owner])
                _release_resource_group(owner)
                if options.linear_diagnostics:
                    pool = cp.get_default_memory_pool()
                    print("BVP_CUPY_RESOURCE_CLEANUP", {
                        "owner": owner, "released_factors": created,
                        "registry_before": before, "registry_after": len(_OBJECT_STORE),
                        "cupy_used_bytes": pool.used_bytes(),
                        "cupy_pool_total_bytes": pool.total_bytes(),
                    }, flush=True)

    return wrapped


##################################################################################################
# CUPY wrappers

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg

    @struct.dataclass
    class __SolverCtx:
        handle: jnp.ndarray

    def __cupy_ilu_application_audit(handle, step, rhs, wrapped_rhs, wrapped_repeat,
                                     x, y, wrapped_x, wrapped_y,
                                     matrix_data, matrix_cols, matrix_ptr, initial_guess,
                                     constrained_dofs, *, maxiter, tol, atol):
        """Untimed bridge/triangular audit; all arguments arrive as host arrays.

        This checks the existing CuPy ILU(0) path, with no permutations. It
        neither changes factors nor substitutes a preconditioner for the solve.
        """
        factor = __retrieve_object(handle)
        with cp.cuda.Device(factor.L.data.device.id):
            if factor.perm_r is not None or factor.perm_c is not None:
                raise ValueError("ILU application audit expects unpermuted CuPy ILU(0) factors; permutation-aware triangular diagnostics are not implemented.")
            b, px, py = cp.asarray(rhs), cp.asarray(x), cp.asarray(y)
            direct = factor.solve(b)
            repeated = factor.solve(b)
            mx, my = factor.solve(px), factor.solve(py)
            combined = factor.solve(px + py)
            lower_solution = cplinalg.spsolve_triangular(factor.L, b, lower=True)
            triangular = cplinalg.spsolve_triangular(factor.U, lower_solution, lower=False)

            def norm(v):
                # Avoid a sum-of-squares overflow hiding the diagnostic scale.
                scale = cp.max(cp.abs(v))
                value = float(scale.item())
                if value == 0. or not np.isfinite(value):
                    return value
                return float((scale * cp.linalg.norm(v / scale)).item())

            def relative_difference(a, reference):
                return norm(a - reference) / max(norm(reference), 1e-300)

            def symmetry(a, ma, c, mc):
                left = float(cp.vdot(a, mc).real.item())
                right = float(cp.vdot(c, ma).real.item())
                return abs(left - right) / max(abs(left) + abs(right), 1e-300)

            wrapped = cp.asarray(wrapped_rhs)
            wx, wy = cp.asarray(wrapped_x), cp.asarray(wrapped_y)
            report = {
                "step": int(step), "factor_handle": int(handle),
                "probe_space": "free DOFs only; constrained components zero",
                "rhs_finite": bool(cp.isfinite(b).all().item()),
                "direct_finite": bool(cp.isfinite(direct).all().item()),
                "callback_finite": bool(cp.isfinite(wrapped).all().item()),
                "explicit_triangular_finite": bool(cp.isfinite(triangular).all().item()),
                "direct_rhs_norm": norm(direct), "callback_rhs_norm": norm(wrapped),
                "callback_vs_direct_relative": relative_difference(wrapped, direct),
                "callback_vs_direct_max_abs": float(cp.max(cp.abs(wrapped - direct)).item()),
                "callback_repeat_relative": relative_difference(cp.asarray(wrapped_repeat), wrapped),
                "direct_repeat_relative": relative_difference(repeated, direct),
                "probe_x_callback_vs_direct_relative": relative_difference(wx, mx),
                "probe_y_callback_vs_direct_relative": relative_difference(wy, my),
                "explicit_triangular_vs_direct_relative": relative_difference(triangular, direct),
                "lower_relative_residual": norm(factor.L @ lower_solution - b) / max(norm(b), 1e-300),
                "upper_relative_residual": norm(factor.U @ triangular - lower_solution) / max(norm(lower_solution), 1e-300),
                "direct_factor_product_relative_residual": norm(factor.L @ (factor.U @ direct) - b) / max(norm(b), 1e-300),
                "direct_linearity_relative": relative_difference(combined, mx + my),
                "direct_symmetry_defect": symmetry(px, mx, py, my),
                "callback_symmetry_defect": symmetry(px, wx, py, wy),
                "rhs_quadratic_form": float(cp.vdot(b, direct).real.item()),
                "probe_x_quadratic_form": float(cp.vdot(px, mx).real.item()),
                "probe_y_quadratic_form": float(cp.vdot(py, my).real.item()),
                "note": "Symmetry/positivity probes are not a proof of SPD. Direct reference is synchronized and host-staged; diagnostic overhead is intentional.",
            }
            print("BVP_CUPY_ILU_APPLICATION", report, flush=True)
            if int(step) == 0:
                matrix = cpsparse.csr_matrix(
                    (cp.asarray(matrix_data), cp.asarray(matrix_cols), cp.asarray(matrix_ptr)),
                    shape=factor.shape,
                )
                guess = cp.asarray(initial_guess)
                constrained = cp.asarray(constrained_dofs)
                ax, ay = matrix @ px, matrix @ py
                diag = matrix.diagonal()
                free_mask = cp.ones(b.size, dtype=cp.bool_)
                free_mask[constrained] = False
                print("BVP_CUPY_FREE_OPERATOR", {
                    "negative_free_diagonal": int(cp.count_nonzero(diag[free_mask] < 0).item()),
                    "zero_free_diagonal": int(cp.count_nonzero(diag[free_mask] == 0).item()),
                    "A_symmetry_defect": symmetry(px, ax, py, ay),
                    "xAx": float(cp.vdot(px, ax).real.item()),
                    "yAy": float(cp.vdot(py, ay).real.item()),
                }, flush=True)

                bound = max(float(atol), float(tol) * norm(b))
                tx = guess.copy()
                r = b - matrix @ tx
                z = factor.solve(r)
                p = z.copy()
                gamma = float(cp.vdot(r, z).real.item())
                for iteration in range(1, 9):
                    rn = norm(r)
                    if not np.isfinite(rn) or rn <= bound or not np.isfinite(gamma):
                        break
                    ap = matrix @ p
                    pap = float(cp.vdot(p, ap).real.item())
                    if pap == 0. or gamma == 0.:
                        print("BVP_CUPY_PCG_TRACE_BREAKDOWN", {"iteration": iteration, "rMz": gamma, "pAp": pap}, flush=True)
                        break
                    alpha = gamma / pap
                    tx = tx + alpha * p
                    r = r - alpha * ap
                    z = factor.solve(r)
                    next_gamma = float(cp.vdot(r, z).real.item())
                    beta = next_gamma / gamma
                    true_r = b - matrix @ tx
                    print("BVP_CUPY_PCG_TRACE", {
                        "iteration": iteration, "rMz": gamma, "pAp": pap,
                        "alpha": alpha, "beta": beta,
                        "recursive_residual": norm(r), "true_residual": norm(true_r),
                        "constrained_true_residual": norm(true_r[constrained]) if constrained.size else 0.,
                    }, flush=True)
                    p, gamma = z + beta * p, next_gamma

                preconditioner = cplinalg.LinearOperator(matrix.shape, matvec=factor.solve, dtype=matrix.dtype)
                iterations = [0]

                def progress(candidate):
                    iterations[0] += 1
                    if iterations[0] <= 8 or iterations[0] % 1000 == 0:
                        print("BVP_CUPY_CG_PROGRESS", {
                            "iteration": iterations[0], "true_residual": norm(b - matrix @ candidate),
                        }, flush=True)

                # CuPy releases use either tol or rtol. Select the installed
                # signature explicitly, without changing the requested value.
                parameters = inspect.signature(cplinalg.cg).parameters
                tolerance_keyword = "rtol" if "rtol" in parameters else "tol"
                if tolerance_keyword not in parameters:
                    raise RuntimeError("CuPy cg exposes neither tol nor rtol; inspect its installed API before running this diagnostic.")
                print("BVP_CUPY_CG_BEGIN", {
                    "maxiter": int(maxiter), "rtol": float(tol), "atol": float(atol),
                    "true_bound": bound, "same_matrix_rhs_initial_guess_and_factors": True,
                }, flush=True)
                solution, status = cplinalg.cg(
                    matrix, b, x0=guess.copy(), M=preconditioner,
                    atol=float(atol), maxiter=int(maxiter), callback=progress,
                    **{tolerance_keyword: float(tol)},
                )
                final_norm = norm(b - matrix @ solution)
                print("BVP_CUPY_CG_RESULT", {
                    "status": int(status), "iterations": iterations[0],
                    "true_residual": final_norm, "true_bound": bound,
                    "solution_finite": bool(cp.isfinite(solution).all().item()),
                    "accepted": bool(np.isfinite(final_norm) and np.isfinite(bound) and final_norm <= bound),
                }, flush=True)

    def __solve_cpu(A: jsparse.COO, b: jnp.ndarray):
        """
        Sparse direct solve for system A*x = b for a CPU backend.
        Returns the solution, x.
        """
        A_jax_csr = coo_to_csr(A)
        A_csr = scipy.sparse.csr_matrix(
            (
                np.array(A_jax_csr.data),
                np.array(A_jax_csr.indices),
                np.array(A_jax_csr.indptr),
            ),
            shape=(A.shape[0], A.shape[1]),
        )
        return scipy.sparse.linalg.spsolve(A_csr, b)

    @jax.jit
    def __cupy_spsolve(A: jsparse.CSR, b: jnp.ndarray):

        def kernel(ctx, out, A: jsparse.CSR, b):
            A_cp = cpsparse.csr_matrix(
                (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
                shape=A.shape,
            )
            A_cp.has_canonical_format = True
            # cp.savetxt("A_cp.csv", A_cp.todense())
            cp.asarray(out)[...] = cplinalg.spsolve(A_cp, cp.asarray(b))

        out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)
        cupy_callback = buffer_callback(kernel, out_type)
        return cupy_callback(A, b)

    @jax.jit
    def __solve_gpu(A: jsparse.COO, b: jnp.ndarray):
        """
        Sparse direct solve for system A*x = b for a GPU backend.
        Returns the solution, x.
        """
        A_csr = coo_to_csr(A)
        return __cupy_spsolve(A_csr, b)

    def __spsolve(A: jsparse.COO, b: jnp.ndarray) -> jnp.ndarray:
        """
        Sparse direct solve for system A*x = b.
        Returns the solution, x.
        """
        match jextend.backend.get_backend().platform:
            case "cpu":
                return jnp.array(__solve_cpu(A, b))
            case "gpu":
                return __solve_gpu(A, b)
        raise Exception(
            f"Backend {jextend.backend.get_backend().platform} unsupported."
        )

    def __cupy_spilu_init_impl(A: jsparse.CSR, resource_owner, *, diagnostics=False, complete_setup=False):
        if diagnostics:
            free, total = cp.cuda.runtime.memGetInfo()
            print("BVP_CUPY_ILU_SETUP_BEGIN", {
                "owner": int(resource_owner), "n": A.shape[0], "nnz": A.data.size,
                "live_registry_objects": len(_OBJECT_STORE),
                "cupy_used_bytes": cp.get_default_memory_pool().used_bytes(),
                "cupy_pool_total_bytes": cp.get_default_memory_pool().total_bytes(),
                "device_free_bytes": free, "device_total_bytes": total,
            }, flush=True)
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_cp.has_canonical_format = True
        ilu_obj = cplinalg.spilu(A_cp, fill_factor=1.0)
        if diagnostics:
            upper_diagonal = ilu_obj.U.diagonal()
            print("BVP_CUPY_ILU_FACTORS", {
                "matrix_finite": bool(cp.isfinite(A_cp.data).all().item()),
                "L_finite": bool(cp.isfinite(ilu_obj.L.data).all().item()),
                "U_finite": bool(cp.isfinite(ilu_obj.U.data).all().item()),
                "U_zero_diagonal": int(cp.count_nonzero(upper_diagonal == 0).item()),
                "U_min_abs_diagonal": float(cp.min(cp.abs(upper_diagonal)).item()),
            }, flush=True)
        if complete_setup:
            # One setup-stream wait per factorization, not a device-wide wait
            # per Krylov application. Makes factor uploads ready for XLA's stream.
            cp.cuda.get_current_stream().synchronize()
        return __store_object(ilu_obj, resource_owner)

    @partial(jax.jit, static_argnames=("diagnostics", "complete_setup"))
    def __cupy_spilu_init(A: jsparse.COO, *, diagnostics=False, resource_owner=None, complete_setup=False) -> __SolverCtx:
        result_info = jax.ShapeDtypeStruct((), jnp.int64)
        owner = jnp.asarray(-1, dtype=jnp.int64) if resource_owner is None else resource_owner
        # Implicit linear-solve AD differentiates A and b, not the numerical
        # preconditioner's factorization algorithm. Keep the active A used by
        # custom_linear_solve untouched; stop only this setup callback input.
        setup_matrix = jax.tree_util.tree_map(jax.lax.stop_gradient, A)
        handle = jax.pure_callback(partial(__cupy_spilu_init_impl, diagnostics=diagnostics,
                                          complete_setup=complete_setup), result_info, coo_to_csr(setup_matrix), owner)
        return __SolverCtx(handle=handle)

    def __cupy_splu_init_impl(A: jsparse.CSR):
        A_cp = cpsparse.csr_matrix(
            (cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_cp.has_canonical_format = True
        ilu_obj = cplinalg.splu(A_cp)
        return __store_object(ilu_obj)

    @jax.jit
    def __cupy_splu_init(A: jsparse.COO) -> __SolverCtx:
        result_info = jax.ShapeDtypeStruct((), jnp.int64)
        handle = jax.pure_callback(__cupy_splu_init_impl, result_info, coo_to_csr(A))
        return __SolverCtx(handle=handle)

    class __JaxCudaStream:
        """Borrow XLA's stream via CUDA stream protocol v0; never destroy it."""

        def __init__(self, context):
            self.context = context
            value = getattr(context, "stream", None)
            if value is None:
                raise RuntimeError("JAX buffer callback context has no stream handle. Use ilu_synchronize=True as the correctness reference and inspect the installed JAX/jaxlib APIs.")
            self.ptr = int(value)

        def __cuda_stream__(self):
            return (0, self.ptr)

    def __cupy_solve_impl(ctx, out, handle: jnp.ndarray, b: jnp.ndarray, *, synchronize=False, use_jax_stream=False):
        if use_jax_stream:
            if not hasattr(cp.cuda.Stream, "from_external"):
                raise RuntimeError("Same-stream ILU requires CuPy Stream.from_external (CuPy 14+). Select the synchronized reference or use a compatible CuPy version; no stream fallback is performed.")
            device_type, device_id = out.__dlpack_device__()
            if int(device_type) != 2:
                raise RuntimeError("Same-stream CuPy ILU requires CUDA device buffers; this path does not support CPU/ROCm buffers.")
            with cp.cuda.Device(int(device_id)):
                stream = cp.cuda.Stream.from_external(__JaxCudaStream(ctx))
                with stream:
                    # Includes scalar-handle read, both triangular solves and
                    # output assignment. No device-wide fences in this branch.
                    cupy_obj = __retrieve_object(cp.asarray(handle))
                    if cupy_obj.L.data.device.id != int(device_id):
                        raise RuntimeError("ILU factors and callback buffers are on different CUDA devices; rebuild factors for this device.")
                    output = cp.asarray(out)
                    output[...] = cupy_obj.solve(cp.asarray(b))
            return
        if synchronize:
            output = cp.asarray(out)
            with cp.cuda.Device(output.device.id):
                # Complete JAX producers before consuming their device buffers.
                cp.cuda.runtime.deviceSynchronize()
                cupy_obj = __retrieve_object(cp.asarray(handle))
                output[...] = cupy_obj.solve(cp.asarray(b))
                # Complete both triangular solves AND the output copy before
                # returning control to the consuming JAX recurrence.
                cp.cuda.runtime.deviceSynchronize()
            return
        # Retrieve the opaque object using the handle
        cupy_obj = __retrieve_object(cp.asarray(handle))
        cp.asarray(out)[...] = cupy_obj.solve(cp.asarray(b))

    @partial(jax.jit, static_argnames=("synchronize", "use_jax_stream"))
    def __cupy_solve(ctx: __SolverCtx, b: jnp.ndarray, *, synchronize=False, use_jax_stream=False):
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        callback = partial(__cupy_solve_impl, synchronize=synchronize, use_jax_stream=use_jax_stream)
        return buffer_callback(callback, result_info)(ctx.handle, b)


##################################################################################################
# AMGX wrappers

if PYAMX_AVAILABLE:
    import pyamgx
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    pyamgx.initialize()

    # Refer to https://github.com/NVIDIA/AMGX/tree/main/src/configs for examples
    cfg = pyamgx.Config().create_from_dict(
        {
            "config_version": 2,
            "solver": {
                "preconditioner": {"scope": "amg", "solver": "NOSOLVER"},
                "use_scalar_norm": 1,
                "solver": "PCG",
                "print_solve_stats": 1,
                "obtain_timings": 1,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "scope": "main",
                "max_iters": 100,
                "tolerance": 1e-06,
                "norm": "L2",
            },
        }
    )

    rsc = pyamgx.Resources().create_simple(cfg)

    @struct.dataclass
    class __AMGXCtx:
        cfg_handle: jnp.ndarray
        rsc_handle: jnp.ndarray
        A_handle: jnp.ndarray
        b_handle: jnp.ndarray
        x_handle: jnp.ndarray
        solver_handle: jnp.ndarray

    def __amgx_init_impl(
        A: jsparse.COO,
    ) -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64, np.int64]:
        A_amgx = pyamgx.Matrix().create(rsc)
        A_cp = cpsparse.csr_matrix(
            (
                cp.asarray(A.data),
                (cp.asarray(A.row), cp.asarray(A.col)),
            ),  # cp.asarray(A.indices), cp.asarray(A.indptr)),
            shape=A.shape,
        )
        A_amgx.upload_CSR(A_cp)

        b_amgx = pyamgx.Vector().create(rsc)
        x_amgx = pyamgx.Vector().create(rsc)

        solver = pyamgx.Solver().create(rsc, cfg)
        solver.setup(A_amgx)

        return (
            __store_object(cfg),
            __store_object(rsc),
            __store_object(A_amgx),
            __store_object(b_amgx),
            __store_object(x_amgx),
            __store_object(solver),
        )

    @jax.jit
    def __amgx_init(A: jsparse.COO) -> __AMGXCtx:
        result_info = (
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
            jax.ShapeDtypeStruct((), jnp.int64),
        )
        cfg_handle, rsc_handle, A_handle, b_handle, x_handle, solver_handle = (
            jax.pure_callback(__amgx_init_impl, result_info, A)  # coo_to_csr(A)
        )
        return __AMGXCtx(
            cfg_handle=cfg_handle,
            rsc_handle=rsc_handle,
            A_handle=A_handle,
            b_handle=b_handle,
            x_handle=x_handle,
            solver_handle=solver_handle,
        )

    def __amgx_solve_impl(
        ctx,
        out,
        b_handle: jnp.ndarray,
        x_handle: jnp.ndarray,
        solver_handle: jnp.ndarray,
        b: jnp.ndarray,
        x0: jnp.ndarray,
    ):
        # Retrieve the opaque object using the handle
        b_amgx = __retrieve_object(cp.asarray(b_handle))
        x_amgx = __retrieve_object(cp.asarray(x_handle))
        solver_amgx = __retrieve_object(cp.asarray(solver_handle))

        b_amgx.upload(cp.asarray(b))
        x_amgx.upload(cp.asarray(x0).get())
        print("x_amgx", cp.asarray(x_amgx.download()))
        solver_amgx.solve(b_amgx, x_amgx, zero_initial_guess=False)

        # x = cp.zeros_like(cp.asarray(b))
        cp.asarray(out)[...] = cp.asarray(x_amgx.download())

    @jax.jit
    def __amgx_solve(ctx: __AMGXCtx, b: jnp.ndarray, x0: jnp.ndarray):
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return buffer_callback(__amgx_solve_impl, result_info)(
            ctx.x_handle, ctx.b_handle, ctx.solver_handle, b, x0
        )

    def __amgx_finalize_impl(
        A_handle: jnp.ndarray,
        b_handle: jnp.ndarray,
        x_handle: jnp.ndarray,
        solver_handle: jnp.ndarray,
        cfg_handle: jnp.ndarray,
        rsc_handle: jnp.ndarray,
    ):
        import pyamgx

        A_amgx = __retrieve_object(cp.asarray(A_handle))
        b_amgx = __retrieve_object(cp.asarray(b_handle))
        x_amgx = __retrieve_object(cp.asarray(x_handle))
        solver_amgx = __retrieve_object(cp.asarray(solver_handle))
        rsc = __retrieve_object(cp.asarray(rsc_handle))
        cfg = __retrieve_object(cp.asarray(cfg_handle))

        A_amgx.destroy()
        x_amgx.destroy()
        b_amgx.destroy()
        solver_amgx.destroy()
        rsc.destroy()
        cfg.destroy()
        # pyamgx.finalize()

    def __amgx_finalize(ctx: __AMGXCtx):
        jax.pure_callback(
            __amgx_finalize_impl,
            ctx.A_handle,
            ctx.x_handle,
            ctx.b_handle,
            ctx.solver_handle,
            ctx.cfg_handle,
            ctx.rsc_handle,
        )


##################################################################################################
# PYPARDISO wrappers

if PYPARDISO_AVAILABLE:
    import pypardiso
    import cupy as cp

    def __pypardiso_solve_impl(
        # ctx, <- buffer_callback implementation
        # out, <- buffer_callback implementation
        A_data: jnp.ndarray,
        A_row: jnp.ndarray,
        A_col: jnp.ndarray,
        b: jnp.ndarray,
    ):
        A_scipy = scipy.sparse.csr_matrix(
            (
                cp.asarray(A_data).get().astype(np.float64),
                (
                    cp.asarray(A_row).get().astype(np.int32),
                    cp.asarray(A_col).get().astype(np.int32),
                ),
            ),
            shape=(b.shape[0], b.shape[0]),
        )
        b_np = cp.asarray(b).get().astype(np.float64)
        result = pypardiso.spsolve(A_scipy, b_np)
        # cp.asarray(out)[...] = cp.asarray(result) <- buffer_callback implementation
        return result

    @jax.jit
    def __pypardiso_solve(A: jsparse.COO, b: jnp.ndarray):
        """
        # Import Note
        The buffer_callback would theoretically be more efficient, but there is a bug in JAX resulting
        in bizzare values in the arrays.  If jax.debug.print is not called to view the arrays, they become
        uninitiailized values within __pypardiso_solve_impl. It is like the XLA compiler thinks the arrays
        are not used and the computation branch is pruned. Consequently, I will use a pure_callback variant.
        Fortunately, for this solver, the cost is low since the solve is performed on the host anyway.
        """
        result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return jax.pure_callback(
            __pypardiso_solve_impl, result_info, A.data, A.row, A.col, b
        )
        # result_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
        # jax.debug.print("A.row - jax {}", A.row)
        # jax.debug.print("A.data - jax {}", A.data)
        # return buffer_callback(
        #    __pypardiso_solve_impl, result_info, command_buffer_compatible=False
        # )(A.data, A.row, A.col, b)

jax.tree_util.register_pytree_node(
    SolverOptions,
    lambda x: ((), x),
    lambda aux, _: aux
)
jax.tree_util.register_pytree_node(
    LinearSolverType,
    lambda x: ((), x),
    lambda aux, _: aux
)
jax.tree_util.register_pytree_node(
    PreconditionerType,
    lambda x: ((), x),
    lambda aux, _: aux
)
