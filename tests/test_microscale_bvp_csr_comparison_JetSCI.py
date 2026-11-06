"""Full corrected microscale BVP: PETSc vs matrix-free JAX vs assembled CSR.

Linux/CUDA from fea-in-jax:
  python -m pytest -s -q tests/test_microscale_bvp_csr_comparison_JetSCI.py

FEA_JETSCI_MICROSCALE_REPEATS=5 and FEA_JETSCI_MICROSCALE_WARMUPS=2.
Same FEA Newton driver for all JAX arms; only the explicit linear-solver option
changes. CSR assembly, sorting and conversion are INSIDE each solve/JVP, not
a frozen-operator shortcut. Geometry/preprocessing and first-call compilation
are separate. PETSc uses the stable public differentiable_solve API.

Compares primal displacement, solve+one material JVP, and the reusable tangent
from jax.linearize. The last phase follows the older benchmark's semantics:
JAX linearizes its converged Newton computation; it is not a new implicit
nonlinear derivative implementation. No reverse-AD claim or speed assertion.
"""

import json
import os
from time import perf_counter

from helper import *
import jetsci
from jetsci.petsc_snes import differentiable_snes as diagnostics
from fe_jax.fea import solve_nonlinear_step
from fe_jax.sparse_matrix import coo_to_csr
from test_microscale_bvp_ad_benchmark_oriented_JetSCI import prepare_microscale_ad_problem
import pytest

pytestmark = pytest.mark.slow


def test_microscale_bvp_csr_comparison():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Run on the Linux CUDA/PETSc worker; no CPU solver substitution.")
    (x0, ebc, maps, constraints, nnz, residual_func, force,
     residual, jacobian, phi, direction, weights, repeats) = prepare_microscale_ad_problem()
    warmups = int(os.environ.get("FEA_JETSCI_MICROSCALE_WARMUPS", "2"))
    if warmups < 1:
        raise ValueError("FEA_JETSCI_MICROSCALE_WARMUPS must be positive to separate compilation from warm timings.")
    if constraints.P.nse:
        raise ValueError("This comparison requires fixed Dirichlet constraints, not MPCs; do not substitute an incomplete assembled operator.")
    rtol, atol, maxiter = 1e-6, 1e-14, 10000
    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        nonlinear_relative_tol=rtol, nonlinear_absolute_tol=atol,
        linear_max_iter=maxiter, linear_relative_tol=rtol, linear_absolute_tol=atol,
    )
    step = jax.jit(solve_nonlinear_step, static_argnames=("solver_options", "jacobian_nnz"))

    def make_jax_solve(method):
        settings = SolverOptions(
            linear_solve_type=method, linear_precond_type=PreconditionerType.JACOBI,
            nonlinear_relative_tol=rtol, nonlinear_absolute_tol=atol,
            linear_max_iter=maxiter, linear_relative_tol=rtol, linear_absolute_tol=atol,
        )

        def solve(active):
            return step(
                element_residual_func=residual_func, ebc=ebc.replace(material_params=active),
                assembly_map_b=maps, jacobian_nnz=nnz, u_0_g=x0,
                constraints=constraints, solver_options=settings, f_ext=force,
            )[0]
        return jax.jit(solve)

    def petsc_solve(active):
        return jetsci.differentiable_solve(options, residual, jacobian, x0, active)[0]

    solvers = {
        "PETSc": petsc_solve,
        "JAX_matrix_free_with_info": make_jax_solve(LinearSolverType.CG_JAX_SCIPY_W_INFO),
        "JAX_matrix_free_native": make_jax_solve(LinearSolverType.CG_JAX_SCIPY),
        "JAX_CSR_native": make_jax_solve(LinearSolverType.CG_JAX_SCIPY_CSR),
    }

    def ready(result):
        return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)

    def timed(function):
        start = perf_counter()
        result = ready(function())
        return result, perf_counter() - start

    def emit(kind, **fields):
        print("BVP_CSR_" + kind + " " + json.dumps(fields), flush=True)

    initial_norm = float(jnp.linalg.norm(residual(phi, x0)))
    residual_bound = max(atol, rtol * initial_norm)

    def validate(phase, backend, result, reference):
        actual_leaves = jax.tree_util.tree_leaves(result)
        expected_leaves = jax.tree_util.tree_leaves(reference)
        for actual in actual_leaves:
            assert bool(jnp.all(jnp.isfinite(actual))), (phase, backend, "nonfinite result")
        # Gate on the governing equations BEFORE comparing two approximate
        # solutions. Near-zero field components are not useful relative scales.
        if phase == "primal":
            norm = float(jnp.linalg.norm(residual(phi, result)))
            emit("CHECK", phase=phase, backend=backend, true_residual=norm,
                 bound=residual_bound, accepted=norm <= residual_bound,
                 response=float(jnp.vdot(weights, result)))
            assert norm <= residual_bound, (phase, backend, norm, residual_bound)
        elif phase == "solve_plus_jvp":
            u, u_dot = result
            r_dot = jax.jvp(residual, (phi, u), (direction, u_dot))[1]
            parameter_rhs = jax.jvp(lambda active: residual(active, u), (phi,), (direction,))[1]
            # Differentiate finite-tolerance equilibria; allow explicitly stated
            # accumulation slack, never hide a failure by timing another solver.
            bound = 10 * max(atol, rtol * float(jnp.linalg.norm(parameter_rhs)))
            norm = float(jnp.linalg.norm(r_dot))
            emit("CHECK", phase=phase, backend=backend, tangent_residual=norm,
                 bound=bound, accepted=norm <= bound,
                 response_jvp=float(jnp.vdot(weights, u_dot)))
            assert norm <= bound, (phase, backend, norm, bound)

        for leaf, (actual, expected) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
            actual_host, expected_host = np.asarray(actual), np.asarray(expected)
            assert actual_host.shape == expected_host.shape, (phase, backend, "field shape mismatch")
            delta = (actual_host - expected_host).reshape(-1)
            reference_field = expected_host.reshape(-1)
            error_l2 = float(np.linalg.norm(delta))
            error_inf = float(np.max(np.abs(delta)))
            reference_l2 = float(np.linalg.norm(reference_field))
            reference_inf = float(np.max(np.abs(reference_field)))
            # Keep the original 0.5% comparison tolerance and 1e-13 absolute
            # per-component floor, but apply relative scaling to the FIELD,
            # not independently to tiny/zero individual components. Requiring
            # both norms prevents a single bad DOF hiding in an L2 average.
            l2_bound = 1e-13 * np.sqrt(delta.size) + 5e-3 * reference_l2
            inf_bound = 1e-13 + 5e-3 * reference_inf
            accepted = error_l2 <= l2_bound and error_inf <= inf_bound
            emit("FIELD_CHECK", phase=phase, backend=backend, leaf=leaf,
                 error_l2=error_l2, error_inf=error_inf,
                 reference_l2=reference_l2, reference_inf=reference_inf,
                 relative_l2=error_l2 / reference_l2 if reference_l2 else None,
                 l2_bound=l2_bound, inf_bound=inf_bound, accepted=accepted)
            assert accepted, (phase, backend, leaf, "PETSc field comparison",
                              error_l2, l2_bound, error_inf, inf_bound)

    def compare(phase, functions):
        first, samples, first_results = {}, {name: [] for name in functions}, {}
        for name, function in functions.items():
            result, first[name] = timed(function)
            first_results[name] = result
            validate(phase, name, result, first_results["PETSc"])
            for _ in range(warmups):
                timed(function)
        names = list(functions)
        for iteration in range(repeats):
            for name in names[iteration % len(names):] + names[:iteration % len(names)]:
                result, elapsed = timed(functions[name])
                samples[name].append(elapsed)
                # Completion is timed, but independent validation is not.
                validate(phase, name, result, first_results["PETSc"])
        medians = {name: float(np.median(values)) for name, values in samples.items()}
        emit("TIMING", phase=phase, first_s=first, samples_s=samples, median_s=medians,
             jax_csr_over_petsc=medians["JAX_CSR_native"] / medians["PETSc"],
             matrix_free_over_csr=medians["JAX_matrix_free_native"] / medians["JAX_CSR_native"],
             matrix_free_info_over_native=medians["JAX_matrix_free_with_info"] / medians["JAX_matrix_free_native"])
        return first_results

    diagnostics.set_jvp_diagnostics(False)
    try:
        # Independent correctness gate before interpreting any speed results.
        # This is untimed setup verification, not a matrix used by timed solves.
        matrix = jacobian(phi, x0)
        csr = coo_to_csr(matrix)
        probe = jnp.sin(jnp.arange(x0.size, dtype=x0.dtype) + 1)
        expected_mv = jax.jvp(lambda x: residual(phi, x), (x0,), (probe,))[1]
        error = float(jnp.linalg.norm(csr @ probe - expected_mv))
        bound = 1e-10 * float(jnp.linalg.norm(expected_mv)) + 1e-9
        emit("OPERATOR_CHECK", error=error, bound=bound, n=x0.size, stored_entries=matrix.nse)
        assert error <= bound, "Assembled CSR differs from the residual JVP; do not benchmark different operators."
        del matrix, csr, probe, expected_mv
        _, options = jetsci.differentiable_solve(options, residual, jacobian, x0, phi)
        ready((phi, direction, weights, x0))
        emit("CONFIG", dofs=x0.size, repeats=repeats, warmups=warmups,
             backends=list(solvers), rtol=rtol, atol=atol, maxiter=maxiter,
             mesh="corrected VTK-to-Basix triangle6", matrix_shift=0,
             csr_cost="Jacobian assembly and COO->CSR sorting/conversion inside every linear solve",
             field_comparison="global L2 and infinity norms; rtol=5e-3, per-component absolute floor=1e-13; independent residual checks unchanged",
             comparison="PETSc public eager API vs compiled JAX FEA; setup/compilation reported separately")
        compare("primal", {name: (lambda solve=solve: solve(phi)) for name, solve in solvers.items()})

        def make_jvp(solve, compile_it):
            def evaluate(active, tangent):
                return jax.jvp(solve, (active,), (tangent,))
            return jax.jit(evaluate) if compile_it else evaluate

        jvps = {name: make_jvp(solve, name != "PETSc") for name, solve in solvers.items()}
        compare("solve_plus_jvp", {name: (lambda fn=fn: fn(phi, direction)) for name, fn in jvps.items()})

        tangents = {}
        for name, solve in solvers.items():
            start = perf_counter()
            value, tangent = jax.linearize(solve, phi)
            ready(value)
            tangents[name] = jax.jit(tangent) if name != "PETSc" else tangent
            emit("LINEARIZE_SETUP", backend=name, wall_s=perf_counter() - start)
        compare("saved_linearization_jvp", {
            name: (lambda fn=fn: fn(direction)) for name, fn in tangents.items()})
    finally:
        diagnostics.set_jvp_diagnostics(False)
        if options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(options.solver_key)
