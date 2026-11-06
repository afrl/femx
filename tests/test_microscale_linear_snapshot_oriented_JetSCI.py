"""Freeze the corrected microscale equilibrium and compare its linear operators.

Separate corrected benchmark; preserves the original snapshot test and output.

Run from fea-in-jax on Linux/CUDA:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -s -q \
    tests/test_microscale_linear_snapshot_oriented_JetSCI.py

FEA_JETSCI_SNAPSHOT_REPEATS=3, FEA_JETSCI_SNAPSHOT_WARMUP_MAX=8,
FEA_JETSCI_SNAPSHOT_MATVECS=10. FEA_JETSCI_SNAPSHOT_PATH selects the saved NPZ.
The original operator is never shifted, symmetrized, or densified. Equilibrium,
Jacobian assembly, derivative RHS construction and file export are untimed setup.
Headline solves use native JAX CG without iteration-history bookkeeping. Extra
CSR/matrix-free arms run FEA's existing CG-with-info to measure that distinction.
All JIT numerical data are explicit operands. No PETSc/JAX implementation edits.
"""

import json
import os
from pathlib import Path
from threading import RLock
from time import perf_counter

import cupy as cp
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from jax.experimental.buffer_callback import buffer_callback
import numpy as np
import pytest
from petsc4py import PETSc
from scipy import sparse

import jetsci
from jetsci.lifecycle import build_linear_solver_with_reuse, destroy_linear_solver
from test_microscale_bvp_ad_benchmark_oriented_JetSCI import prepare_microscale_ad_problem
from fe_jax.fea import calculate_residual_w_constraints
from fe_jax.solve_cg import cg as cg_with_info

pytestmark = pytest.mark.slow


def ready(tree):
    return jax.tree_util.tree_map(
        lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, tree,
    )


def measure(function, arguments):
    start = perf_counter()
    result = ready(function(*arguments))
    return result, perf_counter() - start


def benchmark(label, functions, arguments, repeats, warmup_max):
    first, warmups, samples, results = {}, {}, {}, {}
    for name, function in functions.items():
        results[name], first[name] = measure(function, arguments)
        warmups[name], samples[name] = [], []
        for _ in range(warmup_max):
            _, elapsed = measure(function, arguments)
            warmups[name].append(elapsed)
            recent = warmups[name][-3:]
            if len(recent) == 3 and max(recent) <= 1.25 * min(recent):
                break
        else:
            pytest.fail(f"{label}/{name}: warmup did not stabilize: {warmups[name]}. Increase FEA_JETSCI_SNAPSHOT_WARMUP_MAX.")
    for index in range(repeats):
        names = list(functions) if index % 2 == 0 else list(reversed(functions))
        for name in names:
            _, elapsed = measure(functions[name], arguments)
            samples[name].append(elapsed)
    print("SNAPSHOT_TIMING " + json.dumps({
        "phase": label, "first_s": first, "warmups_s": warmups,
        "samples_s": samples, "median_s": {k: float(np.median(v)) for k, v in samples.items()},
    }), flush=True)
    return results


def test_microscale_linear_snapshot_oriented():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("Requires Linux/CUDA JAX and PETSc; no CPU fallback.")
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("Only a single MPI rank/device is supported by this benchmark.")
    repeats = int(os.environ.get("FEA_JETSCI_SNAPSHOT_REPEATS", "3"))
    warmup_max = int(os.environ.get("FEA_JETSCI_SNAPSHOT_WARMUP_MAX", "8"))
    matvec_repeats = int(os.environ.get("FEA_JETSCI_SNAPSHOT_MATVECS", "10"))
    if repeats < 1 or matvec_repeats < 1 or warmup_max < 3:
        raise ValueError("Require REPEATS>=1, MATVECS>=1, WARMUP_MAX>=3.")
    (x0, ebc, maps, constraints, nnz, kernel, force,
     residual, jacobian, phi, direction, weights, _) = prepare_microscale_ad_problem()
    rtol, atol, maxiter = 1e-6, 1e-14, 10000
    nonlinear_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_relative_tol=rtol, linear_absolute_tol=atol, linear_max_iter=maxiter,
    )
    start = perf_counter()
    try:
        x_star, nonlinear_options = jetsci.differentiable_solve(
            nonlinear_options, residual, jacobian, x0, phi,
        )
        x_star = ready(x_star)
        equilibrium_s = perf_counter() - start
        assembled = ready(jacobian(phi, x_star))
        host = sparse.coo_matrix((np.asarray(assembled.data),
                                  (np.asarray(assembled.row), np.asarray(assembled.col))),
                                 shape=assembled.shape).tocsr()
        host.sum_duplicates()
        host.eliminate_zeros()
        host.sort_indices()
    finally:
        if nonlinear_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(nonlinear_options.solver_key)

    # Explicit runtime FEA data reproduces the same residual without closing
    # over the mesh/material arrays in each compiled matvec/CG variant.
    def runtime_residual(p, x, active_ebc, active_maps, active_constraints, active_force):
        return calculate_residual_w_constraints(
            u_f=x, element_residual_func=kernel,
            ebc=active_ebc.replace(material_params=p), assembly_map_b=active_maps,
            constraints=active_constraints, f_ext=active_force,
        )[0]

    runtime_residual = jax.jit(runtime_residual)
    residual_at_star = ready(runtime_residual(phi, x_star, ebc, maps, constraints, force))
    initial_norm = float(jnp.linalg.norm(runtime_residual(phi, x0, ebc, maps, constraints, force)))
    # Compare relative to the initial residual, not to near-zero converged
    # entries where equivalent compiled summation orders can differ in ULPs.
    residual_difference = float(jnp.linalg.norm(residual_at_star - residual(phi, x_star)))
    assert residual_difference <= 1e-9 * initial_norm + 1e-12, residual_difference
    equilibrium_norm = float(jnp.linalg.norm(residual_at_star))
    assert equilibrium_norm <= 5 * max(atol, 1e-6 * initial_norm), (equilibrium_norm, initial_norm)

    @jax.jit
    def parameter_rhs(p, d, x, active_ebc, active_maps, active_constraints, active_force):
        return -jax.jvp(lambda q: runtime_residual(q, x, active_ebc, active_maps,
                                                 active_constraints, active_force), (p,), (d,))[1]

    forward_rhs = ready(parameter_rhs(phi, direction, x_star, ebc, maps, constraints, force))
    adjoint_rhs = weights  # d(weights @ x)/dx from the existing microscale test.
    diagonal = host.diagonal()
    diagonal_stats = {
        "min": float(np.min(diagonal)), "max": float(np.max(diagonal)),
        "positive": int(np.count_nonzero(diagonal > 0)),
        "negative": int(np.count_nonzero(diagonal < 0)),
        "zero": int(np.count_nonzero(diagonal == 0)),
        "nonfinite_entries": int(np.count_nonzero(~np.isfinite(host.data))),
        "nonpositive_dof_examples": np.flatnonzero(diagonal <= 0)[:12].tolist(),
    }
    print("SNAPSHOT_DIAGONAL " + json.dumps(diagonal_stats), flush=True)
    # Geometry diagnostics are untimed and do not change connectivity,
    # quadrature, signed integration measures, or the original operator.
    for batch in range(ebc.B):
        coordinates = np.asarray(ebc.get_x(batch))
        basis_derivatives = np.asarray(ebc.get_dphi_dxi(batch))
        derivative_layout = "eqnp" if basis_derivatives.ndim == 4 else "qnp"
        element_jacobians = np.einsum("end," + derivative_layout + "->eqpd",
                                      coordinates, basis_derivatives)
        determinants = np.linalg.det(element_jacobians)
        print("SNAPSHOT_GEOMETRY " + json.dumps({
            "batch": batch, "determinant_min": float(determinants.min()),
            "determinant_max": float(determinants.max()),
            "negative_determinants": int(np.count_nonzero(determinants < 0)),
            "zero_determinants": int(np.count_nonzero(determinants == 0)),
            "nonfinite_determinants": int(np.count_nonzero(~np.isfinite(determinants))),
            "quadrature_weight_min": float(np.asarray(ebc.get_weights(batch)).min()),
        }), flush=True)
    difference = host - host.T
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    csr = jsparse.CSR((jnp.asarray(host.data), jnp.asarray(host.indices, dtype=jnp.int32),
                      jnp.asarray(host.indptr, dtype=jnp.int32)), shape=host.shape)
    diag = jnp.asarray(diagonal)
    operands = (csr, diag, phi, x_star, ebc, maps, constraints, force)
    config = {"n": host.shape[0], "nnz": host.nnz, "rtol": rtol, "atol": atol,
              "maxiter": maxiter, "pc": "jacobi", "shift": 0,
              "equilibrium_s": equilibrium_s, "equilibrium_residual_norm": equilibrium_norm,
              "forward_rhs_norm": float(jnp.linalg.norm(forward_rhs)),
              "adjoint_rhs_norm": float(jnp.linalg.norm(adjoint_rhs)),
              "symmetry_max_abs_error": symmetry_error, "diagonal": diagonal_stats}
    path = Path(os.environ.get("FEA_JETSCI_SNAPSHOT_PATH",
                               Path(__file__).parent / "output" / "microscale_linear_snapshot_oriented.npz")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=host.data, indices=host.indices, indptr=host.indptr, shape=host.shape,
             diagonal=diagonal, x_star=np.asarray(x_star), phi=np.asarray(phi),
             parameter_direction=np.asarray(direction), forward_rhs=np.asarray(forward_rhs),
             adjoint_rhs=np.asarray(adjoint_rhs), metadata=json.dumps(config))
    print("SNAPSHOT_CONFIG " + json.dumps({**config, "saved_npz": str(path)}), flush=True)
    if diagonal_stats["nonfinite_entries"] or diagonal_stats["zero"]:
        raise ValueError(
            f"Jacobi cannot use this snapshot: {diagonal_stats}. Saved {path}. "
            "Inspect nonfinite entries/zero diagonal DOFs before benchmarking; "
            "no shift or solver fallback has been applied."
        )
    assert symmetry_error <= 1e-10 * max(1., np.max(np.abs(host.data))), symmetry_error
    if diagonal_stats["negative"]:
        raise ValueError(
            f"Corrected geometry must produce a positive elasticity diagonal; got {diagonal_stats}. "
            f"Inspect saved {path} before interpreting any CG timings."
        )

    def make_matvec(transpose):
        def csr_matvec(v, operator, diag, p, x, active_ebc, active_maps, active_constraints, active_force):
            return operator.T @ v if transpose else operator @ v

        def matrix_free(v, operator, diag, p, x, active_ebc, active_maps, active_constraints, active_force):
            def R(y):
                return runtime_residual(p, y, active_ebc, active_maps, active_constraints, active_force)
            if transpose:
                return jax.vjp(R, x)[1](v)[0]
            return jax.jvp(R, (x,), (v,))[1]

        return jax.jit(csr_matvec), jax.jit(matrix_free)

    linear_options = jetsci.LinearSolverOptions(
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_relative_tol=rtol, linear_absolute_tol=atol, linear_max_iter=maxiter,
    )
    coo = host.tocoo()
    solver, linear_options = build_linear_solver_with_reuse(linear_options, jetsci.COOData(
        shape=jnp.asarray(host.shape), rows=jnp.asarray(coo.row), cols=jnp.asarray(coo.col),
        vals=jnp.asarray(coo.data),
    ))
    lock = RLock()
    try:
        solver.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        solver.ksp.setUp()
        # No matrix construction/updates in the timed callbacks below.
        def make_petsc_callback(transpose, matvec=False):
            def callback(ctx, out, v):
                with lock:
                    if not matvec:
                        solver.solve_from_device_buffers(out=out, rhs=v, transpose=transpose)
                        if solver.ksp.getConvergedReason() <= 0:
                            raise RuntimeError(f"Frozen PETSc CG did not converge: reason={solver.ksp.getConvergedReason()}, iterations={solver.ksp.getIterationNumber()}. Inspect the snapshot; no fallback.")
                        return
                    v_cp, out_cp = cp.from_dlpack(v, copy=False), cp.asarray(out)
                    v_vec = PETSc.Vec().createWithDLPack(v_cp, size=v_cp.size)
                    out_vec = PETSc.Vec().createWithDLPack(out_cp, size=out_cp.size)
                    try:
                        if transpose:
                            solver.operator_matrix.multTranspose(v_vec, out_vec)
                        else:
                            solver.operator_matrix.mult(v_vec, out_vec)
                    finally:
                        out_vec.destroy()
                        v_vec.destroy()
            return callback

        def make_petsc_function(transpose, matvec=False):
            callback = make_petsc_callback(transpose, matvec)
            @jax.jit
            def apply(v, operator, diag, p, x, active_ebc, active_maps, active_constraints, active_force):
                return buffer_callback(callback, jax.ShapeDtypeStruct(v.shape, v.dtype),
                                       vmap_method="sequential")(v)
            return apply

        def make_cg(matvec, with_info=False):
            @jax.jit
            def solve(b, operator, diag, p, x, active_ebc, active_maps, active_constraints, active_force):
                args = (operator, diag, p, x, active_ebc, active_maps, active_constraints, active_force)
                A = lambda v: matvec(v, *args)
                M = lambda v: v / diag
                if with_info:
                    return cg_with_info(A, b, M=M, tol=rtol, atol=atol, maxiter=maxiter)
                return jax.scipy.sparse.linalg.cg(A, b, M=M, tol=rtol, atol=atol, maxiter=maxiter)[0]
            return solve

        rng = np.random.default_rng(441)
        for label, transpose, b in (("forward_derivative", False, forward_rhs),
                                    ("adjoint", True, adjoint_rhs)):
            csr_mv, mf_mv = make_matvec(transpose)
            matvecs = {"PETSc": make_petsc_function(transpose, matvec=True),
                       "JAX_CSR": csr_mv, "JAX_matrix_free": mf_mv}
            # Separate fixed probes prevent operator agreement from depending
            # only on the solution or on a tiny derivative RHS.
            for probe in range(3):
                v = jnp.asarray(rng.normal(size=x0.size))
                reference = ready(csr_mv(v, *operands))
                for name, function in matvecs.items():
                    value = ready(function(v, *operands))
                    relative = float(jnp.linalg.norm(value - reference) / jnp.linalg.norm(reference))
                    boundary_error = float(jnp.max(jnp.abs((value - reference)[constraints.dep_dofs])))
                    print("SNAPSHOT_OPERATOR_CHECK " + json.dumps({"system": label, "probe": probe,
                          "backend": name, "relative_error": relative, "boundary_max_abs_error": boundary_error}), flush=True)
                    assert relative < 1e-8, (name, relative)
                    assert boundary_error < 1e-10, (name, boundary_error)
            benchmark(label + "_matvec", matvecs, (v, *operands), matvec_repeats, warmup_max)
            print("SNAPSHOT_NOTE PETSc matvec timings include one device callback and Vec wrappers; KSP does not pay that boundary on every iteration.", flush=True)
            solves = {"PETSc": make_petsc_function(transpose),
                      "JAX_CSR": make_cg(csr_mv), "JAX_matrix_free": make_cg(mf_mv),
                      "JAX_CSR_with_info": make_cg(csr_mv, with_info=True),
                      "JAX_matrix_free_with_info": make_cg(mf_mv, with_info=True)}
            results = benchmark(label + "_solve", solves, (b, *operands), repeats, warmup_max)
            reference = results["PETSc"]
            threshold = max(atol, rtol * float(jnp.linalg.norm(b)))
            for name, result in results.items():
                if name.endswith("_with_info"):
                    solution, info = result
                    iterations = int(info["iterations"])
                else:
                    solution = result
                    iterations = solver.ksp.getIterationNumber() if name == "PETSc" else None
                residual_norm = float(jnp.linalg.norm(csr_mv(solution, *operands) - b))
                relative_solution_difference = float(jnp.linalg.norm(solution - reference) /
                                                     jnp.maximum(jnp.linalg.norm(reference), jnp.finfo(reference.dtype).tiny))
                print("SNAPSHOT_SOLVE_CHECK " + json.dumps({"system": label, "backend": name,
                      "true_residual_norm": residual_norm, "requested_bound": threshold,
                      "relative_to_petsc": relative_solution_difference, "iterations": iterations,
                      "iteration_note": "Native JAX CG does not expose iteration counts; with-info arm reports its own count."}), flush=True)
                assert residual_norm <= 5 * threshold, (label, name, residual_norm, threshold)
                assert relative_solution_difference < 5e-3, (label, name, relative_solution_difference)
    finally:
        destroy_linear_solver(linear_options.solver_key)
