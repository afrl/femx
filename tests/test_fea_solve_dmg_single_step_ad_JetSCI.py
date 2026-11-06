"""Single equilibrium-step AD benchmark. Run in the Linux CUDA/PETSc environment.

pytest -s -q tests/test_fea_solve_dmg_single_step_ad_JetSCI.py

FEA_JETSCI_SINGLE_REPEATS (default 5) controls warm repetitions.
FEA_JETSCI_SINGLE_STRAIN (default 0.006) controls the prescribed strain.
Mesh preprocessing, callback construction, and PETSc initialization happen once.
"""

import os
from pathlib import Path
from time import perf_counter

from helper import *
import jetsci
import pyvista as pv
from fe_jax.fea import (
    SolverOptions, LinearSolverType, preprocess_bvp, build_nonlinear_objects,
    solve_nonlinear_step,
)
from fe_jax.implicit_differentiation import solve_nonlinear_quasi_step_autodiff_vjp
from fe_jax.linear_elasticity_dmg import (
    damage_elastic_isotropic_vmap, elastic_orthotropic, linear_elasticity_residual,
)


def test_single_damage_step_ad_jetsci():
    jax.config.update("jax_enable_x64", True)
    repeats = int(os.environ.get("FEA_JETSCI_SINGLE_REPEATS", "5"))
    strain = float(os.environ.get("FEA_JETSCI_SINGLE_STRAIN", "0.006"))
    if repeats < 1 or not np.isfinite(strain) or strain <= 0:
        raise ValueError("Set a positive repeat count and positive finite prescribed strain.")
    setup_start = perf_counter()
    mesh = pv.read(str(Path(__file__).parent / "meshes" / "IGFEM_1fib.vtk"))
    points = np.asarray(mesh.points, dtype=np.float64)[:, :2]
    matrix_id = np.max(mesh.cell_data["materials"])
    cells = {(matrix, kind): [] for matrix in (True, False) for kind in (5, 9)}
    for index, kind in enumerate(mesh.celltypes):
        if kind in (5, 9):
            cells[(mesh.cell_data["materials"][index] == matrix_id, kind)].append(
                reorder_cell_basix(mesh, mesh.get_cell(index).point_ids)
            )
    params = jnp.array([3000.0, 200000.0], dtype=jnp.float64)
    # Change the stiffness contrast, rather than scaling both moduli together
    # (which can leave a displacement-controlled elastic solution unchanged).
    direction = jnp.array([3.0, -200.0], dtype=jnp.float64)
    batches, matrix_indices, fiber_indices = [], [], []
    offset = 0
    for is_matrix, kind in ((True, 5), (True, 9), (False, 5), (False, 9)):
        connectivity = np.asarray(cells[(is_matrix, kind)], dtype=np.int64).reshape(-1, 3 if kind == 5 else 4)
        q = 1 if kind == 5 else 4
        fe = FiniteElementType(
            cell_type=CellType.triangle if kind == 5 else CellType.quadrilateral,
            family=ElementFamily.P, basis_degree=1,
            lagrange_variant=LagrangeVariant.equispaced,
            quadrature_type=QuadratureType.default, quadrature_degree=1 if kind == 5 else 2,
        )
        values = jnp.array(
            [3000., .39, .95, 2., 10., .35, .04, 79., 62., 2e4, 12.]
            if is_matrix else [200000., 23.1e3, .2, 8.96e3, 8.27e3], dtype=jnp.float64,
        )
        material = jnp.broadcast_to(values, (len(connectivity), q, len(values)))
        state = jnp.zeros((len(connectivity), q, 11 if is_matrix else 7), dtype=jnp.float64)
        if is_matrix:
            state = state.at[..., 10].set(0.5)
        indices = offset + np.arange(0, material.size, len(values), dtype=np.int64)
        (matrix_indices if is_matrix else fiber_indices).extend(indices.tolist())
        offset += material.size
        batches.append(ElementBatch(
            fe_type=fe, connectivity_en=connectivity, n_dofs_per_basis=2,
            constitutive_model=damage_elastic_isotropic_vmap if is_matrix else elastic_orthotropic,
            material_params=material, internal_state=state,
        ))
    matrix_indices = jnp.asarray(matrix_indices, dtype=jnp.int32)
    fiber_indices = jnp.asarray(fiber_indices, dtype=jnp.int32)
    left = np.flatnonzero(points[:, 0] == points[:, 0].min())
    right = np.flatnonzero(points[:, 0] == points[:, 0].max())
    displacement = strain * np.ptp(points[:, 0])
    bcs = [DirichletBC(index=i, component=component,
                       value=displacement if component == 0 and i in right else 0.0)
           for component in range(2) for i in np.concatenate((left, right))]
    ebc, maps, constraints, nnz, residual_func, force = preprocess_bvp(
        vertices_vd=points, element_batches=batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=bcs, multipoint_constraints=None,
    )
    x0 = jnp.zeros(2 * len(points), dtype=jnp.float64)
    residual, jacobian, residual_isv = build_nonlinear_objects(
        element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
        jacobian_nnz=nnz, u_0_g=x0, constraints=constraints, f_ext=force,
    )

    def material(p):
        return ebc.material_params.at[matrix_indices].set(p[0]).at[fiber_indices].set(p[1])

    jax_options = SolverOptions(
        linear_solve_type=LinearSolverType.DENSE_INVERSE_JNP,
        nonlinear_max_iter=50, nonlinear_relative_tol=1e-8, nonlinear_absolute_tol=1e-12,
    )
    petsc_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_GMRES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        nonlinear_max_iter=50, nonlinear_relative_tol=1e-8, nonlinear_absolute_tol=1e-12,
        linear_relative_tol=1e-10, linear_absolute_tol=1e-12, linear_max_iter=10000,
    )

    def jax_step(p, *, reverse=False):
        kernel = solve_nonlinear_quasi_step_autodiff_vjp if reverse else solve_nonlinear_step
        return kernel(
            element_residual_func=residual_func, ebc=ebc.replace(material_params=material(p)),
            assembly_map_b=maps, jacobian_nnz=nnz, u_0_g=x0,
            constraints=constraints, solver_options=jax_options, f_ext=force,
        )[0]

    def petsc_step(p):
        return jetsci.differentiable_solve(
            petsc_options, residual, jacobian, x0, material(p),
        )[0]

    weights = jnp.sin(jnp.arange(x0.size, dtype=x0.dtype) + 1)
    weights = weights / jnp.linalg.norm(weights)
    jax_solve = jax.jit(jax_step)
    jax_objective = lambda p: jnp.vdot(weights, jax_step(p, reverse=True))
    petsc_objective = lambda p: jnp.vdot(weights, petsc_step(p))

    def ready(result):
        return jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)

    def measure(function):
        start = perf_counter()
        result = ready(function())
        return result, perf_counter() - start

    def compare(label, petsc_function, jax_function):
        p, p_first = measure(petsc_function)
        j, j_first = measure(jax_function)
        timings = {"PETSc": [], "JAX": []}
        for repeat in range(repeats):
            order = (("PETSc", petsc_function), ("JAX", jax_function))
            for backend, function in (order if repeat % 2 == 0 else reversed(order)):
                _, elapsed = measure(function)
                timings[backend].append(elapsed)
        p_median, j_median = (float(np.median(timings[key])) for key in ("PETSc", "JAX"))
        print(f"{label}: first PETSc={p_first:.6f}s JAX={j_first:.6f}s; "
              f"warm median PETSc={p_median:.6f}s JAX={j_median:.6f}s; JAX/PETSc={j_median / p_median:.3f}x")
        for actual, expected in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(j), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-9, err_msg=label)
        return p, j

    try:
        # Build one persistent solver before any transforms. Its initialization
        # cost is reported separately and never hidden in a warm measurement.
        _, petsc_options = jetsci.differentiable_solve(
            petsc_options, residual, jacobian, x0, material(params),
        )
        ready((params, x0))
        print(f"Single damage step: {x0.size} DOFs, strain={strain}, dt=0.5, repeats={repeats}")
        print(f"Setup + PETSc initialization: {perf_counter() - setup_start:.6f}s")
        print("Same zero initial guess and fixed incoming state on every call; no timestepping or optimization.")
        print("JAX baseline: dense inverse; PETSc: GMRES/Jacobi. First calls may include compilation.")
        p_solution, j_solution = compare("primal solve", lambda: petsc_step(params), lambda: jax_solve(params))
        for backend, u in (("PETSc", p_solution), ("JAX", j_solution)):
            r, state = ready(residual_isv(material(params), u))
            r0 = residual(material(params), x0)
            threshold = max(1e-12, 1e-8 * float(jnp.linalg.norm(r0)))
            norm = float(jnp.linalg.norm(r))
            print(f"{backend}: residual norm={norm:.6e}; max matrix damage="
                  f"{max(float(jnp.max(s[..., 6])) for s in state[:2] if s.size):.6e}")
            assert norm <= 10 * threshold, f"{backend} did not reach comparable equilibrium: {norm}"

        jax_jvp = jax.jit(lambda p, v: jax.jvp(jax_solve, (p,), (v,)))
        compare("complete solve + JVP", lambda: jax.jvp(petsc_step, (params,), (direction,)),
                lambda: jax_jvp(params, direction))
        p_grad = jax.value_and_grad(petsc_objective)
        j_grad = jax.jit(jax.value_and_grad(jax_objective))
        compare("complete solve + reverse gradient", lambda: p_grad(params), lambda: j_grad(params))

        # These closures reuse the nonlinear linearization, separating the
        # derivative application from finding the primal equilibrium.
        _, p_tangent = jax.linearize(petsc_step, params)
        _, j_tangent = jax.linearize(jax_solve, params)
        j_tangent = jax.jit(j_tangent)
        compare("JVP application at saved equilibrium", lambda: p_tangent(direction), lambda: j_tangent(direction))
        _, p_pullback = jax.vjp(petsc_objective, params)
        _, j_pullback = jax.vjp(jax.jit(jax_objective), params)
        j_pullback = jax.jit(j_pullback)
        compare("VJP application at saved equilibrium", lambda: p_pullback(jnp.array(1.0)),
                lambda: j_pullback(jnp.array(1.0)))
    finally:
        if petsc_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(petsc_options.solver_key)


if __name__ == "__main__":
    test_single_damage_step_ad_jetsci()
