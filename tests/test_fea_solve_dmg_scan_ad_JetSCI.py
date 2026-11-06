"""Compare JetSCI PETSc and JAX derivatives for a damage load history.

Forward AD scans the same JAX timestep solver as ``solve_bvp_quasi``. Reverse AD and
optimization use the custom implicit timestep VJP from ``test_fea_solve_dmg_scan_ad``.
PETSc advances through the public API in an eager Python loop. Timings include
each complete objective/derivative evaluation, with all outputs synchronized.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

import jetsci
from helper import *
from fe_jax.fea import (
    SolverOptions,
    LinearSolverType,
    build_nonlinear_objects,
    compute_ISV_be,
    preprocess_bvp,
    solve_nonlinear_step,
)
from fe_jax.linear_elasticity_dmg import *
from fe_jax.utils import compute_stress_strain_curve
from fe_jax.implicit_differentiation import solve_nonlinear_quasi_step_autodiff_vjp


jax.config.update("jax_enable_x64", True)


def test_fea_solve_dmg_scan_autodiff_jetsci():
    """Compare derivatives, timings, and projected optimization trajectories.

    FEA_JETSCI_AD_TIMESTEPS: 20 by default (500 for the legacy workload).
    FEA_JETSCI_AD_REPEATS: 2 warm timing repetitions by default.
    FEA_JETSCI_OPT_STEPS: 3 by default; 0 skips optimization, 50 matches legacy.
    """
    num_fib = "1"
    # Keep the default practical for the GPU regression test.  Use 500 when
    # reproducing the full legacy scan_ad workload.
    t_total = int(os.environ.get("FEA_JETSCI_AD_TIMESTEPS", "20"))
    repeats = int(os.environ.get("FEA_JETSCI_AD_REPEATS", "2"))
    optimization_steps = int(os.environ.get("FEA_JETSCI_OPT_STEPS", "3"))
    if t_total < 1 or repeats < 1 or optimization_steps < 0:
        raise ValueError("Timesteps/repeats must be positive and optimization steps nonnegative.")
    strain_max = 0.012
    params = jnp.array([3000.0, 200000.0], dtype=jnp.float64)
    params_dot = jnp.array([1.0, 100.0], dtype=jnp.float64)

    mesh = pv.read(f"tests/meshes/IGFEM_{num_fib}fib.vtk")
    points = np.asarray(mesh.points, dtype=np.float64)[:, :2]
    matrix_id = jnp.max(mesh.cell_data["materials"])
    groups = {"matrix_tri": [], "matrix_quad": [], "fiber_tri": [], "fiber_quad": []}
    for cell_id, cell_type in enumerate(mesh.celltypes):
        material = mesh.cell_data["materials"][cell_id]
        prefix = "matrix" if material == matrix_id else "fiber"
        if cell_type == 5:
            groups[f"{prefix}_tri"].append(reorder_cell_basix(mesh, mesh.get_cell(cell_id).point_ids))
        elif cell_type == 9:
            groups[f"{prefix}_quad"].append(reorder_cell_basix(mesh, mesh.get_cell(cell_id).point_ids))
    for key, cells in groups.items():
        groups[key] = np.asarray(cells)

    fe_tri = FiniteElementType(
        cell_type=CellType.triangle, family=ElementFamily.P, basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default, quadrature_degree=1,
    )
    fe_quad = FiniteElementType(
        cell_type=CellType.quadrilateral, family=ElementFamily.P, basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default, quadrature_degree=2,
    )
    length = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    dxs = jnp.arange(1, t_total + 1, dtype=jnp.float64) * strain_max * length / t_total
    lhs = np.where(points[:, 0] == np.min(points[:, 0]))[0]
    rhs = np.where(points[:, 0] == np.max(points[:, 0]))[0]
    boundary_points = np.concatenate((lhs, rhs))

    def make_bcs(rhs_dx):
        return [
            DirichletBC(index=i, component=j, value=rhs_dx if j == 0 and i in rhs else 0.0)
            for j in range(2) for i in boundary_points
        ]

    def make_batches(active_params):
        def material_and_state(cells, q, is_matrix):
            if is_matrix:
                material = jnp.zeros((cells.shape[0], q, 11), dtype=jnp.float64)
                material = material.at[:, :, 0].set(active_params[0])
                material = material.at[:, :, 1:].set(jnp.array([0.39, .95, 2., 10., .35, .04, 79., 62., 2e4, 12.]))
                state = jnp.zeros((cells.shape[0], q, 11), dtype=jnp.float64)
                state = state.at[..., 10].set(10.0 / t_total)
                return material, state
            material = jnp.zeros((cells.shape[0], q, 5), dtype=jnp.float64)
            material = material.at[:, :, 0].set(active_params[1])
            material = material.at[:, :, 1:].set(jnp.array([23.1e3, .2, 8.96e3, 8.27e3]))
            return material, jnp.zeros((cells.shape[0], q, 7), dtype=jnp.float64)

        batches = []
        for cells, fe, q, matrix in (
            (groups["matrix_tri"], fe_tri, 1, True), (groups["matrix_quad"], fe_quad, 4, True),
            (groups["fiber_tri"], fe_tri, 1, False), (groups["fiber_quad"], fe_quad, 4, False),
        ):
            material, state = material_and_state(cells, q, matrix)
            batches.append(ElementBatch(
                fe_type=fe, connectivity_en=cells,
                constitutive_model=damage_elastic_isotropic_vmap if matrix else elastic_orthotropic,
                material_params=material, internal_state=state, n_dofs_per_basis=2,
            ))
        return batches

    def objective_from_history(isv_history, batches):
        s11, _, _ = compute_stress_strain_curve(isv_history, batches, points)
        return -jnp.max(s11)

    def jax_objective(active_params):
        return jax_history_objective(active_params, implicit_vjp=False)

    # Match the timestep AD optimization test, whose custom_vjp cannot be
    # used with jvp. Keep its reverse reference distinct from the JVP check.
    jax_step_options = SolverOptions(
        linear_solve_type=LinearSolverType.DENSE_INVERSE_JNP,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
    )
    jax_implicit_step = jax.jit(
        solve_nonlinear_quasi_step_autodiff_vjp,
        static_argnames=("solver_options", "jacobian_nnz"),
    )
    jax_forward_step = jax.jit(
        solve_nonlinear_step,
        static_argnames=("solver_options", "jacobian_nnz"),
    )
    reference_batches = make_batches(params)
    reference_ebc, reference_maps, reference_constraints, reference_nnz, reference_residual, reference_force = preprocess_bvp(
        vertices_vd=points, element_batches=reference_batches,
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=make_bcs(dxs), multipoint_constraints=None,
    )

    @jax.jit
    def flat_material(active_params):
        return jnp.concatenate([batch.material_params.ravel() for batch in make_batches(active_params)])

    @jax.jit
    def finish_history(history):
        # Stress averaging uses only the fixed connectivity and geometry from
        # these batches. Material dependence is already in the explicit ISVs.
        return objective_from_history(history, reference_batches)

    def make_state_update(residual_isv):
        # Construct both transforms once, not on each history evaluation.
        checkpointed = jax.checkpoint(residual_isv)

        @jax.jit
        def update(data, u):
            _, state = checkpointed(data, u)
            return jnp.concatenate([s.ravel() for s in state]), compute_ISV_be(state)

        return update

    # One static geometry/sparsity context for the entire history. Boundary
    # values and external loads travel in the same explicit operand as
    # material parameters and incoming constitutive state.
    plan_start = time.perf_counter()
    x0 = jnp.zeros(2 * points.shape[0], dtype=jnp.float64)
    all_boundary = reference_constraints.g
    if all_boundary.ndim == 1:
        all_boundary = jnp.broadcast_to(all_boundary, (t_total, all_boundary.size))
    all_force = reference_force.loads
    if all_force.ndim == 1:
        all_force = jnp.broadcast_to(all_force, (t_total, all_force.size))

    @jax.jit
    def pack_step(material, state, boundary, loads):
        return jnp.concatenate((material, state, boundary, loads))

    residual, jacobian, residual_isv = build_nonlinear_objects(
        element_residual_func=reference_residual, ebc=reference_ebc,
        assembly_map_b=reference_maps, jacobian_nnz=reference_nnz, u_0_g=x0,
        constraints=reference_constraints.replace(g=all_boundary[0]),
        f_ext=reference_force.replace(loads=all_force[0]),
        differentiate_internal_state=True, explicit_boundary_data=True,
    )
    shared_step = (residual, jacobian, make_state_update(residual_isv))
    petsc_steps = [shared_step] * t_total
    print(f"PETSc shared history function construction: {time.perf_counter() - plan_start:.6f} s")

    def jax_optimization_objective(active_params):
        return jax_history_objective(active_params, implicit_vjp=True)

    def jax_history_objective(active_params, *, implicit_vjp):
        batches = make_batches(active_params)
        ebc = reference_ebc.replace(
            material_params=flat_material(active_params),
        )
        constraints, f_ext = reference_constraints, reference_force
        loads = f_ext.loads
        if loads.ndim == 1:
            loads = jnp.broadcast_to(loads, (t_total, loads.shape[0]))

        def step(carry, inputs):
            u_prev, current = carry
            g, force = inputs
            timestep_solve = jax_implicit_step if implicit_vjp else jax_forward_step
            u, state, _, _, _ = timestep_solve(
                element_residual_func=reference_residual, ebc=current,
                assembly_map_b=reference_maps, jacobian_nnz=reference_nnz, u_0_g=u_prev,
                constraints=constraints.replace(g=g), solver_options=jax_step_options,
                f_ext=f_ext.replace(loads=force),
            )
            next_ebc = current.replace(internal_state=jnp.concatenate([s.ravel() for s in state]))
            return (u, next_ebc), compute_ISV_be(state)

        _, history = jax.lax.scan(
            step, (jnp.zeros(2 * points.shape[0], dtype=jnp.float64), ebc),
            (constraints.g, loads),
        )
        return objective_from_history(history, batches)

    def timed_evaluation(label, function, *inputs):
        start = time.perf_counter()
        result = function(*inputs)
        jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.6f} s")
        return result, elapsed

    def benchmark_pair(label, petsc_function, jax_function):
        # Report first calls separately. PETSc prewarming below is excluded.
        # Alternate execution order for the warm measurements.
        timed_evaluation(f"{label} PETSc first call", petsc_function, params)
        timed_evaluation(f"{label} JAX first call", jax_function, params)
        times = {"PETSc": [], "JAX": []}
        results = {}
        functions = {"PETSc": petsc_function, "JAX": jax_function}
        max_warmups = int(os.environ.get("FEA_JETSCI_AD_WARMUP_MAX", "8"))
        if max_warmups < 3:
            raise ValueError("FEA_JETSCI_AD_WARMUP_MAX must be at least 3.")
        warmups = {"PETSc": [], "JAX": []}

        def stable(values):
            return len(values) >= 3 and max(values[-3:]) <= 1.25 * min(values[-3:])

        for iteration in range(max_warmups):
            for backend, function in functions.items():
                if not stable(warmups[backend]):
                    _, elapsed = timed_evaluation(
                        f"{label} {backend} warmup {iteration + 1}", function, params,
                    )
                    warmups[backend].append(elapsed)
            if all(stable(values) for values in warmups.values()):
                break
        print(f"{label} warmup samples: {warmups}; "
              f"steady-state confirmed: { {key: stable(values) for key, values in warmups.items()} }")
        for repeat in range(repeats):
            order = ("PETSc", "JAX") if repeat % 2 == 0 else ("JAX", "PETSc")
            for backend in order:
                results[backend], elapsed = timed_evaluation(
                    f"{label} {backend} warm {repeat + 1}", functions[backend], params,
                )
                times[backend].append(elapsed)
        medians = {backend: float(np.median(values)) for backend, values in times.items()}
        print(f"{label} measurement samples: {times}")
        print(f"{label} warm medians: {medians}; JAX/PETSc = {medians['JAX'] / medians['PETSc']:.3f}x")
        return results["PETSc"], results["JAX"]

    petsc_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000, linear_relative_tol=1e-6, linear_absolute_tol=1e-14,
    )
    def petsc_objective(active_params):
        nonlocal petsc_options
        material_params = flat_material(active_params)
        incoming_state = reference_ebc.internal_state
        u = x0
        isv_history = []
        for index, (residual, jacobian, update_state) in enumerate(petsc_steps):
            # Keep the complete chain rule through constitutive history.
            # Only the continuation guess is treated as a numerical input.
            solve_parameters = pack_step(material_params, incoming_state, all_boundary[index], all_force[index])
            u, petsc_options = jetsci.differentiable_solve(
                petsc_options, residual, jacobian,
                jax.lax.stop_gradient(u), solve_parameters,
            )
            incoming_state, summary = update_state(solve_parameters, u)
            isv_history.append(summary)
        stacked_history = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *isv_history)
        return finish_history(stacked_history)

    try:
        # Initialize once using the same stable functions that the history
        # will use; no separate prewarm function set is created.
        residual, jacobian, _ = petsc_steps[0]
        _, petsc_options = jetsci.differentiable_solve(
            petsc_options, residual, jacobian, x0,
            pack_step(flat_material(params), reference_ebc.internal_state, all_boundary[0], all_force[0]),
        )

        print(f"Comparison: 1 fiber, {t_total} timesteps, params={params}, warm repeats={repeats}")
        print("JAX: dense inverse + compiled scan; PETSc: CG/Jacobi + eager timestep loop")
        print("Timings include complete history evaluation; this is an application comparison, not isolated solver timing.")
        jax_forward = jax.jit(jax_optimization_objective)
        petsc_objective_value, jax_objective_value = benchmark_pair("objective", petsc_objective, jax_forward)
        np.testing.assert_allclose(petsc_objective_value, jax_objective_value, rtol=1e-5, atol=1e-8)
        (petsc_value, petsc_jvp), (jax_value, jax_jvp) = benchmark_pair(
            "JVP",
            lambda active: jax.jvp(petsc_objective, (active,), (params_dot,)),
            jax.jit(lambda active: jax.jvp(jax_objective, (active,), (params_dot,))),
        )
        print("forward objective: PETSc =", petsc_value, "JAX =", jax_value)
        print("forward directional derivative: PETSc =", petsc_jvp, "JAX =", jax_jvp)
        np.testing.assert_allclose(petsc_value, jax_value, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(petsc_jvp, jax_jvp, rtol=5e-3, atol=1e-7)

        petsc_value_grad = jax.value_and_grad(petsc_objective)
        jax_value_grad = jax.jit(jax.value_and_grad(jax_optimization_objective))
        (petsc_value_vjp, petsc_grad), (jax_value_vjp, jax_grad) = benchmark_pair(
            "reverse value_and_grad", petsc_value_grad, jax_value_grad,
        )
        print("reverse gradient: PETSc =", petsc_grad, "JAX =", jax_grad)
        np.testing.assert_allclose(petsc_value_vjp, jax_value_vjp, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(petsc_grad, jax_grad, rtol=5e-3, atol=1e-7)

        # Same projected-gradient update as the legacy optimization test.
        # Correct its inverted fiber bounds (230000, 26000) to (230000, 260000).
        # Project both starting points identically, since the original 200000
        # starting fiber modulus is below the specified feasible range.
        lower = jnp.array([2500.0, 230000.0])
        upper = jnp.array([4500.0, 260000.0])
        learning_rate = jnp.array([1e4, 1e7])
        start_params = jnp.clip(params, lower, upper)
        print(f"Optimization: {optimization_steps} steps; bounds={lower}, {upper}; start={start_params}")

        def optimize(value_grad, backend):
            active = start_params
            losses, gradients, parameter_history = [], [], []
            for iteration in range(optimization_steps):
                (loss, gradient), _ = timed_evaluation(
                    f"{backend} optimization step {iteration + 1}", value_grad, active,
                )
                losses.append(loss)
                gradients.append(gradient)
                parameter_history.append(active)
                active = jnp.clip(active - learning_rate * gradient, lower, upper)
            return active, jnp.stack(parameter_history), jnp.stack(losses), jnp.stack(gradients)

        if optimization_steps:
            petsc_optimization, petsc_seconds = timed_evaluation(
                "PETSc complete optimization", lambda: optimize(petsc_value_grad, "PETSc"),
            )
            jax_optimization, jax_seconds = timed_evaluation(
                "JAX complete optimization", lambda: optimize(jax_value_grad, "JAX"),
            )
            for label, actual, expected in zip(
                ("final parameters", "parameter trajectory", "loss trajectory", "gradient trajectory"),
                petsc_optimization, jax_optimization, strict=True,
            ):
                print(f"Optimization {label}: PETSc={actual}; JAX={expected}")
                np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-7, err_msg=label)
            print(f"Optimization JAX/PETSc wall time = {jax_seconds / petsc_seconds:.3f}x")
    finally:
        if petsc_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(petsc_options.solver_key)


if __name__ == "__main__":
    test_fea_solve_dmg_scan_autodiff_jetsci()
