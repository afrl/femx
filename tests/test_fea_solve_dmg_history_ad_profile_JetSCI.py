"""Stable damage-history AD profiling; Linux CUDA/PETSc only.

Run: env FEA_JETSCI_AD_TIMESTEPS=20 FEA_JETSCI_AD_REPEATS=2 python -m pytest -s -q tests/test_fea_solve_dmg_history_ad_profile_JetSCI.py
Set FEA_JETSCI_PROFILE_LOCAL_VJPS=0 to skip standalone local VJP probes.
"""
import os
import time
import json
from jetsci.petsc_snes import differentiable_snes as diagnostics

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
from fe_jax.sparse_linear_solve import PreconditionerType


jax.config.update("jax_enable_x64", True)


def test_damage_history_ad_profile():
    """Profile stable forward/reverse histories without changing solve rules."""
    num_fib = "1"
    # Keep the default practical for the GPU regression test.  Use 500 when
    # reproducing the full legacy scan_ad workload.
    t_total = int(os.environ.get("FEA_JETSCI_AD_TIMESTEPS", "20"))
    repeats = int(os.environ.get("FEA_JETSCI_AD_REPEATS", "2"))
    if t_total < 1 or repeats < 1:
        raise ValueError("Timesteps and repeats must be positive.")
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
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY,
        linear_precond_type=PreconditionerType.JACOBI,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000, linear_relative_tol=1e-6, linear_absolute_tol=1e-14,
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


    def ready(result):
        def wait(leaf):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
            return leaf
        return jax.tree_util.tree_map(wait, result)

    def measure(function):
        start = time.perf_counter()
        result = ready(function())
        return result, time.perf_counter() - start

    def normal_pair(label, petsc_function, jax_function):
        p, p_first = measure(petsc_function)
        j, j_first = measure(jax_function)
        print("AD_FIRST " + json.dumps({"phase": label, "PETSc_s": p_first, "JAX_s": j_first}), flush=True)
        for actual, expected in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(j), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-7, err_msg=label)
        max_warmups = int(os.environ.get("FEA_JETSCI_AD_WARMUP_MAX", "8"))
        if max_warmups < 3:
            raise ValueError("FEA_JETSCI_AD_WARMUP_MAX must be at least 3 to check consecutive stability.")
        warmups = {"PETSc": [], "JAX": []}

        def stable(values):
            window = values[-3:]
            return len(window) == 3 and max(window) <= 1.25 * min(window)

        for _ in range(max_warmups):
            for backend, function in (("PETSc", petsc_function), ("JAX", jax_function)):
                if not stable(warmups[backend]):
                    _, elapsed = measure(function)
                    warmups[backend].append(elapsed)
            if all(stable(values) for values in warmups.values()):
                break
        print("AD_WARMUP " + json.dumps({"phase": label, "samples_s": warmups,
              "stable": {key: stable(values) for key, values in warmups.items()}}), flush=True)
        samples = {"PETSc": [], "JAX": []}
        for iteration in range(repeats):
            order = (("PETSc", petsc_function), ("JAX", jax_function))
            for backend, function in (order if iteration % 2 == 0 else reversed(order)):
                _, elapsed = measure(function)
                samples[backend].append(elapsed)
        medians = {key: float(np.median(values)) for key, values in samples.items()}
        print("AD_WARM " + json.dumps({"phase": label, "samples_s": samples, "medians_s": medians,
              "steady_state_confirmed": {key: stable(values) for key, values in warmups.items()}}), flush=True)

    def instrumented(label, function):
        diagnostics.set_jvp_diagnostics(True)
        rows = []
        try:
            for iteration in range(repeats + 1):
                diagnostics.reset_ad_diagnostics()
                _, wall = measure(function)
                row = {"phase": label, "iteration": iteration, "wall_s": wall,
                       **diagnostics.get_ad_diagnostics()}
                rows.append(row)
                print("AD_STAGES " + json.dumps(row), flush=True)
                direction = "forward" if label == "complete JVP" else "reverse"
                assert row.get(direction + "_linear_solve_count", 0) >= t_total, (
                    "Missing derivative callback measurements; migrate the updated "
                    "JetSCI differentiable_snes.py before running this profile.", row,
                )
                assert row.get("forward_ksp_failed_count", 0) == 0, row
                assert row.get("reverse_ksp_failed_count", 0) == 0, row
                buffer_prefix = direction + "_device_buffer_"
                exclusive_stages = (
                    "buffer_views_s", "vec_create_s", "initial_guess_s",
                    "petsc_ksp_solve_s", "vec_destroy_s",
                )
                required = (*exclusive_stages, "total_s", "ksp_host_call_s",
                            "ksp_completion_wait_s", "diagnostic_sync_wait_s")
                for field in required:
                    assert buffer_prefix + field in row, (
                        "Missing device-buffer timings. Migrate both updated JetSCI "
                        "petsc_snes/solver.py and petsc_snes/differentiable_snes.py.", field,
                    )
                    assert row[buffer_prefix + field] >= 0, (field, row)
                np.testing.assert_allclose(
                    sum(row[buffer_prefix + field] for field in exclusive_stages),
                    row[buffer_prefix + "total_s"], rtol=1e-6, atol=1e-8,
                    err_msg="Device-buffer exclusive stages must account for their total",
                )
            # First instrumented execution is reported but excluded here.
            warm = rows[1:]
            aggregate = {key: float(np.median([r[key] for r in warm]))
                         for key in warm[0] if key.endswith("_s") or key.endswith("_count") or key.endswith("_iterations")}
            print("AD_STAGE_MEDIANS " + json.dumps({"phase": label, **aggregate}), flush=True)
            print("AD_KSP_BREAKDOWN " + json.dumps({
                "phase": label,
                **{name: value for name, value in aggregate.items()
                   if "_device_buffer_" in name},
                "note": "Synchronized diagnostic timings; host-call and sync-wait fields overlap stage totals.",
            }), flush=True)
        finally:
            diagnostics.set_jvp_diagnostics(False)

    def local_vjp_probes():
        # Diagnostic snapshots only: one current (parameters/state, solution)
        # pair per step. No PETSc matrices or Newton iteration tapes retained.
        incoming, u = reference_ebc.internal_state, x0
        material_params = flat_material(params)
        probes = []
        shared_residual, _, shared_update = shared_step

        @jax.jit
        def residual_probe(data, u):
            _, pullback = jax.vjp(lambda p: shared_residual(p, u), data)
            return pullback(jnp.ones_like(u))

        @jax.jit
        def state_probe(data, u):
            output, pullback = jax.vjp(shared_update, data, u)
            cotangent = jax.tree_util.tree_map(
                lambda value: jnp.ones_like(value) / max(value.size, 1), output,
            )
            return pullback(cotangent)

        for index, (residual, jacobian, update) in enumerate(petsc_steps):
            data = pack_step(material_params, incoming, all_boundary[index], all_force[index])
            u, _ = jetsci.differentiable_solve(
                petsc_options, residual, jacobian, jax.lax.stop_gradient(u), data,
            )
            incoming, _ = update(data, u)
            ready((data, u, incoming))

            probes.append((index + 1, data, u, residual_probe, state_probe))
        totals = {"residual_vjp_s": 0.0, "state_update_vjp_s": 0.0}
        for step, data, u, residual_probe, state_probe in probes:
            # Compile once; the probes use stable inputs and functions.
            _, residual_first = measure(lambda: residual_probe(data, u))
            _, state_first = measure(lambda: state_probe(data, u))
            residual_times, state_times = [], []
            for _ in range(repeats):
                _, elapsed = measure(lambda: residual_probe(data, u))
                residual_times.append(elapsed)
                _, elapsed = measure(lambda: state_probe(data, u))
                state_times.append(elapsed)
            row = {"step": step, "residual_vjp_first_s": residual_first,
                   "state_update_vjp_first_s": state_first,
                   "residual_vjp_s": float(np.median(residual_times)),
                   "state_update_vjp_s": float(np.median(state_times))}
            for key in totals:
                totals[key] += row[key]
            print("AD_LOCAL_VJP " + json.dumps(row), flush=True)
        print("AD_LOCAL_VJP_TOTALS " + json.dumps(totals), flush=True)

    diagnostics.set_jvp_diagnostics(False)
    try:
        residual, jacobian, _ = petsc_steps[0]
        _, petsc_options = jetsci.differentiable_solve(
            petsc_options, residual, jacobian, x0,
            pack_step(flat_material(params), reference_ebc.internal_state, all_boundary[0], all_force[0]),
        )
        print(f"Damage AD profile: timesteps={t_total}, repeats={repeats}; stable functions and compiled postprocessing", flush=True)
        print("Linear solvers: PETSc CG/Jacobi vs JAX CG/Jacobi; "
              "rtol=1e-6, atol=1e-14, maxiter=10000 (including the JAX adjoint)", flush=True)
        print("AD_WARM records have diagnostics OFF. AD_STAGES insert GPU barriers/explicit KSP setup.", flush=True)
        print("Stage totals overlap wall time. Unaccounted wall time includes JAX differentiation, dispatch, and other work.", flush=True)
        p_jvp = lambda: jax.jvp(petsc_objective, (params,), (params_dot,))
        j_jvp = jax.jit(lambda p: jax.jvp(jax_objective, (p,), (params_dot,)))
        p_reverse = jax.value_and_grad(petsc_objective)
        j_reverse = jax.jit(jax.value_and_grad(jax_optimization_objective))
        j_objective = jax.jit(jax_optimization_objective)
        normal_pair("objective", lambda: petsc_objective(params), lambda: j_objective(params))
        normal_pair("complete JVP", p_jvp, lambda: j_jvp(params))
        normal_pair("complete reverse", lambda: p_reverse(params), lambda: j_reverse(params))

        instrumented("complete JVP", p_jvp)
        instrumented("complete reverse", lambda: p_reverse(params))
        # Reuse one reverse linearization to exclude nonlinear forward work.
        diagnostics.set_jvp_diagnostics(False)
        (value, pullback), build_seconds = measure(lambda: jax.vjp(petsc_objective, params))
        print(f"AD_PULLBACK_BUILD seconds={build_seconds:.6f}; objective={value}", flush=True)
        (_, gradient) = p_reverse(params)
        actual = ready(pullback(jnp.array(1.0, dtype=params.dtype)))[0]
        np.testing.assert_allclose(actual, gradient, rtol=5e-3, atol=1e-7)
        instrumented("saved reverse pullback", lambda: pullback(jnp.array(1.0, dtype=params.dtype)))
        if os.environ.get("FEA_JETSCI_PROFILE_LOCAL_VJPS", "1") == "1":
            print("Local VJPs use synthetic cotangents: standalone cost probes, not an additive history breakdown.", flush=True)
            local_vjp_probes()
    finally:
        diagnostics.set_jvp_diagnostics(False)
        if petsc_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(petsc_options.solver_key)


if __name__ == "__main__":
    test_damage_history_ad_profile()

