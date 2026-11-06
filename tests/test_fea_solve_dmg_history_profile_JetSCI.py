"""Forward damage-history diagnosis, Linux GPU/PETSc only.

Run from fea-in-jax with compilation logs merged into the output:
  env JAX_LOG_COMPILES=1 FEA_JETSCI_AD_TIMESTEPS=20 \\
    python -m pytest -s -q tests/test_fea_solve_dmg_history_profile_JetSCI.py > damage_profile.txt 2>&1

Two full histories in each mode by default. No Newton graphs or matrices are
retained per step; prepared mode keeps only stable per-step function/layout objects.
"""
import os
import time
import json
from jetsci.petsc_snes.solver_lifecycle import get_petsc_solver_objects_from_key, destroy_petsc_solver

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

import jetsci
from helper import *
from fe_jax.fea import (
    build_nonlinear_objects,
    compute_ISV_be,
    preprocess_bvp,
)
from fe_jax.linear_elasticity_dmg import *
from fe_jax.utils import compute_stress_strain_curve


jax.config.update("jax_enable_x64", True)


def test_damage_history_profile():
    """Diagnose callback construction and nonlinear cost over repeated histories."""
    num_fib = "1"
    # Keep the default practical for the GPU regression test.  Use 500 when
    # reproducing the full legacy scan_ad workload.
    t_total = int(os.environ.get("FEA_JETSCI_AD_TIMESTEPS", "20"))
    passes = int(os.environ.get("FEA_JETSCI_PROFILE_PASSES", "2"))
    if t_total < 1 or passes < 2:
        raise ValueError("Use positive timesteps and at least two profile passes.")
    strain_max = 0.012
    params = jnp.array([3000.0, 200000.0], dtype=jnp.float64)

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


    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000, linear_relative_tol=1e-6, linear_absolute_tol=1e-14,
    )

    def ready(tree):
        def block(leaf):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
            return leaf
        return jax.tree_util.tree_map(block, tree)

    def construct(batches, dx):
        start = time.perf_counter()
        data = preprocess_bvp(
            vertices_vd=points, element_batches=batches,
            element_residual_func=linear_elasticity_residual,
            boundary_conditions=make_bcs(dx), multipoint_constraints=None,
        )
        ebc, maps, constraints, nnz, residual_func, f_ext = data
        ready((ebc, maps, constraints, f_ext))
        preprocessing = time.perf_counter() - start
        start = time.perf_counter()
        functions = build_nonlinear_objects(
            element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
            jacobian_nnz=nnz, u_0_g=jnp.zeros(2 * points.shape[0]),
            constraints=constraints, f_ext=f_ext, differentiate_internal_state=True,
        )
        return ebc, functions, preprocessing, time.perf_counter() - start

    def history(mode, pass_number, plan=None):
        nonlocal options
        rows, isv_history = [], []
        batches = make_batches(params)
        u = jnp.zeros(2 * points.shape[0], dtype=jnp.float64)
        ready((batches, u))
        start_history = time.perf_counter()
        for index, dx in enumerate(dxs):
            start_step = time.perf_counter()
            print(f"DAMAGE_STEP_BEGIN mode={mode} pass={pass_number} step={index + 1}", flush=True)
            if plan is None:
                ebc, functions, preprocessing, construction = construct(batches, dx)
                incoming = ebc.internal_state
            else:
                ebc, functions, _, _ = plan[index]
                preprocessing = construction = 0.0
                incoming = jnp.concatenate([batch.internal_state.ravel() for batch in batches])
            residual, jacobian, residual_isv = functions
            packing_start = time.perf_counter()
            data = jnp.concatenate((ebc.material_params, incoming))
            ready(data)
            packing = time.perf_counter() - packing_start
            previous = {}
            if options.solver_key is not None:
                solver, _ = get_petsc_solver_objects_from_key(options.solver_key)
                previous = dict(solver.callback_stats or {})
            start_solve = time.perf_counter()
            u, options = jetsci.differentiable_solve(options, residual, jacobian, u, data)
            ready(u)
            solve_wall = time.perf_counter() - start_solve
            solver, _ = get_petsc_solver_objects_from_key(options.solver_key)
            stats = dict(solver.callback_stats or {})
            snes = solver.snes
            start_state = time.perf_counter()
            # Match the history comparison's pure JAX state-update path.
            _, state = jax.checkpoint(residual_isv)(data, u)
            ready(state)
            state_time = time.perf_counter() - start_state
            batches = [batch.replace(internal_state=s) for batch, s in zip(batches, state, strict=True)]
            start_post = time.perf_counter()
            summary = ready(compute_ISV_be(state))
            isv_history.append(summary)
            post_time = time.perf_counter() - start_post
            row = {
                "mode": mode, "pass": pass_number, "step": index + 1,
                "preprocess_s": preprocessing, "function_build_s": construction,
                "parameter_pack_s": packing, "public_solve_s": solve_wall,
                "snes_s": solver.last_diagnostics["snes_solve_s"],
                "state_update_s": state_time, "postprocess_s": post_time,
                "residual_callback_s": stats.get("residual_total_s", 0) - previous.get("residual_total_s", 0),
                "jacobian_callback_s": stats.get("jacobian_total_s", 0) - previous.get("jacobian_total_s", 0),
                "residual_calls": stats.get("residual_calls", 0) - previous.get("residual_calls", 0),
                "jacobian_calls": stats.get("jacobian_calls", 0) - previous.get("jacobian_calls", 0),
                "nonlinear_iterations": int(snes.getIterationNumber()),
                "linear_iterations_total": int(snes.getLinearSolveIterations()),
                "linear_iterations_last": int(snes.getKSP().getIterationNumber()),
                "snes_reason": int(snes.getConvergedReason()),
                "residual_norm": float(snes.getFunctionNorm()),
                "max_matrix_damage": max(float(jnp.max(s[..., 6])) for s in state[:2] if s.size),
            }
            row["step_wall_s"] = time.perf_counter() - start_step
            rows.append(row)
            print("DAMAGE_STEP " + json.dumps(row), flush=True)
            assert row["snes_reason"] > 0, f"Unconverged step: {row}"
        states = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *isv_history)
        stress, _, _ = ready(compute_stress_strain_curve(states, batches, points))
        objective = float(-jnp.max(stress))
        total = time.perf_counter() - start_history
        summary = {"mode": mode, "pass": pass_number, "history_wall_s": total, "objective": objective}
        for key in ("preprocess_s", "function_build_s", "public_solve_s", "snes_s",
                    "state_update_s", "residual_callback_s", "jacobian_callback_s",
                    "nonlinear_iterations", "linear_iterations_total"):
            summary[key] = sum(row[key] for row in rows)
        print("DAMAGE_PASS " + json.dumps(summary), flush=True)
        return objective

    print(f"Damage history profile: {t_total} timesteps; {passes} passes per mode; 1 fiber", flush=True)
    print(f"JAX compilation logging enabled: {jax.config.jax_log_compiles}", flush=True)
    print("Callback times overlap SNES/public solve times; do not add overlapping columns.", flush=True)
    print("Run with JAX_LOG_COMPILES=1; compilation messages on stderr can be correlated with DAMAGE_STEP_BEGIN.", flush=True)
    print("Timing includes diagnostic output and stage synchronization; use the ordinary benchmark for headline performance.", flush=True)
    objectives = []
    try:
        # Reproduce the original loop before introducing stable function objects.
        for iteration in range(1, passes + 1):
            objectives.append(history("rebuild_each_step", iteration))
        start = time.perf_counter()
        plan = [construct(make_batches(params), dx) for dx in dxs]
        print(f"Prepared-plan preprocessing + function construction: {time.perf_counter() - start:.6f}s", flush=True)
        # Identical dynamics: only function/mesh-object construction is hoisted.
        # Changing current state remains an explicit input on every step.
        for iteration in range(1, passes + 1):
            objectives.append(history("reuse_step_functions", iteration, plan))
        np.testing.assert_allclose(objectives, objectives[0], rtol=1e-8, atol=1e-8)
    finally:
        if options.solver_key is not None:
            destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    test_damage_history_profile()

