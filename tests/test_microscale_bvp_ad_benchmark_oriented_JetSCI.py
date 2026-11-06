"""Corrected VTK-to-Basix microscale JVP comparison; Linux CUDA/PETSc only.

Separate corrected benchmark: the original test and mesh are unchanged.

Run: pytest -s -q tests/test_microscale_bvp_ad_benchmark_oriented_JetSCI.py
Set FEA_JETSCI_MICROSCALE_REPEATS to change the five warm repetitions.
"""
import os
from helper import *
import json
from microscale_orientation import (
    orient_vtk_triangle6_for_basix, quadratic_triangle, geometry_at, determinant_summary,
)
import jetsci
from jetsci.petsc_snes import differentiable_snes as diagnostics
from fe_jax.fea import preprocess_bvp, build_nonlinear_objects, solve_nonlinear_step
from fe_jax.linear_elasticity import (
    elastic_isotropic as stateless_elastic_isotropic,
    elastic_orthotropic as stateless_elastic_orthotropic,
)
import pytest
import time

pytestmark = pytest.mark.slow


def prepare_microscale_ad_problem():
    """Shared mesh/material setup for the AD and frozen-operator benchmarks."""
    jax.config.update("jax_enable_x64", True)

    # import jax.extend
    # print(jax.extend.backend.get_backend().platform)

    # from jax_smi import initialise_tracking
    # initialise_tracking()

    # Read in the mesh
    mesh = meshio.read(get_mesh("microscale_2D_r1.vtk"))
    points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
    cells = np.array(mesh.cells[0].data, dtype=np.uint64)
    mesh.cell_data["DomainIDs"][0] = np.array(
        mesh.cell_data["DomainIDs"][0], dtype=np.int64
    )
    cell_domain_ids = mesh.cell_data["DomainIDs"][0].flatten()
    print("# DoFs = ", 2 * points.shape[0])

    # Sizes of arrays
    U = 2  # number of solution components
    V = points.shape[0]  # number of vertices
    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=2,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=3,
    )
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points

    # VTK triangle6 edge numbering differs from Basix. Convert BEFORE
    # splitting domains and constructing either backend's element batches.
    if len(mesh.cells) != 1 or mesh.cells[0].type != "triangle6":
        raise ValueError("This corrected benchmark expects one VTK triangle6 block; do not apply its conversion to other cell types.")
    basix_element = quadratic_triangle()
    cells, orientation_report = orient_vtk_triangle6_for_basix(points, cells, basix_element)
    reference_points, _ = get_quadrature(fe_type=fe_type)
    determinants = geometry_at(points, cells, basix_element, reference_points)[1]
    if not np.isfinite(determinants).all() or np.any(determinants <= 0):
        raise ValueError("Corrected connectivity still has nonpositive/nonfinite geometric determinants. Inspect the mesh; no absolute determinant or solver fallback is used.")
    print("ORIENTED_MESH " + json.dumps({
        **orientation_report, "quadrature_determinants": determinant_summary(determinants),
        "mesh": "microscale_2D_r0.vtk",
    }), flush=True)

    # Define node sets
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
    right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
    bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
    top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]

    # Boundary conditions:
    # - Fix left nodes along x-direction
    # - Fix right nodes such that the model is subjected to 1% strain along x-axis
    # - Fix bottom nodes along y-direction
    # - Fix top nodes along y-direction
    right_u_val = (max_xy[0] - min_xy[0]) / 100.0
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, index=i, component=0, value=0.0)
            for i in left_points
        ]
        + [
            DirichletBC(
                bc_type=BCType.NODE, index=i, component=0, value=right_u_val
            )
            for i in right_points
        ]
        + [
            DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
            for i in bottom_points
        ]
        + [
            DirichletBC(bc_type=BCType.NODE, index=i, component=1, value=0.0)
            for i in top_points
        ]
    )

    # Extract cells for each subdomain
    matrix_cells = cells[cell_domain_ids == 0]
    fiber_cells = cells[cell_domain_ids == 1]

    # Set material properties
    @jax.jit
    def get_properties():
        # Neat 5220 Epoxy
        matrix_mat_params_eqm = jnp.zeros(shape=(matrix_cells.shape[0], Q, 2))
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3.45e9)  # E
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 1].set(0.35)  # nu
        # IM7 Fiber
        fiber_mat_params_eqm = jnp.zeros(shape=(fiber_cells.shape[0], Q, 4))
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(26e9)  # E_xx
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(26e9)  # E_yy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(
            0.7218543046357615
        )  # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(7.55e9)  # G_xy
        return (matrix_mat_params_eqm, fiber_mat_params_eqm)

    matrix_mat_params_eqm, fiber_mat_params_eqm = get_properties()
    print(fiber_mat_params_eqm[0, 0, :])

    @jax.tree_util.Partial
    @jax.jit
    def elastic_isotropic_vmap_adapter(
        eps_qdd, material_params, internal_state_qi
    ):
        """Adapt stateless isotropic elasticity to the damage residual API."""
        stress_qdd, _ = jax.vmap(stateless_elastic_isotropic)(
            eps_qdd, material_params
        )
        return stress_qdd, internal_state_qi

    @jax.tree_util.Partial
    @jax.jit
    def elastic_orthotropic_vmap_adapter(
        eps_qdd, material_params, internal_state_qi
    ):
        """Adapt stateless orthotropic elasticity to the damage residual API."""
        stress_qdd, _ = jax.vmap(stateless_elastic_orthotropic)(
            eps_qdd, material_params
        )
        return stress_qdd, internal_state_qi

    def make_element_batches():
        return [
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=matrix_cells,
                # This benchmark supplies only (E, nu) and no damage state.
                # The adapter matches linear_elasticity_dmg's batched keyword
                # interface while leaving the empty state unchanged.
                constitutive_model=elastic_isotropic_vmap_adapter,
                material_params=matrix_mat_params_eqm,
            ),
            ElementBatch(
                fe_type=fe_type,
                n_dofs_per_basis=2,
                connectivity_en=fiber_cells,
                constitutive_model=elastic_orthotropic_vmap_adapter,
                material_params=fiber_mat_params_eqm,
            ),
        ]


    repeats = int(os.environ.get("FEA_JETSCI_MICROSCALE_REPEATS", "5"))
    if repeats < 1:
        raise ValueError("FEA_JETSCI_MICROSCALE_REPEATS must be positive.")
    x0 = jnp.zeros(V * U, dtype=jnp.float64)
    ebc, maps, constraints, nnz, residual_func, force = preprocess_bvp(
        vertices_vd=points, element_batches=make_element_batches(),
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=bcs, multipoint_constraints=None,
    )
    residual, jacobian, _ = build_nonlinear_objects(
        element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
        jacobian_nnz=nnz, u_0_g=x0, constraints=constraints, f_ext=force,
    )
    phi = ebc.material_params
    direction = 1e-3 * phi * jnp.sin(jnp.arange(phi.size, dtype=phi.dtype) + 1)
    weights = jnp.linspace(0.5, 1.5, x0.size, dtype=x0.dtype)
    weights = weights / jnp.linalg.norm(weights)
    return (x0, ebc, maps, constraints, nnz, residual_func, force,
            residual, jacobian, phi, direction, weights, repeats)


def test_microscale_bvp_ad_benchmark_oriented():
    (x0, ebc, maps, constraints, nnz, residual_func, force,
     residual, jacobian, phi, direction, weights, repeats) = prepare_microscale_ad_problem()
    jax_options = SolverOptions(
        linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
        linear_precond_type=PreconditionerType.JACOBI,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000, linear_relative_tol=1e-6, linear_absolute_tol=1e-14,
    )
    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        nonlinear_relative_tol=1e-6, nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000, linear_relative_tol=1e-6, linear_absolute_tol=1e-14,
    )
    step = jax.jit(solve_nonlinear_step, static_argnames=("solver_options", "jacobian_nnz"))

    def petsc_solve(active):
        return jetsci.differentiable_solve(options, residual, jacobian, x0, active)[0]

    def jax_solve(active):
        return step(
            element_residual_func=residual_func,
            ebc=ebc.replace(material_params=active), assembly_map_b=maps,
            jacobian_nnz=nnz, u_0_g=x0, constraints=constraints,
            solver_options=jax_options, f_ext=force,
        )[0]

    def petsc_response(active):
        return jnp.vdot(weights, petsc_solve(active))

    def jax_response(active):
        return jnp.vdot(weights, jax_solve(active))

    def ready(result):
        return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)

    def timed(function):
        start = time.perf_counter()
        result = ready(function())
        return result, time.perf_counter() - start

    def compare(label, petsc_function, jax_function):
        p, p_first = timed(petsc_function)
        j, j_first = timed(jax_function)
        print(f"{label} first calls: PETSc={p_first:.6f}s JAX={j_first:.6f}s")
        for actual, expected in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(j), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-13, err_msg=label)
        times = {"PETSc": [], "JAX": []}
        for iteration in range(repeats):
            order = (("PETSc", petsc_function), ("JAX", jax_function))
            for backend, function in (order if iteration % 2 == 0 else reversed(order)):
                _, elapsed = timed(function)
                times[backend].append(elapsed)
        for backend, samples in times.items():
            print(f"{label} {backend}: samples={samples}; median={np.median(samples):.6f}s; "
                  f"min={min(samples):.6f}s; max={max(samples):.6f}s")
        print(f"{label} warm JAX/PETSc = {np.median(times['JAX']) / np.median(times['PETSc']):.3f}x")
        return p, j

    diagnostics.set_jvp_diagnostics(False)
    try:
        start = time.perf_counter()
        _, options = jetsci.differentiable_solve(options, residual, jacobian, x0, phi)
        ready((phi, direction, weights, x0))
        print(f"PETSc initialization + prewarm: {time.perf_counter() - start:.6f}s")
        print(f"Microscale AD benchmark: {x0.size} DOFs; warm repeats={repeats}")
        print("Same CG/Jacobi tolerances, fixed zero guess and prepared mesh for both backends.")
        print("PETSc public eager API versus compiled JAX FEA; setup excluded, first calls separate.")
        jax_primal = jax.jit(jax_response)
        jax_jvp = jax.jit(lambda active, tangent: jax.jvp(jax_response, (active,), (tangent,)))
        compare("primal response", lambda: petsc_response(phi), lambda: jax_primal(phi))
        p, j = compare(
            "solve + one JVP",
            lambda: jax.jvp(petsc_response, (phi,), (direction,)),
            lambda: jax_jvp(phi, direction),
        )
        print(f"Directional derivative: PETSc={p[1]}; JAX={j[1]}")
        _, p_tangent = jax.linearize(petsc_response, phi)
        _, j_tangent = jax.linearize(jax_response, phi)
        j_tangent = jax.jit(j_tangent)
        compare("JVP at saved equilibrium", lambda: p_tangent(direction), lambda: j_tangent(direction))

        # Instrumentation inserts GPU barriers and explicit KSP setup. Run it
        # separately, so these numbers never contaminate headline wall times.
        print("Separate diagnostic pass (GPU-synchronized stage timings):")
        diagnostic_samples = []
        diagnostics.set_jvp_diagnostics(True)
        for iteration in range(2):
            _, elapsed = timed(lambda: jax.jvp(petsc_response, (phi,), (direction,)))
            stages = diagnostics.get_jvp_diagnostics()
            diagnostic_samples.append(stages)
            print(f"PETSc diagnostic {iteration + 1}: wall={elapsed:.6f}s; stages={stages}")
        for name in diagnostic_samples[-1]:
            if name.endswith("_s"):
                print(f"diagnostic median {name}={np.median([s[name] for s in diagnostic_samples]):.6f}s")
        print("Diagnostic stages exclude some JAX tracing/dispatch overhead; do not expect an exact wall-time sum.")
    finally:
        diagnostics.set_jvp_diagnostics(False)
        if options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    test_microscale_bvp_ad_benchmark_oriented()
