from helper import *
from fe_jax.linear_elasticity import (
    elastic_isotropic as stateless_elastic_isotropic,
    elastic_orthotropic as stateless_elastic_orthotropic,
)
import pytest
import time

pytestmark = pytest.mark.slow


def test_microscale_bvp():
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

    u_0 = jnp.zeros(shape=(V * U))

    # Solve the boundary value problem

    def block_bvp_result(result):
        u, residual, element_batches = result
        u.block_until_ready()
        residual.block_until_ready()
        return u, residual, element_batches

    def time_solve(label, solve_func, n_calls=3):
        start = time.perf_counter()
        result = block_bvp_result(solve_func())
        first_call_time = time.perf_counter() - start

        times = []
        for _ in range(n_calls):
            start = time.perf_counter()
            result = block_bvp_result(solve_func())
            times.append(time.perf_counter() - start)

        print(
            f"{label}: first call {first_call_time:.6f} sec, "
            f"avg {sum(times) / len(times):.6f} sec "
            f"(n={n_calls}, min={min(times):.6f}, max={max(times):.6f})"
        )
        return result, first_call_time, times

    def build_public_differentiable_bvp(element_batches):
        """Build the FEA residual/Jacobian consumed by JetSCI's public API."""
        (
            ebc,
            assembly_map_b,
            constraint_system,
            jacobian_nnz,
            element_residual_func,
            f_ext,
        ) = preprocess_bvp(
            vertices_vd=points,
            element_batches=element_batches,
            element_residual_func=linear_elasticity_residual,
            boundary_conditions=bcs,
            multipoint_constraints=None,
        )
        residual_for_phi, jacobian_for_phi, residual_isv_for_phi = (
            build_nonlinear_objects(
                element_residual_func=element_residual_func,
                ebc=ebc,
                assembly_map_b=assembly_map_b,
                jacobian_nnz=jacobian_nnz,
                u_0_g=u_0,
                constraints=constraint_system,
                f_ext=f_ext,
            )
        )
        return (
            residual_for_phi,
            jacobian_for_phi,
            residual_isv_for_phi,
            ebc.material_params,
        )

    def run_petsc():
        element_batches = make_element_batches()
        residual_for_phi, jacobian_for_phi, residual_isv_for_phi, phi = (
            build_public_differentiable_bvp(element_batches)
        )
        petsc_options = jetsci.SolverOptions(
            nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
            linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
            linear_solver_type=jetsci.LinearSolverType.PETSC_GMRES,
            nonlinear_absolute_tol=1e-14,
            linear_max_iter=10000,
            linear_relative_tol=1e-6,
            linear_absolute_tol=1e-14,
        )
        try:
            u_petsc, petsc_options = jetsci.differentiable_solve(
                petsc_options,
                residual_for_phi,
                jacobian_for_phi,
                u_0,
                phi,
            )
            residual_petsc, new_internal_state_eqi = residual_isv_for_phi(
                phi, u_petsc
            )
            element_batches = [
                batch.replace(internal_state=internal_state)
                for batch, internal_state in zip(
                    element_batches, new_internal_state_eqi, strict=True
                )
            ]
            return u_petsc, residual_petsc, element_batches
        finally:
            if petsc_options.solver_key is not None:
                jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(
                    petsc_options.solver_key
                )

    def run_jax():
        return solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=make_element_batches(),
            u_0_g=u_0,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY,
                linear_precond_type=PreconditionerType.JACOBI,
            ),
        )



    petsc_result, _, _ = time_solve(
        "PETSc differentiable_solve", run_petsc, n_calls=3
    )
    u_petsc, residual_petsc, _ = petsc_result

    jax_result, _, _ = time_solve("JAX solve_bvp", run_jax, n_calls=2)
    u, residual, element_batches = jax_result

    residual_petsc_norm = jnp.linalg.norm(residual_petsc)
    residual_jax_norm = jnp.linalg.norm(residual)
    print("|R| PETSc = ", residual_petsc_norm)
    print("|R| JAX   = ", residual_jax_norm)
    print("|u_petsc - u_jax| = ", jnp.linalg.norm(u_petsc - u))

    diff_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        nonlinear_relative_tol=1e-6,
        nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000,
        linear_relative_tol=1e-6,
        linear_absolute_tol=1e-14,
    )
    residual_for_phi, jacobian_for_phi, _, phi_0 = build_public_differentiable_bvp(
        make_element_batches()
    )
    x_0 = u_0

    try:
        # Populate solver_key once. Subsequent primal, JVP, and VJP calls
        # reuse this public differentiable_solve context.
        _, diff_options = jetsci.differentiable_solve(
            diff_options,
            residual_for_phi,
            jacobian_for_phi,
            x_0,
            phi_0,
        )
        response_weights = jnp.linspace(
            0.5,
            1.5,
            x_0.shape[0],
            dtype=x_0.dtype,
        )
        response_weights = response_weights / jnp.linalg.norm(response_weights)
        phi_direction = jnp.sin(
            jnp.arange(phi_0.shape[0], dtype=phi_0.dtype) + 1.0
        )
        phi_dot = 1e-3 * phi_0 * phi_direction

        def scalar_response(active_phi):
            u_active, _ = jetsci.differentiable_solve(
                diff_options,
                residual_for_phi,
                jacobian_for_phi,
                x_0,
                active_phi,
            )
            return jnp.vdot(response_weights, u_active)

        def make_element_batches_from_phi(active_phi):
            matrix_size = matrix_mat_params_eqm.size
            matrix_phi = active_phi[:matrix_size].reshape(matrix_mat_params_eqm.shape)
            fiber_phi = active_phi[matrix_size:].reshape(fiber_mat_params_eqm.shape)
            return [
                ElementBatch(
                    fe_type=fe_type,
                    n_dofs_per_basis=2,
                    connectivity_en=matrix_cells,
                    constitutive_model=elastic_isotropic_vmap_adapter,
                    material_params=matrix_phi,
                ),
                ElementBatch(
                    fe_type=fe_type,
                    n_dofs_per_basis=2,
                    connectivity_en=fiber_cells,
                    constitutive_model=elastic_orthotropic_vmap_adapter,
                    material_params=fiber_phi,
                ),
            ]

        def jax_scalar_response(active_phi):
            u_active, _, _ = solve_bvp(
                element_residual_func=linear_elasticity_residual,
                vertices_vd=points,
                element_batches=make_element_batches_from_phi(active_phi),
                u_0_g=u_0,
                boundary_conditions=bcs,
                solver_options=SolverOptions(
                    linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
                    linear_precond_type=PreconditionerType.JACOBI,
                    nonlinear_relative_tol=1e-6,
                    nonlinear_absolute_tol=1e-14,
                    linear_max_iter=10000,
                    linear_relative_tol=1e-6,
                    linear_absolute_tol=1e-14,
                ),
            )
            return jnp.vdot(response_weights, u_active)

        PETScStartTime = time.time()
        response, response_dot = jax.jvp(
            scalar_response,
            (phi_0,),
            (phi_dot,),
        )
        response.block_until_ready()
        response_dot.block_until_ready()
        PETScDiffTime = time.time() - PETScStartTime

        eps = 1.0
        response_plus = scalar_response(phi_0 + eps * phi_dot)
        response_minus = scalar_response(phi_0 - eps * phi_dot)
        finite_difference_dot = (response_plus - response_minus) / (2.0 * eps)
        finite_difference_dot.block_until_ready()

        JaxStartTime = time.time()
        jax_response, jax_response_dot = jax.jvp(
            jax_scalar_response,
            (phi_0,),
            (phi_dot,),
        )
        jax_response.block_until_ready()
        jax_response_dot.block_until_ready()
        JaxDiffTime = time.time() - JaxStartTime

        derivative_abs_error = jnp.abs(response_dot - finite_difference_dot)
        derivative_rel_error = derivative_abs_error / (
            jnp.abs(finite_difference_dot) + 1e-14
        )
        finite_difference_signal = jnp.abs(response_plus - response_minus)
        expected_finite_difference_signal = jnp.abs(2.0 * eps * response_dot)
        petsc_jax_abs_error = jnp.abs(response_dot - jax_response_dot)
        petsc_jax_rel_error = petsc_jax_abs_error / (
            jnp.abs(jax_response_dot) + 1e-14
        )
        print("PETSc d response / d phi dot =", response_dot)
        print("JAX d response / d phi dot =", jax_response_dot)
        print("Finite-difference d response / d phi dot =", finite_difference_dot)
        print("Finite-difference response signal =", finite_difference_signal)
        print("Expected finite-difference response signal =", expected_finite_difference_signal)
        print("PETSc vs finite-difference abs error =", derivative_abs_error)
        print("PETSc vs finite-difference rel error =", derivative_rel_error)
        print("PETSc vs JAX abs error =", petsc_jax_abs_error)
        print("PETSc vs JAX rel error =", petsc_jax_rel_error)
        print("PETSc vs Jax time:",PETScDiffTime,"vs",JaxDiffTime)
        assert (float(petsc_jax_abs_error) < 1e-10) or (
            float(petsc_jax_rel_error) < 5e-2
        )
        if float(expected_finite_difference_signal) > 1e-10:
            assert (float(derivative_abs_error) < 1e-10) or (
                float(derivative_rel_error) < 5e-2
            )
        else:
            print(
                "Skipping finite-difference assertion because the expected response "
                "difference is below the solve-noise threshold."
            )
    finally:
        solver_key = diff_options.solver_key
        if solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)

    # Make sure the solution matches at the Dirichlet BCs
    dirichlet_dofs = np.array([U * bc.index + bc.component for bc in bcs])
    dirichlet_values = np.array([bc.value for bc in bcs])
    assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()


test_microscale_bvp()

