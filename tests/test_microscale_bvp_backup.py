from helper import *
from fe_jax.linear_elasticity import (
    elastic_isotropic as stateless_elastic_isotropic,
    elastic_orthotropic as stateless_elastic_orthotropic,
)
import pytest
import time
import jetsci
import json
from petsc4py import PETSc

pytestmark = pytest.mark.slow


def test_microscale_bvp():
    jax.config.update("jax_enable_x64", True)

    # import jax.extend
    # print(jax.extend.backend.get_backend().platform)

    # from jax_smi import initialise_tracking
    # initialise_tracking()

    # Read in the mesh
    mesh = meshio.read(get_mesh("microscale_2D_r0.vtk"))
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
        basis_degree=2,  #1 for r3 and 2 foe all others
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
        """Adapt stateless isotropic elasticity to the FEA residual API."""
        stress_qdd, _ = jax.vmap(stateless_elastic_isotropic)(
            eps_qdd, material_params
        )
        return stress_qdd, internal_state_qi

    @jax.tree_util.Partial
    @jax.jit
    def elastic_orthotropic_vmap_adapter(
        eps_qdd, material_params, internal_state_qi
    ):
        """Adapt stateless orthotropic elasticity to the FEA residual API."""
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

    def run_jax(linear_solve_type=LinearSolverType.CG_JAX_SCIPY_CSR):
        return solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=make_element_batches(),
            u_0_g=u_0,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=linear_solve_type,
                linear_precond_type=PreconditionerType.ILU_CUPY,
                nonlinear_relative_tol=1e-10,
                nonlinear_absolute_tol=1e-10,
                linear_max_iter=25000,
                linear_relative_tol=1e-10,
                linear_absolute_tol=1e-10,
            ),
        )

    initial_residual_norm = None

    def run_petsc(diagnostics=False):
        nonlocal initial_residual_norm
        element_batches = make_element_batches()
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
        petsc_options = jetsci.SolverOptions(
            nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
            linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
            linear_solver_type=jetsci.LinearSolverType.PETSC_LGMRES,
            linear_preconditioner_side="right",
            linear_norm_type="unpreconditioned",
            nonlinear_relative_tol=1e-12,
            nonlinear_absolute_tol=1e-12,
            linear_max_iter=20000,
            linear_relative_tol=1e-12,
            linear_absolute_tol=1e-12,
        )
        try:
            phi = ebc.material_params
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
            if diagnostics:
                # Extra untimed solve: do not put diagnostic reductions,
                # synchronization or KSP.view() inside benchmark samples.
                u_petsc.block_until_ready()
                final_norm = float(jnp.linalg.norm(residual_petsc))
                initial_residual_norm = float(jnp.linalg.norm(residual_for_phi(phi, u_0)))
                solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(
                    petsc_options.solver_key
                )
                snes, ksp = solver.snes, solver.snes.getKSP()

                def enum_name(enum, value):
                    return "|".join(
                        name for name in dir(enum)
                        if not name.startswith("_")
                        and isinstance(getattr(enum, name), int)
                        and getattr(enum, name) == int(value)
                    ) or "UNKNOWN"

                print("BVP_PETSC_DIAGNOSTICS " + json.dumps({
                    "untimed_extra_solve": True,
                    "snes_reason": int(snes.getConvergedReason()),
                    "snes_reason_name": enum_name(PETSc.SNES.ConvergedReason, snes.getConvergedReason()),
                    "newton_iterations": snes.getIterationNumber(),
                    "snes_function_norm": snes.getFunctionNorm(),
                    "ksp_reason": int(ksp.getConvergedReason()),
                    "ksp_reason_name": enum_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason()),
                    "last_ksp_iterations": ksp.getIterationNumber(),
                    "total_ksp_iterations": snes.getLinearSolveIterations(),
                    "ksp_reported_residual": ksp.getResidualNorm(),
                    "ksp_type": ksp.getType(),
                    "pc_type": ksp.getPC().getType(),
                    "pc_side": enum_name(PETSc.PC.Side, ksp.getPCSide()),
                    "norm_type": enum_name(PETSc.KSP.NormType, ksp.getNormType()),
                    # petsc4py exposes setGMRESRestart but not a getter in
                    # the target version; ksp.view() below reports restart.
                    "ksp_tolerances": list(ksp.getTolerances()),
                    "initial_true_residual": initial_residual_norm,
                    "final_true_residual": final_norm,
                }), flush=True)
                ksp.view()
            updated_batches = [
                batch.replace(internal_state=internal_state)
                for batch, internal_state in zip(
                    element_batches, new_internal_state_eqi, strict=True
                )
            ]
            return u_petsc, residual_petsc, updated_batches
        finally:
            if petsc_options.solver_key is not None:
                jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(
                    petsc_options.solver_key
                )

    petsc_result, petsc_firstTime, petsc_times = time_solve(
        "PETSc GMRES/ILU differentiable_solve", run_petsc, n_calls=3
    )
    
    #jax_result, jax_firstTime, jax_times = time_solve("JAX solve_bvp", run_jax, n_calls=3)
    
    # The CSR option assembles the sparse Jacobian and converts COO to CSR
    # inside solve_bvp, so this timing includes assembly/conversion as well.
    csr_result, csr_firstTime, csr_times = time_solve(
        "JAX CSR GMRES/CuPy ILU(0) solve_bvp",
        lambda: run_jax(LinearSolverType.GMRES_JAX_SCIPY_CSR),
        n_calls=3,
    )
    u_petsc, residual_petsc, _ = petsc_result
    #u, residual, element_batches = jax_result
    u_csr, residual_csr, _ = csr_result


    # Inspect a separate execution only after both timed paths finish.
    run_petsc(diagnostics=True)

    residual_petsc_norm = jnp.linalg.norm(residual_petsc)
    print("|R| PETSc = ", residual_petsc_norm)
    print("petsc times", petsc_times)
    print("petsc CSR avg time", jnp.average(jnp.asarray(petsc_times[1:3])))
    print("jax CSR times", csr_times)
    print("jax CSR avg time", jnp.average(jnp.asarray(csr_times[1:3])))
    residual_csr_norm = jnp.linalg.norm(residual_csr)
    print("|R| JAX CSR = ", residual_csr_norm)

    exit(1)

    for backend, final_residual in (("PETSc", residual_petsc), ("JAX_CSR", residual_csr)):
        final_norm = float(jnp.linalg.norm(final_residual))
        bound = max(1e-14, 1e-12 * initial_residual_norm)
        print("BVP_RESIDUAL_CHECK " + json.dumps({
            "backend": backend,
            "initial_true_residual": initial_residual_norm,
            "final_true_residual": final_norm,
            "relative_residual": final_norm / initial_residual_norm if initial_residual_norm else None,
            "requested_bound": bound,
            "accepted": final_norm <= bound,
        }), flush=True)

    residual_petsc_norm = jnp.linalg.norm(residual_petsc)
    #residual_jax_norm = jnp.linalg.norm(residual)
    #solution_abs_error = jnp.linalg.norm(u_petsc - u)
    #solution_rel_error = solution_abs_error / (jnp.linalg.norm(u) + 1e-14)
    print("|R| PETSc = ", residual_petsc_norm)
    #print("|R| JAX   = ", residual_jax_norm)
    #print("|u_petsc - u_jax| = ", solution_abs_error)
    #print("relative solution error = ", solution_rel_error)
    print("petsc times", petsc_times)
    #print("jax times", jax_times)
    print("jax CSR times", csr_times)
    residual_csr_norm = jnp.linalg.norm(residual_csr)
    #csr_solution_abs_error = jnp.linalg.norm(u_csr - u)
    #csr_solution_rel_error = csr_solution_abs_error / (jnp.linalg.norm(u) + 1e-14)
    print("|R| JAX CSR = ", residual_csr_norm)
    #print("|u_csr - u_jax| = ", csr_solution_abs_error)
    #print("CSR relative solution error = ", csr_solution_rel_error)

    #assert bool(jnp.isfinite(residual_petsc_norm))
    #assert bool(jnp.isfinite(residual_jax_norm))
    #assert (float(solution_abs_error) < 1e-12) or (
    #    float(solution_rel_error) < 1e-5
    #)
    #assert bool(jnp.isfinite(residual_csr_norm))
    #assert (float(csr_solution_abs_error) < 1e-12) or (
    #    float(csr_solution_rel_error) < 1e-5
    #)

    # Make sure the solution matches at the Dirichlet BCs
    dirichlet_dofs = np.array([U * bc.index + bc.component for bc in bcs])
    dirichlet_values = np.array([bc.value for bc in bcs])
    #assert jnp.isclose(u[dirichlet_dofs], dirichlet_values).all()
    #assert jnp.isclose(u_csr[dirichlet_dofs], dirichlet_values).all()

    # Write output
    #mesh.point_data["u"] = u.reshape((points.shape[0], U))
    #mesh.write(get_output("test_microscale_bvp_out.vtk"))

test_microscale_bvp()
