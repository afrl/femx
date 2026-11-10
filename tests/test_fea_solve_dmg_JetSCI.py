import meshio
import numpy as np

# import fea_traditional as test
from helper import *
from fe_jax.igfem_mesh import *

import os
import pyvista as pv
import time
import json
import dataclasses
jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.write_vtk import *
from fe_jax.plot_results import plot_comparison
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
 
import jetsci

def test_fea_solve_dmg():
    args = {}
    num_fib = 3200    # 1, 2, 4, 9, 16, 23, 33, 36, 46, 49, 60, 64, 100
    args['t_total']  = 500
    args['dir_path'] = f"jetsci/jetsci_t{args['t_total']}_{num_fib}fib"
    args['strain_max'] = 0.012
    dt = 10/args['t_total']

    # Create the directory (and any necessary parent directories)
    args['out_dir'] = os.path.join("tests/output",args['dir_path'])
    args['vtk_dir'] = os.path.join(args['out_dir'],"vtks")
    args['ref_dir'] = os.path.join("tests/IGFEM_ref",f"{num_fib}fib_t{args['t_total']}")
    os.makedirs(args['out_dir'], exist_ok=True)
    os.makedirs(args['vtk_dir'], exist_ok=True)

    # Read in the mesh (IGFEM)
    mesh     = pv.read(f'tests/meshes/IGFEM_{num_fib}fib.vtk')
    vtk_mesh = mesh.copy()
    if comparison_backend is None:
        vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{0}.vtk")

    points = np.array(mesh.points, dtype=np.float64)[:,0:2]
    cells  = np.array(mesh.cells, dtype=np.uint64)
    print("# DoFs = ", 2 * points.shape[0])

    # {material: {shape: (cells, cell_ids)}} with only the shapes present in the mesh (tri, quad, polygon5, ...)
    cell_groups = sort_cells_by_material(mesh)


    length = (np.max(mesh.points[:,0]) - np.min(mesh.points[:,0]))

    # Sizes of arrays
    U = 2  # number of solution components
    V = points.shape[0]  # number of vertices
    E = cells.shape[0]  # number of elements
    M = 11  # number of material parameters
    F = V * U  # number of DoFs

    strain_increment = args['strain_max'] * length / args['t_total']

    # Define Dirichlet boundary conditions
    '''
    this BC locks the dx AND dy on both lhs and rhs 
    '''
    # LHS and RHS
    LHS = np.where(points[:,0]==np.min(points[:,0]))[0]
    RHS = np.where(points[:,0]==np.max(points[:,0]))[0]
    boundary_points = np.concatenate((LHS,RHS))
    dirichlet_values_init = np.zeros((boundary_points.shape[0], 2))
    n_LHS = len(LHS)
    n_vals = dirichlet_values_init.shape[0]
    dirichlet_values = np.empty(2 * n_vals)
    dirichlet_values[:n_vals] = dirichlet_values_init[:, 0]
    dirichlet_values[n_vals:] = dirichlet_values_init[:, 1]
    # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
    # Constrain every boundary point to have a random displacement
    dirichlet_bcs = [
        DirichletBC(index=i, component=j, value=1.0)
        for j in range(U)
        for i in boundary_points
    ]

    '''    
    # An array that is (# of constrainted DoFs, 2) with structure [point index][component of solution]
    dirichlet_bcs = np.zeros((U * boundary_points.shape[0], 2), dtype=np.uint64)
    for i,bp in enumerate(boundary_points):
        for j in range(U):
            dirichlet_bcs[U * i + j, 0] = bp
            dirichlet_bcs[U * i + j, 1] = j
    # Values of the Dirichlet boundary conditions matching 'dirichlet_bcs'
    dirichlet_values_init = np.zeros((boundary_points.shape[0], 2))
    '''    

    # Set material properties
    @partial(jax.jit, static_argnames=('Q',))
    def matrix_properties(cells, Q: int):
        # Neat 5220 Epoxy
        matrix_mat_params_eqm = jnp.zeros(shape=(cells.shape[0], Q, 11))
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 0].set(3900)     # E   MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 1].set(0.39)     # nu
        # Epoxy damage properties
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 2].set(0.95)     # P1, A
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 3].set(2.0)      # P2, B
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 4].set(10.0)     # mu_visc
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 5].set(0.35)     # eps_c
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 6].set(0.04)     # eps_t
        # Epoxy hardening properties
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 7].set(79)       # sigmay_c  MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 8].set(62)       # sigmay_t  MPa
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 9].set(2e4)      # H_ro, a
        matrix_mat_params_eqm = matrix_mat_params_eqm.at[:, :, 10].set(12)      # n_ro, b
        return matrix_mat_params_eqm

    @partial(jax.jit, static_argnames=('Q',))
    def fiber_properties(cells, Q: int):
        # IM7 Fiber
        fiber_mat_params_eqm = jnp.zeros(shape=(cells.shape[0], Q, 5))
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 0].set(233e3)  # E_xx
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 1].set(23.1e3) # E_yy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 2].set(0.2)    # nu_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 3].set(8.96e3) # G_xy
        fiber_mat_params_eqm = fiber_mat_params_eqm.at[:, :, 4].set(8.27e3) # G_yz
        return fiber_mat_params_eqm

    # Intialize internal_state_eqi
    @partial(jax.jit, static_argnames=('Q',))
    def matrix_ISV(cells, Q: int):
        matrix_internal_state_eqi = jnp.zeros(shape=(cells.shape[0], Q, 11))
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,0:3].set(0)    # e11, e22, e12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,3:6].set(0)    # s11, s22, s12
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,6]  .set(0)    # D
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,7]  .set(0)    # Y
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,8]  .set(0)    # tau0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,9]  .set(0)    # vM0
        matrix_internal_state_eqi = matrix_internal_state_eqi.at[...,10] .set(dt)   # dt
        return matrix_internal_state_eqi

    @partial(jax.jit, static_argnames=('Q',))
    def fiber_ISV(cells, Q: int):
        fiber_internal_state_eqi = jnp.zeros(shape=(cells.shape[0], Q, 7))
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,0:3].set(0)     # e11
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,3:6].set(0)     # e22
        fiber_internal_state_eqi = fiber_internal_state_eqi.at[...,6]  .set(-0.2)  # D
        return fiber_internal_state_eqi

    # (constitutive model, material parameters, initial ISV) of each material group
    materials = {
        "matrix": (damage_elastic_isotropic_vmap, matrix_properties, matrix_ISV),
        "fiber":  (elastic_orthotropic,           fiber_properties,  fiber_ISV),
    }

    # One element batch per (material, cell shape) present in the mesh
    element_batches = []
    batch_cell_ids = []     # mesh cell index of each element, per batch
    for name, shapes in cell_groups.items():
        constitutive_model, get_properties, init_ISV = materials[name]
        for shape, (cells_en, cell_ids) in shapes.items():
            fe_type = igfem_element_type(shape)
            Q = get_quadrature(fe_type=fe_type)[0].shape[0]    # number of quadrature points
            element_batches.append(ElementBatch(
                fe_type=fe_type,
                connectivity_en=cells_en,
                constitutive_model=constitutive_model,
                material_params=get_properties(cells_en, Q),
                internal_state=init_ISV(cells_en, Q),
                n_dofs_per_basis=2,
            ))
            batch_cell_ids.append(cell_ids)
            print(f"{name} {shape}: {cells_en.shape[0]} cells, Q = {Q}")

    u_prev = jnp.array(jnp.reshape(vtk_mesh['displacement'][:,:2],-1), dtype=jnp.float64)

    print('Start Deformation Loop')
    # Keep these options across increments so JetSCI reuses the PETSc SNES/KSP
    # context. The solve itself intentionally goes through JetSCI's public
    # differentiable_solve API rather than the FEA-specific PETSc wrapper.
    petsc_solver_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
        # linear_precond_type=jetsci.PETScPreconditionerType.ILU,
        linear_solve_type=jetsci.PETScLinearSolverType.CG,
        nonlinear_absolute_tol=1e-14,
        linear_max_iter=10000,
        linear_relative_tol=1e-6,
        linear_absolute_tol=1e-14,
    )

    runtime_check = None
    if comparison_backend is not None:
        from damage_runtime_support import DamageRuntimeCheck
        jax_format = os.environ.get("FEA_DAMAGE_JAX_FORMAT", "matrix_free")
        if jax_format not in ("matrix_free", "csr"):
            raise ValueError("FEA_DAMAGE_JAX_FORMAT must be matrix_free or csr; no fallback.")
        requested_restart = int(os.environ.get("FEA_DAMAGE_GMRES_RESTART", "30"))
        if requested_restart < 1:
            raise ValueError("FEA_DAMAGE_GMRES_RESTART must be a positive integer.")
        gmres_restart = min(requested_restart, int(u_prev.size))
        method = os.environ.get("FEA_DAMAGE_METHOD", "cg")
        pc = os.environ.get("FEA_DAMAGE_PC", "jacobi")
        if method not in ("cg", "gmres") or pc not in ("none", "jacobi", "ilu"):
            raise ValueError("Set FEA_DAMAGE_METHOD=cg|gmres and FEA_DAMAGE_PC=none|jacobi|ilu. These shared settings select both backends; no fallback is performed.")
        petsc_method = os.environ.get("FEA_DAMAGE_PETSC_METHOD", method)
        petsc_pc = os.environ.get("FEA_DAMAGE_PETSC_PC", pc)
        jax_method = os.environ.get("FEA_DAMAGE_JAX_METHOD", method)
        jax_pc = os.environ.get("FEA_DAMAGE_JAX_PC", pc)
        petsc_methods = {"cg": jetsci.LinearSolverType.PETSC_CG,
                         "gmres": jetsci.LinearSolverType.PETSC_GMRES,
                         "lgmres": jetsci.LinearSolverType.PETSC_LGMRES}
        petsc_pcs = {"none": jetsci.PreconditionerType.PETSC_NONE,
                     "jacobi": jetsci.PreconditionerType.PETSC_JACOBI,
                     "ilu": jetsci.PreconditionerType.PETSC_ILU,
                     "icc": jetsci.PreconditionerType.PETSC_ICC}
        jax_methods = {
            ("cg", "matrix_free"): LinearSolverType.CG_JAX_SCIPY_W_INFO,
            ("cg", "csr"): LinearSolverType.CG_JAX_SCIPY_CSR,
            ("gmres", "matrix_free"): LinearSolverType.GMRES_JAX_SCIPY,
            ("gmres", "csr"): LinearSolverType.GMRES_JAX_SCIPY_CSR,
        }
        jax_pcs = {"none": PreconditionerType.NONE, "jacobi": PreconditionerType.JACOBI,
                   "ilu": PreconditionerType.ILU_CUPY}
        if petsc_method not in petsc_methods or petsc_pc not in petsc_pcs:
            raise ValueError("Set FEA_DAMAGE_PETSC_METHOD=cg|gmres|lgmres and FEA_DAMAGE_PETSC_PC=none|jacobi|ilu|icc; no fallback.")
        if (jax_method, jax_format) not in jax_methods or jax_pc not in jax_pcs:
            raise ValueError("Set FEA_DAMAGE_JAX_METHOD=cg|gmres and FEA_DAMAGE_JAX_PC=none|jacobi|ilu; no fallback.")
        petsc_solver_options = dataclasses.replace(
            petsc_solver_options, linear_solver_type=petsc_methods[petsc_method],
            linear_preconditioner_type=petsc_pcs[petsc_pc],
        )
        common = {name: getattr(petsc_solver_options, name) for name in (
            "nonlinear_max_iter", "nonlinear_relative_tol", "nonlinear_absolute_tol",
            "linear_max_iter", "linear_relative_tol", "linear_absolute_tol",
        )}
        jax_solver_options = SolverOptions(
            linear_solve_type=jax_methods[jax_method, jax_format],
            linear_precond_type=jax_pcs[jax_pc],
            gmres_solve_method="incremental", gmres_stopping_test="true_residual",
            gmres_restart=gmres_restart,
            ilu_synchronize=True, **common,
        )
        check_rtol = float(os.environ.get("FEA_DAMAGE_CHECK_RTOL", "1e-4"))
        check_atol = float(os.environ.get("FEA_DAMAGE_CHECK_ATOL", "1e-8"))
        if not np.isfinite([check_rtol, check_atol]).all() or min(check_rtol, check_atol) < 0:
            raise ValueError("Damage agreement tolerances must be finite and nonnegative.")
        config = dict(
            fibers=num_fib, timesteps=args['t_total'], strain_max=args['strain_max'], dt=dt,
            dofs=int(u_prev.size), method=method, pc=pc, jax_format=jax_format,
            petsc_method=petsc_method, petsc_pc=petsc_pc,
            jax_method=jax_method, jax_pc=jax_pc,
            jax_gmres_solve_method="incremental", jax_gmres_stopping="true_residual",
            gmres_restart_requested=requested_restart,
            petsc_gmres_restart=gmres_restart if petsc_method in ("gmres", "lgmres") else None,
            jax_gmres_restart=gmres_restart if jax_method == "gmres" else None,
            jax_ilu_device_synchronization=jax_pc == "ilu",
            common_tolerances=common, check_rtol=check_rtol, check_atol=check_atol,
            x64=bool(jax.config.x64_enabled),
            scope="sum of blocked increment solves including per-step preprocessing, assembly and state update; mesh setup, validation, reference I/O and final solver destruction excluded",
        )
        runtime_check = DamageRuntimeCheck(comparison_backend, os.environ["FEA_DAMAGE_COMPARE_DIR"], check_rtol, check_atol)
        print("DAMAGE_RUNTIME_CONFIG " + json.dumps(dict(backend=comparison_backend, **config)), flush=True)

    try:
        for i in range(1,args['t_total']+1):
            dirichlet_values[n_LHS:n_vals] = strain_increment * i
            for bc in range(len(dirichlet_bcs)):
                dirichlet_bcs[bc].value = dirichlet_values[bc]

            if runtime_check is not None:
                incoming_batches = list(element_batches)
                jax.block_until_ready((u_prev, [batch.internal_state for batch in incoming_batches]))
                step_start = time.perf_counter()
            if comparison_backend == "JAX":
                u, residual, element_batches = solve_bvp(
                    element_residual_func=linear_elasticity_residual,
                    vertices_vd=points, element_batches=list(incoming_batches),
                    u_0_g=u_prev, boundary_conditions=dirichlet_bcs,
                    solver_options=jax_solver_options,
                )
            else:
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
                    boundary_conditions=dirichlet_bcs,
                    multipoint_constraints=None,
                )
                residual_for_phi, jacobian_for_phi, residual_isv_for_phi = (
                    build_nonlinear_objects(
                        element_residual_func=element_residual_func,
                        ebc=ebc,
                        assembly_map_b=assembly_map_b,
                        jacobian_nnz=jacobian_nnz,
                        u_0_g=u_prev,
                        constraints=constraint_system,
                        f_ext=f_ext,
                    )
                )

                # `phi` remains an explicit argument so this follows the public
                # differentiable-solve path used by applications that differentiate
                # with respect to material parameters.
                phi = ebc.material_params
                if (runtime_check is not None
                        and petsc_method in ("gmres", "lgmres")
                        and "FEA_DAMAGE_GMRES_RESTART" in os.environ):
                    # Configure the live KSP before the first public solve,
                    # without running a separate initialization solve. This
                    # test-level setting survives the public API's reuse path.
                    if petsc_solver_options.solver_key is None:
                        live_solver, petsc_solver_options = jetsci.build_solver_with_reuse(
                            petsc_solver_options,
                            jax.tree_util.Partial(residual_for_phi, phi),
                            jax.tree_util.Partial(jacobian_for_phi, phi), u_prev,
                        )
                    else:
                        live_solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(
                            petsc_solver_options.solver_key,
                        )
                    live_solver.snes.getKSP().setGMRESRestart(gmres_restart)
                u, petsc_solver_options = jetsci.differentiable_solve(
                    petsc_solver_options,
                    residual_for_phi,
                    jacobian_for_phi,
                    u_prev,
                    phi,
                )
                residual, new_internal_state_eqi = residual_isv_for_phi(phi, u)
                element_batches = [
                    batch.replace(internal_state=internal_state)
                    for batch, internal_state in zip(
                        element_batches, new_internal_state_eqi, strict=True
                    )
                ]

            if runtime_check is not None:
                jax.block_until_ready((u, residual, [batch.internal_state for batch in element_batches]))
                elapsed = time.perf_counter() - step_start
                # Rebuild an independent residual with the INCOMING state, not
                # the state just produced by this increment. All checks are untimed.
                check_ebc, check_maps, check_constraints, check_nnz, check_func, check_force = preprocess_bvp(
                    vertices_vd=points, element_batches=incoming_batches,
                    element_residual_func=linear_elasticity_residual,
                    boundary_conditions=dirichlet_bcs, multipoint_constraints=None,
                )
                check_residual, _, _ = build_nonlinear_objects(
                    element_residual_func=check_func, ebc=check_ebc, assembly_map_b=check_maps,
                    jacobian_nnz=check_nnz, u_0_g=u_prev, constraints=check_constraints, f_ext=check_force,
                )
                initial_norm = float(jnp.linalg.norm(check_residual(check_ebc.material_params, u_prev)))
                final_norm = float(jnp.linalg.norm(check_residual(check_ebc.material_params, u)))
                bound = max(common["nonlinear_absolute_tol"], common["nonlinear_relative_tol"] * initial_norm)

                def report_failure():
                    # Failure-only diagnostics: the solve and its timer have
                    # finished, and the live SNES still belongs to this step.
                    print("DAMAGE_FAILED_STEP " + json.dumps(dict(
                        backend=comparison_backend, step=i,
                        prescribed_displacement=float(strain_increment * i),
                        correction_norm=float(jnp.linalg.norm(u-u_prev)),
                        incoming_max_matrix_damage=max(
                            (float(jnp.max(batch.internal_state[..., 6]))
                             for batch in incoming_batches[:2] if batch.internal_state.size), default=0.0),
                        settings=config,
                    )), flush=True)
                    if comparison_backend == "PETSc":
                        from petsc4py import PETSc
                        solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(
                            petsc_solver_options.solver_key,
                        )
                        snes = solver.snes
                        ksp = snes.getKSP()

                        def enum_name(enum, value):
                            return "|".join(name for name in dir(enum)
                                            if not name.startswith("_")
                                            and isinstance(getattr(enum, name), int)
                                            and getattr(enum, name) == int(value)) or "UNKNOWN"

                        print("DAMAGE_PETSC_FAILURE " + json.dumps(dict(
                            step=i, snes_reason=int(snes.getConvergedReason()),
                            snes_reason_name=enum_name(PETSc.SNES.ConvergedReason, snes.getConvergedReason()),
                            newton_iterations=int(snes.getIterationNumber()),
                            snes_function_norm=float(snes.getFunctionNorm()),
                            ksp_reason=int(ksp.getConvergedReason()),
                            ksp_reason_name=enum_name(PETSc.KSP.ConvergedReason, ksp.getConvergedReason()),
                            last_ksp_iterations=int(ksp.getIterationNumber()),
                            total_ksp_iterations=int(snes.getLinearSolveIterations()),
                            ksp_reported_residual=float(ksp.getResidualNorm()),
                            ksp_type=ksp.getType(), pc_type=ksp.getPC().getType(),
                            pc_side=enum_name(PETSc.PC.Side, ksp.getPCSide()),
                            norm_type=enum_name(PETSc.KSP.NormType, ksp.getNormType()),
                            ksp_tolerances=list(ksp.getTolerances()),
                        )), flush=True)
                        ksp.view()

                runtime_check.step(i, elapsed, u, element_batches, initial_norm, final_norm, bound,
                                   on_failure=report_failure)
            u_prev = u

            print("Time step =", i)

            # # write and save to vtk
            vtk_mesh = write2VTK_avg(args,mesh,u,element_batches,batch_cell_ids)
            vtk_mesh.save(args['vtk_dir'] + f"/fea_solve_out_{i}.vtk")
    finally:
        solver_key = petsc_solver_options.solver_key
        if solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)
    # zip_folder(args['vtk_dir'], args['vtk_dir']+'.zip')
    if runtime_check is not None:
        runtime_check.finish(config)
    n_total_dofs = u.shape[0]

    return n_total_dofs, args, vtk_mesh

if __name__ == "__main__":
    t_start = time.perf_counter()

    # with jax.profiler.trace("./jax-trace-jetsci", create_perfetto_trace=True):
    n_total_dofs, args, vtk_mesh = test_fea_solve_dmg()

    t_end = time.perf_counter()
    total_time = t_end - t_start
    print("Time used:", total_time)

    with open(os.path.join(args['out_dir'], "statistics.txt"), "w") as f:
        f.write(f"Number of dofs: {n_total_dofs}\n")
        f.write(f"Total Solver time: {total_time} seconds\n")

    # Plot against the IGFEM reference, outside the timed region
    plot_comparison(args, vtk_mesh)
