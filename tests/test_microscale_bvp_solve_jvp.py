"""Prepared primal and solve-plus-JVP comparison; run on Linux/CUDA.

python -m pytest -s -q tests/test_microscale_bvp_solve_jvp.py
FEA_BVP_AD_MESH selects the mesh (default r3); FEA_BVP_AD_REPEATS defaults to 3.
Preserves the forward test geometry/connectivity; no mesh repair is applied.
Both backends exclude preprocessing/setup and include assembly during solves.
PETSc defaults to LGMRES/Jacobi; JAX defaults to CSR incremental GMRES/CuPy ILU.
FEA_BVP_PETSC_METHOD=cg|gmres|lgmres and FEA_BVP_PETSC_PC=none|jacobi|ilu select PETSc.
FEA_BVP_JAX_METHOD=cg|gmres and FEA_BVP_JAX_PC=none|jacobi|ilu select JAX.
ILU synchronization and per-call factor cleanup are included in timings.
"""
from helper import *
from fe_jax.linear_elasticity import (
    elastic_isotropic as stateless_elastic_isotropic,
    elastic_orthotropic as stateless_elastic_orthotropic,
)
import pytest
import time
import jetsci
import json
import os
from fe_jax.fea import solve_nonlinear_step
from fe_jax.sparse_linear_solve import cupy_factor_scope

pytestmark = pytest.mark.slow


degree = 1

def test_microscale_bvp_solve_jvp():
    jax.config.update("jax_enable_x64", True)

    # import jax.extend
    # print(jax.extend.backend.get_backend().platform)

    # from jax_smi import initialise_tracking
    # initialise_tracking()

    # Read in the mesh
    mesh_name = os.environ.get("FEA_BVP_AD_MESH", "microscale_2D_r3.vtk")
    mesh = meshio.read(get_mesh(mesh_name))
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
        basis_degree=degree,
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


    x0 = jnp.zeros(V * U, dtype=jnp.float64)
    (ebc, assembly_maps, constraints, nnz, element_residual, force) = preprocess_bvp(
        vertices_vd=points, element_batches=make_element_batches(),
        element_residual_func=linear_elasticity_residual,
        boundary_conditions=bcs, multipoint_constraints=None,
    )
    residual_phi, jacobian_phi, _ = build_nonlinear_objects(
        element_residual_func=element_residual, ebc=ebc,
        assembly_map_b=assembly_maps, jacobian_nnz=nnz,
        u_0_g=x0, constraints=constraints, f_ext=force,
    )

    @jax.jit
    def pack(p):
        # Independent dimensionless stiffness multipliers for matrix/fiber.
        # Scale fiber E_xx, E_yy and G_xy together, not its Poisson ratio.
        matrix = matrix_mat_params_eqm.at[:, :, 0].multiply(p[0])
        fiber = fiber_mat_params_eqm.at[:, :, 0:2].multiply(p[1])
        fiber = fiber.at[:, :, 3].multiply(p[1])
        return jnp.concatenate((matrix.ravel(), fiber.ravel()))

    def residual(p, x):
        return residual_phi(pack(p), x)

    def jacobian(p, x):
        return jacobian_phi(pack(p), x)

    common = dict(
        nonlinear_relative_tol=1e-10, nonlinear_absolute_tol=1e-12,
        linear_relative_tol=1e-10, linear_absolute_tol=1e-12,
        linear_max_iter=20000, nonlinear_max_iter=10,
    )
    petsc_method = os.environ.get("FEA_BVP_PETSC_METHOD", "bcgs")
    petsc_pc = os.environ.get("FEA_BVP_PETSC_PC", "amg")
    petsc_methods = {"cg": jetsci.LinearSolverType.PETSC_CG,
                     "gmres": jetsci.LinearSolverType.PETSC_GMRES,
                     "bcgs":jetsci.LinearSolverType.PETSC_BCGS,
                     "lgmres": jetsci.LinearSolverType.PETSC_LGMRES}
    petsc_pcs = {"none": jetsci.PreconditionerType.PETSC_NONE,
                 "jacobi": jetsci.PreconditionerType.PETSC_JACOBI,
                 "amg": jetsci.PreconditionerType.PETSC_AMG,
                 "ilu": jetsci.PreconditionerType.PETSC_ILU}
    if petsc_method not in petsc_methods or petsc_pc not in petsc_pcs:
        raise ValueError("Set FEA_BVP_PETSC_METHOD=cg|gmres|lgmres and FEA_BVP_PETSC_PC=none|jacobi|ilu.")
    memory_backend = os.environ.get("FEA_BVP_JVP_MEMORY_WORKER")
    if memory_backend is not None and memory_backend not in ("PETSc", "JAX_CSR"):
        raise ValueError("FEA_BVP_JVP_MEMORY_WORKER must be PETSc or JAX_CSR.")
    petsc_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=petsc_methods[petsc_method],
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_AMG,
        **common,
    )
    jax_method = os.environ.get("FEA_BVP_JAX_METHOD", "gmres")
    jax_pc = os.environ.get("FEA_BVP_JAX_PC", "ilu")
    methods = {"cg": LinearSolverType.CG_JAX_SCIPY_CSR,
               "gmres": LinearSolverType.GMRES_JAX_SCIPY_CSR}
    pcs = {"none": PreconditionerType.NONE, "jacobi": PreconditionerType.JACOBI,
           "ilu": PreconditionerType.ILU_CUPY}
    if jax_method not in methods or jax_pc not in pcs:
        raise ValueError("Set FEA_BVP_JAX_METHOD=cg|gmres and FEA_BVP_JAX_PC=none|jacobi|ilu; no fallback is used.")
    stream_flag = os.environ.get("FEA_BVP_ILU_STREAM", "0")
    if stream_flag not in ("0", "1"):
        raise ValueError("FEA_BVP_ILU_STREAM must be 0 or 1.")
    use_jax_stream = stream_flag == "1"
    jax_options = SolverOptions(
        linear_solve_type=methods[jax_method], linear_precond_type=pcs[jax_pc],
        gmres_solve_method="incremental", gmres_stopping_test="true_residual",
        ilu_synchronize=not use_jax_stream, ilu_use_jax_stream=use_jax_stream, **common,
    )
    p = jnp.ones(2, dtype=jnp.float64)
    direction = jnp.array([0.01, -0.01], dtype=p.dtype)
    repeats = int(os.environ.get("FEA_BVP_AD_REPEATS", "3"))
    if repeats < 1:
        raise ValueError("FEA_BVP_AD_REPEATS must be at least 1 to measure warm timings.")

    def block(result):
        return jax.tree_util.tree_map(lambda a: a.block_until_ready(), result)

    def execute(call, arguments):
        # Owner is dynamic: no compilation per repetition. Retain factors until
        # the complete primal/JVP has finished, not just the primal solve.
        with cupy_factor_scope() as owner:
            return block(call(*arguments, owner))

    def validate_result(name, result):
        u, u_dot = result
        initial_norm = float(jnp.linalg.norm(residual(p, x0)))
        bound = max(common["nonlinear_absolute_tol"],
                    common["nonlinear_relative_tol"] * initial_norm)
        final_norm = float(jnp.linalg.norm(residual(p, u)))
        parameter_rhs = jax.jvp(lambda q: residual(q, u), (p,), (direction,))[1]
        # Independent tangent-equilibrium identity: R_x u_dot + R_p p_dot = 0.
        tangent_residual = jax.jvp(residual, (p, u), (direction, u_dot))[1]
        tangent_norm = float(jnp.linalg.norm(tangent_residual))
        tangent_bound = max(1e-12, 1e-6 * float(jnp.linalg.norm(parameter_rhs)))
        finite_metrics = bool(np.isfinite([initial_norm, final_norm, bound,
                                           tangent_norm, tangent_bound]).all())
        primal_finite = bool(jnp.isfinite(u).all())
        tangent_finite = bool(jnp.isfinite(u_dot).all())
        print("BVP_AD_CHECK " + json.dumps({
            "backend": name, "initial_residual": initial_norm,
            "final_residual": final_norm, "primal_bound": bound,
            "tangent_residual": tangent_norm, "tangent_bound": tangent_bound,
            "accepted": bool(finite_metrics and primal_finite and tangent_finite
                             and final_norm <= bound and tangent_norm <= tangent_bound),
        }), flush=True)
        assert primal_finite, name
        assert tangent_finite, name
        assert finite_metrics, f"{name}: nonfinite residual norm or validation bound"
        assert final_norm <= bound, f"{name}: primal did not converge; reject benchmark"
        assert tangent_norm <= tangent_bound, f"{name}: invalid JVP; reject benchmark"

    try:
        # Required eager construction for JetSci's traced public API.
        # The isolated JAX worker must never construct/run a PETSc solver.
        if memory_backend != "JAX_CSR":
            initial_u, petsc_options = jetsci.differentiable_solve(
                petsc_options, residual, jacobian, x0, p,
            )
            initial_u.block_until_ready()
            del initial_u

        @jax.jit
        def petsc_solve(active_p, resource_owner):
            del resource_owner
            return jetsci.differentiable_solve(
                petsc_options, residual, jacobian, x0, active_p,
            )[0]

        @jax.jit
        def jax_solve(active_p, resource_owner):
            # Same prepared geometry/layout, with material parameters explicit.
            return solve_nonlinear_step(
                element_residual_func=element_residual,
                ebc=ebc.replace(material_params=pack(active_p)),
                assembly_map_b=assembly_maps, jacobian_nnz=nnz,
                u_0_g=x0, constraints=constraints,
                solver_options=jax_options, f_ext=force,
                resource_owner=resource_owner,
            )[0]

        functions = {"PETSc": petsc_solve, "JAX_CSR": jax_solve}
        jvps = {
            name: jax.jit(lambda q, d, owner, solve=solve: jax.jvp(
                lambda active_p: solve(active_p, owner), (q,), (d,),
            ))
            for name, solve in functions.items()
        }
        print("BVP_AD_CONFIG " + json.dumps({
            "mesh": mesh_name, "degree": degree, "dofs": x0.size,
            "PETSc": f"{petsc_method}/{petsc_pc}", "JAX_CSR": f"{jax_method}/{jax_pc}",
            "jax_gmres_method": "incremental", "jax_gmres_stopping": "true_residual",
            "ilu_device_synchronization": not use_jax_stream,
            "ilu_use_jax_stream": use_jax_stream,
            "factor_lifetime": "one complete blocked primal/JVP call; cleanup included",
            "settings": common, "repeats": repeats,
            "scope": "prepared full primal solve and full solve + one JVP; setup excluded for both",
        }), flush=True)
        if memory_backend is not None:
            from bvp_memory_support import measure_route
            measure_route(
                memory_backend,
                lambda: execute(jvps[memory_backend], (p, direction)),
                block, jax, lambda result: validate_result(memory_backend, result),
                inputs=(p, direction, x0, ebc), workload="solve_plus_jvp",
                scope="prepared nonlinear forward solve + one full displacement JVP; includes CuPy factor cleanup; geometry preparation and PETSc eager initialization excluded from sampled phase but retained memory included in baseline/residency; no other backend executed",
            )
            return
        results = {}
        for phase, calls, arguments in (
            ("primal", functions, (p,)),
            ("solve_plus_jvp", jvps, (p, direction)),
        ):
            first, samples = {}, {name: [] for name in calls}
            for name, call in calls.items():
                start = time.perf_counter()
                execute(call, arguments)
                first[name] = time.perf_counter() - start
            # Alternate backend order to reduce fixed ordering/thermal bias.
            for repeat in range(repeats):
                names = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                for name in names:
                    start = time.perf_counter()
                    results[phase, name] = execute(calls[name], arguments)
                    samples[name].append(time.perf_counter() - start)
            medians = {name: float(np.median(values)) for name, values in samples.items()}
            print("BVP_AD_TIMING " + json.dumps({
                "phase": phase, "first_s": first, "samples_s": samples,
                "median_s": medians,
                "jax_over_petsc": medians["JAX_CSR"] / medians["PETSc"],
            }), flush=True)

        for name in functions:
            validate_result(name, results["solve_plus_jvp", name])

        for index, label in ((0, "displacement"), (1, "directional derivative")):
            actual = results["solve_plus_jvp", "PETSc"][index]
            reference = results["solve_plus_jvp", "JAX_CSR"][index]
            relative_error = float(jnp.linalg.norm(actual - reference)) / max(
                float(jnp.linalg.norm(actual)), float(jnp.linalg.norm(reference)), 1e-30,
            )
            print(f"BVP_AD_AGREEMENT {label}: relative_L2_error={relative_error:.9g}", flush=True)
            assert relative_error < 5e-3, f"{label}: backends disagree"
    finally:
        if petsc_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(petsc_options.solver_key)


if __name__ == "__main__":
    test_microscale_bvp_solve_jvp()

