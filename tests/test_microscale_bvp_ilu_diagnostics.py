"""Single diagnostic BVP solve, before any timing or PETSc solve.

Default: current r3 geometry, JAX CSR CG with CuPy ILU(0).
FEA_BVP_DIAGNOSTIC_MESH selects r0/r1/r2/r3; degree follows cell type.
FEA_BVP_JAX_METHOD=cg|gmres; FEA_BVP_JAX_PC=ilu|jacobi|none.
Existing GMRES method/restart/stopping environment variables are supported.
ILU audit includes free-DOF probes, eight-step JAX/CuPy PCG replays and
one full direct CuPy CG reference on the first Newton step. These add
diagnostic runtime; progress prints every 1000 reference CG iterations.
No orientation correction or installed-library patch is applied here.
Run directly or with pytest -s; failures intentionally propagate.
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
from petsc4py import PETSc

pytestmark = pytest.mark.slow


def test_microscale_bvp_ilu_diagnostics():
    jax.config.update("jax_enable_x64", True)

    # import jax.extend
    # print(jax.extend.backend.get_backend().platform)

    # from jax_smi import initialise_tracking
    # initialise_tracking()

    # Read in the mesh
    mesh_name = os.environ.get("FEA_BVP_DIAGNOSTIC_MESH", "microscale_2D_r3.vtk")
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

    u_0 = jnp.zeros(shape=(V * U))

    # Solve the boundary value problem

    def block_bvp_result(result):
        u, residual, element_batches = result
        u.block_until_ready()
        residual.block_until_ready()
        return u, residual, element_batches

    jax_method = os.environ.get("FEA_BVP_JAX_METHOD", "cg")
    jax_pc = os.environ.get("FEA_BVP_JAX_PC", "ilu")
    jax_methods = {"cg": LinearSolverType.CG_JAX_SCIPY_CSR,
                   "gmres": LinearSolverType.GMRES_JAX_SCIPY_CSR}
    jax_pcs = {"none": PreconditionerType.NONE, "jacobi": PreconditionerType.JACOBI,
               "ilu": PreconditionerType.ILU_CUPY}
    if jax_method not in jax_methods or jax_pc not in jax_pcs:
        raise ValueError("Set FEA_BVP_JAX_METHOD=cg|gmres and FEA_BVP_JAX_PC=none|jacobi|ilu; no fallback is performed.")
    gmres_method = os.environ.get("FEA_BVP_JAX_GMRES_SOLVE_METHOD", "batched")
    gmres_restart = int(os.environ.get("FEA_BVP_JAX_GMRES_RESTART", "30"))
    gmres_stopping = os.environ.get("FEA_BVP_JAX_GMRES_STOPPING", "true_residual")
    sync_flag = os.environ.get("FEA_BVP_ILU_SYNC", "1")
    stream_flag = os.environ.get("FEA_BVP_ILU_STREAM", "0")
    if stream_flag not in ("0", "1"):
        raise ValueError("FEA_BVP_ILU_STREAM must be 0 or 1.")
    use_jax_stream = stream_flag == "1"
    if sync_flag not in ("0", "1"):
        raise ValueError("FEA_BVP_ILU_SYNC must be 0 (original path) or 1 (device synchronization experiment).")

    def run_jax(linear_solve_type=LinearSolverType.CG_JAX_SCIPY_CSR,
                *, diagnostics=False, solve_method=gmres_method):
        return solve_bvp(
            element_residual_func=linear_elasticity_residual,
            vertices_vd=points,
            element_batches=make_element_batches(),
            u_0_g=u_0,
            boundary_conditions=bcs,
            solver_options=SolverOptions(
                linear_solve_type=linear_solve_type,
                linear_precond_type=jax_pcs[jax_pc],
                nonlinear_relative_tol=1e-10,
                nonlinear_absolute_tol=1e-12,
                linear_max_iter=20000,
                linear_relative_tol=1e-10,
                linear_absolute_tol=1e-12,
                gmres_solve_method=solve_method,
                gmres_restart=gmres_restart,
                gmres_stopping_test=gmres_stopping,
                linear_diagnostics=diagnostics,
                ilu_application_diagnostics=diagnostics,
                ilu_synchronize=sync_flag == "1" and not use_jax_stream,
                ilu_use_jax_stream=use_jax_stream,
            ),
        )


    print("BVP_DIAGNOSTIC_CONFIG " + json.dumps({
        "mesh": mesh_name, "degree": degree, "dofs": int(u_0.size),
        "solver": jax_method, "pc": jax_pc,
        "gmres_method": gmres_method, "restart": gmres_restart,
        "stopping_test": gmres_stopping,
        "ilu_device_synchronization": sync_flag == "1" and not use_jax_stream,
        "ilu_use_jax_stream": use_jax_stream,
        "scope": "one complete solve_bvp; diagnostics on from the first call; no benchmark warmups",
        "ilu_reference": "direct CuPy CG + free-DOF probes + short PCG replays; diagnostic only",
    }), flush=True)
    try:
        result = block_bvp_result(run_jax(jax_methods[jax_method], diagnostics=True))
        jax.effects_barrier()
        u, residual, _ = result
        print("BVP_DIAGNOSTIC_RESULT " + json.dumps({
            "residual_norm": float(jnp.linalg.norm(residual)),
            "solution_finite": bool(jnp.all(jnp.isfinite(u))),
        }), flush=True)
    except Exception as error:
        print("BVP_DIAGNOSTIC_FAILED " + json.dumps({
            "type": type(error).__name__, "message": str(error),
        }), flush=True)
        raise


if __name__ == "__main__":
    test_microscale_bvp_ilu_diagnostics()

