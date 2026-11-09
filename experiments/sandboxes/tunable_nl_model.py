import jax
import jax.numpy as jnp
import meshio


import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")

from fe_jax import *


outputVTU = False

def custom_nl_scalar_diffusion(nl_kernel):
    """
    Factory for defining arbitrary nonlinear diffusion consitutive relations.

    returns a jax.tree_util.Partial object, ready to be used in the ElementBatch based solver flow
    """
    @jax.jit
    def custom_constitutive_law(dT_dx_ud, T_u, material_params_m):
        k = material_params_m[0]
        df_dx_ud = jnp.einsum("u,ud->ud", nl_kernel(T_u), dT_dx_ud)

        return k * df_dx_ud

    return jax.tree_util.Partial(custom_constitutive_law)    



if __name__=="__main__":
    U = 1 # Solution dimensionality (always 1)
    n=400 # Mesh size (nxn square grid)
    constitutive_model=custom_nl_scalar_diffusion(
        lambda x: 1 #1+60*x**2+5*x**4+1.4*x**6
    ) # Should be even, positive
    PARAM_RANGE_FACTOR = 3 # number of orders of magnitude parameters span (upper bound 1e11)
    BC_SCALE = 1 # upper bound for dirichlet BC values

    rng = np.random.default_rng() # for random BC and parameter vals

    mesh = meshio.read("microscale_2D_r0.vtk")
    points = np.array(mesh.points, dtype=np.float32)[:, 0:2]
    cells = np.array(mesh.cells[0].data, dtype=np.uint64)
    mesh.cell_data["DomainIDs"][0] = np.array(
        mesh.cell_data["DomainIDs"][0], dtype=np.int64
    )
    cell_domain_ids = mesh.cell_data["DomainIDs"][0].flatten()

    # Extract cells for each subdomain
    matrix_cells = cells[cell_domain_ids == 0]
    fiber_cells = cells[cell_domain_ids == 1]

    # Define node sets
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    left_points = np.isclose(points[:, 0], min_xy[0], atol=1e-16).nonzero()[0]
    right_points = np.isclose(points[:, 0], max_xy[0], atol=1e-16).nonzero()[0]
    bottom_points = np.isclose(points[:, 1], min_xy[1], atol=1e-16).nonzero()[0]
    top_points = np.isclose(points[:, 1], max_xy[1], atol=1e-16).nonzero()[0]

    # Boundary conditions:
    # - Fix left nodes to 0
    # - Fix right nodes 100
    # - Leave bottom and top flux free
    bcs = (
        [
            DirichletBC(bc_type=BCType.NODE, index=i, component=0, value=0.0)
            for i in left_points
        ]
        + [
            DirichletBC(
                bc_type=BCType.NODE, index=i, component=0, value=100.0
            )
            for i in right_points
        ]
    )

    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=2,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=3,
    )

    E = cells.shape[0]  # number of elements
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=matrix_cells,
            constitutive_model=constitutive_model,
            material_params=0.148 * jnp.ones((matrix_cells.shape[0],Q,1)),
        ),
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=fiber_cells,
            constitutive_model=constitutive_model,
            material_params=100. * jnp.ones((fiber_cells.shape[0],Q,1)),
        ),
    ]
    
    u_0_g = jnp.full(shape=(points.shape[0]*element_batches[0].n_dofs_per_basis),fill_value=0.0) 
    u, residual, element_batch = solve_bvp(
        element_residual_func=heat_conduction_residual,
        vertices_vd=points,
        element_batches=element_batches,
        u_0_g=u_0_g,
        boundary_conditions=bcs,
        plot_convergence=False,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_CSR,
            nonlinear_max_iter=200,
            linear_relative_tol=1e-12,
            nonlinear_absolute_tol=1e-12,
            linear_max_iter=5000,
            linear_precond_type=PreconditionerType.JACOBI,
            gmres_solve_method="incremental",
            linear_diagnostics=True,
        ),
    )
    print("Final residual norm:", float(jnp.linalg.norm(residual)))
    print("Solution finite:", bool(jnp.isfinite(u).all()))

    if outputVTU == True:
        meshio.write_points_cells(
            "tunable_nl_solution_JetSCI.vtu",
            points=np.column_stack((vertices_vd, np.zeros(len(vertices_vd)))),
            cells=[("quad", cells[:, [0, 1, 3, 2]])],
            point_data={"T": np.asarray(u).reshape(-1)},
        )
    
    print("Solution range:", float(jnp.min(u)), float(jnp.max(u)))
