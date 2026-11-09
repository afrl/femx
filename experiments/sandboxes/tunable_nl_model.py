import jax
import jax.numpy as jnp
import meshio
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
        lambda x: 1+60*x**2+5*x**4+1.4*x**6
    ) # Should be even, positive
    PARAM_RANGE_FACTOR = 3 # number of orders of magnitude parameters span (upper bound 1e11)
    BC_SCALE = 1 # upper bound for dirichlet BC values

    rng = np.random.default_rng() # for random BC and parameter vals


    vertices_vd = np.vstack(
        [
            np.ravel(vec)
            for vec in np.meshgrid(
                np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
            )
        ],
        dtype=float,
    ).T
    cells = np.vstack(
        [
            (
                np.array([i, i + 1, i + n + 1, i + n + 2])
                if (i + 1) % (n + 1) != 0
                else np.zeros((0, 4), dtype=int)
            )
            for i in np.arange(vertices_vd.shape[0] - n - 1, dtype=int)
        ],
        dtype=int,
    )
    boundary_points = np.hstack(
        [
            np.arange(n + 1),
            vertices_vd.shape[0] - 1 - np.arange(n + 1),
            np.array(
                [[(n + 1) * i, (n + 1) * (i + 1) - 1] for i in range(1, n)]
            ).reshape(-1),
        ]
    )
    fe_type = FiniteElementType(
        cell_type=CellType.quadrilateral,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default,
        quadrature_degree=2,
    )    

    E = cells.shape[0]  # number of elements
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points
    phi_qn, _ = eval_basis_and_derivatives(fe_type, get_quadrature(fe_type)[0])

    def element_heat_residual(u_nd, x_nd, dphi_dxi_qnp, W_q,
                              material_params, internal_state_qi, constitutive_model):
        return heat_conduction_residual(
            u_nd, x_nd, phi_qn, dphi_dxi_qnp, W_q,
            material_params, internal_state_qi, constitutive_model,
        )

    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=constitutive_model,
            material_params=10**(11-PARAM_RANGE_FACTOR * rng.random((E,Q,1))),
        ),
    ]

    dirichlet_bcs=[]
    for j in range(U):
        dirichlet_bcs = dirichlet_bcs + [
            DirichletBC(index=i, component=j, value=BC_SCALE * rng.random())
            for i in boundary_points
        ]
    u_0_g = jnp.full(shape=(vertices_vd.shape[0]*element_batches[0].n_dofs_per_basis),fill_value=jnp.mean(jnp.array([bc.value for bc in dirichlet_bcs]))) 
    u, residual, element_batch = solve_bvp(
        element_residual_func=jax.tree_util.Partial(element_heat_residual),
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        u_0_g=u_0_g,
        boundary_conditions=dirichlet_bcs,
        plot_convergence=False,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY_CSR,
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