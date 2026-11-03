import jax
import jax.numpy as jnp
from fe_jax import *
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
    U = 1
    n=200
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
    constitutive_model=custom_nl_scalar_diffusion(
        lambda x: x+20*x**3+1*x**5+0.2*x**7
    )
    rng = np.random.default_rng(1423232)
    E = cells.shape[0]  # number of elements
    Q = get_quadrature(fe_type=fe_type)[0].shape[0]  # number of quadrature points
    element_batches = [
        ElementBatch(
            fe_type=fe_type,
            n_dofs_per_basis=U,
            connectivity_en=cells,
            constitutive_model=constitutive_model,
            material_params=10**(11-0.1*rng.random((E,Q,1))),
        ),
    ]
    dirichlet_bcs=[]
    for j in range(U):
        dirichlet_bcs = dirichlet_bcs + [
            DirichletBC(index=i, component=j, value=rng.random())
            for i in boundary_points
        ]

    u, residual, element_batch = solve_bvp(
        element_residual_func=heat_conduction_residual,
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        u_0_g=jnp.full(shape=(vertices_vd.shape[0]*element_batches[0].n_dofs_per_basis),fill_value=jnp.mean(jnp.array([bc.value for bc in dirichlet_bcs]))),
        boundary_conditions=dirichlet_bcs,
        plot_convergence=False,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            nonlinear_max_iter=200,
            linear_relative_tol=1e-1
            # linear_precond_type=fea.PreconditionerType.JACOBI,
        ),
    )