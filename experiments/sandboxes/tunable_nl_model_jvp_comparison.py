"""Compare JAX CSR and JetSci solution JVPs on identical seeded inputs."""
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", type=int, default=200,
                        help="Elements per side of the square mesh (default: 200).")
    parser.add_argument("--param-range-factor", type=float, default=1,
                        help="Conductivity: 10**(11 - factor * U[0,1)) (default: 1).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mesh_size < 1:
        parser.error("--mesh-size must be positive")
    if not math.isfinite(args.param_range_factor) or args.param_range_factor < 0:
        parser.error("--param-range-factor must be finite and nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def main(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jetsci
    from fe_jax import (
        FiniteElementType, CellType, ElementFamily, LagrangeVariant, QuadratureType,
        get_quadrature, eval_basis_and_derivatives, ElementBatch, DirichletBC,
        heat_conduction_residual, preprocess_bvp, build_nonlinear_objects,
        solve_nonlinear_step, SolverOptions, LinearSolverType, PreconditionerType,
    )
    from tunable_nl_model import custom_nl_scalar_diffusion

    n = args.mesh_size
    rng = np.random.default_rng(args.seed)
    vertices = np.stack(np.meshgrid(np.linspace(0, 1, n + 1),
                                   np.linspace(0, 1, n + 1)), axis=-1).reshape(-1, 2)
    origins = np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]
    cells = (origins.reshape(-1, 1) + np.array([0, 1, n + 1, n + 2])).astype(int)
    boundary = np.concatenate((
        np.arange(n + 1), len(vertices) - 1 - np.arange(n + 1),
        np.array([[(n + 1) * i, (n + 1) * (i + 1) - 1]
                  for i in range(1, n)], dtype=int).reshape(-1),
    ))
    fe_type = FiniteElementType(
        cell_type=CellType.quadrilateral, family=ElementFamily.P, basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default, quadrature_degree=2,
    )
    quadrature_points, _ = get_quadrature(fe_type)
    phi_qn, _ = eval_basis_and_derivatives(fe_type, quadrature_points)
    model = custom_nl_scalar_diffusion(lambda x: 1 + 60*x**2 + 5*x**4 + 1.4*x**6)
    material = 10**(11 - args.param_range_factor * rng.random(
        (len(cells), len(quadrature_points), 1)))
    bcs = [DirichletBC(index=i, component=0, value=rng.random()) for i in boundary]
    x0 = jnp.full((len(vertices),), np.mean([bc.value for bc in bcs]))

    def batches():
        return [ElementBatch(fe_type=fe_type, n_dofs_per_basis=1,
                             connectivity_en=cells, constitutive_model=model,
                             material_params=material.copy())]

    @jax.tree_util.Partial
    def element_residual(u_nd, x_nd, dphi_dxi_qnp, W_q,
                         material_params, internal_state_qi, constitutive_model):
        return heat_conduction_residual(
            u_nd, x_nd, phi_qn, dphi_dxi_qnp, W_q,
            material_params, internal_state_qi, constitutive_model,
        )

    common = dict(nonlinear_max_iter=200, nonlinear_relative_tol=1e-12,
                  nonlinear_absolute_tol=1e-12, linear_max_iter=5000,
                  linear_relative_tol=1e-12, linear_absolute_tol=1e-12)
    ebc, maps, constraints, nnz, residual_func, force = preprocess_bvp(
        vertices_vd=vertices, element_batches=batches(),
        element_residual_func=element_residual, boundary_conditions=bcs,
    )
    residual_fn, jacobian_fn, _ = build_nonlinear_objects(
        element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
        jacobian_nnz=nnz, u_0_g=x0, constraints=constraints, f_ext=force,
    )
    phi = ebc.material_params
    phi_dot = 0.01 * phi * jnp.sin(jnp.arange(phi.size, dtype=phi.dtype))
    initial_norm = float(jnp.linalg.norm(residual_fn(phi, x0)))
    bound = max(common["nonlinear_absolute_tol"],
                common["nonlinear_relative_tol"] * initial_norm)
    print(f"Mesh: {n} x {n}; PARAM_RANGE_FACTOR: {args.param_range_factor}; seed: {args.seed}", flush=True)
    print("Derivative: phi(epsilon) = phi + epsilon * (0.01 * phi * sin(arange(phi.size)))")
    petsc_options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_LGMRES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI, **common,
    )
    jax_options = SolverOptions(
        linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY_CSR,
        linear_precond_type=PreconditionerType.JACOBI,
        gmres_solve_method="incremental", **common,
    )
    try:
        # Initialize the solver key required by JetSci before JAX tracing.
        initial_u, petsc_options = jetsci.differentiable_solve(
            petsc_options, residual_fn, jacobian_fn, x0, phi,
        )
        initial_u.block_until_ready()
        solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(petsc_options.solver_key)
        if solver.snes.getConvergedReason() <= 0:
            raise RuntimeError(f"Cannot differentiate failed equilibrium: SNES reason {int(solver.snes.getConvergedReason())}")

        def petsc_solve(parameters):
            return jetsci.differentiable_solve(
                petsc_options, residual_fn, jacobian_fn, x0, parameters,
            )[0]

        def jax_solve(parameters):
            return solve_nonlinear_step(
                element_residual_func=residual_func,
                ebc=ebc.replace(material_params=parameters), assembly_map_b=maps,
                jacobian_nnz=nnz, u_0_g=x0, constraints=constraints,
                solver_options=jax_options, f_ext=force,
            )[0]

        petsc_result = jax.jvp(petsc_solve, (phi,), (phi_dot,))
        jax.block_until_ready(petsc_result)
        snes_reason = int(solver.snes.getConvergedReason())
        print(f"JetSci SNES reason: {snes_reason}", flush=True)
        if snes_reason <= 0:
            raise RuntimeError("JetSci JVP primal solve failed; reject derivative comparison")
        jax_result = jax.jvp(jax_solve, (phi,), (phi_dot,))
        jax.block_until_ready(jax_result)

        for name, (u, u_dot) in (("JetSci LGMRES/AMG", petsc_result),
                                  ("JAX CSR GMRES/Jacobi", jax_result)):
            residual_norm = float(jnp.linalg.norm(residual_fn(phi, u)))
            tangent = jax.jvp(residual_fn, (phi, u), (phi_dot, u_dot))[1]
            finite = bool(jnp.isfinite(u).all() & jnp.isfinite(u_dot).all())
            print(f"{name}: residual={residual_norm}; bound={bound}; "
                  f"JVP norm={float(jnp.linalg.norm(u_dot))}; "
                  f"tangent residual={float(jnp.linalg.norm(tangent))}; finite={finite}", flush=True)
            if not finite or not math.isfinite(residual_norm) or residual_norm > bound:
                raise RuntimeError(f"{name}: invalid equilibrium/JVP; reject comparison")

        for index, label in ((0, "Solution"), (1, "JVP")):
            a, b = petsc_result[index], jax_result[index]
            error = float(jnp.linalg.norm(a - b))
            scale = max(float(jnp.linalg.norm(a)), float(jnp.linalg.norm(b)), 1e-30)
            print(f"{label} difference L2: {error}; relative difference: {error / scale}")
    finally:
        if petsc_options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(petsc_options.solver_key)


if __name__ == "__main__":
    main(parse_args())
