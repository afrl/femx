"""Fit the conductivity field to a target equilibrium using JetSci value_and_grad."""
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

    ebc, maps, constraints, nnz, residual_func, force = preprocess_bvp(
        vertices_vd=vertices, element_batches=batches(),
        element_residual_func=element_residual, boundary_conditions=bcs,
    )
    residual_fn, jacobian_fn, _ = build_nonlinear_objects(
        element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
        jacobian_nnz=nnz, u_0_g=x0, constraints=constraints, f_ext=force,
    )
    base_material = ebc.material_params
    theta = jnp.zeros_like(base_material)

    def material_params(parameters):
        # Fix the unidentifiable global log-conductivity offset.
        return base_material * jnp.exp(parameters - jnp.mean(parameters))

    centers = vertices[cells].mean(axis=1)
    target_pattern = 0.25 * np.sin(2*np.pi*centers[:, 0]) - 0.20 * np.sin(2*np.pi*centers[:, 1])
    target_theta = jnp.asarray(np.repeat(target_pattern[:, None], len(quadrature_points), axis=1).ravel())
    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_LGMRES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        nonlinear_relative_tol=1e-12, nonlinear_absolute_tol=1e-12,
        nonlinear_max_iter=200, linear_relative_tol=1e-12,
        linear_absolute_tol=1e-10, linear_max_iter=5000,
    )
    print(f"Mesh: {n} x {n}; material unknowns: {theta.size}; "
          f"PARAM_RANGE_FACTOR: {args.param_range_factor}; seed: {args.seed}", flush=True)

    def solve(parameters):
        return jetsci.differentiable_solve(
            options, residual_fn, jacobian_fn, x0, material_params(parameters),
        )[0]

    def check_equilibrium(parameters, u):
        solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(options.solver_key)
        actual = material_params(parameters)
        residual = float(jnp.linalg.norm(residual_fn(actual, u)))
        initial = float(jnp.linalg.norm(residual_fn(actual, x0)))
        bound = max(options.nonlinear_absolute_tol, options.nonlinear_relative_tol * initial)
        reason = int(solver.snes.getConvergedReason())
        # Allow 5% for independent residual reevaluation at a threshold crossing.
        if reason <= 0 or not bool(jnp.isfinite(u).all()) or not math.isfinite(residual) or residual > 1.05 * bound:
            raise RuntimeError(f"Invalid equilibrium: SNES={reason}, residual={residual}, bound={bound}. Inspect solver convergence before optimizing.")

    try:
        initial_u, options = jetsci.differentiable_solve(
            options, residual_fn, jacobian_fn, x0, material_params(theta),
        )
        initial_u.block_until_ready()
        check_equilibrium(theta, initial_u)
        target_u = jax.lax.stop_gradient(solve(target_theta))
        target_u.block_until_ready()
        check_equilibrium(target_theta, target_u)
        response_scale = float(jnp.linalg.norm(target_u - initial_u))
        if not math.isfinite(response_scale) or response_scale < 1e-10:
            raise RuntimeError("Target change is too small to resolve; increase the target perturbation.")

        def objective(parameters):
            u = solve(parameters)
            mismatch = (u - target_u) / response_scale
            centered = parameters - jnp.mean(parameters)
            # Weak regularization toward the initial material field.
            value = 0.5 * jnp.vdot(mismatch, mismatch) + 0.5e-6 * jnp.mean(centered**2)
            return value, u

        value_and_grad = jax.value_and_grad(objective, has_aux=True)
        initial_loss = None
        for step in range(200):
            (value, u), gradient = value_and_grad(theta)
            jax.block_until_ready((value, u, gradient))
            check_equilibrium(theta, u)
            if not bool(jnp.isfinite(gradient).all()) or not math.isfinite(float(value)):
                raise RuntimeError("Nonfinite reverse-mode gradient/loss; inspect the adjoint solve.")
            current_loss = float(value)
            if step == 0:
                initial_loss = current_loss
            print(f"Step {step}: loss={current_loss:.9g}; gradient norm={float(jnp.linalg.norm(gradient)):.9g}", flush=True)
            if step == 0:
                # Check reverse mode against one independent forward-mode direction.
                direction = jnp.sin(jnp.arange(theta.size, dtype=theta.dtype))
                direction /= jnp.linalg.norm(direction)
                _, loss_dot = jax.jvp(lambda p: objective(p)[0], (theta,), (direction,))
                reverse_dot = jnp.vdot(gradient, direction)
                print(f"Derivative check: grad dot v={float(reverse_dot)}; JVP={float(loss_dot)}", flush=True)
                np.testing.assert_allclose(np.asarray(reverse_dot), np.asarray(loss_dot), rtol=1e-3, atol=1e-7)
            if current_loss < initial_loss * 1e-4:
                break
            largest = float(jnp.max(jnp.abs(gradient)))
            if largest == 0:
                raise RuntimeError("Zero gradient before target fit; inspect material identifiability.")
            # Basic gradient descent. Bound the largest initial log-k change at 0.1.
            for trial in range(10):
                candidate = theta - (0.1 * 0.5**trial / largest) * gradient
                candidate -= jnp.mean(candidate)
                candidate_value, candidate_u = objective(candidate)
                jax.block_until_ready((candidate_value, candidate_u))
                check_equilibrium(candidate, candidate_u)
                if math.isfinite(float(candidate_value)) and float(candidate_value) < current_loss:
                    theta = candidate
                    break
            else:
                raise RuntimeError("No decreasing gradient step found; inspect derivative accuracy and equilibrium noise.")

        final_value, final_u = objective(theta)
        jax.block_until_ready((final_value, final_u))
        check_equilibrium(theta, final_u)
        print(f"Initial loss: {initial_loss}; final loss: {float(final_value)}")
        print("Relative target-change error:", float(jnp.linalg.norm(final_u - target_u)) / response_scale)
        optimized_material = material_params(theta)
        print("Optimized conductivity range:", float(optimized_material.min()), float(optimized_material.max()))
        assert float(final_value) < initial_loss, "Optimization did not improve the target equilibrium fit"
    finally:
        if options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(options.solver_key)


if __name__ == "__main__":
    main(parse_args())
