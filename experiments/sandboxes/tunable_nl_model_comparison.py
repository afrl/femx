"""Run JAX CSR and JetSci SNES on the same seeded nonlinear diffusion problem."""
import argparse
import math
from time import perf_counter


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
        solve_bvp, SolverOptions, LinearSolverType, PreconditionerType,
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
    def prepare_petsc():
        ebc, maps, constraints, nnz, residual_func, force = preprocess_bvp(
            vertices_vd=vertices, element_batches=batches(),
            element_residual_func=element_residual, boundary_conditions=bcs,
        )
        functions = build_nonlinear_objects(
            element_residual_func=residual_func, ebc=ebc, assembly_map_b=maps,
            jacobian_nnz=nnz, u_0_g=x0, constraints=constraints, f_ext=force,
        )
        return ebc, functions

    # Reference functions are used only for validation after both timed solves.
    jax.block_until_ready((material, x0))
    print(f"Mesh: {n} x {n}; DOFs: {len(vertices)}; "
          f"PARAM_RANGE_FACTOR: {args.param_range_factor}; seed: {args.seed}", flush=True)
    options = jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_solver_type=jetsci.LinearSolverType.PETSC_LGMRES,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_AMG, **common,
    )
    start = perf_counter()
    ebc, (residual_fn, jacobian_fn, residual_isv_fn) = prepare_petsc()
    phi = ebc.material_params
    try:
        u_petsc, options = jetsci.differentiable_solve(
            options, residual_fn, jacobian_fn, x0, phi,
        )
        petsc_residual, petsc_states = residual_isv_fn(phi, u_petsc)
        jax.block_until_ready((u_petsc, petsc_residual, petsc_states))
        petsc_seconds = perf_counter() - start
        solver, _ = jetsci.petsc_snes.solver_lifecycle.get_petsc_solver_objects_from_key(options.solver_key)
        print(f"SNES reason: {int(solver.snes.getConvergedReason())}; "
              f"Newton iterations: {solver.snes.getIterationNumber()}; "
              f"KSP reason: {int(solver.snes.getKSP().getConvergedReason())}", flush=True)
    finally:
        if options.solver_key is not None:
            jetsci.petsc_snes.solver_lifecycle.destroy_petsc_solver(options.solver_key)

    start = perf_counter()
    u_jax, jax_residual, jax_batches = solve_bvp(
        vertices_vd=vertices, element_batches=batches(),
        element_residual_func=element_residual, boundary_conditions=bcs, u_0_g=x0,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.GMRES_JAX_SCIPY_CSR,
            linear_precond_type=PreconditionerType.JACOBI,
            gmres_solve_method="incremental", **common,
        ),
    )
    jax.block_until_ready((u_jax, jax_residual, [b.internal_state for b in jax_batches]))
    jax_seconds = perf_counter() - start
    print("First-call times include preprocessing, assembly, solve and residual/state "
          "evaluation, including compilation; validation, output and PETSc destruction excluded.")
    print(f"JetSci: {petsc_seconds:.6f} s; JAX CSR: {jax_seconds:.6f} s; "
          f"JAX/JetSci: {jax_seconds / petsc_seconds:.3f}x")
    initial_norm = float(jnp.linalg.norm(residual_fn(phi, x0)))
    bound = max(common["nonlinear_absolute_tol"],
                common["nonlinear_relative_tol"] * initial_norm)
    print(f"Initial residual: {initial_norm}; acceptance bound: {bound}")

    def report(name, u):
        norm = float(jnp.linalg.norm(residual_fn(phi, u)))
        accepted = bool(jnp.isfinite(u).all()) and math.isfinite(norm) and norm <= bound
        print(f"{name}: residual={norm}; accepted={accepted}; "
              f"solution range=[{float(u.min())}, {float(u.max())}]", flush=True)

    report("JetSci LGMRES/Jacobi", u_petsc)
    report("JAX CSR GMRES/Jacobi", u_jax)
    difference = float(jnp.linalg.norm(u_petsc - u_jax))
    scale = max(float(jnp.linalg.norm(u_petsc)), float(jnp.linalg.norm(u_jax)), 1e-30)
    print(f"Solution difference L2: {difference}; relative difference: {difference / scale}")


if __name__ == "__main__":
    main(parse_args())
