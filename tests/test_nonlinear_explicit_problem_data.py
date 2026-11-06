"""Regression for one FEA callback set with explicit per-step boundary data."""

from helper import *
import pytest
from fe_jax.fea import preprocess_bvp, build_nonlinear_objects
from fe_jax.linear_elasticity_dmg import damage_elastic_isotropic_vmap, linear_elasticity_residual


def test_explicit_boundary_data_matches_step_closures_and_derivatives():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    fe = FiniteElementType(
        cell_type=CellType.triangle, family=ElementFamily.P, basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.default, quadrature_degree=1,
    )
    material = jnp.array([3000., .39, .95, 2., 10., .35, .04, 79., 62., 2e4, 12.])
    state = jnp.zeros((1, 1, 11), dtype=jnp.float64).at[..., 10].set(.5)
    batches = [ElementBatch(
        fe_type=fe, connectivity_en=np.array([[0, 1, 2]]), n_dofs_per_basis=2,
        constitutive_model=damage_elastic_isotropic_vmap,
        material_params=material[None, None, :], internal_state=state,
    )]
    ebc, maps, constraints, nnz, kernel, force = preprocess_bvp(
        vertices_vd=points, element_batches=batches, element_residual_func=linear_elasticity_residual,
        boundary_conditions=[
            DirichletBC(index=0, component=0, value=.001),
            DirichletBC(index=0, component=1, value=0.),
            DirichletBC(index=1, component=1, value=0.),
            NeumannBC(index=1, component=0, value=.1),
            NeumannBC(index=2, component=1, value=.05),
        ], multipoint_constraints=None,
    )
    x = jnp.array([.001, 0., .002, 0., -.0003, .001])

    def build(g, loads, **options):
        return build_nonlinear_objects(
            element_residual_func=kernel, ebc=ebc, assembly_map_b=maps,
            jacobian_nnz=nnz, u_0_g=x, constraints=constraints.replace(g=g),
            f_ext=force.replace(loads=loads), differentiate_internal_state=True, **options,
        )

    residual, jacobian, residual_isv = build(constraints.g, force.loads, explicit_boundary_data=True)
    base = jnp.concatenate((ebc.material_params, ebc.internal_state))
    all_parameters = jnp.concatenate((base, constraints.g, force.loads))
    boundaries = (base.size, base.size + constraints.g.size)

    # Alternate loads/g with the SAME function objects. The reference keeps
    # the old per-step closure contract. Incoming state/material are explicit
    # in both, so state sensitivities must also survive the extension.
    for factor in (1., 1.4, .8):
        active = all_parameters.at[0].multiply(factor)
        active = active.at[ebc.material_params.size + 6].set(.05 * factor)
        active = active.at[boundaries[0]:].multiply(factor)
        old_r, old_j, old_isv = build(
            active[boundaries[0]:boundaries[1]], active[boundaries[1]:],
        )
        np.testing.assert_allclose(residual(active, x), old_r(active[:base.size], x), atol=1e-11)
        new_matrix, old_matrix = jacobian(active, x), old_j(active[:base.size], x)
        np.testing.assert_allclose(new_matrix.data, old_matrix.data, atol=1e-11)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(residual_isv(active, x)),
            jax.tree_util.tree_leaves(old_isv(active[:base.size], x)), strict=True,
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-11)

    # Check actual force/boundary sensitivities, not just equal primal values.
    direction = jnp.zeros_like(all_parameters)
    direction = direction.at[boundaries[0]:boundaries[1]].set(1e-3)
    direction = direction.at[boundaries[1]:].set(.1)
    scalar = jax.jit(lambda p: jnp.vdot(jnp.arange(1., 7.), residual(p, x)))
    _, tangent = jax.jvp(scalar, (all_parameters,), (direction,))
    gradient = jax.grad(scalar)(all_parameters)
    eps = 1e-4
    fd = (scalar(all_parameters + eps * direction) - scalar(all_parameters - eps * direction)) / (2 * eps)
    assert float(jnp.linalg.norm(gradient[boundaries[0]:boundaries[1]])) > 0
    assert float(jnp.linalg.norm(gradient[boundaries[1]:])) > 0
    np.testing.assert_allclose(tangent, fd, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), tangent, rtol=1e-8, atol=1e-10)

    # A saved earlier pullback must keep its own numerical boundary/load data
    # after the same residual functions have evaluated a later step.
    _, earlier_pullback = jax.vjp(scalar, all_parameters)
    later = all_parameters.at[boundaries[0]:].multiply(1.7)
    scalar(later).block_until_ready()
    np.testing.assert_allclose(earlier_pullback(jnp.array(1.))[0], gradient, atol=1e-10)

    # The optional boundary packing also works without packing internal state.
    material_residual, _, _ = build_nonlinear_objects(
        element_residual_func=kernel, ebc=ebc, assembly_map_b=maps,
        jacobian_nnz=nnz, u_0_g=x, constraints=constraints, f_ext=force,
        explicit_boundary_data=True,
    )
    material_parameters = jnp.concatenate((ebc.material_params, constraints.g, force.loads))
    np.testing.assert_allclose(material_residual(material_parameters, x), residual(all_parameters, x))
    with pytest.raises(ValueError, match="flat nonlinear parameter"):
        residual(base, x)
