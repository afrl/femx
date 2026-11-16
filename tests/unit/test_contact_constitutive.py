import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fe_jax.contact.constitutive import (
    contact_stiffness_constant,
    contact_stiffness_exponential,
    contact_stiffness_linear,
    contact_stiffness_piecewise_linear,
    contact_stiffness_piecewise_quadratic,
    contact_stiffness_tanh,
    elastic_contact_truss_constant,
    elastic_contact_truss_exponential,
    elastic_contact_truss_linear,
    elastic_contact_truss_piecewise_linear,
    elastic_contact_truss_piecewise_quadratic,
    elastic_contact_truss_tanh,
)
from fe_jax.contact.contact_types import ContactMaterialSpec
from fe_jax.contact.params import (
    CONTACT_ACTIVE_PARAM,
    CONTACT_AREA_PARAM,
    CONTACT_C_TO_D_PARAM,
    CONTACT_E_C_PARAM,
    CONTACT_E_MIN_PARAM,
    CONTACT_M_TO_D_PARAM,
    CONTACT_RADIUS_0_PARAM,
    CONTACT_RADIUS_1_PARAM,
    CONTACT_SEARCH_ALPHA_PARAM,
    build_contact_material_params,
)


@pytest.fixture
def standard_contact_spec():
    return ContactMaterialSpec(
        E_c=1000.0,
        area=0.01,
        M_to_D_ratio=1.1,
        C_to_D_ratio=1.0,
        search_alpha=1.5,
        E_min=1.0,
    )


@pytest.fixture
def standard_params_1d(standard_contact_spec):
    contact_cells = jnp.array([[0, 1]], dtype=jnp.int32)
    point_radii = jnp.array([0.5, 0.5], dtype=jnp.float64)
    params = build_contact_material_params(
        contact_cells=contact_cells,
        point_radii=point_radii,
        spec=standard_contact_spec,
    )
    return params[0]


class TestContactStiffnessModels:
    """Direct tests for individual contact stiffness functions."""

    def test_constant_stiffness(self, standard_params_1d, standard_contact_spec):
        for d in [0.5, 1.0, 1.2, 2.0]:
            E = contact_stiffness_constant(jnp.array(d), standard_params_1d)
            assert np.isclose(E, standard_contact_spec.E_c)

    def test_linear_stiffness(self, standard_params_1d, standard_contact_spec):
        # Total radius R = 1.0
        # For d < R: E = E_max * (1 - d/R)
        # For d >= R: E = 0
        E_in = contact_stiffness_linear(jnp.array(0.5), standard_params_1d)
        assert np.isclose(E_in, standard_contact_spec.E_c * 0.5)

        E_contact = contact_stiffness_linear(jnp.array(1.0), standard_params_1d)
        assert np.isclose(E_contact, 0.0)

        E_out = contact_stiffness_linear(jnp.array(1.2), standard_params_1d)
        assert np.isclose(E_out, 0.0)

    def test_piecewise_linear_continuity(self, standard_params_1d, standard_contact_spec):
        total_radius = 1.0
        d_hard = total_radius * standard_contact_spec.C_to_D_ratio  # 1.0
        d_ramp = total_radius * standard_contact_spec.M_to_D_ratio  # 1.1
        d_search = total_radius * standard_contact_spec.search_alpha  # 1.5

        # At d = d_hard, E = E_c
        E_hard = contact_stiffness_piecewise_linear(jnp.array(d_hard), standard_params_1d)
        assert np.isclose(E_hard, standard_contact_spec.E_c)

        # At d = d_ramp, both segments equal E_min
        eps = 1e-7
        E_ramp_left = contact_stiffness_piecewise_linear(jnp.array(d_ramp - eps), standard_params_1d)
        E_ramp_right = contact_stiffness_piecewise_linear(jnp.array(d_ramp + eps), standard_params_1d)
        assert np.isclose(E_ramp_left, standard_contact_spec.E_min, atol=1e-2)
        assert np.isclose(E_ramp_right, standard_contact_spec.E_min, atol=1e-2)
        assert np.isclose(E_ramp_left, E_ramp_right, atol=1e-2)

        # Beyond search radius, E = 0
        E_beyond = contact_stiffness_piecewise_linear(jnp.array(d_search + 0.1), standard_params_1d)
        assert np.isclose(E_beyond, 0.0)

    def test_piecewise_quadratic_c1_smoothness(self, standard_params_1d, standard_contact_spec):
        total_radius = 1.0
        d_hard = total_radius * standard_contact_spec.C_to_D_ratio  # 1.0
        d_ramp = total_radius * standard_contact_spec.M_to_D_ratio  # 1.1
        d_search = total_radius * standard_contact_spec.search_alpha  # 1.5

        # At hard contact distance
        E_hard = contact_stiffness_piecewise_quadratic(jnp.array(d_hard), standard_params_1d)
        assert np.isclose(E_hard, standard_contact_spec.E_c)

        # C^0 continuity at d_ramp
        eps = 1e-7
        E_ramp_left = contact_stiffness_piecewise_quadratic(jnp.array(d_ramp - eps), standard_params_1d)
        E_ramp_right = contact_stiffness_piecewise_quadratic(jnp.array(d_ramp + eps), standard_params_1d)
        assert np.isclose(E_ramp_left, standard_contact_spec.E_min, atol=1e-2)
        assert np.isclose(E_ramp_right, standard_contact_spec.E_min, atol=1e-2)
        assert np.isclose(E_ramp_left, E_ramp_right, atol=1e-2)

        # C^1 derivative continuity at d_ramp (analytical slope is mu = E_min / (d_ramp - d_search))
        mu_expected = standard_contact_spec.E_min / (d_ramp - d_search)
        grad_fn = jax.grad(contact_stiffness_piecewise_quadratic, argnums=0)
        dE_left = grad_fn(jnp.array(d_ramp - 1e-8), standard_params_1d)
        dE_right = grad_fn(jnp.array(d_ramp + 1e-8), standard_params_1d)
        assert np.isclose(dE_left, mu_expected, rtol=1e-2)
        assert np.isclose(dE_right, mu_expected, rtol=1e-2)
        assert np.isclose(dE_left, dE_right, rtol=1e-2)

        # Beyond search radius
        E_beyond = contact_stiffness_piecewise_quadratic(jnp.array(d_search + 0.1), standard_params_1d)
        assert np.isclose(E_beyond, 0.0)

    def test_exponential_stiffness(self, standard_params_1d, standard_contact_spec):
        total_radius = 1.0
        d_hard = total_radius * standard_contact_spec.C_to_D_ratio  # 1.0
        d_ramp = total_radius * standard_contact_spec.M_to_D_ratio  # 1.1

        E_hard = contact_stiffness_exponential(jnp.array(d_hard), standard_params_1d)
        assert np.isclose(E_hard, standard_contact_spec.E_c, rtol=1e-5)

        E_ramp = contact_stiffness_exponential(jnp.array(d_ramp), standard_params_1d)
        assert np.isclose(E_ramp, standard_contact_spec.E_min, rtol=1e-5)

        # Strictly monotonic decreasing with separation distance
        d_vals = np.linspace(0.8, 1.5, 10)
        E_vals = [float(contact_stiffness_exponential(jnp.array(d), standard_params_1d)) for d in d_vals]
        assert all(E_vals[i] > E_vals[i + 1] for i in range(len(E_vals) - 1))

    def test_tanh_stiffness(self, standard_params_1d, standard_contact_spec):
        total_radius = 1.0
        d_hard = total_radius * standard_contact_spec.C_to_D_ratio  # 1.0
        d_ramp = total_radius * standard_contact_spec.M_to_D_ratio  # 1.1

        E_hard = contact_stiffness_tanh(jnp.array(d_hard), standard_params_1d)
        assert np.isclose(E_hard, standard_contact_spec.E_c, rtol=1e-2)

        E_ramp = contact_stiffness_tanh(jnp.array(d_ramp), standard_params_1d)
        assert np.isclose(E_ramp, standard_contact_spec.E_min, rtol=1e-2)

        # Monotonic decreasing with separation distance
        d_vals = np.linspace(0.8, 1.4, 10)
        E_vals = [float(contact_stiffness_tanh(jnp.array(d), standard_params_1d)) for d in d_vals]
        assert all(E_vals[i] >= E_vals[i + 1] for i in range(len(E_vals) - 1))


class TestContactKernels:
    """Tests for the elastic contact truss constitutive kernels."""

    @pytest.mark.parametrize(
        "kernel",
        [
            elastic_contact_truss_constant,
            elastic_contact_truss_linear,
            elastic_contact_truss_piecewise_linear,
            elastic_contact_truss_piecewise_quadratic,
            elastic_contact_truss_exponential,
            elastic_contact_truss_tanh,
        ],
    )
    def test_inactive_element_yields_zero_stress(self, kernel, standard_contact_spec):
        contact_cells = jnp.array([[0, 1]], dtype=jnp.int32)
        point_radii = jnp.array([0.5, 0.5], dtype=jnp.float64)
        # Explicitly set active = 0
        params = build_contact_material_params(
            contact_cells=contact_cells,
            point_radii=point_radii,
            spec=standard_contact_spec,
            active=jnp.array([0.0]),
        )[0]

        # Heavy penetration
        x_nd = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
        u_nd = jnp.zeros_like(x_nd)
        eps_dd = jnp.zeros((3, 3))

        stress_dd, internal_state = kernel(eps_dd, params, x_nd, u_nd)
        assert np.allclose(stress_dd, 0.0)
        assert internal_state.size == 0

    @pytest.mark.parametrize(
        "kernel",
        [
            elastic_contact_truss_constant,
            elastic_contact_truss_linear,
            elastic_contact_truss_piecewise_linear,
            elastic_contact_truss_piecewise_quadratic,
            elastic_contact_truss_exponential,
            elastic_contact_truss_tanh,
        ],
    )
    def test_penetration_produces_compressive_stress(self, kernel, standard_params_1d):
        # Initial distance L_ref = 1.4 (search radius), deformed distance L_cur = 0.8
        x_nd = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=jnp.float64)
        u_nd = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.6]], dtype=jnp.float64)
        eps_dd = jnp.zeros((3, 3), dtype=jnp.float64)

        stress_dd, _ = kernel(eps_dd, standard_params_1d, x_nd, u_nd)
        # Compression along z-axis => negative normal stress in direction of element
        assert stress_dd[2, 2] < 0.0
        # Off-axis stress components should be zero for an axial truss along z
        assert np.allclose(stress_dd[0, :], 0.0)
        assert np.allclose(stress_dd[1, :], 0.0)

    @pytest.mark.parametrize(
        "kernel",
        [
            elastic_contact_truss_constant,
            elastic_contact_truss_linear,
            elastic_contact_truss_piecewise_linear,
            elastic_contact_truss_piecewise_quadratic,
            elastic_contact_truss_exponential,
            elastic_contact_truss_tanh,
        ],
    )
    def test_autodiff_and_jit_compatibility(self, kernel, standard_params_1d):
        x_nd = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], dtype=jnp.float64)
        u_nd = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.3]], dtype=jnp.float64)
        eps_dd = jnp.zeros((3, 3), dtype=jnp.float64)

        # Test JIT execution
        jit_kernel = jax.jit(kernel)
        stress_jit, _ = jit_kernel(eps_dd, standard_params_1d, x_nd, u_nd)
        stress_eager, _ = kernel(eps_dd, standard_params_1d, x_nd, u_nd)
        assert np.allclose(stress_jit, stress_eager)

        # Test gradient / Jacobian with respect to displacement
        def stress_norm(u):
            s, _ = kernel(eps_dd, standard_params_1d, x_nd, u)
            return jnp.sum(s**2)

        grad_u = jax.grad(stress_norm)(u_nd)
        assert not jnp.isnan(grad_u).any()
        assert not jnp.isinf(grad_u).any()
