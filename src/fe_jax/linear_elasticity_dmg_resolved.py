"""
Event-resolving variant of damage_elastic_isotropic_vmap (linear_elasticity_dmg.py).

Unlike the event-aware model (linear_elasticity_dmg_event.py), this changes the forward result, by O(dt),
so that the result is a continuous function of the material parameters. Plain AD then differentiates
it exactly, and AD agrees with small-h FD. Two things change, both on the step where the damage or yield
criterion changes sign:
  1. Partial onset step: the switched update (damage, or the plastic tangent increment) is applied only
     for the fraction of the crossing interval on the active side of the criterion, instead of all or
     nothing.
  2. Latched onset values: tau0 and vM0 are taken at the crossing point instead of at the overshot state
     where the lagged check first sees the crossing.

The crossing point is the exact zero of the criterion along the straight path between the last two
states. Both criteria are quadratic in the state, so three evaluations give the criterion exactly along
the path.

The criteria are lagged one step, as in the original, so the crossing lies between the state two steps
back and the previous state. Damage stays explicit, and the stress stays linear in the current strain.

ISV layout (17): [eps(3), stress(3), d, Y, tau0, vM0, dt, eps_prev(3), stress_prev(3)]
"""
import jax
import jax.numpy as jnp

from .utils import rank2_tensor_to_voigt, rank2_voigt_to_tensor
from .dmg_eqs_vmap import *
from .linear_elasticity_dmg_event import _safe_sqrt

N_ISV_RESOLVED = 17


def _crossing(criterion, x_prev, x_curr, material_params_m):
    """
    Locates a sign change of a criterion along the straight path from x_prev to x_curr.

    Returns
    -------
    crossed     : the criterion changed sign
    s           : position of the zero along the path, 0 at x_prev and 1 at x_curr; 0 if not crossed
    on_fraction : fraction of the path on the active side (1 - s on loading, s on unloading)
    """
    c0 = criterion(x_prev, material_params_m)
    c_mid = criterion(0.5 * (x_prev + x_curr), material_params_m)
    c1 = criterion(x_curr, material_params_m)
    crossed = (c0 > 0) != (c1 > 0)

    # c(s) = a s^2 + b s + c0 along the path
    a = 2 * (c0 + c1 - 2 * c_mid)
    b = c1 - c0 - a
    # A sign change gives two distinct real roots, one of them in [0, 1]. Without one, the guards keep the
    # unused values and their derivatives finite, since jnp.where backpropagates a zero cotangent into them.
    disc = jnp.where(crossed, b * b - 4 * a * c0, 1.0)
    q = -0.5 * (b + jnp.where(b >= 0, 1.0, -1.0) * jnp.sqrt(disc))   # |q| >= sqrt(disc) / 2 > 0
    root_1 = c0 / q
    a_nonzero = jnp.abs(a) > 1e-12 * jnp.abs(q)
    root_2 = jnp.where(a_nonzero, q / jnp.where(a_nonzero, a, 1.0), 2.0)
    s = jnp.where((root_1 >= 0) & (root_1 <= 1), root_1, root_2)
    s = jnp.where(crossed, jnp.clip(s, 0.0, 1.0), 0.0)
    return crossed, s, jnp.where(c1 > 0, 1 - s, s)


def _resolved_switch(on_flag, crossed, on_fraction, on, off):
    """lax.cond(on_flag, on, off), except on a crossing step, where the on branch gets weight on_fraction."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(crossed, b + on_fraction * (a - b), jnp.where(on_flag, a, b)), on, off
    )


def calc_d_resolved(Y, d, tau0, dt, material_params_m, eps_cd, crossed, tau_cross, C_ss):
    """calc_d with tau0 latched at the crossing point (0.999 * tau_cross) on a crossing step."""
    A    = material_params_m[2]
    B    = material_params_m[3]
    mu   = material_params_m[4]

    DE = jnp.dot(C_ss, eps_cd)
    strain_energy = 0.5 * jnp.dot(eps_cd, DE)
    # At zero strain (first step) this branch is evaluated but not used. The guards keep its
    # derivative finite, since jnp.where (unlike lax.cond) backpropagates a zero cotangent into it.
    tau = _safe_sqrt(2 * strain_energy)
    tau_den = jnp.where(tau > 0, tau, 1.0)

    update_tau0 = jnp.where(tau0 == 0, jnp.where(crossed, tau_cross, tau) * 0.999, tau0)

    G = 1 - update_tau0/tau_den*(1-A) - A*jnp.exp(B*(update_tau0-tau))

    update_d = jax.lax.cond(G-Y >= 0,
                    lambda x: d + (mu * dt / (1 + mu * dt)) * (G - Y),
                    lambda x: d,
                    operand=None)

    curr_damage_thresh = (Y + mu*dt*G) / (1 + mu*dt)
    update_Y = jax.lax.cond(curr_damage_thresh > Y,
                    lambda x: curr_damage_thresh,
                    lambda x: Y,
                    operand=None)

    return update_Y, update_d, update_tau0


def calc_update_Dmat_resolved(material_params_qm, vM0, stress_cd, crossed, vM_cross, C_ss):
    """calc_update_Dmat with vM0 latched at the crossing point (vM_cross) on a crossing step."""
    # At zero stress (first step) this branch is evaluated but not used; a placeholder stress keeps
    # its derivative finite (see calc_d_resolved)
    nonzero = jnp.any(stress_cd != 0)
    stress_cd = jnp.where(nonzero, stress_cd, jnp.array([1.0, 0.0, 0.0]))
    von_Mises = calc_von_Mises(stress_cd)
    vM0 = jnp.where(vM0 == 0, jnp.where(crossed, vM_cross, von_Mises), vM0)

    H = calc_H(von_Mises, vM0, material_params_qm)
    df_dsigma = calc_dfdsigma(stress_cd, von_Mises)
    temp_tensor = calc_temp_tensor(df_dsigma)
    temp_scalar = calc_temp_scalar(df_dsigma, C_ss)
    C_ss_increment = calc_Dmat_bar_increment(temp_tensor, C_ss, H, temp_scalar)

    return vM0, C_ss_increment


@jax.tree_util.Partial
@jax.jit
def update_one_q_resolved(eps_dd, state_i, mat_param_m, C_ss):
        # Extract state
        prev_eps     = state_i[:3]
        prev_stress  = state_i[3:6]
        d, Y, tau0, vM0, dt = state_i[6:11]
        prev2_eps    = state_i[11:14]
        prev2_stress = state_i[14:17]

        # === DAMAGE ===
        dmg_on = calc_dmg_crit(prev_eps, mat_param_m) > 0
        dmg_crossed, s_dmg, on_dmg = _crossing(calc_dmg_crit, prev2_eps, prev_eps, mat_param_m)
        eps_cross = prev2_eps + s_dmg * (prev_eps - prev2_eps)
        tau_cross = _safe_sqrt(jnp.dot(eps_cross, jnp.dot(C_ss, eps_cross)))
        Y_on, d_on, tau0_on = calc_d_resolved(Y, d, tau0, dt, mat_param_m, prev_eps, dmg_crossed, tau_cross, C_ss)
        update_Y, update_d = _resolved_switch(dmg_on, dmg_crossed, on_dmg, (Y_on, d_on), (Y, d))
        # tau0 is a latched value, not an accumulated one, so it is not weighted
        update_tau0 = jnp.where(dmg_on, tau0_on, tau0)

        # === YIELD ===
        yield_on = calc_yield_crit(prev_stress, mat_param_m) > 0
        yield_crossed, s_yield, on_yield = _crossing(calc_yield_crit, prev2_stress, prev_stress, mat_param_m)
        s11, s22, s12 = prev2_stress + s_yield * (prev_stress - prev2_stress)
        vM_cross = _safe_sqrt(s11**2 + s22**2 - s11 * s22 + 3.0 * s12**2)
        vM0_on, delta_C_on = calc_update_Dmat_resolved(mat_param_m, vM0, prev_stress, yield_crossed, vM_cross, C_ss)
        delta_C = _resolved_switch(yield_on, yield_crossed, on_yield, delta_C_on, jnp.zeros_like(C_ss))
        update_vM0 = jnp.where(yield_on, vM0_on, vM0)

        # === STRESS ===
        eps_vogit = rank2_tensor_to_voigt(eps_dd)
        d_eps_vogit = eps_vogit - prev_eps

        d_stress_vogit = (C_ss + delta_C) @ d_eps_vogit
        stress_vogit = prev_stress + d_stress_vogit
        stress_dmg = (1 - update_d) * stress_vogit
        stress_dd = rank2_voigt_to_tensor(stress_dmg)

        # === Update state ===
        update_state_i = state_i.at[0:3].set(eps_vogit)
        update_state_i = update_state_i.at[3:6].set(stress_vogit)
        update_state_i = update_state_i.at[6].set(update_d)
        update_state_i = update_state_i.at[7].set(update_Y)
        update_state_i = update_state_i.at[8].set(update_tau0)
        update_state_i = update_state_i.at[9].set(update_vM0)
        update_state_i = update_state_i.at[11:14].set(prev_eps)
        update_state_i = update_state_i.at[14:17].set(prev_stress)

        return stress_dd, update_state_i


@jax.tree_util.Partial
@jax.jit
def damage_elastic_isotropic_resolved_vmap(eps_qdd: jnp.ndarray, material_params: jnp.ndarray, internal_state_qi: jnp.ndarray):
    """
    Event-resolving damage_elastic_isotropic_vmap (2D plane strain only). The forward result differs from
    the original by O(dt) and is continuous in the material parameters. internal_state_qi has
    N_ISV_RESOLVED = 17 columns.
    """
    if eps_qdd.shape[2] != 2:
        raise RuntimeError("The event-resolving damage model is implemented for 2D plane strain only.")

    E  = material_params[..., 0]
    nu = material_params[..., 1]
    zero = jnp.zeros_like(nu)
    coef = E / ((1 + nu) * (1 - 2 * nu))
    C_qss = (coef * jnp.array(
            [
                [1 - nu, nu,     zero],
                [nu,     1 - nu, zero],
                [zero,   zero,   (1 - 2 * nu) / 2.0]
            ]
        )).transpose((2, 0, 1))

    update_stress_qdd, update_internal_state_qi = jax.vmap(update_one_q_resolved)(
                                                            eps_qdd,                # (Q, 2, 2)
                                                            internal_state_qi,      # (Q, 17)
                                                            material_params,        # (Q, M)
                                                            C_qss)                  # (Q, 3, 3)
    return update_stress_qdd, update_internal_state_qi
