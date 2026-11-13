
import jax
import jax.numpy as jnp
from functools import partial
from . import params

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_linear(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_max = material_params_m[..., params.CONTACT_E_C_PARAM]
    radius = material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]
    E = jnp.where(d<radius,E_max - E_max/radius*d,0)
    return E

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_constant(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_max = material_params_m[..., params.CONTACT_E_C_PARAM]
    return E_max

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_piecewise_linear(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_c= material_params_m[..., params.CONTACT_E_C_PARAM] # stiffness at physical contact 
    total_radius =  material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]

    ramp_up_distance      = total_radius*material_params_m[...,params.CONTACT_M_TO_D_PARAM]
    hard_contact_distance = total_radius*material_params_m[...,params.CONTACT_C_TO_D_PARAM]
    search_radius         = total_radius*material_params_m[..., params.CONTACT_SEARCH_ALPHA_PARAM] # search_radius

    E_min = material_params_m[...,params.CONTACT_E_MIN_PARAM]

    seg1 = E_c + (E_c-E_min)/(hard_contact_distance - ramp_up_distance)*(d-hard_contact_distance)
    seg2 = E_min/(ramp_up_distance-search_radius)*(d-search_radius)
    # E = ()*(jnp.where(d<ramp_up_distance,1,0))
    # E += ()*(jnp.where(d>ramp_up_distance,1,0)-jnp.where(d>search_radius,1,0))
    return jnp.where(
        d<ramp_up_distance,
        seg1,
        jnp.where(d<search_radius,seg2, 0.0),
    )

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_piecewise_quadratic(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_c= material_params_m[..., params.CONTACT_E_C_PARAM] # stiffness at physical contact 
    total_radius =  material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]

    ramp_up_distance      = total_radius*material_params_m[..., params.CONTACT_M_TO_D_PARAM]
    hard_contact_distance = total_radius*material_params_m[..., params.CONTACT_C_TO_D_PARAM]
    search_radius         = total_radius*material_params_m[..., params.CONTACT_SEARCH_ALPHA_PARAM] # search_radius

    E_min = material_params_m[..., params.CONTACT_E_MIN_PARAM]

    mu = E_min/(ramp_up_distance-search_radius)
    alpha = (E_c - E_min + mu*(ramp_up_distance-hard_contact_distance))/((hard_contact_distance-ramp_up_distance)**2)
    beta = mu - 2*alpha*ramp_up_distance
    gamma = E_min + alpha*ramp_up_distance**2 - mu*ramp_up_distance

    seg1 = alpha*d**2 + beta*d + gamma
    seg2 = E_min/(ramp_up_distance-search_radius)*(d-search_radius)
    # E = ()*(jnp.where(d<ramp_up_distance,1,0))
    # E += ()*(jnp.where(d>ramp_up_distance,1,0)-jnp.where(d>search_radius,1,0))
    return jnp.where(
        d<ramp_up_distance,
        seg1,
        jnp.where(d<search_radius,seg2, 0.0),
    )

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_exponential(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_c= material_params_m[..., params.CONTACT_E_C_PARAM] # stiffness at physical contact
    total_radius = material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]

    ramp_up_distance      = total_radius*material_params_m[..., params.CONTACT_M_TO_D_PARAM]
    hard_contact_distance = total_radius*material_params_m[..., params.CONTACT_C_TO_D_PARAM]

    E_min = material_params_m[..., params.CONTACT_E_MIN_PARAM]

    alpha =  jnp.exp((hard_contact_distance*jnp.log(E_min) - ramp_up_distance*jnp.log(E_c))/(hard_contact_distance-ramp_up_distance))
    r = (jnp.log(E_min)-jnp.log(E_c))/(hard_contact_distance-ramp_up_distance)
    return alpha*jnp.exp(-r*d)

@jax.tree_util.Partial
@jax.jit
def __contact_stiffness_tanh(
    d: jnp.ndarray,
    material_params_m: jnp.ndarray
) -> float:
    E_c= material_params_m[..., params.CONTACT_E_C_PARAM] # stiffness at physical contact
    total_radius = material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]

    ramp_up_distance      = total_radius*material_params_m[..., params.CONTACT_M_TO_D_PARAM]
    hard_contact_distance = total_radius*material_params_m[..., params.CONTACT_C_TO_D_PARAM]

    E_min = material_params_m[..., params.CONTACT_E_MIN_PARAM]

    sharpness_factor = 2/(hard_contact_distance - ramp_up_distance)*jnp.arctanh(2*E_min/E_c - 1)
    return E_c/2*(jnp.tanh(-sharpness_factor*(d-(ramp_up_distance + hard_contact_distance)/2)) + 1)

@jax.jit
def __elastic_contact_truss_kernel(
    contact_stiffness_model: jax.tree_util.Partial,
    eps_dd: jnp.ndarray,
    material_params_m: jnp.ndarray,
    x_nd: jnp.ndarray,
    u_nd: jnp.ndarray,
):
    """
    A constitive relation for contact elements which is basically
    an elastic truss where stiffness depends on the length.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    material_params_m : material parameters, ndarray[float, (M,)]
    x_nd          : coordinates, ndarray[float, (N, D)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """

    # E_max = material_params_m[..., 0]
    # A = material_params_m[..., 1]
    active = material_params_m[..., params.CONTACT_ACTIVE_PARAM]
    A = material_params_m[...,params.CONTACT_AREA_PARAM]*active

    # radius = material_params_m[..., 2]
    # E = lambda d: jnp.where(d<radius,E_max - E_max/radius*d,0)
    # Assumes the node number puts the endpoints as first and last entries. 
    dx_cur_state = (x_nd+u_nd)[-1,:]-(x_nd+u_nd)[0,:]
    L_cur_state = jnp.linalg.norm(dx_cur_state)
    safe_eps = jnp.finfo(L_cur_state.dtype).eps
    L_cur_state_safe = jnp.maximum(L_cur_state, safe_eps)
    l_cur_state = dx_cur_state/L_cur_state_safe

    dx_ref_state = x_nd[-1,:]-x_nd[0,:]
    L_ref_state = jnp.linalg.norm(dx_ref_state)
    L_ref_state_safe = jnp.maximum(L_ref_state, safe_eps)
    l_ref_state = dx_ref_state/L_ref_state_safe

    # P_dd = jnp.outer(l_cur,l_cur)
    P_dd = jnp.outer(l_ref_state,l_cur_state)

    # dx_d   = (x_nd+u_nd)[-1,:]-(x_nd+u_nd)[0,:]
    # l_d = dx_d/jnp.sqrt(jnp.dot(dx_d,dx_d))

    # P_dd = jnp.outer(l_d,l_d)
    # eps_a = jnp.einsum("i,ij,j->", l_d, eps_dd, l_d)

    total_radius =  material_params_m[..., params.CONTACT_RADIUS_0_PARAM] + material_params_m[..., params.CONTACT_RADIUS_1_PARAM]
    search_radius         = total_radius*material_params_m[..., params.CONTACT_SEARCH_ALPHA_PARAM] # search_radius
    ramp_up_distance      = total_radius*material_params_m[...,params.CONTACT_M_TO_D_PARAM]
    L_ref = jnp.maximum(
        search_radius,
        # ramp_up_distance,
        L_ref_state_safe, #distance at first contact
    )
    # eps_a = (L_ref - jnp.linalg.norm(dx_d))/L_ref
    # eps_a = jnp.maximum(0,L_ref - jnp.linalg.norm(dx_d))
    # eps_a = jnp.minimum(0,jnp.linalg.norm(dx_d) - L_ref)/L_ref #unilateral compression
    eps_a = (L_cur_state - L_ref)/L_ref # bilateral

    stress_dd = contact_stiffness_model(L_cur_state_safe,material_params_m)*A*eps_a*P_dd
    stress_dd = jnp.where(active > 0.0, stress_dd, jnp.zeros_like(stress_dd))

    # jax.debug.print(
    #     'x_nd = {x}\nd = {d}, E = {E}, E*eps_a = {EA}\n',
    #     x = x_nd,
    #     d = jnp.linalg.norm(dx_d),
    #     E = contact_stiffness_model(jnp.linalg.norm(dx_d),material_params_m),
    #     EA = contact_stiffness_model(jnp.linalg.norm(dx_d),material_params_m)*eps_a
    # )

    return stress_dd, jnp.array([])  # no internal state

# Contact formulations to use for 1D contact elements
elastic_contact_truss_constant            = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_constant)
elastic_contact_truss_linear              = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_linear)
elastic_contact_truss_piecewise_linear    = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_piecewise_linear)
elastic_contact_truss_piecewise_quadratic = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_piecewise_quadratic)
elastic_contact_truss_exponential         = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_exponential)
elastic_contact_truss_tanh                = jax.tree_util.Partial(__elastic_contact_truss_kernel, __contact_stiffness_tanh)


