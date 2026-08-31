import jax
import jax.numpy as jnp
from typing import Callable
from .setup import is_required

@jax.tree_util.Partial
@jax.jit
def nonlinear_isotropic_gas(dT_dx_ud,T_u,material_params_m):
    """
    Simplified nonlinear thermal conductivity model based on kinetic theory. Requires T to be absolute temperature and primarily valid far from 0
    """
    k0 = material_params_m[0]
    T0 = material_params_m[1]
    k_u = k0 * jnp.sqrt(jnp.abs(T_u)/T0) # abs saves a nan explosion if T drops negative, but worth noting that is intrinsically nonphysical
    return jnp.einsum("u,ud -> ud",k_u,dT_dx_ud)

@jax.tree_util.Partial
@jax.jit
def cubic_thermal_conductivity(dT_dx_ud,T_u,material_params_m):
    """
    Nonlinear thermal conductivity valid for solids at ~10K. 
    """
    k0 = material_params_m[0]
    T0 = material_params_m[1]
    k_u = k0 * (T_u/T0)**3
    return jnp.einsum("u,ud -> ud",k_u,dT_dx_ud)


@jax.tree_util.Partial
@jax.jit
def fouriers_law_linearisotropic(dT_dx_ud,material_params_m):
    """
    Linear isotropic thermal conductivity
    """
    k = material_params_m[0]
    return k * dT_dx_ud

@jax.tree_util.Partial
@jax.jit
def heat_conduction_residual(
    u_nd : jnp.ndarray,
    x_nd: jnp.ndarray,
    phi_qn: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: Callable,    
):
    u_nu = u_nd # Scalar field, U=1 not D, but for consistency it is stored with a size 1 axis.
    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    constitutive_args = []
    in_axes=[]
    if is_required(constitutive_model, "dT_dx_ud"):
        dT_dx_qud = jnp.einsum("qnd,nu->qud", dphi_dx_qnd, u_nu)
        constitutive_args.append(dT_dx_qud)
        in_axes.append(0)

    if is_required(constitutive_model, "T_u"):
        T_qu = jnp.einsum("qn,nu -> qu",phi_qn,u_nu)
        constitutive_args.append(T_qu)
        in_axes.append(0)    

    if is_required(constitutive_model, "material_params_m"):
        constitutive_args.append(material_params_qm)
        in_axes.append(0 if material_params_qm.ndim==2 else None)

    if is_required(constitutive_model, "internal_state_i"):
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)
  
    heat_flux_qud = jax.vmap(constitutive_model,in_axes=tuple(in_axes))(*constitutive_args)
    dphi_dx_heat_flux_qnu = jnp.einsum("qnd,qud-> qnu",dphi_dx_qnd,heat_flux_qud)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nu = jnp.einsum("qnu,q-> nu",dphi_dx_heat_flux_qnu,det_JxW_q)
    return R_nu, internal_state_qi