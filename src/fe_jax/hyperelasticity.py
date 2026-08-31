import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import is_required
from .linear_elasticity import elastic_isotropic

@jax.tree_util.Partial
@jax.jit
def st_venant_kirchhoff(F_dd: jnp.ndarray, material_params_m: jnp.ndarray):
    """
    A constitutive relation for an isotropic St. Venant-Kirchhoff hyperelastic solid. Identical to linear elasticity, except using nonlinear strain.
    First argument is deformation gradient, rather than strain, since other hyperelastic models are not naturally in terms of even nonlinear strain.
    """
    eps_dd = 0.5 * (
        jnp.einsum("di,dj -> ij", F_dd, F_dd)
        - jnp.eye(F_dd.shape[1])
    )
    stress_dd = elastic_isotropic(eps_dd,material_params_m)[0]
    return stress_dd


@jax.tree_util.Partial
@jax.jit
def mooney_rivlin(F_dd: jnp.ndarray, material_params_m: jnp.ndarray):
    """
    A constitutive model for Mooney-Rivlin hyperelasticity, assumes plane strain/linear strain for the lower dimensional cases,
    essentially assuming a block form of the deformation gradient with the relevant dimensions as provided by the argument F_qdd in the upper left,
    and 1s along the remaining diagonal.
    TODO:
    Since we only want the upper left block of the stress as well, we can essentially work only on that block,
    which means that the computation is independent of dimension. This may not be valid, perhaps confirm how 2D hyperelasticity is "usually" done
    """
    C1_q = material_params_m[0]
    C2_q = material_params_m[1]
    D1_q = material_params_m[2]

    if F_dd.shape[1] <= 3:
        J = jnp.linalg.det(F_dd)
        Jm13 = J ** (-1 / 3)
        Fbar_dd =Jm13*F_dd
        Bbar_dd = jnp.einsum("dj,dk -> jk", Fbar_dd, Fbar_dd)
        I1bar = Bbar_dd.trace()
        I2bar = 0.5 * (I1bar**2 - jnp.einsum("ij,ji ->", Bbar_dd, Bbar_dd))
        BbarFbar_dd = jnp.einsum("jd,dk -> jk", Bbar_dd, Fbar_dd)
        Fbarinv_dd = jnp.linalg.inv(Fbar_dd).transpose()
        FinvTcoefficients = 2 / D1_q * Jm13 ** (-2) * (J - 1) - 2 / 3 * Jm13 * (
            C1_q * I1bar + 2 * C2_q * I2bar
        )
        Fcoefficients = 2 * Jm13 * (C1_q + I1bar * C2_q)
        BFcoefficients = 2 * Jm13 * C2_q
        stress_qdd = (FinvTcoefficients*Fbarinv_dd
            + Fcoefficients*Fbar_dd
            + BFcoefficients*BbarFbar_dd
        )
    else:
        raise RuntimeError("Deformation Gradient must be at most 3D")
    return stress_qdd

@jax.tree_util.Partial
@jax.jit
def hyperelasticity_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: Callable,
):
    """
    Residual function that computes the residual for the weak form corresponding to hyperelasticity


    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]
    constitutive_relation : constitutive stress-strain relation, arguments
                  (F_qdd: jnp.ndarray, material_params_qm: jnp.ndarray) where F_qdd is the deformation gradient, dx/dX

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """

    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]
    assert P == D
    # Formulation assumes solid elements otherwise a different approach is needed (i.e. shells)

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    F_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd + x_nd)
    constitutive_args = []
    in_axes = []

    if is_required(constitutive_model, "F_dd"):
        constitutive_args.append(F_qdd)
        in_axes.append(0)

    if is_required(constitutive_model, "material_params_m"):
        constitutive_args.append(material_params_qm)
        if material_params_qm.ndim == 1:
            in_axes.append(None)
        else:
            in_axes.append(0)

    if is_required(constitutive_model, "internal_state_i"):
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)
    stress_qdd = jax.vmap(constitutive_model,in_axes=tuple(in_axes))(*constitutive_args)
    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, internal_state_qi
