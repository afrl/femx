import jax
import jax.numpy as jnp
from functools import partial
from .element_batch import StrainMeasure
from .utils import (
    rank2_tensor_to_voigt,
    rank2_voigt_to_tensor,
    is_required,
    debug_print,
)

@jax.tree_util.Partial
@jax.jit
def elastic_isotropic(eps_dd: jnp.ndarray, material_params_m: jnp.ndarray):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    material_params_m : material parameters, ndarray[float, (M,)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """

    E = material_params_m[..., 0]
    nu = material_params_m[..., 1]
    G = 0.5 * E / (1.0 + nu)
    if eps_dd.shape[1] == 1:  # 1D
        C_ss = jnp.array([[E]])
    elif eps_dd.shape[1] == 2:  # 2D
        S_ss = jnp.array(
            [
                [1.0 / E, -nu / E, 0.0],
                [-nu / E, 1.0 / E, 0.0],
                [0.0, 0.0, 1.0 / G],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = E/((1.0-2.0*nu)*(1.0+nu))*jnp.array(
            [
                [1.0-nu, nu    , 0.0            ],
                [nu    , 1.0-nu, 0.0            ],
                [0.0   , 0.0   , (1.0-2.0*nu)/nu]
            ]
        )
    elif eps_dd.shape[1] == 3:  # 3D
        S_ss = jnp.array(
            [
                [1.0 / E, -nu / E, -nu / E, 0.0, 0.0, 0.0],
                [-nu / E, 1.0 / E, -nu / E, 0.0, 0.0, 0.0],
                [-nu / E, -nu / E, 1.0 / E, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / G, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / G, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = E/((1.0-2.0*nu)*(1.0+nu))*jnp.array(
            [
                [1.0-nu, nu    , nu    , 0.0            , 0.0             , 0.0             ],
                [nu    , 1.0-nu, nu    , 0.0            , 0.0             , 0.0             ],
                [nu    , nu    , 1.0-nu, 0.0            , 0.0             , 0.0             ],
                [0.0   , 0.0   , 0.0   , (1.0-2.0*nu)/nu, 0.0             , 0.0             ],
                [0.0   , 0.0   , 0.0   , 0.0            , (1.0-2.0*nu)/2.0, 0.0             ],
                [0.0   , 0.0   , 0.0   , 0.0            , 0.0             , (1.0-2.0*nu)/2.0]
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([])  # no internal state


@jax.tree_util.Partial
@jax.jit
def elastic_orthotropic(eps_dd: jnp.ndarray, material_params_m: jnp.ndarray):
    """
    A constitive relation for a linear elastic orthotropic material.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    mat_params_m : material parameters, ndarray[float, (M,)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """
    if eps_dd.shape[1] == 1:  # 1D
        assert (
            material_params_m.shape[-1] == 1
        ), f"Orthotropic elasticity in 1D requires 1 material parameter, received {material_params_m.shape[-1]}"

        C_ss = material_params_m[0]

    elif eps_dd.shape[1] == 2:  # 2D
        assert (
            material_params_m.shape[-1] == 4
        ), f"Orthotropic elasticity in 2D requires 4 material parameters, received {material_params_m.shape[-1]}"

        E_xx = material_params_m[0]
        E_yy = material_params_m[1]
        nu_xy = material_params_m[2]
        G_xy = material_params_m[3]

        S_ss = jnp.array(
            [
                [1.0 / E_xx, -nu_xy / E_xx, 0.0],
                [-nu_xy / E_xx, 1.0 / E_yy, 0.0],
                [0.0, 0.0, 1.0 / G_xy],
            ]
        )
        #C_ss = jnp.linalg.inv(S_ss)
        # For the direct def'n of C_ss we also need nu_yx, nu_zy, and nu_zx.
        # These can be determined from the above parameters along with symmetry. 
        nu_yx = E_yy/E_xx*nu_xy

        Delta =  (1 - nu_xy*nu_yx)/(E_xx*E_yy)
        C_ss = (1/Delta)*jnp.array(
            [
                [1.0/E_yy  , nu_yx/E_yy, 0.0       ],
                [nu_xy/E_xx, 1.0/E_xx  , 0.0       ],
                [0.0       , 0.0       , Delta*G_xy]
            ]
        )

    elif eps_dd.shape[1] == 3:  # 3D
        assert (
            material_params_m.shape[-1] == 9
        ), f"Orthotropic elasticity in 3D requires 9 material parameters, received {material_params_m.shape[-1]}"

        E_xx = material_params_m[0]
        E_yy = material_params_m[1]
        E_zz = material_params_m[2]
        nu_xy = material_params_m[3]
        nu_yz = material_params_m[4]
        nu_xz = material_params_m[5]
        G_xy = material_params_m[6]
        G_yz = material_params_m[7]
        G_xz = material_params_m[9]
        # Note: inv could be avoided if it is a bottleneck, see:
        # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
        S_ss = jnp.array(
            [
                [1.0 / E_xx, -nu_xy / E_xx, -nu_xz / E_xx, 0.0, 0.0, 0.0],
                [-nu_xy / E_xx, 1.0 / E_yy, -nu_yz / E_yy, 0.0, 0.0, 0.0],
                [-nu_xz / E_xx, -nu_yz / E_yy, 1.0 / E_zz, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / G_yz, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / G_xz, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / G_xy],
            ]
        )
        # For the direct def'n of C_ss we also need nu_yx, nu_zy, and nu_zx.
        # These can be determined from the above parameters along with symmetry. 
        nu_yx = E_yy/E_xx*nu_xy
        nu_zy = E_zz/E_yy*nu_yz
        nu_zx = E_zz/E_xx*nu_xz

        Delta =  (1 - nu_xy*nu_yx - nu_yz*nu_zy - nu_xz*nu_zx - 2*nu_xy*nu_yz*nu_zx)/(E_xx*E_yy*E_zz)
        #C_ss = jnp.linalg.inv(S_ss)
        C_ss = (1/Delta)*jnp.array(
            [
                [(1.0-nu_yz*nu_zy)/(E_yy*E_zz)  , (nu_yx+nu_zx*nu_yz)/(E_yy*E_zz), (nu_zx+nu_yx*nu_zy)/(E_yy*E_zz), 0.0, 0.0, 0.0],
                [(nu_xy+nu_xz*nu_zy)/(E_xx*E_zz), (1.0-nu_zx*nu_xz)/(E_xx*E_zz)  , (nu_zy+nu_zx*nu_xy)/(E_xx*E_zz), 0.0, 0.0, 0.0],
                [(nu_xz+nu_xy*nu_yz)/(E_xx*E_yy), (nu_yz+nu_xz*nu_yx)/(E_xx*E_yy), (1.0-nu_xy*nu_yx)/(E_xx*E_yy)  , 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0 , Delta*G_yz ,0.0       , 0.0],
                [0.0, 0.0, 0.0 , 0.0        ,Delta*G_xz, 0.0],
                [0.0, 0.0, 0.0 , 0.0        ,0.0       , Delta*G_xy],
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_dd = rank2_voigt_to_tensor(
        jnp.einsum("si,i->s", C_ss, rank2_tensor_to_voigt(eps_dd))
    )
    return stress_dd, jnp.array([])  # no internal state


@jax.tree_util.Partial
@jax.jit
def linear_elasticity_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: jax.tree_util.Partial,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    material_params : material parameters, ndarray[float, (Q, M)] or ndarray[float, (M,)]
    constitutive_relation : constitutive stress-strain relation, function with arguments
                  (eps_dd: jnp.ndarray, material_params: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """

    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]
    assert (
        P == D
    ), f"Number of dimensions in the parametric coordinate system of the element must match the dimension of the problem, {P} != {D}"
    # Formulation assumes solid elements otherwise a different approach is needed (i.e. shells)

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd)
    eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))

    constitutive_args = []
    in_axes = []

    if is_required(constitutive_model, "eps_dd"):
        constitutive_args.append(eps_qdd)
        in_axes.append(0)

    if is_required(constitutive_model, "material_params_m"):
        constitutive_args.append(material_params)
        if material_params.ndim == 1:
            in_axes.append(None)
        else:
            in_axes.append(0)

    if is_required(constitutive_model, "internal_state_i"):
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)

    constitutive_model_vmap = jax.vmap(constitutive_model, in_axes=tuple(in_axes))
    stress_qdd, new_internal_state_qi = constitutive_model_vmap(*constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, new_internal_state_qi

@jax.tree_util.Partial
@partial(jax.jit, static_argnames=("strain_measure",))
def elastic_truss(
        eps_dd: jnp.ndarray,
        material_params_m: jnp.ndarray,
        internal_state_i: jnp.ndarray,
        x_nd: jnp.ndarray,
        u_nd: jnp.ndarray,
        strain_measure: StrainMeasure | str = StrainMeasure.GREEN_LAGRANGE,
):
    """
    A constitive relation for a an elastic truss.

    Parameters
    ----------
    eps_dd       : infinitesimal strain tensor, ndarray[float, (D, D)]
    material_params_m : material parameters, ndarray[float, (M,)]
    x_nd          : coordinates, ndarray[float, (N, D)]

    Returns
    -------
    stress_dd  : stress tensor, ndarray[float, (D, D)]
    """

    eps_pre = internal_state_i[..., 0] if internal_state_i.size >= 1 else 0
    eps_accum = internal_state_i[..., 1] if internal_state_i.size >= 2 else 0

    E = material_params_m[..., 0]
    A = material_params_m[..., 1]
    # Assumes the node number puts the endpoints as first and last entries. 
    dx_cur = (x_nd+u_nd)[-1,:]-(x_nd+u_nd)[0,:]
    L_cur = jnp.linalg.norm(dx_cur)
    l_cur = dx_cur/L_cur

    dx_ref = x_nd[-1,:]-x_nd[0,:]
    L_ref = jnp.linalg.norm(dx_ref)
    l_ref = dx_ref/L_ref

    # P_dd = jnp.outer(l_cur,l_cur)
    P_dd = jnp.outer(l_ref,l_cur)
    # P_dd = jnp.outer(l_ref,l_ref)
    #jax.debug.print("eps_dd = \n{eps_dd}", eps_dd=eps_dd)


    strain_measure = StrainMeasure.from_value(strain_measure)

    if strain_measure == StrainMeasure.GREEN_LAGRANGE:
        # eps_a = jnp.einsum("i,ij,j->", l_ref, eps_dd, l_ref)
        eps_a = (L_cur**2 - L_ref**2)/(2*L_ref**2)
        eps_total_internal = eps_accum + eps_a + 2*eps_accum*eps_a
    elif strain_measure == StrainMeasure.LINEAR:
        eps_a = jnp.einsum("i,ij,j->", l_cur, eps_dd, l_cur)
        eps_total_internal = eps_accum + eps_a
    elif strain_measure == StrainMeasure.ENGINEERING:
        eps_a = L_cur / L_ref - 1.0
        eps_total_internal = (1.0 + eps_accum) * (1.0 + eps_a) - 1.0
        # eps_total_internal = eps_accum + eps_a
    elif strain_measure == StrainMeasure.LOGARITHMIC:
        eps_a = jnp.log(L_cur/L_ref)
        eps_total_internal = eps_accum + eps_a
    else:
        raise ValueError(f"Option {strain_measure} is not a valid choice for StrainMeasure.")

    stress_dd = E*A*(eps_total_internal + eps_pre)*P_dd


    if internal_state_i.size >= 2:
        internal_state_i = internal_state_i.at[..., 1].set(eps_total_internal)
    return stress_dd, internal_state_i  # no internal state

@jax.tree_util.Partial
@partial(jax.jit, static_argnames=("strain_measure",))
def linear_truss_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: jax.tree_util.Partial,
    strain_measure: StrainMeasure | str = StrainMeasure.GREEN_LAGRANGE,
    contact_stiffness_model: jax.tree_util.Partial | None = None,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    material_params : material parameters, ndarray[float, (Q, M)] or ndarray[float, (M,)]
    constitutive_relation : constitutive stress-strain relation, function with arguments
                  (eps_dd: jnp.ndarray, material_params: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """
    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    g_q11 = jnp.einsum("qpd,qrd->qpr",J_qpd,J_qpd)
    det_J_q = jnp.sqrt(jnp.linalg.det(g_q11))
    dphi_dx_qnd = (dphi_dxi_qnp@J_qpd)/g_q11
    # def lstsq_one(J_pd,dphi_dxi_np):
    #     dphi_dx_nd = jnp.linalg.lstsq(J_pxit() dphi_dxi_np.T)[0]
    #     return dphi_dx_nd.T

    # dphi_dx_qnd = jax.vmap(lstsq_one, in_axes=(0,0))(J_qpd,dphi_dxi_qnp)

    du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd)
    strain_measure = StrainMeasure.from_value(strain_measure)

    if strain_measure == StrainMeasure.GREEN_LAGRANGE:
        eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)) + du_dx_qdd.transpose((0, 2, 1))@du_dx_qdd )
    elif strain_measure in (
        StrainMeasure.LINEAR,
        StrainMeasure.LOGARITHMIC,
        StrainMeasure.ENGINEERING,
    ):
        eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))
    else:
        raise ValueError(f"Option {strain_measure} is not a valid choice for StrainMeasure.")

    constitutive_args = []
    in_axes = []

    if is_required(constitutive_model, "eps_dd"):
        constitutive_args.append(eps_qdd)
        in_axes.append(0)

    if is_required(constitutive_model, "material_params_m"):
        constitutive_args.append(material_params)
        if material_params.ndim == 1:
            in_axes.append(None)
        else:
            in_axes.append(0)

    if is_required(constitutive_model, "internal_state_i"):
        constitutive_args.append(internal_state_qi)
        in_axes.append(0)

    if is_required(constitutive_model, "x_nd"):
        constitutive_args.append(x_nd)
        in_axes.append(None)

    if is_required(constitutive_model, "u_nd"):
        constitutive_args.append(u_nd)
        in_axes.append(None)

    if is_required(constitutive_model, "strain_measure"):
        constitutive_args.append(strain_measure)
        in_axes.append(None)

    if is_required(constitutive_model, "contact_stiffness_model"):
        constitutive_args.append(contact_stiffness_model)
        in_axes.append(None)

    constitutive_model_vmap = jax.vmap(constitutive_model, in_axes=tuple(in_axes))
    stress_qdd, new_internal_state_qi = constitutive_model_vmap(*constitutive_args)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd, new_internal_state_qi


def stiff_matrix(material_params_m: jnp.ndarray, x_nd: jnp.ndarray):
    """
    Compute the stiffness matrix directly from John Whitcomb's notes.

    Parameters
    ----------
    material_params_m : material parameters, ndarray[float, (M,)]
    x_nd          : coordinates, ndarray[float, (N, D)]

    Returns
    -------
    K_global  : stiffness matrix, ndarray[float, (D, D)]
    """

    E = material_params_m[..., 0]
    A = material_params_m[..., 1]
    # Assumes the node number puts the endpoints as first and last entries. 
    dx_d = x_nd[-1,:]-x_nd[0,:]
    L = jnp.linalg.norm(dx_d)

    if x_nd.shape[1] == 1:  # 1D
        K_global = E*A/L*jnp.array([[1,-1],[-1,1]])
    elif x_nd.shape[1] == 2:  # 2D
        K_global =  (E*A/L**3)*jnp.array(
            [
                [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1]],
                [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1]],
                [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1]],
                [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1]],
            ]
        )
    elif x_nd.shape[1] == 3:  # 3D
        K_global = (E*A/L**3)*jnp.array(
            [
                [ dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2],-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2]],
                [ dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2],-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2]],
                [ dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2],-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2]],
                [-dx_d[0]*dx_d[0],-dx_d[0]*dx_d[1],-dx_d[0]*dx_d[2], dx_d[0]*dx_d[0], dx_d[0]*dx_d[1], dx_d[0]*dx_d[2]],
                [-dx_d[0]*dx_d[1],-dx_d[1]*dx_d[1],-dx_d[1]*dx_d[2], dx_d[0]*dx_d[1], dx_d[1]*dx_d[1], dx_d[1]*dx_d[2]],
                [-dx_d[0]*dx_d[2],-dx_d[1]*dx_d[2],-dx_d[2]*dx_d[2], dx_d[0]*dx_d[2], dx_d[1]*dx_d[2], dx_d[2]*dx_d[2]],
            ]
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    return K_global


@jax.tree_util.Partial
@jax.jit
def stiffness_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: jax.tree_util.Partial,
    # u_nd: jnp.ndarray,
    # x_nd: jnp.ndarray,
    # material_params: jnp.ndarray,
    # internal_state_qi: jnp.ndarray,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity using the stiffness matrix.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    material_params : material parameters, ndarray[float, (Q, M)] or ndarray[float, (M,)]
                  (eps_dd: jnp.ndarray, material_params: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """
    E = material_params[..., 0]
    A = material_params[..., 1]
    dx = x_nd[1,:]-x_nd[0,:]
    L = jnp.linalg.norm(dx)

    T= jnp.vstack((jnp.hstack((dx/L,0*dx)),jnp.hstack((0*dx,dx/L))))
    K = E*A/L*jnp.array([[1,-1],[-1,1]])
    K_global = jnp.einsum("ni,ij,jm->nm",T.T,K,T)
    #K_direct = stiff_matrix(material_params_m=material_params,x_nd=x_nd)

    #assert jnp.isclose(K_direct,K_global).all(), "Computing stiffness matrix from T^T K T and direct element-by-element implementation do not match."
    R_nd = jnp.einsum("dk,k->d",K_global,u_nd.reshape((-1))).reshape((-1,x_nd.shape[1]))

    new_internal_state_qi = internal_state_qi
    return R_nd, new_internal_state_qi
