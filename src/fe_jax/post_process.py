import jax
import jax.numpy as jnp

from .utils import get_element_areas


def calculate_global_stress(ISV_be_history, element_batches, points):
    """
    Computes the area-averaged in-plane stress history over the whole mesh.

    The stress components are read from the internal state variables
    (indices 3, 4, 5 = sigma_11, sigma_22, sigma_12) and weighted by element
    area, so every element batch contributes in proportion to the area it covers.

    Parameters
    ----------
    ISV_be_history   : sequence (one entry per batch) of float arrays with shape
                       (N_t, N_e, N_isv), internal state of every element at every step
    element_batches  : sequence of ``ElementBatch`` matching ``ISV_be_history``
    points           : float array with shape (N_v, 2), nodal coordinates

    Returns
    -------
    s11_global, s22_global, s12_global : float arrays with shape (N_t,)
    """
    stress_area_sum = 0.0  # sum over elements of stress * area, shape (N_t, 3)
    total_area = 0.0
    for ISV_tei, batch in zip(ISV_be_history, element_batches):
        area_e = get_element_areas(batch.connectivity_en, points, batch.fe_type)
        stress_tec = ISV_tei[..., 3:6]  # (s11, s22, s12) of every element
        stress_area_sum += jnp.sum(stress_tec * area_e[None, :, None], axis=1)
        total_area += jnp.sum(area_e)
    stress_global_tc = stress_area_sum / total_area
    return stress_global_tc[:, 0], stress_global_tc[:, 1], stress_global_tc[:, 2]


def calculate_strength(s11_global_t, drop_tol=0.01):
    """
    Computes the strength as the first peak of the global s11 curve.

    The first peak is the largest s11 before the curve first falls more than
    ``drop_tol`` times the largest s11 so far below it, i.e. the first local
    maximum that is followed by a load drop of more than ``drop_tol`` (relative).
    Smaller dips, e.g. from single elements reaching the yield or damage
    criterion, do not count. With ``drop_tol = 0`` this is the first local
    maximum. If the curve never drops that far, the strength is the maximum of
    the whole curve.

    Unlike the maximum over the whole loading, this does not jump between
    near-equal later peaks when the parameters or the step size change. The
    peak step is found without derivatives, so the derivative is that of s11
    at the peak step.

    Parameters
    ----------
    s11_global_t : float array with shape (N_t,), global s11 after each load step
                   (e.g. the first output of ``calculate_global_stress``)
    drop_tol     : float, relative size of the load drop that ends the first peak

    Returns
    -------
    strength     : float scalar, in stress units
    """
    s11 = jax.lax.stop_gradient(s11_global_t)
    running_max = jax.lax.cummax(s11)
    dropped = s11 < running_max - drop_tol * jnp.abs(running_max)
    # First step past the peak, or the end of the loading if the curve never drops that far
    n_t = s11.shape[0]
    k_end = jnp.where(jnp.any(dropped), jnp.argmax(dropped), n_t)
    k_peak = jnp.argmax(jnp.where(jnp.arange(n_t) < k_end, s11, -jnp.inf))
    return s11_global_t[k_peak]


def calculate_toughness(s11_global_t, strain_max):
    """
    Computes the toughness, the area under the global s11 vs applied strain curve.

    The loading is assumed to ramp the applied strain linearly from 0 to
    ``strain_max`` in ``N_t`` equal steps, with ``s11_global_t[k]`` the stress
    after step ``k + 1``. The stress at zero strain is taken as 0. The area is
    integrated with the trapezoidal rule and is differentiable.

    Parameters
    ----------
    s11_global_t : float array with shape (N_t,), global s11 after each load step
                   (e.g. the first output of ``calculate_global_stress``)
    strain_max   : float, applied strain at the last step

    Returns
    -------
    toughness    : float scalar, in stress units (MPa = MJ/m^3 for stress in MPa)
    """
    strain_t = jnp.linspace(0.0, strain_max, s11_global_t.shape[0] + 1)
    return jnp.trapezoid(jnp.concatenate([jnp.zeros(1), s11_global_t]), strain_t)
