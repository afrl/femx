"""
AD sensitivities of s11 to the 11 matrix (epoxy) parameters at a single material point, through the
whole loading history.

The point is loaded along a proportional strain path (e11 to STRAIN_MAX, e22 = E22_RATIO * e11,
engineering shear e12 = E12_RATIO * e11) in T_TOTAL steps, starting from zero ISVs. jax.jacfwd
differentiates the whole lax.scan, so the derivative of s11 at step n includes how each parameter
changed every earlier step.

Saved to tests/output/test_dmg_grad/dmg_grad_mp_t{T_TOTAL}.npz:
    data        (T, 22)  [:, 0:11]  ISV after step n (isv_names)
                         [:, 11:22] AD ds11_n/dp, unscaled (param_names); multiply by theta for
                                    p * ds11/dp in MPa
    columns     (22,)    names of the data columns
    s11         (T,)     damaged s11 = (1 - D) * ISV s11, the quantity being differentiated
    applied_strain, theta, param_names, isv_names
and the figures dmg_grad_param_grads.png (theta * ds11/dp) and dmg_grad_isvs.png next to it.
The finite-difference counterpart is test_fd_sensitivity.py, which imports the setup from here.

Run from the repo root:
    python tests/test_dmg_grad.py
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)   #   True or comment out both works. False cause incorrect result
from fe_jax.linear_elasticity_dmg import damage_elastic_isotropic_vmap
from fe_jax.plot_results import plot_dmg_sensitivity

OUT_DIR = "tests/output/test_dmg_grad"

Q = 1
T_TOTAL = 500
STRAIN_MAX = 0.081    # final e11
# Fixed ratios, so the path does not move when nu is perturbed. With these, yield starts near step 68
# and damage near step 291, both between steps; an onset exactly on a step puts the criterion at 0 up
# to round-off and FD flips it.
E22_RATIO = -0.39     # lateral contraction, -nu * e11
E12_RATIO = 0.5       # engineering shear
applied_strain = STRAIN_MAX * np.arange(1, T_TOTAL + 1) / T_TOTAL

param_names = ['E', 'nu', 'P1', 'P2', 'mu_visc', 'eps_c', 'eps_t',
               'sigmay_c', 'sigmay_t', 'H_ro', 'n_ro']
isv_names = ['e11', 'e22', 'e12', 's11', 's22', 's12',
             'D', 'Y', 'tau0', 'vM0', 'dt']


def nominal_params():
    # Neat 5220 Epoxy
    params_qm = jnp.zeros(shape=(Q, 11))
    params_qm = params_qm.at[:, 0].set(3900)     # E   MPa
    params_qm = params_qm.at[:, 1].set(0.39)     # nu
    # Epoxy damage properties
    params_qm = params_qm.at[:, 2].set(0.95)     # P1, A
    params_qm = params_qm.at[:, 3].set(2.0)      # P2, B
    params_qm = params_qm.at[:, 4].set(10.0)     # mu_visc
    params_qm = params_qm.at[:, 5].set(0.35)     # eps_c
    params_qm = params_qm.at[:, 6].set(0.04)     # eps_t
    # Epoxy hardening propertie
    params_qm = params_qm.at[:, 7].set(79)       # sigmay_c  MPa
    params_qm = params_qm.at[:, 8].set(62)       # sigmay_t  MPa
    params_qm = params_qm.at[:, 9].set(2e4)      # H_ro, a
    params_qm = params_qm.at[:, 10].set(12)      # n_ro, b
    return params_qm


def initial_isv():
    dt = 10/T_TOTAL
    isv_qi = jnp.zeros(shape=(Q, 11))
    isv_qi = isv_qi.at[...,0:3].set(0)    # e11, e22, e12
    isv_qi = isv_qi.at[...,3:6].set(0)    # s11, s22, s12
    isv_qi = isv_qi.at[...,6]  .set(0)    # D       damage parameter
    isv_qi = isv_qi.at[...,7]  .set(0)    # Y       damage threshold
    isv_qi = isv_qi.at[...,8]  .set(0)    # tau0    initial damage threshold
    isv_qi = isv_qi.at[...,9]  .set(0)    # vM0
    isv_qi = isv_qi.at[...,10] .set(dt)   # dt
    return isv_qi


def make_run():
    """Jitted run(params_qm) -> (damaged s11 (T,), ISV after each step (T, 11)) for the whole loading."""
    isv_qi = initial_isv()
    # Proportional strain path, (T, Q, 2, 2); the tensor holds half the engineering shear
    eps_dd = jnp.array([[1.0, 0.5 * E12_RATIO], [0.5 * E12_RATIO, E22_RATIO]])
    eps_tqdd = jnp.asarray(applied_strain)[:, None, None, None] * eps_dd

    @jax.jit
    def run(params_qm):
        def step(isv_qi, eps_qdd):
            stress_qdd, isv_qi = damage_elastic_isotropic_vmap(eps_qdd, params_qm, isv_qi)
            return isv_qi, (stress_qdd[0, 0, 0], isv_qi[0])
        _, (s11_t, isv_ti) = jax.lax.scan(step, isv_qi, eps_tqdd)
        return s11_t, isv_ti

    return run


def test_ad():
    params_qm = nominal_params()
    run = make_run()

    s11_t, isv_ti = run(params_qm)
    # d s11_n / d p_m at every step, (T, Q, M) -> (T, M)
    grad_tm = jax.jit(jax.jacfwd(lambda p: run(p)[0]))(params_qm)[:, 0, :]

    theta = np.asarray(params_qm[0])
    data = np.concatenate([np.asarray(isv_ti), np.asarray(grad_tm)], axis=1)   # (T, 22)
    columns = isv_names + [f"ds11/d{name}" for name in param_names]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"dmg_grad_mp_t{T_TOTAL}.npz")
    np.savez(out_path, data=data, columns=np.array(columns), s11=np.asarray(s11_t),
             applied_strain=applied_strain, theta=theta,
             param_names=np.array(param_names), isv_names=np.array(isv_names))

    yield_step = int(np.argmax(data[:, 9] > 0))
    damage_step = int(np.argmax(data[:, 8] > 0))
    print(f"Saved {data.shape} to {out_path}")
    print(f"first yield at step {yield_step} ({100 * applied_strain[yield_step]:.2f} %), "
          f"damage criterion at step {damage_step} ({100 * applied_strain[damage_step]:.2f} %)")
    for k, name in enumerate(param_names):
        print(f"{name:9s} max|p ds11/dp|  AD {np.max(np.abs(theta[k] * data[:, 11 + k])):10.3f}")

    plot_dmg_sensitivity(applied_strain, theta * data[:, 11:22], data[:, 0:11], OUT_DIR, 'dmg_grad', 'AD')

    return


if __name__ == "__main__":
    t_start = time.perf_counter()
    test_ad()
    print("Time used:", time.perf_counter() - t_start)
