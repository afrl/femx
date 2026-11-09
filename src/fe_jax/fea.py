from .setup import *
from .utils import *
from .element_batch_collection import *
from .solve_cg import cg as cg_w_info
from .sparse_matrix import *
from .sparse_linear_solve import *
from .constraints import *
from .constraint_system import *
from .dof_enumeration import *
from .boundary_conditions import *

import jetsci
import jetsci.petsc_snes as petsc_snes
from jetsci.petsc_snes import differentiable_snes

import jax.numpy as jnp
import jax
import jax.experimental.sparse as jsparse

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any

from flax import struct



@partial(jax.jit, static_argnames=["i", "E", "N", "U"])
def __get_jacobian_indices(
    i: int,
    E: tuple[int, ...],
    N: tuple[int, ...],
    U: tuple[int, ...],
    connectivity: jnp.ndarray,
    EN_offsets: jnp.ndarray,
):
    dof_map = ebc_get_dof_map(
        i=i,
        E=E,
        N=N,
        U=U,
        connectivity=connectivity,
        EN_offsets=EN_offsets,
    )
    cols, rows = jax.vmap(jnp.meshgrid)(dof_map, dof_map)
    return jnp.vstack([rows.ravel(), cols.ravel()]).T


@jax.jit
def _calculate_jacobian_unique_nnz(
    ebc: ElementBatchCollection,
):
    """
    Returns the number of non-zeros in the Jacobian for a collection of batches of elements,
    ignoring any effect of constraints on the sparsity pattern.
    """
    non_zero_indices = jnp.vstack(
        [
            __get_jacobian_indices(
                i, ebc.E, ebc.N, ebc.U, ebc.connectivity, ebc.EN_offsets
            )
            for i in range(ebc.B)
        ]
    )
    # Get the permutation that sorts the non-zero entries (sorted by row then col)
    perm = jnp.lexsort((non_zero_indices[:, 1], non_zero_indices[:, 0]))
    # Sort the non-zero indices
    non_zero_indices = non_zero_indices[perm]
    # An array of non_zero_indices.shape[0]-1 that is a[i+1] - a[i]
    diff = jnp.diff(non_zero_indices, axis=0)
    # Boolean mask indicating if each (row, col) value is unique, shape=A.col.shape
    uniq_mask = jnp.append(True, (diff != 0).any(axis=1))
    return jnp.sum(uniq_mask)


@jax.jit
def _calculate_jacobian_batch_element_kernel(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    u_enu: jnp.ndarray,
    x_end: jnp.ndarray,
    phi_qn: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian matrices for a batch of elements without any modification
    of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func: residual function emerging from weak form of governing equations
    constitutive_model: a partial function for the constitutive model
    u_enu: element node solution displacement array, ndarray[float, (E, N, U)]
    x_end: element node coordinate array, ndarray[float, (E, N, D)]
    dphi_dxi_qnp: basis function derivatives at quadrature points, ndarray[float, (Q, N, P)]
    W_q: quadrature weights, ndarray[float, (Q,)]
    material_params: material parameters for the batch
    internal_state: internal state variables for the batch

    Returns
    -------
    out: batch of element-level Jacobian matrices, ndarray[float, (E, N * U, N * U)]
    """

    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]

    # Note: reshaped to be (# elements, # dofs per element) so that the jacfwd produces a
    # (# dofs per element, # dofs per element) matrix for each element.
    # Assumption: # dofs per element is N * U
    u_et = u_enu.reshape(E, N * U)

    # Note: captures dphi_dxi_qnp, W_q, and constitutive_model
    @jax.jit
    def residual_kernel(u_t, x_nd, material_params, internal_state_qi):
        u_nd = u_t.reshape(N, U)
        resid_args=[]
        if is_required(element_residual_func,"u_nd"):
            resid_args.append(u_nd)
        if is_required(element_residual_func,"x_nd"):
            resid_args.append(x_nd)
        if is_required(element_residual_func,"phi_qn"):
            resid_args.append(phi_qn)
        if is_required(element_residual_func,"dphi_dxi_qnp"):
            resid_args.append(dphi_dxi_qnp)
        if is_required(element_residual_func,"W_q"):
            resid_args.append(W_q)
        if is_required(element_residual_func,"material_params"):
            resid_args.append(material_params)
        if is_required(element_residual_func,"internal_state_qi"):
            resid_args.append(internal_state)
        if is_required(element_residual_func,"constitutive_model"):
            resid_args.append(constitutive_model)

        R_nu = element_residual_func(*resid_args)[0]
        return R_nu.reshape(N * U)

    J_ett = jax.vmap(
        jax.jacfwd(residual_kernel, argnums=0),
        in_axes=(
            0,
            0,
            None if material_params.ndim == 1 else 0,
            None if internal_state.ndim < 3 else 0,
        ),
    )(u_et, x_end, material_params, internal_state)

    assert J_ett.shape == (
        E,
        N * U,
        N * U,
    ), f"Expected shape {(E, N * U, N * U)}, but received {J_ett.shape}"

    return J_ett


@jax.jit
def _calculate_jacobian_coo_terms_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    phi_qn:jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    dof_map_enu: jnp.ndarray,
    assembly_map: AssemblyMap,
    u_f: jnp.ndarray,
):
    u_enu = transform_global_unraveled_to_element_node(
        assembly_map, u_f
    )

    dof_map = dof_map_enu.reshape(x_end.shape[0], -1)
    # debug_print(dof_map)
    cols, rows = jax.vmap(jnp.meshgrid)(dof_map, dof_map)
    # debug_print(rows)
    # debug_print(cols)

    J_ett = _calculate_jacobian_batch_element_kernel(
        element_residual_func=element_residual_func,
        constitutive_model=constitutive_model,
        u_enu=u_enu,
        x_end=x_end,
        phi_qn=phi_qn,
        dphi_dxi_qnp=dphi_dxi_qnp,
        W_q=W_q,
        material_params=material_params,
        internal_state=internal_state,
    )
    # debug_print(J_ett)

    return (J_ett, rows, cols)


def calculate_jacobian_wo_constraints(
    u_f: jnp.ndarray,
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
    precomputed_jacobian_nnz: int,
):

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    J_ett, rows, cols = zip(
        *[
            _calculate_jacobian_coo_terms_batch(
                element_residual_func=element_residual_func,
                constitutive_model=ebc.constitutive_models[i],
                material_params=ebc.get_material_params(i),
                internal_state=ebc.get_internal_state(i),
                x_end=ebc.get_x(i),
                phi_qn=ebc.get_phi(i),
                dphi_dxi_qnp=ebc.get_dphi_dxi(i),
                W_q=ebc.get_weights(i),
                dof_map_enu=ebc.get_dof_map(i),
                assembly_map=assembly_map_b[i],
                u_f=u_f,
            )
            for i in range(ebc.B)
        ]
    )
    J_ett = jnp.concatenate([x.ravel() for x in J_ett])
    rows = jnp.concatenate([x.ravel() for x in rows])
    cols = jnp.concatenate([x.ravel() for x in cols])

    # debug_print(J_ett)
    # debug_print(rows)
    # debug_print(cols)

    J_sparse_ff = jsparse.COO(
        (J_ett.ravel(), rows.ravel(), cols.ravel()),
        shape=(u_f.shape[0], u_f.shape[0]),
    )._sort_indices()

    J_sparse_ff = coo_sum_duplicates(
        J_sparse_ff, result_length=precomputed_jacobian_nnz
    )

    return J_sparse_ff


@jax.jit
def _calculate_jacobian_diag_batch_element_kernel(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    u_enu: jnp.ndarray,
    x_end: jnp.ndarray,
    phi_qn: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the element-level jacobian diagonal matrices for a batch of elements without
    any modification of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func: residual function emerging from weak form of governing equations
    constitutive_model: a partial function for the constitutive model
    u_enu: element node solution displacement array, ndarray[float, (E, N, U)]
    x_end: element node coordinate array, ndarray[float, (E, N, D)]
    dphi_dxi_qnp: basis function derivatives at quadrature points, ndarray[float, (Q, N, P)]
    W_q: quadrature weights, ndarray[float, (Q,)]
    material_params: material parameters for each element batch
    internal_state: internal state variables for each element batch

    Returns
    -------
    diag_J_et: batch of element-level Jacobian diagonal vectors, ndarray[float, (E, N * U)]
    """

    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]
    U = u_enu.shape[2]

    # Note: reshaped to be (# elements, # dofs per element) so that the jacfwd produces a
    # (# dofs per element, # dofs per element) matrix for each element.
    # Assumption: # dofs per element is N * U
    u_et = u_enu.reshape(E, N * U)

    # Note: captures dphi_dxi_qnp, W_q, and constitutive_model
    @jax.jit
    def residual_kernel(u_t, x_nd, material_params, internal_state):
        u_nd = u_t.reshape(N, U)
        resid_args=[]
        if is_required(element_residual_func,"u_nd"):
            resid_args.append(u_nd)
        if is_required(element_residual_func,"x_nd"):
            resid_args.append(x_nd)
        if is_required(element_residual_func,"phi_qn"):
            resid_args.append(phi_qn)
        if is_required(element_residual_func,"dphi_dxi_qnp"):
            resid_args.append(dphi_dxi_qnp)
        if is_required(element_residual_func,"W_q"):
            resid_args.append(W_q)
        if is_required(element_residual_func,"material_params"):
            resid_args.append(material_params)
        if is_required(element_residual_func,"internal_state_qi"):
            resid_args.append(internal_state)
        if is_required(element_residual_func,"constitutive_model"):
            resid_args.append(constitutive_model)

        R_nu = element_residual_func(*resid_args)[0]
        return R_nu.reshape(N * U)

    def diag_J(u_t, x_nd, material_params, internal_state):
        return jnp.diagonal(
            jax.jacfwd(residual_kernel, argnums=0)(
                u_t, x_nd, material_params, internal_state
            )
        )

    diag_J_vmap = jax.vmap(
        diag_J,
        in_axes=(
            0,
            0,
            None if material_params.ndim == 1 else 0,
            None if internal_state.ndim < 3 else 0,
        ),
    )

    diag_J_et = diag_J_vmap(u_et, x_end, material_params, internal_state)

    assert diag_J_et.shape == (
        E,
        N * U,
    ), f"Expected shape {(E, N * U)}, but received {diag_J_et.shape}"

    return diag_J_et


@jax.jit
def _calculate_jacobian_diag_coo_terms_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    phi_qn:jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    dof_map_enu: jnp.ndarray,
    assembly_map: AssemblyMap,
    u_f: jnp.ndarray,
):
    u_enu = transform_global_unraveled_to_element_node(
        assembly_map, u_f
    )

    dof_map = dof_map_enu.reshape(x_end.shape[0], -1)
    # debug_print(dof_map)

    diag_J_et = _calculate_jacobian_diag_batch_element_kernel(
        element_residual_func=element_residual_func,
        constitutive_model=constitutive_model,
        u_enu=u_enu,
        x_end=x_end,
        phi_qn=phi_qn,
        dphi_dxi_qnp=dphi_dxi_qnp,
        W_q=W_q,
        material_params=material_params,
        internal_state=internal_state,
    )
    # debug_print(diag_J_et)

    return (diag_J_et, dof_map)


def calculate_jacobian_diag_wo_constraints(
    u_f: jnp.ndarray,
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
):

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    diag_J_et, indices = zip(
        *[
            _calculate_jacobian_diag_coo_terms_batch(
                element_residual_func=element_residual_func,
                constitutive_model=ebc.constitutive_models[i],
                material_params=ebc.get_material_params(i),
                internal_state=ebc.get_internal_state(i),
                x_end=ebc.get_x(i),
                phi_qn=ebc.get_phi(i),
                dphi_dxi_qnp=ebc.get_dphi_dxi(i),
                W_q=ebc.get_weights(i),
                dof_map_enu=ebc.get_dof_map(i),
                assembly_map=assembly_map_b[i],
                u_f=u_f,
            )
            for i in range(ebc.B)
        ]
    )
    # diag_J_et = jnp.vstack(diag_J_et).ravel()
    # indices = jnp.vstack(indices).ravel()
    diag_J_et = jnp.concatenate([x.ravel() for x in diag_J_et])
    indices = jnp.concatenate([x.ravel() for x in indices])

    # debug_print(diag_J_et)
    # debug_print(indices)

    diag_J_f = jnp.zeros_like(u_f)
    diag_J_f = diag_J_f.at[indices].add(diag_J_et)

    return diag_J_f


@jax.jit
def _calculate_residual_wo_constraints_batch(
    element_residual_func: jax.tree_util.Partial,
    constitutive_model: jax.tree_util.Partial,
    material_params: jnp.ndarray,
    internal_state: jnp.ndarray,
    x_end: jnp.ndarray,
    phi_qn:jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    assembly_map: AssemblyMap,
    u_f: jnp.ndarray,
):
    # Extract shape constants needed for args
    E = x_end.shape[0]
    N = x_end.shape[1]
    D = x_end.shape[2]

    assert (
        N == dphi_dxi_qnp.shape[1]
    ), f"Number of nodes per element {N} must match the number of basis functions {dphi_dxi_qnp.shape[1]}."

    u_enu = transform_global_unraveled_to_element_node(assembly_map, u_f)

    resid_args = []
    in_axes = []
    if is_required(element_residual_func,"u_nd"):
        resid_args.append(u_enu)
        in_axes.append(0)
    if is_required(element_residual_func,"x_nd"):
        resid_args.append(x_end)
        in_axes.append(0)
    if is_required(element_residual_func,"phi_qn"):
        resid_args.append(phi_qn)
        in_axes.append(None)
    if is_required(element_residual_func,"dphi_dxi_qnp"):
        resid_args.append(dphi_dxi_qnp)
        in_axes.append(None)
    if is_required(element_residual_func,"W_q"):
        resid_args.append(W_q)
        in_axes.append(None)
    if is_required(element_residual_func,"material_params"):
        resid_args.append(material_params)
        in_axes.append(None if material_params.ndim == 1 else 0)
    if is_required(element_residual_func,"internal_state_qi"):
        resid_args.append(internal_state)
        in_axes.append(None if internal_state.ndim < 3 else 0)
    if is_required(element_residual_func,"constitutive_model"):
        resid_args.append(constitutive_model)
        in_axes.append(None)

    # A vmap'ed version of the element residual function that maps over the elements
    R_vmap = jax.vmap(
        element_residual_func,
        in_axes=tuple(in_axes),
    )

    R_enu, internal_state = R_vmap(*resid_args)

    return R_enu, internal_state


def calculate_residual_wo_constraints(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
    u_f: jnp.ndarray,
):
    """
    Calculates the residual and updated internal state variables without any modification
    of the solution or residual to accomodate constraints.

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    u_f                   : current solution (displacement), ndarray[float, (V * D)]

    Returns
    -------
    R_f                     : residual vector evaluated at the solution, ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    """
    # TODO change the pattern to accept donated arrays to hold R_f and new_internal_state_beqi

    # NOTE This could be slow, measure.  To speed up this section, it might help to
    # add a transform to a batch-level unraveled residual vector and accumulate those,
    # since that operation could be JIT compiled. Then you could loop over the batch level
    # and accumulate them into the global with one more batch-to-global transform.

    result = [
        _calculate_residual_wo_constraints_batch(
            element_residual_func=element_residual_func,
            constitutive_model=ebc.constitutive_models[i],
            material_params=ebc.get_material_params(i),
            internal_state=ebc.get_internal_state(i),
            x_end=ebc.get_x(i),
            phi_qn=ebc.get_phi(i),
            dphi_dxi_qnp=ebc.get_dphi_dxi(i),
            W_q=ebc.get_weights(i),
            assembly_map=assembly_map_b[i],
            u_f=u_f,
        )
        for i in range(ebc.B)
    ]  # for each item, 0: R_end, 1: internal_state

    R_f = jnp.zeros_like(u_f)
    for i in range(ebc.B):
        R_f += transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b[i], v_en=result[i][0]
        )

    new_internal_state_beqi = [result[i][1] for i in range(ebc.B)]
    # TODO split this out into a separate call

    # NOTE here is an alternative implementation leveraging fori, but the index i is a traced
    # array and therefore cannot be used to index into the lists, such as a constitutive_model_b.
    # Keeping this implementation here to revisit for optimization.
    """
    def fori_body(i, R_f) -> jnp.ndarray:
        R_enu, internal_state = _calculate_residual_wo_dirichlet_batch(
            element_residual_func=element_residual_func,
            constitutive_model=constitutive_model_b[i],
            material_params=material_params_beqm[i],
            internal_state=internal_state_beqi[i],
            x_end=x_bend[i],
            dphi_dxi_qnp=dphi_dxi_bqnp[i],
            W_q=W_bq[i],
            assembly_map=assembly_map_b[i],
            u_f=u_f,
        )
        return R_f + transform_element_node_to_global_unraveled_sum(
            assembly_map=assembly_map_b[i], v_en=R_enu
        )

    R_f = jax.lax.fori_loop(
        lower=0, upper=B, body_fun=fori_body, init_val=jnp.zeros_like(u_f), unroll=True
    )
    """

    return R_f, new_internal_state_beqi


def calculate_residual_w_constraints(
    u_f: jnp.ndarray,
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
    constraints: ConstraintSystem,
    f_ext,
):
    """
    Compute the residual vector and updated internal state variables given the current
    solution and state information with constraints applied.

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    u_f                   : current solution (displacement), ndarray[float, (V * D)]
    constraints           : system of linear constraints (MultiPointConstraints and Dirichlet BCs)

    Returns
    -------
    R_f                     : residual vector evaluated at the solution with constraints applied,
                              ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    """
    # Note: this is neccessary to ensure the Jacobian is symmetric. Without this,
    # the autodiff would result in 0's on rows (except on the diagonal) for entries
    # corresponding to Dirichlet BC's, but the columns would be non-zero.
    # debug_print(u_f)_calculate_jacobian_unique_nnz
    u_f_w_constraints = constraints.apply_to_solution(u_f)
    # debug_print(u_f_w_constraints)

    R_f, new_internal_state_beqi = calculate_residual_wo_constraints(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        u_f=u_f_w_constraints,
    )

    # Zero out terms corresponding to Dirichlet BCs and add (solution - what it should be) for those constrained DoFs.
    # This will ensure there will be a 1 on the diagonal of the Jacobian and also return the right residual.
    # debug_print(R_f)
    R_f = f_ext.apply_to_residual(R_f)
    R_f = constraints.apply_to_residual(R_f, u_f)
    # debug_print(R_f)

    return R_f, new_internal_state_beqi


def __extract_first_element(func, u_f):
    """Executes the function and extracts the first element of the returned tuple."""
    return func(u_f=u_f)[0]


def solve_nonlinear_step(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
    jacobian_nnz: int,
    u_0_g: jnp.ndarray,
    constraints: ConstraintSystem,
    solver_options: SolverOptions,
    f_ext,
    resource_owner=None,
):
    """
    Solve the linearized system of equations emerging from the governing equations.
    This can be used within an outer loop to solve linear PDEs across time steps with different
    boundary conditions or to solve a nonlinear problem (via Newton's method for example).

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    jacobian_nnz          : number of non-zeros in the Jacobian matrix
    u_0_g                 : initial solution guess, ndarray[float, (V * D)]
    constraints           : system of linear constraints (MultiPointConstraints and Dirichlet BCs)
    solver_options        : options for the linear and nonlinear solvers

    Returns
    -------
    u_f                     : solution (displacement), ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    R_f                     : residual vector evaluated at the solution, ndarray[float, (V * D)]
    relative_error          : final relative error (L2 norm)
    info                    : solver result information
    """

    # Helpful for debugging array shapes
    # """
    print(f"Global dimensionality : {ebc.D}")
    print(f"# of batches : {ebc.B}")
    # for i in range(ebc.B):
    #     print(
    #         f"For batch {i}:\n\t",
    #         f"Number of elements : {ebc.E[i]}\n\t",
    #         f"Number of nodes / element : {ebc.N[i]}\n\t",
    #         f"Number of quadrature points : {ebc.Q[i]}\n\t",
    #         f"Parametric dimensionality: {ebc.P[i]}\n\t",
    #         f"Number of material parameters per quad point: {ebc.M[i]}",
    #     )
    # """

    # Function that produces (R(u), ISVs)
    residual_isv_func_w_constraints = jax.jit(
        partial(
            calculate_residual_w_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            constraints=constraints,
            f_ext=f_ext,
        )
    )

    # Function that produces R(u)
    residual_func_w_constraints = jax.jit(
        partial(__extract_first_element, residual_isv_func_w_constraints)
    )

    # Function that produces J(u) without Dirichlet BCs and MPCs applied
    jacobian_func_wo_constraints = jax.jit(
        partial(
            calculate_jacobian_wo_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            precomputed_jacobian_nnz=jacobian_nnz,
        )
    )

    # Function that produces diag(J(u)) without Dirichlet BCs and MPCs applied
    jacobian_diag_func_wo_constraints = jax.jit(
        partial(
            calculate_jacobian_diag_wo_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
        )
    )

    R_f, new_internal_state_beqi = residual_isv_func_w_constraints(u_f=u_0_g)
    initial_R_f_norm = jnp.linalg.norm(R_f)

    if solver_options.linear_diagnostics:
        jax.debug.print("BVP_JAX_NEWTON_INITIAL residual={r}", r=initial_R_f_norm, ordered=True)

    def while_cond(args) -> bool:
        nl_iteration, u_f, R_f, new_internal_state_beqi, info = args
        absolute_error = jnp.linalg.norm(R_f)
        relative_error = absolute_error / initial_R_f_norm
        """
        jax.debug.print(
            "End of iteration {x} rel error {y}, abs error {z}",
            x=nl_iteration - 1,
            y=relative_error,
            z=absolute_error,
        )
        jax.debug.print(
            "Convergence criteria: {} {} {}",
            nl_iteration < solver_options.nonlinear_max_iter,
            relative_error > solver_options.nonlinear_relative_tol,
            absolute_error > solver_options.nonlinear_absolute_tol,
        )
        """
        return (
            (nl_iteration < solver_options.nonlinear_max_iter)
            & jnp.isfinite(absolute_error)
            & jnp.isfinite(initial_R_f_norm)
            & (relative_error > solver_options.nonlinear_relative_tol)
            & (absolute_error > solver_options.nonlinear_absolute_tol)
        )

    print(
        "WARNING: If using a solver that requires a Jacobian, Dirichlet BCs are being applied but multi-point constraints are NOT."
    )

    def while_body(
        args: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo]:
        nl_iteration, u_f, R_f, new_internal_state_beqi, info = args

        delta_u, info = linear_solve(
            residual=Residual(
                function=jax.tree_util.Partial(residual_func_w_constraints), #why is this being repassed? 
                dirichlet_bcs_builtin=True,
            ),
            jacobian=Jacobian(
                function=jax.tree_util.Partial(jacobian_func_wo_constraints), #why is this being repassed?
                dirichlet_bcs_builtin=False,
            ),
            jacobian_diagonal=JacobianDiagonl(
                function=jax.tree_util.Partial(jacobian_diag_func_wo_constraints), #why is this being repassed?
                dirichlet_bcs_builtin=False,
            ),
            constraints=constraints,
            solver_options=solver_options,
            solver_info_0=info,
            check_consistency=False,
            x_0=u_f,
            f_ext=f_ext,
            resource_owner=resource_owner,
        )

        u_f = u_f + delta_u
        # u_f = constraints.apply_to_solution(u_f)
        R_f, new_internal_state_beqi = residual_isv_func_w_constraints(u_f=u_f)

        if solver_options.linear_diagnostics:
            jax.debug.print(
                "BVP_JAX_NEWTON_STEP step={step} residual={r} relative_residual={rel} "
                "correction_norm={du} finite={finite}",
                step=nl_iteration + 1, r=jnp.linalg.norm(R_f),
                rel=jnp.linalg.norm(R_f) / initial_R_f_norm,
                du=jnp.linalg.norm(delta_u), finite=jnp.all(jnp.isfinite(u_f)), ordered=True,
            )

        return (
            nl_iteration + 1,
            u_f,
            R_f,
            new_internal_state_beqi,
            info.increment_nl_iteration(),
        )

    _, u_f, R_f, new_internal_state_beqi, info = jax.lax.while_loop(
        cond_fun=while_cond,
        body_fun=while_body,
        init_val=(
            0,
            u_0_g,
            R_f,
            new_internal_state_beqi,
            init_solver_info(solver_options),
        ),
    )

    absolute_error = jnp.linalg.norm(R_f)
    relative_error = absolute_error / initial_R_f_norm

    return (u_f, new_internal_state_beqi, R_f, relative_error, info)


@dataclass
class NeumannCondition:
    """
    Represents a force of value applied at DoF.
    """

    dep_dof: int
    value: float


def convert_boundary_conditions_to_external_load(
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC],
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    dof_enumeration: DofEnumeration,
    n_solution_components: int,
    global_values: List[int] | None = None,
):
    """
    Searches the list of boundary conditions and converts the Neumann
    conditions to data type NeumannCondition.
    """
    external_load = []
    for bc in boundary_conditions:
        if isinstance(bc, NeumannBC):
            if bc.bc_type == BCType.NODE:
                external_load.append(
                    NeumannCondition(
                        dep_dof=n_solution_components * bc.index + bc.component,
                        value=bc.value,
                    )
                )
    return external_load


@struct.dataclass
class LoadSystem:
    dep_dofs: jnp.ndarray
    loads: jnp.ndarray

    @jax.jit
    def apply_to_residual(self, R: jnp.ndarray):
        return R.at[self.dep_dofs].set((R[self.dep_dofs] - self.loads))


def convert_external_load_to_system(
    external_load,
):
    n_loads = len(external_load)
    if n_loads == 0:
        return LoadSystem(
            dep_dofs=jnp.array([], dtype=jnp.int32),
            loads=jnp.array([], dtype=jnp.float32),
        )

    dep_dofs = np.empty(n_loads, dtype=np.int32)
    loads_list = []

    for i, el in enumerate(external_load):
        dep_dofs[i] = el.dep_dof
        loads_list.append(jnp.array(el.value))

    loads = jnp.stack(jnp.broadcast_arrays(*loads_list), axis=-1)

    return LoadSystem(
        dep_dofs=jnp.array(dep_dofs, dtype=jnp.int32),
        loads=loads,
    )


def convert_boundary_conditions(
    boundary_conditions: List[DirichletBC | PeriodicBC],
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    dof_enumeration: DofEnumeration,
    n_solution_components: int,
    global_values: List[int] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
):
    fixed_point_constraints, boundary_multipoint_constraints = (
        convert_boundary_conditions_to_constraints(
            boundary_conditions=boundary_conditions,
            vertices_vd=vertices_vd,
            dof_enumeration=dof_enumeration,
            n_solution_components=n_solution_components,
            global_values=global_values,
        )
    )
    multipoint_constraints = consolidate_multipoint_constraints(
        fixed_point_constraints=fixed_point_constraints,
        multipoint_constraints=[
            *boundary_multipoint_constraints,
            *multipoint_constraints,
        ],
    )

    constraint_system = convert_constraints_to_system(
        fixed_point_constraints=fixed_point_constraints,
        multipoint_constraints=multipoint_constraints,
        n_total_dofs=dof_enumeration.n_local_dofs(),
    )

    external_load = convert_boundary_conditions_to_external_load(
        boundary_conditions=boundary_conditions,
        vertices_vd=vertices_vd,
        dof_enumeration=dof_enumeration,
        n_solution_components=n_solution_components,
        global_values=global_values,
    )

    f_ext = convert_external_load_to_system(external_load)

    return (constraint_system, f_ext)


def preprocess_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    jacobian_nnz: int | None = None,
):
    """
    Converts information from a user-facing format to a JAX-ameniable format.
    """
    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    # For 1D problems, the vertices may be given as a 1D array, so we need to reshape it to a 2D array
    if vertices_vd.ndim == 1:
        vertices_vd = np.expand_dims(vertices_vd, axis=1)

    B = len(element_batches)
    V = vertices_vd.shape[0]
    D = vertices_vd.shape[1]
    U = element_batches[0].n_dofs_per_basis
    n_total_dofs = V * U + sum(global_values)

    # Validate input
    assert D <= 3
    assert len(boundary_conditions) <= D * V
    for b in element_batches:
        assert b.connectivity_en.shape[1] <= V
    for b in element_batches:
        assert (
            b.n_dofs_per_basis == element_batches[0].n_dofs_per_basis
        ), "The current DoF enumeration algorithm requires that the number of DoFs per a basis support point be constant across batches."

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Enumerate degrees of freedom
    # NOTE: this currently assumes that the element_batches contains ALL elements
    # that will exist on this rank for the respective solve. If this is not the case,
    # we will need to construct the enumeration at a point where all element information
    # is known and pass it into this function.
    # NOTE assertion above ensures U is constant across batches
    dof_enumeration = DofEnumeration(
        n_owned_elements=sum([b.connectivity_en.shape[0] for b in element_batches]),
        n_owned_dofs=n_total_dofs,
        n_local_ghost_dofs=0,
        n_exclusive_ghost_dofs=0,
        n_free_global_dofs=sum(global_values),
        free_global_dof_rank_begin=V * U,
        owned_global_dof_begin=0,
        owned_global_dof_end=n_total_dofs,
        rank_to_global_map=jnp.arange(n_total_dofs),
    )

    # Convert element batch information into something ameniable to JAX transforms like JIT
    ebc = batch_to_collection(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        dof_enumeration=dof_enumeration,
    )
    # print(ebc)

    assert (
        np.array(ebc.U) == ebc.U[0]
    ).all(), """The number of DoFs per a point (U) must be the same across all batches.
    To relax this constraint much of the infrastructure code in fea.py would have to be adapted to
    support varying number of DoFs per a batch.
    """

    # Structures for mapping between cell-level arrays and global arrays
    assembly_map_b = [
        mesh_to_sparse_assembly_map(n_vertices=V, cells=b.connectivity_en)
        for b in element_batches
    ]

    # Compute the anticipated number of non-zeros for the assembled Jacobian, which
    # is only needed for solvers that actually form the Jacobian in memory.
    # NOTE: we need a concrete value to specialize for JIT of other functions
    if jacobian_nnz is None:
        jacobian_nnz = int(
            _calculate_jacobian_unique_nnz(ebc=ebc)
        )  # n_vertices=V, ebc=ebc))

    constraint_system, f_ext = convert_boundary_conditions(
        boundary_conditions=boundary_conditions,
        vertices_vd=vertices_vd,
        dof_enumeration=dof_enumeration,
        n_solution_components=ebc.U[0],
        global_values=global_values,
        multipoint_constraints=multipoint_constraints,
    )

    return (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    )

@with_forward_solver_resources
def solve_bvp(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    solver_options: SolverOptions = SolverOptions(),
    plot_convergence: bool = False,
    profile_memory: bool = False,
    _resource_owner=None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[ElementBatch]]:
    """
    Solve a boundary value problem for static linear elasticity.

    Parameters
    ----------
    vertices_vd          : vertices needed for all cells on the rank, ndarray[float, (V, D)]
    element_batches      : batch of elements for this rank
    element_residual_func: residual function emerging from weak form of governing equations
    dirichlet_bcs        : Dirichlet boundary conditions, list[DirichletConstraint]
    multipoint_constraints : Linear constraints between degrees of freedom, list[MultiPointConstraint]
    global_values        : Length of list indicates number of global solution vector-values that will
                           added to the global system (e.g. for periodic BCs). Each entry in the list
                           indicates the number of components for each vector-value.
    u_0_g                : initial guess for the solution, ndarray[float, (V * D)] or None (default, zeros will be used)
    solver_options       : options for the linear/nonlinear solvers
    plot_convergence     : indicates if the convergence history for the linear solver should be
                           plotted via matplotlib as a figure
    profile_memory       : indicates if GPU memory usage should be profiled, which will create *.prof
                           files in the current directory

    Returns
    -------
    u               : solution (displacement), ndarray[float, (V * D)]
    R               : residual vector evaluated at the solution, ndarray[float, (V * D)]
    element_batches : element batches with updated internal state variables
    """
    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
    )

    n_total_dofs = vertices_vd.shape[0] * ebc.U[0] + sum(global_values)

    # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(n_total_dofs,))
    else:
        assert u_0_g.shape == (n_total_dofs,)

    # inner_solve = solve_nonlinear_step
    # if ebc.is_homogeneous:
    #     print("Batches are homogeneous, using JIT compilation for solve_linear_step")
    inner_solve = jax.jit(
        solve_nonlinear_step,
        # donate_argnames="internal_state_beqi",
        static_argnames=["solver_options", "jacobian_nnz"],
    )

    # capture memory usage before
    if profile_memory:
        start_memory_profile("solve_linear_step")

    u, internal_state_beqi, residual, relative_error, info = inner_solve(
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        jacobian_nnz=jacobian_nnz,
        u_0_g=u_0_g,
        constraints=constraint_system,
        solver_options=solver_options,
        f_ext=f_ext,
        resource_owner=_resource_owner,
    )

    if not isinstance(residual, jax.core.Tracer):
        final_norm = float(jnp.linalg.norm(residual))
        if not np.isfinite(final_norm):
            raise FloatingPointError(
                "solve_bvp stopped because the nonlinear residual norm is nonfinite "
                "(including possible norm overflow). No more Newton corrections will "
                "be attempted. Inspect the first failed linear solve and ILU factors; "
                "this result must not be accepted as a converged benchmark."
            )

    # Update internal state variables for the element batches
    # TODO need to update
    # for i, b in enumerate(element_batches):
    # #    b.internal_state = internal_state_beqi[i]
    #    b = b.replace(internal_state=internal_state_beqi[i])

    # What Chennie did for last summer, it worked but not well tested
    for i in range(len(element_batches)):
        element_batches[i] = element_batches[i].replace(
            internal_state=internal_state_beqi[i]
        )

    # capture memory usage after and analyze
    if profile_memory:
        u.block_until_ready()
        stop_memory_profile("solve_linear_step")

    if info.cumulative_linear_iterations > 0:
        print(
            f"Cumulative # of linear solver iterations: {info.cumulative_linear_iterations}"
        )
        if plot_convergence:
            plot_solver_info(opts=solver_options, info=info)

    return (u, residual, element_batches)


def solve_nonlinear_quasi_step(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[jsparse.BCSR],
    jacobian_nnz: int,
    u_0_g: jnp.ndarray,
    constraints: ConstraintSystem,
    solver_options: SolverOptions,
    f_ext,
):
    """
    #TODO this can likely be merged with solve_nonlinear_step. The main diference is here we use scan with a jax.cond
    the reason is reverse auto-diff has issues with while cond. 
    
    #-------------------------------------------
    Solve the linearized system of equations emerging from the governing equations.
    This can be used within an outer loop to solve linear PDEs across time steps with different
    boundary conditions or to solve a nonlinear problem (via Newton's method for example).

    Parameters
    ----------
    element_residual_func : residual function emerging from weak form of governing equations
    ebc                   : collection of element batches containing mesh, property, and state information
    assembly_map_b        : list of assembly maps for each element batch
    jacobian_nnz          : number of non-zeros in the Jacobian matrix
    u_0_g                 : initial solution guess, ndarray[float, (V * D)]
    constraints           : system of linear constraints (MultiPointConstraints and Dirichlet BCs)
    solver_options        : options for the linear and nonlinear solvers

    Returns
    -------
    u_f                     : solution (displacement), ndarray[float, (V * D)]
    new_internal_state_beqi : updated internal state variables for each element batch
    R_f                     : residual vector evaluated at the solution, ndarray[float, (V * D)]
    relative_error          : final relative error (L2 norm)
    info                    : solver result information
    """

    # Helpful for debugging array shapes
    # """
    print(f"Global dimensionality : {ebc.D}")
    print(f"# of batches : {ebc.B}")
    # for i in range(ebc.B):
    #     print(
    #         f"For batch {i}:\n\t",
    #         f"Number of elements : {ebc.E[i]}\n\t",
    #         f"Number of nodes / element : {ebc.N[i]}\n\t",
    #         f"Number of quadrature points : {ebc.Q[i]}\n\t",
    #         f"Parametric dimensionality: {ebc.P[i]}\n\t",
    #         f"Number of material parameters per quad point: {ebc.M[i]}",
    #     )
    # """

    # Function that produces (R(u), ISVs)
    residual_isv_func_w_constraints = jax.jit(
        partial(
            calculate_residual_w_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            constraints=constraints,
            f_ext=f_ext,
        )
    )

    # Function that produces R(u)
    residual_func_w_constraints = jax.jit(
        partial(__extract_first_element, residual_isv_func_w_constraints)
    )

    # Function that produces J(u) without Dirichlet BCs and MPCs applied
    jacobian_func_wo_constraints = jax.jit(
        partial(
            calculate_jacobian_wo_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            precomputed_jacobian_nnz=jacobian_nnz,
        )
    )

    # Function that produces diag(J(u)) without Dirichlet BCs and MPCs applied
    jacobian_diag_func_wo_constraints = jax.jit(
        partial(
            calculate_jacobian_diag_wo_constraints,
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
        )
    )

    R_f, new_internal_state_beqi = residual_isv_func_w_constraints(u_f=u_0_g)
    initial_R_f_norm = jnp.linalg.norm(R_f)

    print(
        "WARNING: If using a solver that requires a Jacobian, Dirichlet BCs are being applied but multi-point constraints are NOT."
    )

    def do_step(
        args: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo, Any],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo, Any]:
        nl_iteration, u_f, R_f, new_internal_state_beqi, info, _ = args

        delta_u, info = linear_solve(
            residual=Residual(
                function=jax.tree_util.Partial(residual_func_w_constraints),
                dirichlet_bcs_builtin=True,
            ),
            jacobian=Jacobian(
                function=jax.tree_util.Partial(jacobian_func_wo_constraints),
                dirichlet_bcs_builtin=False,
            ),
            jacobian_diagonal=JacobianDiagonl(
                function=jax.tree_util.Partial(jacobian_diag_func_wo_constraints),
                dirichlet_bcs_builtin=False,
            ),
            constraints=constraints,
            solver_options=solver_options,
            solver_info_0=info,
            check_consistency=False,
            x_0=u_f,
            f_ext=f_ext,
        )

        u_f_new = u_f + delta_u
        # u_f_new = constraints.apply_to_solution(u_f_new)
        R_f_new, new_internal_state_beqi = residual_isv_func_w_constraints(u_f=u_f_new)

        absolute_error = jnp.linalg.norm(R_f_new)
        relative_error = absolute_error / initial_R_f_norm
        converged = (relative_error <= solver_options.nonlinear_relative_tol) | (
            absolute_error <= solver_options.nonlinear_absolute_tol
        )

        return (
            nl_iteration + 1,
            u_f_new,
            R_f_new,
            new_internal_state_beqi,
            info.increment_nl_iteration(),
            converged,
        )

    def skip_step(
        args: tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo, Any],
    ) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverResultInfo, Any]:
        return args

    def scan_body(carry, _):
        converged = carry[5]
        new_carry = jax.lax.cond(converged, skip_step, do_step, carry)
        return new_carry, None

    initial_converged = initial_R_f_norm <= solver_options.nonlinear_absolute_tol
    init_val = (
        0,
        u_0_g,
        R_f,
        new_internal_state_beqi,
        init_solver_info(solver_options),
        initial_converged,
    )

    final_val, _ = jax.lax.scan(
        f=jax.checkpoint(scan_body),
        init=init_val,
        xs=None,
        length=solver_options.nonlinear_max_iter,
    )

    _, u_f, R_f, new_internal_state_beqi, info, _ = final_val

    absolute_error = jnp.linalg.norm(R_f)
    relative_error = absolute_error / initial_R_f_norm

    return (u_f, new_internal_state_beqi, R_f, relative_error, info)

def calculate_element_averaged_state(internal_state_beqi):
    """
    Averages the internal state of every element over its quadrature points.

    For each batch the output holds (strain 3, stress 3, damage 1) per element, with the
    damage applied to the stress. Fiber batches (< 8 ISVs) get no stress reduction and a
    damage marker of -0.2 for plotting.

    Parameters
    ----------
    internal_state_beqi : sequence (one entry per batch) of arrays with shape (E, Q, I)

    Returns
    -------
    ISV_be : list (one entry per batch) of arrays with shape (E, 7)
    """
    ISV_be = []
    for idx in range(len(internal_state_beqi)):
        state_q = internal_state_beqi[idx]
        
        strain_q = state_q[:, :, 0:3]
        stress_eff_q = state_q[:, :, 3:6]
        damage_q = state_q[:, :, 6]
        # Apply damage to stress only for the matrix (11 ISVs). Fibers (7 ISVs) store D = -0.2
        # as a plotting marker, which must not scale their stress. The shape is static, so a
        # Python if is used: an eager lax.cond with fresh lambdas recompiles on every call.
        if state_q.shape[-1] < 8:
            stress_dmg_q = stress_eff_q
            damage_q = damage_q*0 - 0.2
        else:
            stress_dmg_q = stress_eff_q * (1 - damage_q[:, :, jnp.newaxis])
        
        avg_strain = jnp.mean(strain_q, axis=1)
        avg_stress_dmg = jnp.mean(stress_dmg_q, axis=1)
        avg_damage = jnp.mean(damage_q, axis=1)
        
        avg_state = jnp.concatenate([avg_strain, avg_stress_dmg, avg_damage[:, jnp.newaxis]], axis=1)
        ISV_be.append(avg_state)
        
    return ISV_be


def _bc_time_series(constraint_system: ConstraintSystem, f_ext):
    """
    Returns the Dirichlet offsets g and external loads as (T, ...) arrays with one row per time step.
    A BC without a time series is broadcast to every step.
    """
    all_g = constraint_system.g
    if all_g.ndim == 1:
        all_g = all_g[None, :]

    all_loads = f_ext.loads
    if all_loads.ndim == 1:
        all_loads = jnp.broadcast_to(all_loads[None, :], (all_g.shape[0], all_loads.shape[0]))

    if all_loads.shape[0] > all_g.shape[0]:
        if all_g.shape[0] == 1:
            all_g = jnp.broadcast_to(all_g, (all_loads.shape[0], all_g.shape[1]))
        else:
            raise ValueError("Inconsistent time steps between Dirichlet and Neumann BCs")

    return all_g, all_loads


def solve_bvp_quasi(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    solver_options: SolverOptions = SolverOptions(),
    profile_memory: bool = False,
    jacobian_nnz: int | None = None,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    Solve a quasi-static boundary value problem across time steps using `jax.lax.scan`.

    Parameters
    ----------
    vertices_vd          : vertices needed for all cells on the rank, ndarray[float, (V, D)]
    element_batches      : batch of elements for this rank
    element_residual_func: residual function emerging from weak form of governing equations
    boundary_conditions  : Dirichlet, Neumann, or Periodic boundary conditions (can have array values for time series)
    multipoint_constraints : Linear constraints between degrees of freedom, list[MultiPointConstraint]
    global_values        : Length of list indicates number of global solution vector-values that will
                           added to the global system (e.g. for periodic BCs).
    u_0_g                : initial guess for the solution at the first step, ndarray[float, (V * D)] or None (default, zeros will be used)
    solver_options       : options for the linear/nonlinear solvers
    profile_memory       : indicates if GPU memory usage should be profiled, which will create *.prof
                           files in the current directory

    Returns
    -------
    u_history      : displacement solution history across time steps, ndarray[float, (T, V * D)]
    ISV_be_history : averaged internal state variable history across time steps for each element batch
    """
    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        # jacobian_nnz=jacobian_nnz,
    )

    n_total_dofs = vertices_vd.shape[0] * ebc.U[0] + sum(global_values)

    # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(n_total_dofs,))
    else:
        assert u_0_g.shape == (n_total_dofs,)

    solve_nonlinear_step_jit = jax.jit(
        solve_nonlinear_step,
        static_argnames=["solver_options", "jacobian_nnz"],
    )

    all_g, all_loads = _bc_time_series(constraint_system, f_ext)

    def step_fn(carry, xs):
        u_prev, ebc_curr = carry
        g_i, loads_i = xs
        
        constraint_system_i = constraint_system.replace(g=g_i)
        f_ext_i = f_ext.replace(loads=loads_i)
        
        u, new_internal_state_beqi, residual, relative_error, info = solve_nonlinear_step_jit(
            element_residual_func=element_residual_func,
            ebc=ebc_curr,
            assembly_map_b=assembly_map_b,
            jacobian_nnz=jacobian_nnz,
            u_0_g=u_prev,
            constraints=constraint_system_i,
            solver_options=solver_options,
            f_ext=f_ext_i,
        )

        flat_internal_state = jnp.hstack([isv.ravel() for isv in new_internal_state_beqi])
        ebc_next = ebc_curr.replace(internal_state=flat_internal_state)
        ISV_be = calculate_element_averaged_state(new_internal_state_beqi)
        
        return (u, ebc_next), (u, ISV_be)

    # capture memory usage before
    if profile_memory:
        start_memory_profile("solve_bvp_quasi")

    init_carry = (u_0_g, ebc)
    final_carry, (u_history, ISV_be_history) = jax.lax.scan(step_fn, init_carry, (all_g, all_loads))

    # capture memory usage after and analyze
    if profile_memory:
        u_history.block_until_ready()
        stop_memory_profile("solve_bvp_quasi")

    return (u_history, ISV_be_history)




def solve_bvp_PETSc(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    diagnostics: bool = False,
    petsc_solver_options: jetsci.SolverOptions | None = None,
    destroy_solver: bool = True,
    return_petsc_solver_options: bool = False,
    ):

    solve, phi, u_0_g, residual_for_phi, options, residual_isv_for_phi = build_differentiable_bvp_PETSc_solve(        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        u_0_g=u_0_g,
        diagnostics=diagnostics,
        petsc_solver_options=petsc_solver_options,
    )

    solver_key = options.solver_key
    try:
        output = solve(phi=phi,x0=u_0_g)
        _, new_internal_state_beqi = residual_isv_for_phi(phi, output)
        for i in range(len(element_batches)):
            element_batches[i] = element_batches[i].replace(
                internal_state=new_internal_state_beqi[i]
            )
        residual_at_output = residual_for_phi(output)
        if return_petsc_solver_options:
            return output, residual_at_output, element_batches, options
        return output, residual_at_output, element_batches
    finally:
        if solver_key is not None:
            petsc_snes.differentiable_snes.unregister_primitive_context(solver_key)
        if destroy_solver and solver_key is not None:
            petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)


def build_differentiable_bvp_PETSc_solve(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    diagnostics: bool = False,
    petsc_solver_options: jetsci.SolverOptions | None = None,
    ):
    """Build a differentiable PETSc-backed BVP solve closure.

    Returns `(solve, phi, x0, residual_for_phi, options)`, where `solve(phi, x0)`
    is the custom-JVP SNES primitive and `phi` is the flattened material
    parameter vector from the preprocessed element batch collection.
    """

    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []
    
    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
            vertices_vd=vertices_vd,
            element_batches=element_batches,
            element_residual_func=element_residual_func,
            boundary_conditions=boundary_conditions,
            multipoint_constraints=multipoint_constraints,
            global_values=global_values,
        )
    
    n_total_dofs = vertices_vd.shape[0] * ebc.U[0] + sum(global_values)
    
        # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(n_total_dofs,))
    else:
        assert u_0_g.shape == (n_total_dofs,)

    residual, jacobian, residual_isv_for_phi = build_nonlinear_objects(
            element_residual_func=element_residual_func,
            ebc=ebc,
            assembly_map_b=assembly_map_b,
            jacobian_nnz=jacobian_nnz,
            u_0_g=u_0_g,
            constraints=constraint_system,
            f_ext=f_ext)

    phi = ebc.material_params
    residual_for_phi = jax.tree_util.Partial(residual, phi)
    jacobian_for_phi = jax.tree_util.Partial(jacobian, phi)

    options = petsc_solver_options or _default_petsc_solver_options()

    solver, options = petsc_snes.solver_lifecycle.build_petsc_solver_with_reuse(
            options,
            residual_for_phi,
            jacobian_for_phi,
        )
    solver.diagnostics = diagnostics
    if solver.callback_stats is not None:
        solver.callback_stats["collect_diagnostics"] = diagnostics

    
    primitive = differentiable_snes.DifferentiableSNESPrimitive(
            residual=residual,
            jacobian=jacobian,
            solver_key=options.solver_key,
        )
    solve = petsc_snes.differentiable_snes.make_differentiable_snes_solve(primitive)
    return solve, phi, u_0_g, residual_for_phi, options, residual_isv_for_phi


def _default_petsc_solver_options():
    return jetsci.SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.PETSC_SNES,
        linear_precond_type=jetsci.PETScPreconditionerType.JACOBI,
        linear_solve_type=jetsci.PETScLinearSolverType.CG,
        nonlinear_absolute_tol=1e-14,
        linear_max_iter=5000,
        linear_relative_tol=1e-6,
        linear_absolute_tol=1e-14,
    )


def solve_bvp_quasi_PETSc(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    diagnostics: bool = False,
    petsc_solver_options: jetsci.SolverOptions | None = None,
    jacobian_nnz: int | None = None,
    destroy_solver: bool = True,
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    PETSc SNES counterpart of `solve_bvp_quasi`: solve a quasi-static boundary value problem
    across the time steps of the boundary conditions.

    Returns
    -------
    u_history      : displacement solution history across time steps, ndarray[float, (T, V * D)]
    ISV_be_history : averaged internal state variable history across time steps for each element batch
    """
    run, material_params, options = build_differentiable_bvp_quasi_PETSc_solve(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        u_0_g=u_0_g,
        diagnostics=diagnostics,
        petsc_solver_options=petsc_solver_options,
        jacobian_nnz=jacobian_nnz,
    )

    solver_key = options.solver_key
    try:
        return run(material_params)
    finally:
        if solver_key is not None:
            petsc_snes.differentiable_snes.unregister_primitive_context(solver_key)
        if destroy_solver and solver_key is not None:
            petsc_snes.solver_lifecycle.destroy_petsc_solver(solver_key)


def build_differentiable_bvp_quasi_PETSc_solve(
    vertices_vd: np.ndarray[Any, np.dtype[np.floating[Any]]],
    element_batches: list[ElementBatch],
    element_residual_func: jax.tree_util.Partial,
    boundary_conditions: List[DirichletBC | NeumannBC | PeriodicBC] | None = None,
    multipoint_constraints: List[MultiPointConstraint] | None = None,
    global_values: List[int] | None = None,
    u_0_g: jnp.ndarray | None = None,
    diagnostics: bool = False,
    petsc_solver_options: jetsci.SolverOptions | None = None,
    jacobian_nnz: int | None = None,
    ):
    """Build a differentiable PETSc-backed quasi-static solve over all time steps.

    Returns `(run, material_params, options)`, where `run(material_params)` solves every
    time step with one reused PETSc SNES and returns `(u_history, ISV_be_history)` like
    `solve_bvp_quasi`. `material_params` is the flattened parameter vector of the
    preprocessed element batch collection and `options.solver_key` owns the PETSc solver.

    Each step calls the differentiable SNES primitive with
    phi = [material params, ISVs of the previous step, Dirichlet offsets g_i, external loads_i].
    Everything a step depends on is in phi, so the residual/Jacobian are compiled once and
    the implicit-function sensitivities carry through the ISV history and the BC values.

    NOTE: the JetSCI primitive has no lowering rule, so the time loop is a Python loop
    instead of jax.lax.scan and `run` cannot be wrapped in jax.jit.
    """

    if boundary_conditions is None:
        boundary_conditions = []
    if multipoint_constraints is None:
        multipoint_constraints = []
    if global_values is None:
        global_values = []

    (
        ebc,
        assembly_map_b,
        constraint_system,
        jacobian_nnz,
        element_residual_func,
        f_ext,
    ) = preprocess_bvp(
        vertices_vd=vertices_vd,
        element_batches=element_batches,
        element_residual_func=element_residual_func,
        boundary_conditions=boundary_conditions,
        multipoint_constraints=multipoint_constraints,
        global_values=global_values,
        jacobian_nnz=jacobian_nnz,
    )

    n_total_dofs = vertices_vd.shape[0] * ebc.U[0] + sum(global_values)

    # If an initial guess was not provided, then use zeros
    if u_0_g is None:
        u_0_g = jnp.zeros(shape=(n_total_dofs,))
    else:
        assert u_0_g.shape == (n_total_dofs,)

    all_g, all_loads = _bc_time_series(constraint_system, f_ext)
    ebc = _commit_to_petsc_device(ebc)

    # Offsets of the material params, ISVs and g within phi (loads take the rest)
    phi_splits = np.cumsum(
        [ebc.material_params.size, ebc.internal_state.size, all_g.shape[1]]
    ).tolist()

    def unpack(phi):
        material_params, internal_state, g, loads = jnp.split(phi, phi_splits)
        return (
            ebc.replace(material_params=material_params, internal_state=internal_state),
            constraint_system.replace(g=g),
            f_ext.replace(loads=loads),
        )

    def pack(material_params, internal_state, g, loads):
        return jnp.concatenate([material_params, internal_state, g, loads])

    # Function that produces (R(u), ISVs)
    def residual_isv(phi, x):
        ebc_i, constraints_i, f_ext_i = unpack(phi)
        return _residual_isv_w_constraints_jit(
            u_f=x,
            element_residual_func=element_residual_func,
            ebc=ebc_i,
            assembly_map_b=assembly_map_b,
            constraints=constraints_i,
            f_ext=f_ext_i,
        )

    # PETSc hands over DLPack views of its own vectors, so copy before calling into JAX
    def residual(phi, x):
        return residual_isv(phi, jnp.array(x, copy=True))[0]

    # Function that produces J(u) with Dirichlet BCs applied (MPCs are NOT applied)
    def jacobian(phi, x):
        ebc_i, _, _ = unpack(phi)
        return _jacobian_w_dirichlet_jit(
            u_f=jnp.array(x, copy=True),
            element_residual_func=element_residual_func,
            ebc=ebc_i,
            assembly_map_b=assembly_map_b,
            precomputed_jacobian_nnz=jacobian_nnz,
            dep_dofs=constraint_system.dep_dofs,
        )

    phi_0 = pack(ebc.material_params, ebc.internal_state, all_g[0], all_loads[0])
    options = petsc_solver_options or _default_petsc_solver_options()

    solver, options = petsc_snes.solver_lifecycle.build_petsc_solver_with_reuse(
        options,
        jax.tree_util.Partial(residual, phi_0),
        jax.tree_util.Partial(jacobian, phi_0),
    )
    solver.diagnostics = diagnostics
    if solver.callback_stats is not None:
        solver.callback_stats["collect_diagnostics"] = diagnostics

    primitive = differentiable_snes.DifferentiableSNESPrimitive(
        residual=residual,
        jacobian=jacobian,
        solver_key=options.solver_key,
    )
    solve = petsc_snes.differentiable_snes.make_differentiable_snes_solve(primitive)

    def run(material_params):
        u = u_0_g
        internal_state = ebc.internal_state
        u_history, ISV_be_history = [], []
        for g_i, loads_i in zip(all_g, all_loads):
            phi = pack(material_params, internal_state, g_i, loads_i)
            u = solve(phi, u)
            _, new_internal_state_beqi = residual_isv(phi, u)
            internal_state = jnp.hstack([isv.ravel() for isv in new_internal_state_beqi])
            u_history.append(u)
            ISV_be_history.append(calculate_element_averaged_state(new_internal_state_beqi))

        u_history = jnp.stack(u_history)
        ISV_be_history = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *ISV_be_history)
        return u_history, ISV_be_history

    return run, ebc.material_params, options


# Jitted once at module level so every PETSc callback built below, e.g. once per load step,
# reuses the same compiled code. Everything that changes between calls (ebc, constraints, f_ext)
# must be passed as an argument: jitting a fresh closure misses the cache on every build, and
# arrays captured by a closure are baked into the program as constants.
_residual_isv_w_constraints_jit = jax.jit(calculate_residual_w_constraints)


@partial(jax.jit, static_argnames=("precomputed_jacobian_nnz",))
def _jacobian_w_dirichlet_jit(
    u_f, element_residual_func, ebc, assembly_map_b, precomputed_jacobian_nnz, dep_dofs
):
    J = calculate_jacobian_wo_constraints(
        u_f=u_f,
        element_residual_func=element_residual_func,
        ebc=ebc,
        assembly_map_b=assembly_map_b,
        precomputed_jacobian_nnz=precomputed_jacobian_nnz,
    )
    return apply_dirichlet_bcs_lhs(J, dep_dofs)


def _commit_to_petsc_device(tree):
    """
    PETSc returns solutions through DLPack as arrays committed to a device, so the ISVs computed
    from them are committed too. JAX keys compiled code on placement, so commit the initial
    state up front; otherwise the second step compiles everything again.
    """
    return jax.device_put(tree, jax.devices()[0])


def build_nonlinear_objects(
    element_residual_func: jax.tree_util.Partial,
    ebc: ElementBatchCollection,
    assembly_map_b: list[AssemblyMap],
    jacobian_nnz: int,
    u_0_g: jnp.ndarray,
    constraints: ConstraintSystem,
    f_ext,
    *args,
    differentiate_internal_state: bool = False,
    explicit_boundary_data: bool = False,
    **kwargs,):
    """Build reusable JAX residual/Jacobian/state-update functions.

    By default phi contains only material parameters. With
    differentiate_internal_state=True it contains [material, incoming_state].
    explicit_boundary_data=True appends [constraints.g, f_ext.loads] for one
    timestep, allowing the same function objects to serve an entire history.
    All sections are flat arrays. Fixed geometry, sparsity and constrained/load
    DOF indices remain in the closure; numerical offsets/loads do not.
    """

    # A history-dependent residual must expose its current constitutive state
    # as an operand of the custom solve, not capture a tracer in its closure.
    # Keep only static geometry/layout in that closure. Packing is opt-in to
    # preserve the existing material-only parameter convention.
    material_size = ebc.material_params.size
    state_size = ebc.internal_state.size if differentiate_internal_state else 0
    if differentiate_internal_state:
        ebc = ebc.replace(material_params=None, internal_state=None)
    elif explicit_boundary_data:
        ebc = ebc.replace(material_params=None)
    boundary_size = load_size = 0
    if explicit_boundary_data:
        if constraints.g.ndim != 1 or f_ext.loads.ndim != 1:
            raise ValueError(
                "explicit_boundary_data expects one-timestep vectors for "
                "constraints.g and f_ext.loads, not history arrays. Select "
                "a timestep when constructing the function template."
            )
        boundary_size, load_size = constraints.g.size, f_ext.loads.size
        constraints = constraints.replace(g=None)
        f_ext = f_ext.replace(loads=None)

    def check_parameters(phi):
        expected = material_size + state_size + boundary_size + load_size
        if phi.ndim != 1 or phi.size != expected:
            raise ValueError(
                f"Expected a flat nonlinear parameter vector of length {expected}, "
                f"received shape {phi.shape}. Concatenate material parameters, "
                "incoming state (if enabled), and boundary offsets/loads "
                "(if explicit_boundary_data is enabled)."
            )

    def ebc_with_material_params(phi):
        if differentiate_internal_state or explicit_boundary_data:
            check_parameters(phi)
        if differentiate_internal_state:
            return ebc.replace(
                material_params=phi[:material_size],
                internal_state=phi[material_size:material_size + state_size],
            )
        if explicit_boundary_data:
            return ebc.replace(material_params=phi[:material_size])
        return ebc.replace(material_params=jnp.asarray(phi))

    # Function that produces (R(u), ISVs)
    def residual_isv_func_w_constraints(phi, x):
        active_constraints, active_force = constraints, f_ext
        if explicit_boundary_data:
            offset = material_size + state_size
            active_constraints = constraints.replace(g=phi[offset:offset + boundary_size])
            active_force = f_ext.replace(loads=phi[offset + boundary_size:])
        return calculate_residual_w_constraints(
            u_f=x,
            element_residual_func=element_residual_func,
            ebc=ebc_with_material_params(phi),
            assembly_map_b=assembly_map_b,
            constraints=active_constraints,
            f_ext=active_force,
        )

    # PETSc hands over DLPack views of its own vectors, so copy before calling into JAX

    # Function that produces R(u) with Dirichlet BCs applied
    def R_w_dirichlet(phi, x):
        return residual_isv_func_w_constraints(phi, jnp.array(x, copy=True))[0]

    # Function that produces J(u) with Dirichlet BCs applied (MPCs are NOT applied)
    def J_w_dirichlet(phi, x):
        return _jacobian_w_dirichlet_jit(
            u_f=jnp.array(x, copy=True),
            element_residual_func=element_residual_func,
            ebc=ebc_with_material_params(phi),
            assembly_map_b=assembly_map_b,
            precomputed_jacobian_nnz=jacobian_nnz,
            dep_dofs=constraints.dep_dofs,
        )

    return R_w_dirichlet, J_w_dirichlet, residual_isv_func_w_constraints
