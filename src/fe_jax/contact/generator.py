from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp

from ..profiling import profile_segment, StandardSegment
from ..element_batch import ElementBatch
from ..basis_quadrature import FiniteElementType
from .contact_types import (
    ContactBackend,
    resolve_contact_backend,
    ContactParams,
    ContactMaterialSpec,
)
from .params import (
    build_contact_material_params,
    find_nonzero_length_dummy_contact_pair,
)
from .utils import raise_if_contact_capacity_exhausted
from .scipy_contact import scipy_contact_batch
from .jax_hash_contact import (
    jax_hash_contact_batch,
    pad_hash_cells,
    estimate_jax_hash_rigid_contact_max,
)
from .cupyx_contact import cupyx_contact_batch
from .jztree_contact import jztree_contact_batch
from .warp_contact import (
    warp_contact_batch,
    build_warp_contact_context,
    raise_if_warp_contact_capacity_exhausted,
)


def create_contact_element_generator(
    contact_options: ContactParams,
    vertices_vd: np.ndarray | Callable[[], np.ndarray],
    point_diameters: np.ndarray,
    point_fiber_ids: np.ndarray,
    contact_fe_type: FiniteElementType,
    n_dofs_per_basis: int,
    max_material_E: float,
    max_material_A: float,
) -> Callable[[jnp.ndarray | np.ndarray | None], list[ElementBatch]]:
    """Initializes and returns a callable (u_ref) -> list[ElementBatch] for contact elements.

    Handles backend pre-computation, dummy pair setup, and capacity validation.
    `vertices_vd` may be an ndarray or a callable returning the current deformed coordinates.
    """
    get_vertices = vertices_vd if callable(vertices_vd) else (lambda: vertices_vd)
    initial_vertices = get_vertices()

    if contact_options.contact_search_alpha <= contact_options.M_to_D_ratio:
        raise ValueError("contact_search_alpha must be greater than M_to_D_ratio.")
    if contact_options.M_to_D_ratio <= contact_options.C_to_D_ratio:
        raise ValueError("M_to_D_ratio must be greater than C_to_D_ratio.")

    contact_backend = resolve_contact_backend(
        contact_options.contact_backend,
        auto_uses_newton_warp=False,
    )
    fixed_capacity_backend = contact_backend in (
        ContactBackend.NEWTON_WARP,
        ContactBackend.CUPYX_KDTREE,
        ContactBackend.JAX_HASH,
        ContactBackend.JZTREE,
    )
    if contact_backend in (ContactBackend.NEWTON_WARP, ContactBackend.CUPYX_KDTREE):
        if contact_options.rigid_contact_max is None:
            raise ValueError(
                f"{contact_backend} requires ContactParams.rigid_contact_max to define the fixed contact capacity."
            )

    point_radii = 0.5 * point_diameters
    self_adjacency_block = contact_options.self_adjacency_block

    contact_E_c = contact_options.D_stiffness_to_E_ratio * max_material_E
    contact_A = max_material_A
    contact_E_min = contact_options.M_stiffness_to_E_ratio * max_material_E
    contact_material_spec = ContactMaterialSpec(
        E_c=contact_E_c,
        area=contact_A,
        M_to_D_ratio=contact_options.M_to_D_ratio,
        C_to_D_ratio=contact_options.C_to_D_ratio,
        search_alpha=contact_options.contact_search_alpha,
        E_min=contact_E_min,
    )

    shared_dummy_pair = None
    if fixed_capacity_backend:
        shared_dummy_pair = jnp.asarray(
            find_nonzero_length_dummy_contact_pair(initial_vertices),
            dtype=jnp.int32,
        )

    # 1. SciPy Backend
    if contact_backend == ContactBackend.SCIPY_KDTREE:
        def scipy_generator(u_ref) -> list[ElementBatch]:
            with profile_segment(StandardSegment.CONTACT_SEARCH):
                current_vertices = get_vertices()
                if u_ref is None:
                    current_points = current_vertices
                else:
                    current_points = current_vertices + np.asarray(u_ref).reshape(current_vertices.shape)
                contact_cells = scipy_contact_batch(
                    points=current_points,
                    point_fiber_ids=point_fiber_ids,
                    adjacency_block=self_adjacency_block,
                    point_diameters=point_diameters,
                    search2radius_ratio=contact_options.contact_search_alpha,
                )
                if contact_cells.shape[0] == 0:
                    return []
                contact_material_params = build_contact_material_params(
                    contact_cells=contact_cells,
                    point_radii=point_radii,
                    spec=contact_material_spec,
                )
                return [
                    ElementBatch(
                        fe_type=contact_fe_type,
                        n_dofs_per_basis=n_dofs_per_basis,
                        connectivity_en=contact_cells,
                        constitutive_model=contact_options.contact_constitutive_model,
                        material_params=jnp.array(contact_material_params),
                    )
                ]
        return scipy_generator

    # 2. Warp Backend
    if contact_backend == ContactBackend.NEWTON_WARP:
        if initial_vertices.shape[1] != 3:
            raise ValueError("ContactBackend.NEWTON_WARP requires 3D points.")
        warp_ctx = build_warp_contact_context(
            points=initial_vertices,
            point_diameters=point_diameters,
            contact_search_alpha=contact_options.contact_search_alpha,
            self_adjacency_block=self_adjacency_block,
            point_fiber_ids=point_fiber_ids,
            rigid_contact_max=contact_options.rigid_contact_max,
        )
        def warp_generator(u_ref) -> list[ElementBatch]:
            with profile_segment(StandardSegment.CONTACT_SEARCH):
                current_vertices = get_vertices()
                current_points = current_vertices if u_ref is None else current_vertices + np.asarray(u_ref).reshape(current_vertices.shape)
                contact_cells, active, count, capacity_exhausted = warp_contact_batch(
                    ctx=warp_ctx,
                    current_points_jax=jnp.asarray(current_points),
                    dummy_pair=shared_dummy_pair,
                )
                raise_if_warp_contact_capacity_exhausted(
                    ctx=warp_ctx,
                    count=count,
                    capacity_exhausted=capacity_exhausted,
                )
                contact_material_params = build_contact_material_params(
                    contact_cells=contact_cells,
                    point_radii=point_radii,
                    spec=contact_material_spec,
                    active=active,
                )
                return [
                    ElementBatch(
                        fe_type=contact_fe_type,
                        n_dofs_per_basis=n_dofs_per_basis,
                        connectivity_en=np.asarray(contact_cells),
                        constitutive_model=contact_options.contact_constitutive_model,
                        material_params=jnp.array(contact_material_params),
                    )
                ]
        return warp_generator

    # 3. JAX Hash Backend
    if contact_backend == ContactBackend.JAX_HASH:
        query_radius = float(contact_options.contact_search_alpha * 2.0 * np.max(point_radii))
        jax_hash_grid = pad_hash_cells(
            points=initial_vertices,
            query_radius=query_radius,
            pad_size=contact_options.hash_pad_size,
            cell_capacity_buffer=contact_options.cell_capacity_buffer,
        )
        if contact_options.rigid_contact_max is None:
            contact_options.rigid_contact_max = estimate_jax_hash_rigid_contact_max(
                points=initial_vertices,
                point_fiber_ids=point_fiber_ids,
                point_diameters=point_diameters,
                search2radius_ratio=contact_options.contact_search_alpha,
                adjacency_block=self_adjacency_block,
                grid_params=jax_hash_grid,
                rigid_contact_factor=1.5,
            )
        domain_min, Nx, Ny, Nz, total_cells, C_max = jax_hash_grid
        def jax_hash_generator(u_ref) -> list[ElementBatch]:
            with profile_segment(StandardSegment.CONTACT_SEARCH):
                current_vertices = get_vertices()
                current_points = current_vertices if u_ref is None else current_vertices + np.asarray(u_ref).reshape(current_vertices.shape)
                contact_cells, active, count, capacity_exhausted = jax_hash_contact_batch(
                    points=current_points,
                    point_fiber_ids=point_fiber_ids,
                    adjacency_block=self_adjacency_block,
                    point_diameters=point_diameters,
                    search2radius_ratio=contact_options.contact_search_alpha,
                    rigid_contact_max=contact_options.rigid_contact_max,
                    domain_min=domain_min,
                    Nx=Nx,
                    Ny=Ny,
                    Nz=Nz,
                    total_cells=total_cells,
                    C_max=C_max,
                    dummy_pair=shared_dummy_pair,
                )
                raise_if_contact_capacity_exhausted(
                    capacity=contact_options.rigid_contact_max,
                    count=count,
                    capacity_exhausted=capacity_exhausted,
                    backend_name="JAX Hash",
                )
                contact_material_params = build_contact_material_params(
                    contact_cells=contact_cells,
                    point_radii=point_radii,
                    spec=contact_material_spec,
                    active=active,
                )
                return [
                    ElementBatch(
                        fe_type=contact_fe_type,
                        n_dofs_per_basis=n_dofs_per_basis,
                        connectivity_en=np.asarray(contact_cells),
                        constitutive_model=contact_options.contact_constitutive_model,
                        material_params=jnp.array(contact_material_params),
                    )
                ]
        return jax_hash_generator

    # 4. JZTree Backend
    if contact_backend == ContactBackend.JZTREE:
        def jztree_generator(u_ref) -> list[ElementBatch]:
            with profile_segment(StandardSegment.CONTACT_SEARCH):
                current_vertices = get_vertices()
                current_points = current_vertices if u_ref is None else current_vertices + np.asarray(u_ref).reshape(current_vertices.shape)
                contact_cells, active, count, capacity_exhausted = jztree_contact_batch(
                    points=current_points,
                    point_fiber_ids=point_fiber_ids,
                    adjacency_block=self_adjacency_block,
                    point_diameters=point_diameters,
                    search2radius_ratio=contact_options.contact_search_alpha,
                    rigid_contact_max=contact_options.rigid_contact_max,
                    dummy_pair=shared_dummy_pair,
                    k=contact_options.knn_k,
                )
                raise_if_contact_capacity_exhausted(
                    capacity=contact_options.rigid_contact_max,
                    count=count,
                    capacity_exhausted=capacity_exhausted,
                    backend_name="JZTree",
                )
                contact_material_params = build_contact_material_params(
                    contact_cells=contact_cells,
                    point_radii=point_radii,
                    spec=contact_material_spec,
                    active=active,
                )
                return [
                    ElementBatch(
                        fe_type=contact_fe_type,
                        n_dofs_per_basis=n_dofs_per_basis,
                        connectivity_en=np.asarray(contact_cells),
                        constitutive_model=contact_options.contact_constitutive_model,
                        material_params=jnp.array(contact_material_params),
                    )
                ]
        return jztree_generator

    # 5. CuPy Backend
    if contact_backend == ContactBackend.CUPYX_KDTREE:
        def cupyx_generator(u_ref) -> list[ElementBatch]:
            with profile_segment(StandardSegment.CONTACT_SEARCH):
                current_vertices = get_vertices()
                current_points = current_vertices if u_ref is None else current_vertices + np.asarray(u_ref).reshape(current_vertices.shape)
                contact_cells, active, count, capacity_exhausted = cupyx_contact_batch(
                    points=current_points,
                    point_fiber_ids=point_fiber_ids,
                    adjacency_block=self_adjacency_block,
                    point_diameters=point_diameters,
                    search2radius_ratio=contact_options.contact_search_alpha,
                    rigid_contact_max=contact_options.rigid_contact_max,
                    dummy_pair=shared_dummy_pair,
                )
                raise_if_contact_capacity_exhausted(
                    capacity=contact_options.rigid_contact_max,
                    count=count,
                    capacity_exhausted=capacity_exhausted,
                    backend_name="CuPy k-d tree",
                )
                contact_material_params = build_contact_material_params(
                    contact_cells=contact_cells,
                    point_radii=point_radii,
                    spec=contact_material_spec,
                    active=active,
                )
                return [
                    ElementBatch(
                        fe_type=contact_fe_type,
                        n_dofs_per_basis=n_dofs_per_basis,
                        connectivity_en=np.asarray(contact_cells),
                        constitutive_model=contact_options.contact_constitutive_model,
                        material_params=jnp.array(contact_material_params),
                    )
                ]
        return cupyx_generator

    raise ValueError(f"Unsupported contact backend: {contact_backend}")
