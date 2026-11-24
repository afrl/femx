import numpy as np
from fe_jax.fiber_mechanics.vtms_structs import VTMSBundle, VTMSFabric


def build_custom_hex(
    R: float,
    d: float,
    center: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> np.ndarray:
    """Generate 2D hex-packed fiber positions given radius R and pitch/spacing d."""
    def horizontal_centers(n, d):
        return np.linspace(-(n - 1) * d / 2, (n - 1) * d / 2, n)

    # Layers top to bottom
    n_layers = int(np.floor(R / (d * np.sin(np.pi / 3))))
    if n_layers % 2 == 0:
        n_layers += 1
    y_coords = np.linspace(-R, R, n_layers + 2)[1:-1]
    n_elements_layer = np.zeros(n_layers, dtype=int)
    for i, y in enumerate(y_coords):
        n_elements_layer[i] = int(np.floor(2 * np.sqrt(R**2 - y**2) / d)) + 1
        if i > 0 and n_elements_layer[i] == n_elements_layer[i - 1]:
            n_elements_layer[i] -= 1

    H = []
    for i, y in enumerate(y_coords):
        n = n_elements_layer[i]
        x_coords = horizontal_centers(n, d)
        for x in x_coords:
            H.append([x, y])

    H = np.array(H)
    if rotation != 0.0:
        rot_mat = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        H = H @ rot_mat.T

    # Recenter around provided center
    H = H - np.mean(H, axis=0) + np.array(center)
    return H


def create_straight_fiber(
    n_elements: int,
    x0: tuple | np.ndarray,
    xN: tuple | np.ndarray,
    fiber_id: int = 0,
    cell_shift: int = 0,
    x_shift: np.ndarray | None = None,
    y_shift: np.ndarray | None = None,
    z_shift: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate nodal points, cell connectivity, and IDs for a single fiber."""
    if x_shift is None:
        x_shift = np.zeros(n_elements + 1)
    if y_shift is None:
        y_shift = np.zeros(n_elements + 1)
    if z_shift is None:
        z_shift = np.zeros(n_elements + 1)

    points = np.vstack(
        (
            np.linspace(x0[0], xN[0], n_elements + 1) + x_shift,
            np.linspace(x0[1], xN[1], n_elements + 1) + y_shift,
            np.linspace(x0[2], xN[2], n_elements + 1) + z_shift,
        )
    ).T

    cells = np.array(
        [[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)],
        dtype=np.uint64,
    )
    fiber_ids = np.array([[fiber_id]] * len(points))
    cell_ids = np.array([[fiber_id]] * len(cells))

    return points, cells, fiber_ids, cell_ids


def create_fiber_bundle(
    n_elements: int | list[int] | np.ndarray,
    X0: list[tuple] | np.ndarray,
    XN: list[tuple] | np.ndarray,
    diameters: float | list[float] | np.ndarray = 0.1,
    material_ids: int | list[int] | np.ndarray = 0,
    x_shifts: np.ndarray | list[np.ndarray] | None = None,
    y_shifts: np.ndarray | list[np.ndarray] | None = None,
    z_shifts: np.ndarray | list[np.ndarray] | None = None,
    bundle_name: str = "bundle",
) -> VTMSFabric:
    """Construct a VTMSFabric containing a bundle of straight fibers with optional shifts."""
    n_fibers = len(X0)
    if isinstance(n_elements, (int, np.integer)):
        n_elements_list = [int(n_elements)] * n_fibers
    else:
        n_elements_list = [int(n) for n in n_elements]

    if isinstance(diameters, (float, int, np.number)):
        diameters_arr = np.array([float(diameters)] * n_fibers)
    else:
        diameters_arr = np.array(diameters, dtype=float)

    if isinstance(material_ids, (int, np.integer)):
        material_ids_arr = np.array([int(material_ids)] * n_fibers)
    else:
        material_ids_arr = np.array(material_ids, dtype=int)

    def _get_shift(shifts, idx, n_el):
        if shifts is None:
            return None
        if isinstance(shifts, np.ndarray) and shifts.ndim == 1:
            return shifts
        if isinstance(shifts, (list, tuple)) or (isinstance(shifts, np.ndarray) and shifts.ndim > 1):
            return shifts[idx]
        return None

    point_blocks = []
    cell_blocks = []
    point_id_blocks = []
    cell_id_blocks = []

    vertex_offset = 0

    for fiber_id, (n_el, x0, xN) in enumerate(zip(n_elements_list, X0, XN)):
        x_shift = _get_shift(x_shifts, fiber_id, n_el)
        y_shift = _get_shift(y_shifts, fiber_id, n_el)
        z_shift = _get_shift(z_shifts, fiber_id, n_el)

        points_i, cells_i, point_ids_i, cell_ids_i = create_straight_fiber(
            n_elements=n_el,
            x0=x0,
            xN=xN,
            fiber_id=fiber_id,
            cell_shift=vertex_offset,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)
        point_id_blocks.append(point_ids_i)
        cell_id_blocks.append(cell_ids_i)
        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)

    fiber_offsets = np.concatenate(
        [
            [0],
            np.cumsum([b.shape[0] for b in point_blocks])
        ]
    )

    fabric = VTMSFabric(
        name=bundle_name,
        material_ids=material_ids_arr,
        diameters=diameters_arr,
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0] - 1]),
    )
    return fabric
