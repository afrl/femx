import numpy as np
import jax.numpy as jnp 

def _validate_point_cloud(
    points: jnp.ndarray,
    point_fiber_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    points = np.asarray(points)
    point_fiber_ids = np.asarray(point_fiber_ids)

    if points.ndim != 2 or points.shape[1] not in (1, 2, 3):
        raise ValueError("points must have shape (N_total, 1), (N_total, 2), or (N_total, 3)")
    if point_fiber_ids.ndim != 1:
        raise ValueError("point_fiber_ids must have shape (N_total,)")
    if point_fiber_ids.shape[0] != points.shape[0]:
        raise ValueError("point_fiber_ids must have the same leading dimension as points")

    return points, point_fiber_ids
