"""Test-only triangle6 connectivity checks; no mesh or solver mutation.

VTK/meshio triangle6 edge nodes: (0,1), (1,2), (2,0).
https://vtk.org/doc/nightly/html/classvtkQuadraticTriangle.html
Basix's actual interpolation points determine the target basis ordering.
"""

import basix
import numpy as np


VTK_TRIANGLE6_POINTS = np.array([
    [0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5],
])


def quadratic_triangle():
    return basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 2,
        basix.LagrangeVariant.equispaced,
    )


def _match_reference_nodes(target, source):
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    indices = np.argmin(distances, axis=1)
    if len(np.unique(indices)) != 6 or np.max(distances[np.arange(6), indices]) > 1e-12:
        raise ValueError("Expected six nodal P2 triangle points matching VTK reference nodes. Inspect the Basix element; no guessed permutation is used.")
    return indices


def triangle6_permutations(element):
    """Return target-Basix <- source-VTK and reflected-VTK <- source-VTK."""
    points = np.asarray(element.points)
    if points.shape != (6, 2):
        raise ValueError(f"Expected a nodal scalar P2 triangle with six 2D points, got {points.shape}.")
    np.testing.assert_allclose(element.tabulate(0, points)[0, :, :, 0], np.eye(6), atol=1e-12)
    to_basix = _match_reference_nodes(points, VTK_TRIANGLE6_POINTS)
    reflected = _match_reference_nodes(VTK_TRIANGLE6_POINTS[:, ::-1], VTK_TRIANGLE6_POINTS)
    return to_basix, reflected


def corner_determinants(points, cells):
    corners = points[cells[:, :3], :2]
    edges = corners[:, 1:] - corners[:, :1]
    return edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]


def orient_vtk_triangle6_for_basix(points, cells, element):
    """Return a reordered COPY, preserving each cell's six original nodes.

    This function expects VTK/meshio triangle6 topology; it is not a generic
    repair for arbitrary edge-node ordering, curved/inverted cells, or meshes.
    Curved geometry must still pass determinant checks after permutation.
    """
    points, cells = np.asarray(points), np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 6 or points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected points[n,>=2], triangle6 cells[e,6], got {points.shape}, {cells.shape}.")
    if cells.dtype.kind not in "iu" or np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("Connectivity must contain valid zero-based integer node indices; no reindexing is performed.")
    determinants = corner_determinants(points, cells)
    corners = points[cells[:, :3], :2]
    edge_scale = np.max(np.sum((corners[:, 1:] - corners[:, :1]) ** 2, axis=-1), axis=1)
    if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 100 * np.finfo(float).eps * edge_scale):
        raise ValueError("Degenerate/nonfinite corner triangle found. A node permutation cannot repair degenerate geometry.")
    to_basix, reflected = triangle6_permutations(element)
    ordered = cells.copy()
    negative = determinants < 0
    ordered[negative] = ordered[negative][:, reflected]
    return ordered[:, to_basix], {"vtk_to_basix": to_basix.tolist(),
                                "vtk_reflection": reflected.tolist(),
                                "reflected_cells": int(np.count_nonzero(negative))}


def geometry_at(points, cells, element, reference_points):
    table = np.asarray(element.tabulate(1, reference_points))
    derivatives = table[1:3, :, :, 0].transpose(1, 2, 0)
    coordinates = points[cells, :2]
    jacobians = np.einsum("end,qnp->eqpd", coordinates, derivatives)
    return jacobians, np.linalg.det(jacobians), derivatives


def determinant_summary(determinants):
    return {"min": float(np.min(determinants)), "max": float(np.max(determinants)),
            "negative": int(np.count_nonzero(determinants < 0)),
            "zero": int(np.count_nonzero(determinants == 0)),
            "nonfinite": int(np.count_nonzero(~np.isfinite(determinants)))}


def probe_points():
    # Include edges/vertices and interior probes, not just integration points.
    return np.array([(i / 8, j / 8) for i in range(9) for j in range(9 - i)])
