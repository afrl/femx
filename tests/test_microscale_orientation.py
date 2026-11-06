"""CPU-only geometry tests. Does not import JetSCI/JAX or change the benchmark.

Run: python -m pytest -s -q tests/test_microscale_orientation.py
Outputs ORIENTATION_REPORT with the actual Basix permutation and determinant
statistics before/after a candidate connectivity-only correction.
"""

import json
from pathlib import Path

import basix
import meshio
import numpy as np
import pytest

from microscale_orientation import (
    VTK_TRIANGLE6_POINTS, quadratic_triangle, triangle6_permutations,
    orient_vtk_triangle6_for_basix, corner_determinants, geometry_at,
    determinant_summary, probe_points,
)


@pytest.mark.parametrize("clockwise", [False, True])
@pytest.mark.parametrize("curved", [False, True])
def test_triangle6_permutation_preserves_geometry_and_affine_fields(clockwise, curved):
    element = quadratic_triangle()
    points = VTK_TRIANGLE6_POINTS.copy()
    if curved:
        points[3:] += np.array([[0., -.02], [.01, .01], [-.02, 0.]])
    cells = np.arange(6, dtype=np.int32)[None, :]
    if clockwise:
        _, reflection = triangle6_permutations(element)
        cells = cells[:, reflection]
    original_points, original_cells = points.copy(), cells.copy()
    corrected, report = orient_vtk_triangle6_for_basix(points, cells, element)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(cells, original_cells)
    np.testing.assert_array_equal(np.sort(corrected, axis=1), np.sort(cells, axis=1))
    assert report["reflected_cells"] == int(clockwise)
    probes = probe_points()
    jacobians, determinants, derivatives = geometry_at(points, corrected, element, probes)
    assert np.all(determinants > 0)
    values = np.asarray(element.tabulate(0, probes))[0, :, :, 0]
    physical = np.einsum("qn,end->eqd", values, points[corrected])
    field = 1 + 2 * points[:, 0] - 3 * points[:, 1]
    interpolated = np.einsum("qn,en->eq", values, field[corrected])
    np.testing.assert_allclose(interpolated, 1 + 2 * physical[..., 0] - 3 * physical[..., 1], atol=1e-12)
    inverse_transpose = np.swapaxes(np.linalg.inv(jacobians), -1, -2)
    gradients = np.einsum("eqpd,qnp->eqnd", inverse_transpose, derivatives)
    field_gradient = np.einsum("eqnd,en->eqd", gradients, field[corrected])
    np.testing.assert_allclose(field_gradient, np.broadcast_to([2., -3.], field_gradient.shape), atol=1e-12)
    if not curved:
        np.testing.assert_allclose(determinants, 1., atol=1e-12)


def test_microscale_triangle6_orientation_candidate():
    path = Path(__file__).parent / "meshes" / "microscale_2D_r0.vtk"
    mesh = meshio.read(path)
    # Match the current benchmark's coordinate precision exactly. Merely
    # upgrading precision must not be mistaken for an orientation correction.
    points = np.asarray(mesh.points, dtype=np.float32)[:, :2].astype(np.float64)
    if len(mesh.cells) != 1 or mesh.cells[0].type != "triangle6":
        raise ValueError("Expected the same single triangle6 cell block as the microscale benchmark. Inspect changed mesh topology before applying this candidate.")
    cells = np.asarray(mesh.cells[0].data, dtype=np.int64)
    original = cells.copy()
    element = quadratic_triangle()
    to_basix, _ = triangle6_permutations(element)
    corrected, permutation = orient_vtk_triangle6_for_basix(points, cells, element)
    quadrature, weights = basix.make_quadrature(basix.CellType.triangle, 3)
    quadrature, weights = np.asarray(quadrature), np.asarray(weights)
    raw_det = geometry_at(points, cells, element, quadrature)[1]
    reordered_det = geometry_at(points, cells[:, to_basix], element, quadrature)[1]
    jacobians, corrected_det, derivatives = geometry_at(points, corrected, element, quadrature)
    dense_det = geometry_at(points, corrected, element, probe_points())[1]
    corner_det = corner_determinants(points, cells)
    report = {
        "mesh": path.name, "nodes": len(points), "cells": len(cells), **permutation,
        "clockwise_corner_cells": int(np.count_nonzero(corner_det < 0)),
        "raw_as_basix": determinant_summary(raw_det),
        "vtk_to_basix_only": determinant_summary(reordered_det),
        "corrected_quadrature": determinant_summary(corrected_det),
        "corrected_dense_probes": determinant_summary(dense_det),
    }
    print("ORIENTATION_REPORT " + json.dumps(report), flush=True)
    np.testing.assert_array_equal(cells, original)
    # A correction is only a within-cell permutation. No nodes move, no
    # elements/material labels change position, and no determinant is abs'ed.
    np.testing.assert_array_equal(np.sort(corrected, axis=1), np.sort(original, axis=1))
    for name, determinants in (("quadrature", corrected_det), ("dense probes", dense_det)):
        assert np.isfinite(determinants).all() and np.all(determinants > 0), (
            f"Candidate fails positive geometry at {name}; do not apply it to the solver benchmark.", report,
        )

    # Independent signed-measure check: assemble a scalar diffusion diagonal
    # with this geometry. It must be strictly positive at every used node.
    # This is not a replacement for the later full elasticity/adjoint test.
    inverse_transpose = np.swapaxes(np.linalg.inv(jacobians), -1, -2)
    gradients = np.einsum("eqpd,qnp->eqnd", inverse_transpose, derivatives)
    local_diagonal = np.einsum("eqnd,eqnd,eq,q->en", gradients, gradients, corrected_det, weights)
    diagonal = np.zeros(len(points))
    np.add.at(diagonal, corrected.ravel(), local_diagonal.ravel())
    used = np.unique(corrected)
    assert np.all(diagonal[used] > 0), "Signed integration still produces nonpositive diffusion diagonals."
    print("ORIENTATION_SIGNED_INTEGRATION " + json.dumps({
        "used_nodes": len(used), "positive_diagonal_nodes": int(np.count_nonzero(diagonal[used] > 0)),
        "diagonal_min": float(diagonal[used].min()), "diagonal_max": float(diagonal[used].max()),
    }), flush=True)


def test_degenerate_triangle6_is_not_silently_repaired():
    points = VTK_TRIANGLE6_POINTS.copy()
    points[:, 1] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        orient_vtk_triangle6_for_basix(points, np.arange(6)[None, :], quadratic_triangle())
