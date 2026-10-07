"""Closed-form ball and SOC projector fixtures."""

import numpy as np

from snn_opt import AffineSubspaceProjector, ball_projector, soc_projector


def test_ball_soc_subset_projectors_match_direct_projection():
    x = np.array([9.0, 3.0, 4.0, -2.0, 0.5])
    ball = ball_projector([1, 3], radius=2.0, center=np.array([1.0, -1.0]))
    y = ball.project(x)
    delta = x[[1, 3]] - np.array([1.0, -1.0])
    expected_local = np.array([1.0, -1.0]) + 2.0 * delta / np.linalg.norm(delta)
    np.testing.assert_allclose(y[[1, 3]], expected_local)
    np.testing.assert_array_equal(y[[0, 2, 4]], x[[0, 2, 4]])

    soc = soc_projector(0, [2, 4])
    soc_x = x.copy()
    soc_x[0] = 0.5
    z = soc.project(soc_x)
    t, v = soc_x[0], soc_x[[2, 4]]
    vn = np.linalg.norm(v)
    alpha = (vn + t) / 2.0
    expected_t = alpha
    expected_v = alpha * v / vn
    np.testing.assert_allclose(z[0], expected_t)
    np.testing.assert_allclose(z[[2, 4]], expected_v)
    np.testing.assert_array_equal(z[[1, 3]], soc_x[[1, 3]])

    # The affine-subspace helper uses the standard minimum-distance formula.
    sub = AffineSubspaceProjector(np.array([[1.0, 1.0]]), np.array([1.0]))
    q = sub.project(np.array([0.0, 0.0]))
    np.testing.assert_allclose(q, [0.5, 0.5])
    np.testing.assert_allclose(sub.B @ q, [1.0])
