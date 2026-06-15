import numpy as np

from diffgeo.transform_airfoils import transform


def test_transform_uniform_scale_and_shift():
    points = np.asarray([[[0.0, 0.0], [1.0, 0.1], [0.5, -0.1]]], dtype=np.float32)
    out = transform(points, chord_scale=2.0, shift_x=0.25, shift_y=-0.5)
    expected = np.asarray([[[0.25, -0.5], [2.25, -0.3], [1.25, -0.7]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)
