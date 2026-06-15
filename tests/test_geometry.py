import torch

from diffgeo.geometry import naca0012, polygon_area, polygon_area_numpy, template_tensor


def test_naca_template_shape_and_area():
    points = naca0012(80)
    assert points.shape == (80, 2)
    assert points[:, 0].min() >= -1e-6
    assert points[:, 0].max() <= 1.0 + 1e-6
    assert polygon_area_numpy(points) > 0.01


def test_torch_area_matches_numpy():
    points = template_tensor(80)
    torch_area = polygon_area(points).item()
    np_area = polygon_area_numpy(points.numpy())
    assert abs(torch_area - np_area) < 1e-6
