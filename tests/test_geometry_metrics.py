import numpy as np
import torch

from diffgeo.geometry import max_thickness, max_thickness_numpy, min_surface_gap_numpy, naca0012, surface_order_penalty


def test_max_thickness_numpy_naca0012_close_to_12_percent():
    points = naca0012(200)
    value = max_thickness_numpy(points)
    assert 0.115 < value < 0.125


def test_torch_max_thickness_shape():
    points = torch.from_numpy(naca0012(80)).float().unsqueeze(0)
    value = max_thickness(points)
    assert value.shape == (1,)
    assert value.item() > 0.1


def test_surface_gap_metrics_detect_crossed_surfaces():
    points = naca0012(80)
    assert min_surface_gap_numpy(points) > -1e-6

    crossed = torch.from_numpy(points).float().unsqueeze(0)
    half = crossed.shape[1] // 2
    crossed[:, half:, 1] = 0.1
    penalty = surface_order_penalty(crossed)
    assert penalty.shape == (1,)
    assert penalty.item() > 0.0
