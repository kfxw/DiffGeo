from __future__ import annotations

import numpy as np
import torch


def naca0012(num_points: int = 200) -> np.ndarray:
    if num_points < 8:
        raise ValueError("num_points must be at least 8")
    n_upper = num_points // 2
    n_lower = num_points - n_upper
    theta_u = np.linspace(0.0, np.pi, n_upper, dtype=np.float64)
    x_u = 0.5 * (1.0 + np.cos(theta_u))
    yt_u = _naca_thickness(x_u, thickness=0.12)
    upper = np.stack([x_u, yt_u], axis=1)

    theta_l = np.linspace(np.pi, 0.0, n_lower, dtype=np.float64)
    x_l = 0.5 * (1.0 + np.cos(theta_l))
    yt_l = _naca_thickness(x_l, thickness=0.12)
    lower = np.stack([x_l, -yt_l], axis=1)
    return np.concatenate([upper, lower], axis=0).astype(np.float32)


def _naca_thickness(x: np.ndarray, thickness: float = 0.12) -> np.ndarray:
    # Closed trailing-edge 4-digit NACA thickness equation.
    return 5.0 * thickness * (
        0.2969 * np.sqrt(np.clip(x, 0.0, 1.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def template_tensor(num_points: int, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.from_numpy(naca0012(num_points)).float().to(device)


def polygon_area_signed(points: torch.Tensor) -> torch.Tensor:
    x = points[..., :, 0]
    y = points[..., :, 1]
    return 0.5 * torch.sum(x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1), dim=-1)


def polygon_area(points: torch.Tensor) -> torch.Tensor:
    return polygon_area_signed(points).abs()


def polygon_area_numpy(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def smoothness_loss(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-2] < 3:
        return points.new_tensor(0.0)
    second = points[..., 2:, :] - 2.0 * points[..., 1:-1, :] + points[..., :-2, :]
    return torch.mean(second.pow(2))



def split_airfoil_surfaces_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered TE-upper-LE-lower-TE contour into upper/lower surfaces.

    Returned surfaces are sorted by increasing x and duplicate x locations are
    averaged. This is intended for metrics and reports, not for gradient flow.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        raise ValueError("points must have shape (n, 2) with at least 4 points")
    le_idx = int(np.argmin(pts[:, 0]))
    if le_idx == 0 or le_idx == len(pts) - 1:
        le_idx = len(pts) // 2
    upper = pts[: le_idx + 1]
    lower = pts[le_idx:]
    return _sort_unique_surface(upper), _sort_unique_surface(lower)


def _sort_unique_surface(surface: np.ndarray) -> np.ndarray:
    order = np.argsort(surface[:, 0])
    sorted_surface = surface[order]
    unique_x = []
    unique_y = []
    for x in np.unique(sorted_surface[:, 0]):
        mask = np.isclose(sorted_surface[:, 0], x, atol=1e-10)
        unique_x.append(float(x))
        unique_y.append(float(np.mean(sorted_surface[mask, 1])))
    return np.stack([np.asarray(unique_x), np.asarray(unique_y)], axis=1)


def max_thickness_numpy(points: np.ndarray, num_samples: int = 256) -> float:
    """Return max airfoil thickness from x-interpolated upper/lower surfaces."""
    thickness = surface_thickness_numpy(points, num_samples=num_samples)
    return float(np.max(thickness))


def surface_thickness_numpy(points: np.ndarray, num_samples: int = 256) -> np.ndarray:
    """Return upper-minus-lower thickness values on a common x grid."""
    upper, lower = split_airfoil_surfaces_numpy(points)
    xmin = max(float(upper[:, 0].min()), float(lower[:, 0].min()))
    xmax = min(float(upper[:, 0].max()), float(lower[:, 0].max()))
    if xmax <= xmin:
        return np.asarray([float(np.max(points[:, 1]) - np.min(points[:, 1]))], dtype=np.float64)
    x_grid = np.linspace(xmin, xmax, num_samples)
    y_upper = np.interp(x_grid, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_grid, lower[:, 0], lower[:, 1])
    return y_upper - y_lower


def min_surface_gap_numpy(points: np.ndarray, num_samples: int = 256) -> float:
    """Return the minimum upper-minus-lower surface gap."""
    return float(np.min(surface_thickness_numpy(points, num_samples=num_samples)))


def max_thickness(points: torch.Tensor) -> torch.Tensor:
    """Differentiable fixed-order approximation of max thickness.

    The decoder uses a TE-upper-LE-lower-TE template. Pairing the reversed upper
    half with the lower half gives a stable differentiable thickness estimate for
    guidance and latent refinement. Use `max_thickness_numpy` for final reports.
    """
    thickness = surface_thickness(points)
    return torch.max(thickness, dim=-1).values


def surface_thickness(points: torch.Tensor) -> torch.Tensor:
    """Differentiable fixed-order upper-minus-lower thickness profile."""
    n = points.shape[-2]
    half = n // 2
    upper = points[..., :half, 1].flip(dims=(-1,))
    lower = points[..., half:, 1]
    m = min(upper.shape[-1], lower.shape[-1])
    return upper[..., :m] - lower[..., :m]


def surface_order_penalty(points: torch.Tensor, min_gap: float = 0.0) -> torch.Tensor:
    """Penalize decoded contours whose lower surface crosses above the upper surface."""
    gap = surface_thickness(points)
    return torch.mean(torch.relu(float(min_gap) - gap).pow(2), dim=-1)


def airfoil_metrics_numpy(points: np.ndarray) -> dict[str, float]:
    return {
        "area": polygon_area_numpy(points),
        "max_thickness": max_thickness_numpy(points),
        "min_surface_gap": min_surface_gap_numpy(points),
    }
