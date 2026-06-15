from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_airfoil_names(split_file: str | Path, limit: int | None = None) -> list[str]:
    names: list[str] = []
    with Path(split_file).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.append(stripped.split()[0])
            if limit is not None and len(names) >= limit:
                break
    return names


def resolve_airfoil_file(name: str, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    candidates = [data_dir / name]
    if not name.endswith(".dat"):
        candidates.append(data_dir / f"{name}.dat")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find airfoil '{name}' in {data_dir}")


def read_airfoil_dat(path: str | Path) -> np.ndarray:
    points: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.replace(",", " ").split()
            if len(tokens) < 2:
                continue
            try:
                x = float(tokens[0])
                y = float(tokens[1])
            except ValueError:
                continue
            if not points and _looks_like_surface_count_line(x, y):
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((x, y))
    if len(points) < 3:
        raise ValueError(f"Airfoil file {path} contains fewer than 3 numeric points")
    return np.asarray(points, dtype=np.float32)


def _looks_like_surface_count_line(x: float, y: float) -> bool:
    """Return true for UIUC/Selig count headers such as ``35. 38.``."""
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and x > 1.0
        and y > 1.0
        and abs(x - round(x)) < 1e-6
        and abs(y - round(y)) < 1e-6
    )


def remove_consecutive_duplicates(points: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    if len(points) <= 1:
        return points
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], delta > eps])
    return points[keep]


def normalize_chord(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    xmin = float(points[:, 0].min())
    xmax = float(points[:, 0].max())
    chord = xmax - xmin
    if chord <= 0.0:
        raise ValueError("Airfoil has non-positive chord length")
    points[:, 0] = (points[:, 0] - xmin) / chord
    points[:, 1] = points[:, 1] / chord
    return points


def rotate_to_trailing_edge(points: np.ndarray) -> np.ndarray:
    """Put a high-x point first when a file is not already TE-first."""
    if points[0, 0] > 0.75:
        return points
    idx = int(np.argmax(points[:, 0]))
    return np.concatenate([points[idx:], points[:idx]], axis=0)


def canonicalize_airfoil(points: np.ndarray, num_points: int) -> np.ndarray:
    """Normalize and resample into TE-upper-LE-lower-TE point order."""
    points = normalize_chord(points)
    split = _split_two_surface_format(points)
    if split is None:
        contour = rotate_to_trailing_edge(points)
        le_idx = int(np.argmin(contour[:, 0]))
        if le_idx <= 0 or le_idx >= len(contour) - 1:
            return resample_polyline(contour, num_points)
        first_surface = contour[: le_idx + 1]
        second_surface = contour[le_idx:]
    else:
        first_surface, second_surface = split

    upper, lower = _assign_upper_lower(first_surface, second_surface)
    upper = _orient_surface(upper, start_at_trailing_edge=True)
    lower = _orient_surface(lower, start_at_trailing_edge=False)
    n_upper = num_points // 2
    n_lower = num_points - n_upper
    return np.concatenate(
        [
            resample_polyline(upper, n_upper),
            resample_polyline(lower, n_lower),
        ],
        axis=0,
    ).astype(np.float32)


def _split_two_surface_format(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect LE-to-TE upper/lower blocks used by many UIUC/Selig files."""
    points = remove_consecutive_duplicates(np.asarray(points, dtype=np.float32))
    if len(points) < 6:
        return None
    x = points[:, 0]
    reset_indices = np.where(np.diff(x) < -0.25)[0]
    if len(reset_indices) != 1:
        return None
    split_at = int(reset_indices[0] + 1)
    first = points[:split_at]
    second = points[split_at:]
    if len(first) < 3 or len(second) < 3:
        return None
    if first[0, 0] > 0.25 or second[0, 0] > 0.25:
        return None
    if first[-1, 0] < 0.75 or second[-1, 0] < 0.75:
        return None
    return first, second


def _assign_upper_lower(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_mean, second_mean = _surface_mean_y(first), _surface_mean_y(second)
    if first_mean >= second_mean:
        return first, second
    return second, first


def _surface_mean_y(surface: np.ndarray) -> float:
    return float(np.mean(np.asarray(surface)[:, 1]))


def _orient_surface(surface: np.ndarray, start_at_trailing_edge: bool) -> np.ndarray:
    surface = remove_consecutive_duplicates(np.asarray(surface, dtype=np.float32))
    if len(surface) < 2:
        return surface
    starts_at_te = surface[0, 0] >= surface[-1, 0]
    if starts_at_te != start_at_trailing_edge:
        return surface[::-1].copy()
    return surface


def resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    points = remove_consecutive_duplicates(points)
    if len(points) < 2:
        raise ValueError("Need at least two unique points to resample")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(distances)])
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    points = points[keep]
    s = s[keep]
    if s[-1] <= 0.0:
        raise ValueError("Degenerate airfoil polyline")
    target = np.linspace(0.0, s[-1], num_points, dtype=np.float32)
    x = np.interp(target, s, points[:, 0]).astype(np.float32)
    y = np.interp(target, s, points[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def load_airfoil_array(name: str, data_dir: str | Path, num_points: int) -> np.ndarray:
    path = resolve_airfoil_file(name, data_dir)
    points = read_airfoil_dat(path)
    return canonicalize_airfoil(points, num_points)


class AirfoilDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        num_points: int = 200,
        limit: int | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.num_points = int(num_points)
        self.names = list(names) if names is not None else load_airfoil_names(split_file, limit=limit)
        if not self.names:
            raise ValueError(f"No airfoil names found in {split_file}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, object]:
        name = self.names[index]
        points = load_airfoil_array(name, self.data_dir, self.num_points)
        return {
            "points": torch.from_numpy(points).float(),
            "index": torch.tensor(index, dtype=torch.long),
            "name": name,
        }
