from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_airfoil_grid(points: np.ndarray, path: str | Path, titles: Iterable[str] | None = None, n_cols: int = 4) -> None:
    points = np.asarray(points)
    if points.ndim == 2:
        points = points[None]
    n = points.shape[0]
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.1 * n_rows), squeeze=False)
    title_list = list(titles) if titles is not None else [""] * n
    for i, ax in enumerate(axes.ravel()):
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        if i >= n:
            continue
        xy = points[i]
        closed = np.concatenate([xy, xy[:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], color="#1f77b4", linewidth=1.4)
        ax.set_xlim(-0.05, 1.05)
        y_abs = max(0.08, float(np.max(np.abs(xy[:, 1]))) * 1.2)
        ax.set_ylim(-y_abs, y_abs)
        if i < len(title_list) and title_list[i]:
            ax.set_title(title_list[i], fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curve(values: list[float], path: str | Path, ylabel: str = "loss") -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_airfoils_npz(path: str | Path, points: np.ndarray, latents: np.ndarray | None = None, names: list[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"points": np.asarray(points, dtype=np.float32)}
    if latents is not None:
        payload["latents"] = np.asarray(latents, dtype=np.float32)
    if names is not None:
        payload["names"] = np.asarray(names)
    np.savez(path, **payload)


def save_airfoil_dat_files(points: np.ndarray, directory: str | Path, prefix: str = "sample") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, xy in enumerate(np.asarray(points)):
        out = directory / f"{prefix}_{i:04d}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"{prefix}_{i:04d}\n")
            for x, y in xy:
                f.write(f"{x:.8f} {y:.8f}\n")
