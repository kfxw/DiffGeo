from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import read_airfoil_dat
from .visualization import plot_airfoil_grid, save_airfoil_dat_files, save_airfoils_npz


def load_points(input_path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob("*.dat"))
        if not files:
            raise FileNotFoundError(f"No .dat files found in {path}")
        points = [read_airfoil_dat(p) for p in files]
        lengths = {len(x) for x in points}
        if len(lengths) != 1:
            raise ValueError("All .dat files in a directory input must have the same number of points")
        return np.stack(points).astype(np.float32), [p.stem for p in files]
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        if "points" not in payload.files:
            raise KeyError(f"{path} does not contain a 'points' array")
        names = [str(x) for x in payload["names"]] if "names" in payload.files else [f"airfoil_{i:04d}" for i in range(len(payload["points"]))]
        return payload["points"].astype(np.float32), names
    if path.suffix.lower() == ".dat":
        return read_airfoil_dat(path)[None].astype(np.float32), [path.stem]
    raise ValueError(f"Unsupported input type: {path}")


def transform(points: np.ndarray, chord_scale: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    out *= float(chord_scale)
    out[..., 0] += float(shift_x)
    out[..., 1] += float(shift_y)
    return out


def run(
    input_path: str | Path,
    output_dir: str | Path,
    chord_scale: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    prefix: str = "transformed",
) -> Path:
    points, names = load_points(input_path)
    transformed = transform(points, chord_scale=chord_scale, shift_x=shift_x, shift_y=shift_y)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{prefix}_airfoils.npz"
    save_airfoils_npz(npz_path, transformed, names=names)
    save_airfoil_dat_files(transformed, out_dir / f"{prefix}_dat", prefix=prefix)
    plot_airfoil_grid(transformed[: min(16, len(transformed))], out_dir / f"{prefix}_grid.png", n_cols=4)
    (out_dir / f"{prefix}_transform_report.txt").write_text(
        "\n".join(
            [
                f"input: {input_path}",
                f"num_airfoils: {len(transformed)}",
                f"chord_scale: {float(chord_scale):.8f}",
                f"shift_x: {float(shift_x):.8f}",
                f"shift_y: {float(shift_y):.8f}",
                f"output_npz: {npz_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scale and shift generated or existing 2D airfoil coordinates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chord-scale", type=float, default=1.0)
    parser.add_argument("--shift-x", type=float, default=0.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    parser.add_argument("--prefix", default="transformed")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = run(
        args.input,
        output_dir=args.output_dir,
        chord_scale=args.chord_scale,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        prefix=args.prefix,
    )
    print(f"saved transformed airfoils: {path}")


if __name__ == "__main__":
    main()
