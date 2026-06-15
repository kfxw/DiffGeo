from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_path
from .diffusion import DiffusionSchedule, sample_loop
from .io import get_device, load_autodecoder, load_diffusion
from .visualization import plot_airfoil_grid, save_airfoil_dat_files, save_airfoils_npz


def sample(config_path: str | Path, num_samples: int | None = None, pretrained_dir: str | Path | None = None, output_dir: str | Path | None = None, force_cpu: bool = False) -> Path:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    device = get_device(force_cpu)
    n_samples = int(num_samples if num_samples is not None else cfg["sampling"]["num_samples"])

    if pretrained_dir is not None:
        pretrained = Path(pretrained_dir)
        if not pretrained.is_absolute():
            pretrained = root / pretrained
        autodecoder_path = pretrained / "autodecoder.pt"
        diffusion_path = pretrained / "diffusion.pt"
    else:
        autodecoder_path = resolve_path(cfg["autodecoder"]["checkpoint"], root)
        diffusion_path = resolve_path(cfg["diffusion"]["checkpoint"], root)
    autodecoder, template, _ = load_autodecoder(autodecoder_path, device)
    diffusion, diffusion_ckpt = load_diffusion(diffusion_path, device)
    schedule_cfg = diffusion_ckpt["schedule"]
    schedule = DiffusionSchedule(**schedule_cfg, device=device)
    mean = diffusion_ckpt["latent_mean"].to(device)
    std = diffusion_ckpt["latent_std"].to(device)

    with torch.no_grad():
        z_norm = sample_loop(diffusion, schedule, (n_samples, diffusion.latent_dim), device=device, progress=True)
        z = z_norm * std[None] + mean[None]
        points = autodecoder.decode(z, template).detach().cpu().numpy().astype(np.float32)

    output_dir = resolve_path(output_dir if output_dir is not None else cfg["sampling"]["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "unconditional_samples.npz"
    save_airfoils_npz(npz_path, points=points, latents=z.detach().cpu().numpy())
    save_airfoil_dat_files(points, output_dir / "unconditional_dat", prefix="unconditional")
    plot_airfoil_grid(points[: min(16, len(points))], output_dir / "unconditional_grid.png", n_cols=4)
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate unconditional 2D airfoils with DiffGeo.")
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--pretrained-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = sample(args.config, num_samples=args.num_samples, pretrained_dir=args.pretrained_dir, output_dir=args.output_dir, force_cpu=args.cpu)
    print(f"saved unconditional samples: {path}")


if __name__ == "__main__":
    main()
