from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_path
from .io import load_autodecoder


def encode(config_path: str | Path, force_cpu: bool = False) -> Path:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device("cuda")
    model, _template, checkpoint = load_autodecoder(resolve_path(cfg["autodecoder"]["checkpoint"], root), device)
    latents = model.latents.weight.detach().cpu().numpy().astype(np.float32)
    mean = latents.mean(axis=0).astype(np.float32)
    std = latents.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    names = np.asarray(checkpoint.get("names", [f"shape_{i}" for i in range(latents.shape[0])]))
    latent_path = resolve_path(cfg["diffusion"]["latent_path"], root)
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(latent_path, latents=latents, mean=mean, std=std, names=names)
    return latent_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export learned airfoil latent codes for diffusion training.")
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = encode(args.config, force_cpu=args.cpu)
    print(f"saved latent table: {path}")


if __name__ == "__main__":
    main()
