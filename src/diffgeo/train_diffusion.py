from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import load_config, resolve_path
from .diffusion import DiffusionSchedule, diffusion_x0_loss
from .io import get_device
from .models import DiffusionMLP
from .visualization import plot_curve


def set_seed(seed: int, seed_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config_path: str | Path, force_cpu: bool = False, max_steps: int | None = None) -> Path:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    set_seed(int(cfg.get("seed", 0)), seed_cuda=not force_cpu)
    device = get_device(force_cpu)
    diff_cfg = cfg["diffusion"]

    latent_npz = np.load(resolve_path(diff_cfg["latent_path"], root), allow_pickle=True)
    latents = latent_npz["latents"].astype(np.float32)
    mean = latent_npz["mean"].astype(np.float32)
    std = latent_npz["std"].astype(np.float32)
    normalized = (latents - mean[None]) / std[None]
    latent_tensor = torch.from_numpy(normalized).float().to(device)

    model = DiffusionMLP(
        latent_dim=latent_tensor.shape[1],
        hidden_dim=int(diff_cfg["hidden_dim"]),
        num_layers=int(diff_cfg["num_layers"]),
        time_embed_dim=int(diff_cfg["time_embed_dim"]),
    ).to(device)
    schedule = DiffusionSchedule(
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(diff_cfg["learning_rate"]),
        weight_decay=float(diff_cfg.get("weight_decay", 0.0)),
    )
    steps = int(max_steps if max_steps is not None else diff_cfg["train_steps"])
    batch_size = int(diff_cfg["batch_size"])
    history: list[float] = []

    progress = tqdm(range(1, steps + 1), desc="diffusion")
    for step in progress:
        choice = torch.randint(0, latent_tensor.shape[0], (batch_size,), device=device)
        batch = latent_tensor[choice]
        loss = diffusion_x0_loss(model, schedule, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append(value)
        progress.set_postfix(loss=f"{value:.4e}")

    checkpoint_path = resolve_path(diff_cfg["checkpoint"], root)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.model_config,
            "model_state": model.state_dict(),
            "schedule": {
                "timesteps": schedule.timesteps,
                "beta_start": schedule.beta_start,
                "beta_end": schedule.beta_end,
            },
            "latent_mean": torch.from_numpy(mean),
            "latent_std": torch.from_numpy(std),
            "history": history,
            "config": cfg,
        },
        checkpoint_path,
    )
    out_dir = resolve_path(cfg.get("output_dir", "outputs"), root)
    plot_curve(history, out_dir / "diffusion_training_curve.png", ylabel="x0 prediction loss")
    return checkpoint_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DiffGeo latent-space diffusion model.")
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = train(args.config, force_cpu=args.cpu, max_steps=args.max_steps)
    print(f"saved diffusion checkpoint: {path}")


if __name__ == "__main__":
    main()
