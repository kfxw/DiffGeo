from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, resolve_path
from .data import AirfoilDataset
from .geometry import smoothness_loss, template_tensor
from .io import get_device
from .models import AutoDecoder
from .visualization import plot_airfoil_grid, plot_curve


def set_seed(seed: int, seed_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config_path: str | Path, force_cpu: bool = False, max_epochs: int | None = None, limit: int | None = None) -> Path:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    set_seed(int(cfg.get("seed", 0)), seed_cuda=not force_cpu)
    device = get_device(force_cpu)

    data_cfg = cfg["data"]
    data_limit = limit if limit is not None else data_cfg.get("limit")
    dataset = AirfoilDataset(
        data_dir=resolve_path(data_cfg["data_dir"], root),
        split_file=resolve_path(data_cfg["train_split"], root),
        num_points=int(data_cfg["num_points"]),
        limit=data_limit,
    )
    loader = DataLoader(dataset, batch_size=int(cfg["autodecoder"]["batch_size"]), shuffle=True, num_workers=0)
    model = AutoDecoder(
        num_shapes=len(dataset),
        latent_dim=int(cfg["autodecoder"]["latent_dim"]),
        hidden_dim=int(cfg["autodecoder"]["hidden_dim"]),
        num_layers=int(cfg["autodecoder"]["num_layers"]),
    ).to(device)
    template = template_tensor(int(data_cfg["num_points"]), device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["autodecoder"]["learning_rate"]),
        weight_decay=float(cfg["autodecoder"].get("weight_decay", 0.0)),
    )
    epochs = int(max_epochs if max_epochs is not None else cfg["autodecoder"]["epochs"])
    latent_l2_weight = float(cfg["autodecoder"].get("latent_l2_weight", 0.0))
    smooth_weight = float(cfg["autodecoder"].get("smoothness_weight", 0.0))
    history: list[float] = []

    progress = tqdm(range(1, epochs + 1), desc="autodecoder")
    for epoch in progress:
        epoch_losses: list[float] = []
        for batch in loader:
            indices = batch["index"].to(device)
            target = batch["points"].to(device)
            pred = model(indices, template)
            recon = F.mse_loss(pred, target)
            latent_l2 = model.latents(indices).pow(2).mean()
            smooth = smoothness_loss(pred)
            loss = recon + latent_l2_weight * latent_l2 + smooth_weight * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        progress.set_postfix(loss=f"{mean_loss:.4e}")

    checkpoint_path = resolve_path(cfg["autodecoder"]["checkpoint"], root)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_config": model.model_config,
        "model_state": model.state_dict(),
        "template": template.detach().cpu(),
        "names": dataset.names,
        "history": history,
        "config": cfg,
    }
    torch.save(checkpoint, checkpoint_path)

    out_dir = resolve_path(cfg.get("output_dir", "outputs"), root)
    plot_curve(history, out_dir / "autodecoder_training_curve.png", ylabel="training loss")
    with torch.no_grad():
        preview_batch = next(iter(DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)))
        preview = model(preview_batch["index"].to(device), template).cpu().numpy()
    plot_airfoil_grid(preview, out_dir / "autodecoder_reconstruction_preview.png", titles=dataset.names[: len(preview)], n_cols=4)
    (out_dir / "autodecoder_summary.json").write_text(json.dumps({"num_shapes": len(dataset), "final_loss": history[-1]}, indent=2), encoding="utf-8")
    return checkpoint_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DiffGeo 2D airfoil auto-decoder.")
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = train(args.config, force_cpu=args.cpu, max_epochs=args.max_epochs, limit=args.limit)
    print(f"saved auto-decoder checkpoint: {path}")


if __name__ == "__main__":
    main()
