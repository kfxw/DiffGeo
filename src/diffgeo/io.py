from __future__ import annotations

from pathlib import Path

import torch

from .models import AutoDecoder, DiffusionMLP


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def load_autodecoder(checkpoint_path: str | Path, device: torch.device | str) -> tuple[AutoDecoder, torch.Tensor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AutoDecoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    template = checkpoint["template"].to(device)
    return model, template, checkpoint


def load_diffusion(checkpoint_path: str | Path, device: torch.device | str) -> tuple[DiffusionMLP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiffusionMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
