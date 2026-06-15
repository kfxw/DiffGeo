from __future__ import annotations

import math

import torch
from torch import nn


class AutoDecoder(nn.Module):
    def __init__(self, num_shapes: int, latent_dim: int = 256, hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.num_shapes = int(num_shapes)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.latents = nn.Embedding(self.num_shapes, self.latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=0.01)

        layers: list[nn.Module] = []
        in_dim = self.latent_dim + 2
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.decoder = nn.Sequential(*layers)

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "num_shapes": self.num_shapes,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }

    def decode(self, latent: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must have shape (batch, latent_dim)")
        batch = latent.shape[0]
        if template.ndim == 2:
            template_batch = template.unsqueeze(0).expand(batch, -1, -1)
        elif template.ndim == 3:
            template_batch = template
        else:
            raise ValueError("template must have shape (points, 2) or (batch, points, 2)")
        latent_batch = latent[:, None, :].expand(-1, template_batch.shape[1], -1)
        decoder_input = torch.cat([template_batch, latent_batch], dim=-1)
        delta = self.decoder(decoder_input)
        return template_batch + delta

    def forward(self, indices: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        latent = self.latents(indices)
        return self.decode(latent, template)


class DiffusionMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512, num_layers: int = 4, time_embed_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.time_embed_dim = int(time_embed_dim)

        layers: list[nn.Module] = []
        in_dim = self.latent_dim + self.time_embed_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.latent_dim))
        self.net = nn.Sequential(*layers)

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "time_embed_dim": self.time_embed_dim,
        }

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.time_embed_dim).to(device=x.device, dtype=x.dtype)
        return self.net(torch.cat([x, t_emb], dim=-1))


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / max(half, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
