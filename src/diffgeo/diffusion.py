from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class DiffusionSchedule:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device: str | torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(self.timesteps, self.beta_start, self.beta_end, device=device)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        scale_x = self.sqrt_alpha_bars[timesteps].view(-1, *([1] * (x0.ndim - 1)))
        scale_n = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, *([1] * (x0.ndim - 1)))
        return scale_x * x0 + scale_n * noise


def diffusion_x0_loss(model: torch.nn.Module, schedule: DiffusionSchedule, x0: torch.Tensor) -> torch.Tensor:
    batch = x0.shape[0]
    timesteps = torch.randint(0, schedule.timesteps, (batch,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = schedule.q_sample(x0, timesteps, noise)
    pred_x0 = model(xt, timesteps)
    return F.mse_loss(pred_x0, x0)


def p_sample_x0(model: torch.nn.Module, schedule: DiffusionSchedule, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    pred_x0 = model(x, timesteps)
    beta_t = schedule.betas[timesteps].view(-1, 1)
    alpha_t = schedule.alphas[timesteps].view(-1, 1)
    alpha_bar_t = schedule.alpha_bars[timesteps].view(-1, 1)
    pred_noise = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
    mean = (x - beta_t * pred_noise / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))) / torch.sqrt(alpha_t)
    noise = torch.randn_like(x)
    nonzero = (timesteps > 0).float().view(-1, 1)
    return mean + nonzero * torch.sqrt(beta_t) * noise


def sample_loop(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    latent_shape: tuple[int, int],
    device: torch.device | str,
    guidance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    guidance_scale: float = 1.0,
    progress: bool = False,
) -> torch.Tensor:
    iterator = range(schedule.timesteps - 1, -1, -1)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="sampling")
    x = torch.randn(latent_shape, device=device)
    batch = latent_shape[0]
    model.eval()
    for step in iterator:
        t = torch.full((batch,), step, device=device, dtype=torch.long)
        if guidance_fn is not None and guidance_scale != 0.0:
            with torch.enable_grad():
                guided_x = x.detach().requires_grad_(True)
                pred_x0 = model(guided_x, t)
                energy = guidance_fn(pred_x0)
                grad = torch.autograd.grad(energy.sum(), guided_x)[0]
                x = (guided_x - guidance_scale * schedule.betas[step] * grad).detach()
        with torch.no_grad():
            x = p_sample_x0(model, schedule, x, t)
    return x
