import torch

from diffgeo.diffusion import DiffusionSchedule, diffusion_x0_loss, sample_loop
from diffgeo.models import DiffusionMLP


def test_diffusion_loss_and_sampling_shape():
    model = DiffusionMLP(latent_dim=4, hidden_dim=16, num_layers=1, time_embed_dim=8)
    schedule = DiffusionSchedule(timesteps=5)
    x0 = torch.randn(6, 4)
    loss = diffusion_x0_loss(model, schedule, x0)
    assert torch.isfinite(loss)
    sample = sample_loop(model, schedule, (2, 4), device="cpu")
    assert sample.shape == (2, 4)
