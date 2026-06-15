import torch

from diffgeo.geometry import template_tensor
from diffgeo.models import AutoDecoder, DiffusionMLP


def test_autodecoder_forward_shape():
    model = AutoDecoder(num_shapes=4, latent_dim=8, hidden_dim=16, num_layers=2)
    template = template_tensor(32)
    out = model(torch.tensor([0, 2]), template)
    assert out.shape == (2, 32, 2)


def test_diffusion_mlp_shape():
    model = DiffusionMLP(latent_dim=8, hidden_dim=16, num_layers=2, time_embed_dim=12)
    x = torch.randn(3, 8)
    t = torch.tensor([0, 1, 2])
    out = model(x, t)
    assert out.shape == x.shape
