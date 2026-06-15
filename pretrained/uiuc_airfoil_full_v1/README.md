# UIUC Airfoil Full Pretrained Checkpoints v1

This directory contains the released full-UIUC 2D DiffGeo airfoil assets for direct inference and agent-skill use.

Contents:

- `autodecoder.pt`: full UIUC auto-decoder checkpoint.
- `diffusion.pt`: latent-space diffusion checkpoint trained on exported UIUC latent codes.
- `uiuc_latents.npz`: latent table and normalization statistics.
- `config.yaml`: config snapshot used for the full run.
- `unconditional_grid.png`, `conditional_grid.png`, `conditional_report.txt`: reference inference outputs.

Generate unconditional airfoils:

```bash
diffgeo-sample-unconditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --num-samples 16 \
  --output-dir outputs/pretrained_unconditional
```

Generate conditional airfoils:

```bash
diffgeo-sample-conditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --target-area 0.07 \
  --target-max-thickness 0.12 \
  --num-samples 16 \
  --output-dir outputs/pretrained_conditional
```

If you use these checkpoints or generated airfoils in research or engineering reports, cite both the DiffGeo AIAA Journal paper and the DiffAirfoil AIAA Aviation paper.
