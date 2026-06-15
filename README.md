# DiffGeo

This repository is the 2D airfoil open-source release for **Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model** (AIAA Journal, 2026).

It provides a compact Python package for:

- training a 2D airfoil auto-decoder on the UIUC full training split;
- exporting learned latent codes;
- training a latent-space diffusion model;
- generating unconditional airfoils;
- generating airfoils with normalized area and max-thickness targets;
- scaling and shifting generated airfoil coordinates.

## Repository Layout

```text
DiffGeo/
├── configs/                         # Full UIUC experiment config
├── data/
│   ├── uiuc_airfoils.tar.gz          # UIUC coordinate archive
│   └── splits/                       # train_full_UIUC and test_UIUC split lists
├── pretrained/uiuc_airfoil_full_v1/  # Released full-UIUC checkpoint bundle
├── scripts/                          # Thin CLI wrappers
├── skills/diffgeo-airfoil-generation/
├── src/diffgeo/                      # Package implementation
└── tests/                            # Unit and data-loading tests
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Some minimal Debian/Ubuntu containers do not include `ensurepip`, so `python -m venv` may require `apt install python3.10-venv`. If you cannot modify the image, install dependencies into a project-local target:

```bash
mkdir -p .deps .cache .pip-cache .tmp
export PIP_CACHE_DIR=$PWD/.pip-cache
export XDG_CACHE_HOME=$PWD/.cache
export TMPDIR=$PWD/.tmp
python -m pip install --target .deps -r requirements.txt
export PYTHONPATH=$PWD/src:$PWD/.deps
```

For GPU training, install a PyTorch wheel compatible with the host NVIDIA driver.

## Data

The UIUC airfoil coordinate files are packaged as:

```text
data/uiuc_airfoils.tar.gz
```

Extract the archive before running tests, training, or data-dependent commands:

```bash
tar -xzf data/uiuc_airfoils.tar.gz -C data
```

The extracted path must be:

```text
data/uiuc_airfoils/dat/
```

The release keeps two split files:

```text
data/splits/train_full_UIUC.txt
data/splits/test_UIUC.txt
```

The bundled UIUC coordinates are included for reproducibility of the 2D airfoil experiments. Respect the upstream UIUC airfoil database terms when redistributing derived packages.

## Pretrained Airfoil Generation

The repository includes a full-UIUC pretrained checkpoint bundle:

```text
pretrained/uiuc_airfoil_full_v1
```

Generate unconditional airfoils without retraining:

```bash
diffgeo-sample-unconditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --num-samples 16 \
  --output-dir outputs/pretrained_unconditional
```

Generate airfoils with normalized geometry targets:

```bash
diffgeo-sample-conditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --target-area 0.07 \
  --target-max-thickness 0.12 \
  --num-samples 16 \
  --output-dir outputs/pretrained_conditional
```

Scale and shift generated coordinates as a post-processing step:

```bash
diffgeo-transform-airfoils \
  --input outputs/pretrained_conditional/conditional_samples.npz \
  --chord-scale 1.5 \
  --shift-x 0.25 \
  --shift-y -0.05 \
  --output-dir outputs/pretrained_transformed
```

Area and max-thickness targets use the normalized unit-chord coordinate system. `chord-scale` applies uniform scaling, then `shift-x` and `shift-y` translate all coordinates.

## Full UIUC Reproduction

Run the full training pipeline:

```bash
diffgeo-train-autodecoder --config configs/full_uiuc.yaml
diffgeo-encode-latents --config configs/full_uiuc.yaml
diffgeo-train-diffusion --config configs/full_uiuc.yaml
diffgeo-sample-unconditional --config configs/full_uiuc.yaml --num-samples 64
diffgeo-sample-conditional \
  --config configs/full_uiuc.yaml \
  --target-area 0.07 \
  --target-max-thickness 0.12 \
  --num-samples 64
```

Expected full-run artifacts:

```text
outputs/full_uiuc/checkpoints/autodecoder.pt
outputs/full_uiuc/latents/uiuc_latents.npz
outputs/full_uiuc/checkpoints/diffusion.pt
outputs/full_uiuc/samples/unconditional_grid.png
outputs/full_uiuc/samples/conditional_grid.png
outputs/full_uiuc/samples/conditional_unguided_baseline_grid.png
outputs/full_uiuc/samples/conditional_report.txt
```

The conditional report includes guided and unguided baseline errors for the same sample count. A successful run should report lower guided mean absolute error and no surface-order violations under the reported tolerance.

## Tests

After extracting `data/uiuc_airfoils.tar.gz`, run:

```bash
pytest -q
```

## Agent Skill

The agent-facing skill is located at:

```text
skills/diffgeo-airfoil-generation/SKILL.md
```

It is intended for agents working on airfoil/aerofoil generation, wing-section exploration, chord scaling, coordinate transforms, and simple area/thickness constrained geometry tasks. It uses the pretrained checkpoint bundle by default.

## Citation

```bibtex
@article{wei2026diffgeo,
  title={Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model},
  author={Wei, Zhen and Dufour, Edouard and Pelletier, Colin and Bauerheim, Michael and Fua, Pascal},
  journal={AIAA Journal},
  year={2026}
}
```

If you use generated airfoils or the pretrained model, also cite the DiffAirfoil AIAA Aviation paper:

```bibtex
@inproceedings{dufour2024diffairfoil,
  title={DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization},
  author={Dufour, Edouard and Wei, Zhen and Pelletier, Colin and Bauerheim, Michael and Fua, Pascal},
  booktitle={AIAA Aviation Forum},
  year={2024}
}
```

## License

Code is released under the MIT license.
