<div align="center">

# DiffGeo

**Agent-ready 2D airfoil/aerofoil generation with latent diffusion**

Use an AI agent or LLM workflow to generate airfoil, aerofoil, wing-section, and blade-section geometries with a reproducible Python package, a bundled agent skill, pretrained UIUC checkpoints, and full training scripts.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-Included-2E7D59?style=flat-square)](skills/diffgeo-airfoil-generation/SKILL.md)
[![Pretrained](https://img.shields.io/badge/Pretrained-UIUC%20full-8A5A1F?style=flat-square)](pretrained/uiuc_airfoil_full_v1)
[![License](https://img.shields.io/badge/License-MIT-111827?style=flat-square)](LICENSE)

[English](./README.md) | [简体中文](./README.zh-CN.md)

<img src=".github/assets/diffgeo-readme-hero.png" alt="DiffGeo AI agent airfoil generation with unconditional and area thickness guided aerofoil samples" width="100%"/>

</div>

## Why DiffGeo

DiffGeo is the 2D airfoil open-source release for **Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model** (AIAA Journal, 2026). It is the journal extension of the conference paper **DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization** (AIAA Aviation Forum, 2024). The release is designed for both normal research-code use and agent-driven geometry generation.

The repository provides:

- a generic agent skill for AI agents and LLM workflows that need airfoil generation;
- pretrained full-UIUC checkpoints for immediate unconditional or constrained sampling;
- Python CLIs for generation, coordinate transforms, training, latent export, and reproduction;
- normalized geometry guidance for sectional area and maximum thickness targets;
- `.npz`, `.dat`, `.png`, and report artifacts for downstream aerodynamic workflows.

DiffGeo does not ask the LLM to invent coordinates directly. The agent reads the skill instructions, calls the reproducible DiffGeo tools, and lets the trained latent diffusion model generate the geometry.

## Agent Skill Quickstart

The bundled skill lives at:

```text
skills/diffgeo-airfoil-generation/SKILL.md
```

Use it when an AI agent, LLM coding assistant, or autonomous engineering workflow needs to generate 2D airfoils/aerofoils, wing sections, blade sections, or simple area/thickness constrained geometries.

Example prompt for an agent:

```text
Use the DiffGeo airfoil-generation skill from this repository.
Generate 16 unit-chord airfoils with target area 0.07 and max thickness 0.12,
then export .dat files for downstream aerodynamic analysis.
```

What the skill routes the agent to do:

<img src=".github/assets/diffgeo-agent-workflow.png" alt="AI agent workflow for LLM driven airfoil generation using the DiffGeo skill" width="100%"/>

For agent runtimes that support repo-local skills or reusable tool instructions, register or point the runtime at `skills/diffgeo-airfoil-generation/`. For simpler assistants, paste the contents of `SKILL.md` into the task context and set `DIFFGEO_ROOT` to this repository.

## Python Quickstart

Clone and install from the repository root:

```bash
git clone https://github.com/kfxw/DiffGeo.git
cd DiffGeo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Extract the bundled UIUC coordinate archive:

```bash
tar -xzf data/uiuc_airfoils.tar.gz -C data
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

Generate unconditional samples:

```bash
diffgeo-sample-unconditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --num-samples 16 \
  --output-dir outputs/pretrained_unconditional
```

Scale or shift generated coordinates:

```bash
diffgeo-transform-airfoils \
  --input outputs/pretrained_conditional/conditional_samples.npz \
  --chord-scale 1.5 \
  --shift-x 0.25 \
  --shift-y -0.05 \
  --output-dir outputs/pretrained_transformed
```

Area and max-thickness targets use the normalized unit-chord coordinate system. `chord-scale` applies uniform scaling, then `shift-x` and `shift-y` translate all coordinates.

## Installation Notes

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

## Repository Layout

```text
DiffGeo/
├── .github/assets/                 # README display assets
├── configs/                        # Full UIUC experiment config
├── data/
│   ├── uiuc_airfoils.tar.gz         # UIUC coordinate archive
│   └── splits/                      # train_full_UIUC and test_UIUC split lists
├── pretrained/uiuc_airfoil_full_v1/ # Released full-UIUC checkpoint bundle
├── scripts/                         # Thin CLI wrappers
├── skills/diffgeo-airfoil-generation/
├── src/diffgeo/                     # Package implementation
└── tests/                           # Unit and data-loading tests
```

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

## Pretrained Bundle

The repository includes a full-UIUC pretrained checkpoint bundle:

```text
pretrained/uiuc_airfoil_full_v1
```

Expected generation artifacts include:

```text
*_samples.npz
*_grid.png
*_report.txt
*_dat/*.dat
```

Transform artifacts include:

```text
transformed_airfoils.npz
transformed_grid.png
transformed_transform_report.txt
transformed_dat/*.dat
```

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

## Citation

```bibtex
@article{wei2026diffgeo,
  title={Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model},
  author={Wei, Zhen and Dufour, Edouard and Pelletier, Colin and Bauerheim, Michael and Fua, Pascal},
  journal={AIAA Journal},
  year={2026}
}
```

If you use this work for airfoil-related applications, research, or development, please also cite the DiffGeo conference-version paper, **DiffAirfoil**:

```bibtex
@inproceedings{wei2024diffairfoil,
  title={DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization},
  author={Wei, Zhen and Dufour, Edouard R. and Pelletier, Colin and Fua, Pascal and Bauerheim, Michaël},
  booktitle={AIAA AVIATION FORUM AND ASCEND 2024},
  year={2024},
  doi={10.2514/6.2024-3755}
}
```

## License

Code is released under the MIT license.
