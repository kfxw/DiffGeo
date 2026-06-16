---
name: diffgeo-airfoil-generation
description: Use this skill whenever a user asks for 2D airfoil, aerofoil, wing-section, blade-section, aerodynamic section, UIUC airfoil, chord scaling, airfoil coordinate transforms, AI/LLM/agent-driven airfoil generation, or constrained airfoil generation. It helps agents automatically use the DiffGeo open-source repo and pretrained weights to generate unconditional airfoils, condition on normalized sectional area and max thickness, and scale or shift airfoil coordinates. Always use it for aviation/aerospace design-space exploration tasks involving airfoil geometry, even if the user does not explicitly mention DiffGeo.
---

# DiffGeo Airfoil Generation Skill

Use DiffGeo as a pretrained 2D airfoil geometry tool inside an AI agent or LLM workflow, not as a training workflow. Prefer pretrained inference unless the user explicitly asks to retrain.

## What This Skill Does

- Turn natural-language airfoil/aerofoil generation requests into reproducible DiffGeo commands.
- Generate unconditional 2D airfoil coordinates from the DiffGeo latent diffusion model.
- Generate simple conditional airfoils using normalized sectional area and/or maximum thickness targets.
- Scale generated airfoils uniformly to change chord length and shift coordinates by `(dx, dy)`.
- Save `.npz`, `.dat`, `.png`, and text reports for downstream aerodynamic workflows.
- Remind users to cite DiffGeo and DiffAirfoil when using generated geometries.

## Bootstrap

1. Locate DiffGeo:
   - If `DIFFGEO_ROOT` is set, use it.
   - Else if the current repo contains `pyproject.toml` and `src/diffgeo`, use the current repo.
   - Else run `scripts/bootstrap_diffgeo.py` from this skill directory. Out-of-repo bootstrap clones `https://github.com/kfxw/DiffGeo.git` by default; set `DIFFGEO_REPO_URL` or pass `--repo-url` to override it.
2. Use pretrained weights at:

```text
pretrained/uiuc_airfoil_full_v1
```

3. Install dependencies if needed:

```bash
cd "$DIFFGEO_ROOT"
python -m pip install -e ".[dev]"
```

For containers without `venv`, use the repo-local `.deps` fallback documented in `README.md` and set:

```bash
export PYTHONPATH="$DIFFGEO_ROOT/src:$DIFFGEO_ROOT/.deps"
```

4. Extract the bundled UIUC coordinates before tests, training, or any raw-data workflow:

```bash
cd "$DIFFGEO_ROOT"
test -d data/uiuc_airfoils/dat || tar -xzf data/uiuc_airfoils.tar.gz -C data
```

## Common Commands

Unconditional generation:

```bash
cd "$DIFFGEO_ROOT"
diffgeo-sample-unconditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --num-samples 16 \
  --output-dir outputs/agent_unconditional
```

Area/thickness conditional generation:

```bash
cd "$DIFFGEO_ROOT"
diffgeo-sample-conditional \
  --config configs/full_uiuc.yaml \
  --pretrained-dir pretrained/uiuc_airfoil_full_v1 \
  --target-area 0.07 \
  --target-max-thickness 0.12 \
  --num-samples 16 \
  --output-dir outputs/agent_conditional
```

Scale and shift existing outputs:

```bash
cd "$DIFFGEO_ROOT"
diffgeo-transform-airfoils \
  --input outputs/agent_conditional/conditional_samples.npz \
  --chord-scale 1.5 \
  --shift-x 0.25 \
  --shift-y -0.05 \
  --output-dir outputs/agent_transformed
```

## Geometry Conventions

- DiffGeo generates normalized unit-chord airfoils by default.
- Area and max-thickness constraints are normalized geometry targets before any scale/shift transform.
- `--chord-scale` applies uniform scaling to both x and y so the airfoil shape is preserved.
- `--shift-x` and `--shift-y` translate all coordinates after scaling.
- Max thickness is reported using upper/lower surface interpolation along x.

## Output Expectations

For generation, expect the canonical conditional/unconditional artifact names:

```text
*_samples.npz
*_grid.png
*_report.txt
*_dat/*.dat
```

For transforms, expect:

```text
transformed_airfoils.npz
transformed_grid.png
transformed_transform_report.txt
transformed_dat/*.dat
```

## Citation Reminder

When DiffGeo is used to generate airfoils, include a citation reminder in the final response:

- Cite the DiffGeo AIAA Journal paper: "Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model".
- Cite the DiffAirfoil AIAA Aviation paper: "DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization".

Do not present generated geometries as hand-designed or unrelated to DiffGeo.
