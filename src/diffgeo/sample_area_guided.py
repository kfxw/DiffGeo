from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .config import load_config
from .sample_conditional import sample as sample_conditional


def _default_target_area(config_path: str | Path, target_area: float | None) -> float | None:
    if target_area is not None:
        return target_area
    cfg = load_config(config_path)
    return cfg.get("sampling", {}).get("target_area")


def sample(
    config_path: str | Path,
    target_area: float | None = None,
    num_samples: int | None = None,
    guidance_scale: float | None = None,
    refinement_steps: int | None = None,
    refinement_lr: float | None = None,
    refinement_prior_weight: float | None = None,
    force_cpu: bool = False,
) -> Path:
    warnings.warn(
        "diffgeo-sample-area-guided is deprecated. Use diffgeo-sample-conditional "
        "with --target-area instead. Outputs use the canonical conditional_* names.",
        DeprecationWarning,
        stacklevel=2,
    )
    resolved_target = _default_target_area(config_path, target_area)
    return sample_conditional(
        config_path,
        target_area=resolved_target,
        target_max_thickness=None,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        refinement_steps=refinement_steps,
        refinement_lr=refinement_lr,
        refinement_prior_weight=refinement_prior_weight,
        force_cpu=force_cpu,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated area-only wrapper for conditional 2D airfoil generation. "
            "Prefer diffgeo-sample-conditional --target-area."
        )
    )
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--target-area", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--refinement-steps", type=int, default=None)
    parser.add_argument("--refinement-lr", type=float, default=None)
    parser.add_argument("--refinement-prior-weight", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    warnings.simplefilter("default", DeprecationWarning)
    args = build_argparser().parse_args()
    path = sample(
        args.config,
        target_area=args.target_area,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        refinement_prior_weight=args.refinement_prior_weight,
        force_cpu=args.cpu,
    )
    print(f"saved conditional samples: {path}")


if __name__ == "__main__":
    main()
