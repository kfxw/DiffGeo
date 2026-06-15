from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config, resolve_path
from .diffusion import DiffusionSchedule, sample_loop
from .geometry import airfoil_metrics_numpy, max_thickness, polygon_area, surface_order_penalty
from .io import get_device, load_autodecoder, load_diffusion
from .visualization import plot_airfoil_grid, save_airfoil_dat_files, save_airfoils_npz

SURFACE_ORDER_TOLERANCE = 1e-3


def _resolve_checkpoints(cfg: dict[str, Any], root: Path, pretrained_dir: str | Path | None) -> tuple[Path, Path]:
    if pretrained_dir is not None:
        p = Path(pretrained_dir)
        if not p.is_absolute():
            p = root / p
        return p / "autodecoder.pt", p / "diffusion.pt"
    return resolve_path(cfg["autodecoder"]["checkpoint"], root), resolve_path(cfg["diffusion"]["checkpoint"], root)


def _metric_loss(
    points: torch.Tensor,
    target_area: float | None,
    target_max_thickness: float | None,
    area_weight: float,
    thickness_weight: float,
    surface_order_weight: float,
) -> torch.Tensor:
    loss = points.new_zeros(points.shape[0])
    if target_area is not None:
        scale = max(abs(float(target_area)), 1e-6)
        loss = loss + area_weight * ((polygon_area(points) - float(target_area)) / scale).pow(2)
    if target_max_thickness is not None:
        scale = max(abs(float(target_max_thickness)), 1e-6)
        loss = loss + thickness_weight * ((max_thickness(points) - float(target_max_thickness)) / scale).pow(2)
    if surface_order_weight > 0.0:
        loss = loss + surface_order_weight * surface_order_penalty(points)
    return loss


def _summarize_metrics(points: np.ndarray, target_area: float | None, target_max_thickness: float | None) -> dict[str, float]:
    metrics = [airfoil_metrics_numpy(xy) for xy in points]
    out: dict[str, float] = {}
    for key in ["area", "max_thickness"]:
        values = np.asarray([m[key] for m in metrics], dtype=np.float64)
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"min_{key}"] = float(np.min(values))
        out[f"max_{key}"] = float(np.max(values))
        target = target_area if key == "area" else target_max_thickness
        if target is not None:
            err = np.abs(values - float(target))
            out[f"{key}_mean_abs_error"] = float(np.mean(err))
            out[f"{key}_max_abs_error"] = float(np.max(err))
    gaps = np.asarray([m["min_surface_gap"] for m in metrics], dtype=np.float64)
    out["mean_min_surface_gap"] = float(np.mean(gaps))
    out["min_min_surface_gap"] = float(np.min(gaps))
    out["invalid_surface_order_fraction"] = float(np.mean(gaps < -SURFACE_ORDER_TOLERANCE))
    return out


def _format_optional(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.8f}"


def sample(
    config_path: str | Path,
    pretrained_dir: str | Path | None = None,
    target_area: float | None = None,
    target_max_thickness: float | None = None,
    num_samples: int | None = None,
    output_dir: str | Path | None = None,
    guidance_scale: float | None = None,
    refinement_steps: int | None = None,
    refinement_lr: float | None = None,
    refinement_prior_weight: float | None = None,
    area_weight: float = 1.0,
    thickness_weight: float = 1.0,
    surface_order_weight: float | None = None,
    force_cpu: bool = False,
) -> Path:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    device = get_device(force_cpu)
    sampling_cfg = cfg["sampling"]
    n_samples = int(num_samples if num_samples is not None else sampling_cfg["num_samples"])
    scale = float(guidance_scale if guidance_scale is not None else sampling_cfg.get("guidance_scale", 0.0))
    refine_steps = int(
        refinement_steps if refinement_steps is not None else sampling_cfg.get("latent_refinement_steps", 0)
    )
    refine_lr = float(refinement_lr if refinement_lr is not None else sampling_cfg.get("latent_refinement_lr", 0.05))
    refine_prior = float(
        refinement_prior_weight
        if refinement_prior_weight is not None
        else sampling_cfg.get("latent_refinement_prior_weight", 0.0)
    )
    order_weight = float(
        surface_order_weight
        if surface_order_weight is not None
        else sampling_cfg.get("surface_order_weight", 0.0)
    )
    use_config_targets = target_area is None and target_max_thickness is None
    if use_config_targets:
        target_area = sampling_cfg.get("target_area")
        target_max_thickness = sampling_cfg.get("target_max_thickness")
    target_area = None if target_area is None else float(target_area)
    target_max_thickness = None if target_max_thickness is None else float(target_max_thickness)

    autodecoder_ckpt, diffusion_ckpt_path = _resolve_checkpoints(cfg, root, pretrained_dir)
    autodecoder, template, _ = load_autodecoder(autodecoder_ckpt, device)
    diffusion, diffusion_ckpt = load_diffusion(diffusion_ckpt_path, device)
    schedule = DiffusionSchedule(**diffusion_ckpt["schedule"], device=device)
    mean = diffusion_ckpt["latent_mean"].to(device)
    std = diffusion_ckpt["latent_std"].to(device)

    def guidance(pred_x0_norm: torch.Tensor) -> torch.Tensor:
        z = pred_x0_norm * std[None] + mean[None]
        decoded = autodecoder.decode(z, template)
        return _metric_loss(
            decoded,
            target_area,
            target_max_thickness,
            area_weight,
            thickness_weight,
            order_weight,
        )

    def refine_latents(z_norm_in: torch.Tensor) -> torch.Tensor:
        if refine_steps <= 0 or (target_area is None and target_max_thickness is None):
            return z_norm_in.detach()
        z_var = z_norm_in.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z_var], lr=refine_lr)
        for _ in range(refine_steps):
            z = z_var * std[None] + mean[None]
            decoded = autodecoder.decode(z, template)
            loss = _metric_loss(
                decoded,
                target_area,
                target_max_thickness,
                area_weight,
                thickness_weight,
                order_weight,
            ).mean()
            if refine_prior > 0.0:
                loss = loss + refine_prior * z_var.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return z_var.detach()

    baseline_z_norm = sample_loop(diffusion, schedule, (n_samples, diffusion.latent_dim), device=device, progress=True)
    z_norm = sample_loop(
        diffusion,
        schedule,
        (n_samples, diffusion.latent_dim),
        device=device,
        guidance_fn=guidance if (target_area is not None or target_max_thickness is not None) else None,
        guidance_scale=scale,
        progress=True,
    )
    z_norm = refine_latents(z_norm)

    with torch.no_grad():
        baseline_z = baseline_z_norm * std[None] + mean[None]
        baseline_points = autodecoder.decode(baseline_z, template).detach().cpu().numpy().astype(np.float32)
        z = z_norm * std[None] + mean[None]
        points = autodecoder.decode(z, template).detach().cpu().numpy().astype(np.float32)

    out_dir = resolve_path(output_dir if output_dir is not None else sampling_cfg["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "conditional_samples.npz"
    baseline_npz_path = out_dir / "conditional_unguided_baseline_samples.npz"
    save_airfoils_npz(npz_path, points=points, latents=z.detach().cpu().numpy())
    save_airfoils_npz(baseline_npz_path, points=baseline_points, latents=baseline_z.detach().cpu().numpy())
    save_airfoil_dat_files(points, out_dir / "conditional_dat", prefix="conditional")
    save_airfoil_dat_files(baseline_points, out_dir / "conditional_unguided_baseline_dat", prefix="unguided_baseline")

    guided_metrics = _summarize_metrics(points, target_area, target_max_thickness)
    baseline_metrics = _summarize_metrics(baseline_points, target_area, target_max_thickness)
    titles = [
        f"A={airfoil_metrics_numpy(xy)['area']:.3f}, T={airfoil_metrics_numpy(xy)['max_thickness']:.3f}"
        for xy in points[: min(16, len(points))]
    ]
    baseline_titles = [
        f"A={airfoil_metrics_numpy(xy)['area']:.3f}, T={airfoil_metrics_numpy(xy)['max_thickness']:.3f}"
        for xy in baseline_points[: min(16, len(baseline_points))]
    ]
    plot_airfoil_grid(points[: min(16, len(points))], out_dir / "conditional_grid.png", titles=titles, n_cols=4)
    plot_airfoil_grid(
        baseline_points[: min(16, len(baseline_points))],
        out_dir / "conditional_unguided_baseline_grid.png",
        titles=baseline_titles,
        n_cols=4,
    )

    lines = [
        f"target_area: {_format_optional(target_area)}",
        f"target_max_thickness: {_format_optional(target_max_thickness)}",
        f"guidance_scale: {scale:.8f}",
        f"latent_refinement_steps: {refine_steps}",
        f"latent_refinement_lr: {refine_lr:.8f}",
        f"latent_refinement_prior_weight: {refine_prior:.8f}",
        f"area_weight: {area_weight:.8f}",
        f"thickness_weight: {thickness_weight:.8f}",
        f"surface_order_weight: {order_weight:.8f}",
        f"surface_order_tolerance: {SURFACE_ORDER_TOLERANCE:.8f}",
        f"num_samples: {n_samples}",
    ]
    for prefix, metrics in [("guided", guided_metrics), ("unguided_baseline", baseline_metrics)]:
        for key in sorted(metrics):
            lines.append(f"{prefix}_{key}: {metrics[key]:.8f}")
    improvements = []
    if target_area is not None:
        improvements.append(
            baseline_metrics["area_mean_abs_error"] - guided_metrics["area_mean_abs_error"]
        )
        lines.append(f"area_mean_abs_error_improvement: {improvements[-1]:.8f}")
    if target_max_thickness is not None:
        improvements.append(
            baseline_metrics["max_thickness_mean_abs_error"] - guided_metrics["max_thickness_mean_abs_error"]
        )
        lines.append(f"max_thickness_mean_abs_error_improvement: {improvements[-1]:.8f}")
    if improvements:
        lines.append(f"guided_improves_over_unguided: {str(all(x > 0.0 for x in improvements)).lower()}")
    lines.extend([
        f"guided_samples: {npz_path.name}",
        f"unguided_baseline_samples: {baseline_npz_path.name}",
    ])
    (out_dir / "conditional_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate conditional 2D airfoils with DiffGeo pretrained or trained checkpoints.")
    parser.add_argument("--config", default="configs/full_uiuc.yaml")
    parser.add_argument("--pretrained-dir", default=None)
    parser.add_argument("--target-area", type=float, default=None)
    parser.add_argument("--target-max-thickness", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--refinement-steps", type=int, default=None)
    parser.add_argument("--refinement-lr", type=float, default=None)
    parser.add_argument("--refinement-prior-weight", type=float, default=None)
    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--thickness-weight", type=float, default=1.0)
    parser.add_argument("--surface-order-weight", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = sample(
        args.config,
        pretrained_dir=args.pretrained_dir,
        target_area=args.target_area,
        target_max_thickness=args.target_max_thickness,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        refinement_prior_weight=args.refinement_prior_weight,
        area_weight=args.area_weight,
        thickness_weight=args.thickness_weight,
        surface_order_weight=args.surface_order_weight,
        force_cpu=args.cpu,
    )
    print(f"saved conditional samples: {path}")


if __name__ == "__main__":
    main()
