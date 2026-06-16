#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/kfxw/DiffGeo.git"


def looks_like_diffgeo(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "src" / "diffgeo").is_dir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate or clone the DiffGeo repository for agent airfoil generation.")
    parser.add_argument("--repo-url", default=os.environ.get("DIFFGEO_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--target", default=os.environ.get("DIFFGEO_ROOT", str(Path.home() / ".cache" / "diffgeo" / "DiffGeo")))
    args = parser.parse_args()

    env_root = os.environ.get("DIFFGEO_ROOT")
    if env_root and looks_like_diffgeo(Path(env_root)):
        print(Path(env_root).resolve())
        return
    cwd = Path.cwd()
    if looks_like_diffgeo(cwd):
        print(cwd.resolve())
        return

    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        if not args.repo_url:
            raise SystemExit(
                "DiffGeo repository URL is not set. "
                "Set DIFFGEO_REPO_URL or pass --repo-url before cloning."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", args.repo_url, str(target)])
    if not looks_like_diffgeo(target):
        raise SystemExit(f"{target} does not look like a DiffGeo repository")
    pretrained = target / "pretrained" / "uiuc_airfoil_full_v1"
    missing = [name for name in ["autodecoder.pt", "diffusion.pt", "uiuc_latents.npz"] if not (pretrained / name).exists()]
    if missing:
        raise SystemExit(f"Missing pretrained assets in {pretrained}: {', '.join(missing)}")
    print(target)


if __name__ == "__main__":
    main()
