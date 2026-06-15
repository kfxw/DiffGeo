from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    project_root = Path(cfg.get("project_root", "."))
    if not project_root.is_absolute():
        project_root = (config_path.parent.parent / project_root).resolve()
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(project_root)
    return cfg


def resolve_path(path: str | Path, project_root: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(project_root).resolve() / p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
