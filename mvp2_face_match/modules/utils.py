from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value.strip('"').strip("'")
            cfg[key.strip()] = parsed
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def project_relative_default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "config_face.yaml"


def resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def apply_face_model_profile(cfg: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    updated = dict(cfg)
    selected = str(profile or updated.get("face_model_profile", "stable")).lower()
    if selected not in {"stable", "accurate"}:
        selected = "stable"
    profile_path = updated.get(f"arcface_model_path_{selected}")
    if profile_path:
        updated["arcface_model_path"] = profile_path
    updated["face_model_profile"] = selected
    return updated
