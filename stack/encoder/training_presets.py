"""Training presets loader — loads predefined training intensity profiles.

Usage:
    from stack.encoder.training_presets import load_presets, apply_preset

    presets = load_presets()
    params = apply_preset("quick", {"epochs": 40}, cli_overrides={"epochs": 3})
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PRESETS_PATH = Path(__file__).parents[2] / "training" / "presets.json"


def load_presets(path: str | Path | None = None) -> dict[str, Any]:
    """Load the training presets JSON."""
    p = Path(path) if path else DEFAULT_PRESETS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Presets file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_preset(name: str, path: str | Path | None = None) -> dict[str, Any]:
    presets = load_presets(path)
    available = presets.get("presets", {})
    if name not in available:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(available.keys())}")
    return dict(available[name])


def list_presets(path: str | Path | None = None) -> list[str]:
    return sorted(load_presets(path).get("presets", {}).keys())


def get_model_defaults(model: str, path: str | Path | None = None) -> dict[str, Any]:
    return dict(load_presets(path).get("model_defaults", {}).get(model, {}))


def apply_preset(
    preset_name: str,
    current_params: dict[str, Any] | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
    model_type: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply a preset on top of current parameters. CLI overrides win.

    Priority: cli_overrides > preset > model_defaults > current_params.
    None values in cli_overrides are ignored (not provided).
    """
    merged = dict(current_params or {})

    if model_type:
        for k, v in get_model_defaults(model_type, path).items():
            if cli_overrides is None or k not in cli_overrides or cli_overrides[k] is None:
                merged.setdefault(k, v)

    preset = get_preset(preset_name, path)
    for k, v in preset.items():
        if cli_overrides is None or k not in cli_overrides or cli_overrides[k] is None:
            merged[k] = v

    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                merged[k] = v

    return merged
