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
_METADATA_KEYS = {"note", "description"}
_ALIASES = {"lr": "learning_rate"}


def _normalize_params(values: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if key in _METADATA_KEYS:
            continue
        normalized[_ALIASES.get(key, key)] = value
    return normalized


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
    merged = _normalize_params(current_params)
    normalized_cli = _normalize_params(cli_overrides)

    if model_type:
        for k, v in get_model_defaults(model_type, path).items():
            if k not in normalized_cli or normalized_cli[k] is None:
                merged.setdefault(k, v)

    preset = _normalize_params(get_preset(preset_name, path))
    for k, v in preset.items():
        if k not in normalized_cli or normalized_cli[k] is None:
            merged[k] = v

    if normalized_cli:
        for k, v in normalized_cli.items():
            if v is not None:
                merged[k] = v

    for key in ("epochs", "patience", "batch_size"):
        if key in merged and (not isinstance(merged[key], int) or merged[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if "learning_rate" in merged and float(merged["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")

    return merged
