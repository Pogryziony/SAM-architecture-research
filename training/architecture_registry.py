"""Fail-closed Realizer architecture registry.

Loads ``training/REJECTED_ARCHITECTURES.json`` so training entrypoints can
refuse rejected configs without re-reading scattered pilot reports.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parent / "REJECTED_ARCHITECTURES.json"
SCHEMA_VERSION = "nexus-rejected-architectures-v1"


class ArchitectureBlockedError(RuntimeError):
    """Raised when a rejected architecture is requested for training."""


@lru_cache(maxsize=1)
def load_architecture_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the architecture registry."""
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported architecture registry schema: {raw.get('schema_version')!r}"
        )
    for key in ("rejected", "accepted", "experimental", "policy"):
        if key not in raw:
            raise ValueError(f"architecture registry missing key: {key}")
    return raw


def rejected_ids(registry: dict[str, Any] | None = None) -> set[str]:
    data = registry if registry is not None else load_architecture_registry()
    return {str(item["id"]) for item in data["rejected"]}


def accepted_ids(registry: dict[str, Any] | None = None) -> set[str]:
    data = registry if registry is not None else load_architecture_registry()
    return {str(item["id"]) for item in data["accepted"]}


def find_by_config(
    config_path: str | Path,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the registry entry whose ``config`` matches *config_path*."""
    data = registry if registry is not None else load_architecture_registry()
    needle = Path(config_path).as_posix().replace("\\", "/")
    needle_name = Path(needle).name
    for section in ("rejected", "accepted", "experimental"):
        for item in data[section]:
            cfg = str(item.get("config", "")).replace("\\", "/")
            if not cfg:
                continue
            if cfg == needle or Path(cfg).name == needle_name:
                return {**item, "section": section}
    return None


def assert_training_allowed(
    config_path: str | Path,
    *,
    action: str = "full_training",
    registry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Refuse rejected architectures for disallowed training actions.

    Returns the matched registry entry when found. Returns ``None`` when the
    config is unknown to the registry (callers may still apply their own gates).
    """
    entry = find_by_config(config_path, registry=registry)
    if entry is None:
        return None
    if entry["section"] != "rejected":
        return entry

    allowed = set(entry.get("allowed_actions") or [])
    if action in allowed:
        return entry

    raise ArchitectureBlockedError(
        f"architecture {entry['id']!r} is {entry['status']}: {entry['reason']} "
        f"(action={action!r} not in allowed_actions={sorted(allowed)})"
    )


def describe_production_profiles() -> dict[str, str]:
    """Short map of recommended runtime factories to accepted architecture ids."""
    return {
        "library_default": "synth (historical Stage 2 semantics; not the production QA profile)",
        "ProductionNEXUSConfig.pointer_copy()": "pointer_copy_v3",
        "ProductionNEXUSConfig.comparison_plan()": "comparison_plan_v3",
        "ProductionNEXUSConfig.grounded()": "pointer_copy_v3 + comparison_plan_v3",
    }


__all__ = [
    "ArchitectureBlockedError",
    "REGISTRY_PATH",
    "SCHEMA_VERSION",
    "accepted_ids",
    "assert_training_allowed",
    "describe_production_profiles",
    "find_by_config",
    "load_architecture_registry",
    "rejected_ids",
]
