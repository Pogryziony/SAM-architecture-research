"""Tiny config helper: load YAML into an attribute-accessible dict.

Kept deliberately minimal (no pydantic / hydra) so the POC stays readable.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested defaults.

    Nested dicts are wrapped recursively, so ``cfg.model.memory.top_k`` works.
    """

    def __init__(self, data: Dict[str, Any] | None = None):
        super().__init__()
        data = data or {}
        for k, v in data.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def get(self, name: str, default: Any = None) -> Any:
        return self[name] if name in self else default

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a YAML config file, applying optional shallow/dotted overrides.

    Overrides use dotted keys, e.g. ``{"model.memory.top_k": 8}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(data)
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)
    return cfg


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], Config):
            node[p] = Config({})
        node = node[p]
    node[parts[-1]] = Config._wrap(value)


def merge(base: Config, other: Dict[str, Any]) -> Config:
    """Deep-merge ``other`` into a copy of ``base``."""
    out = Config(copy.deepcopy(base.to_dict()))
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), Config):
            out[k] = merge(out[k], v)
        else:
            out[k] = Config._wrap(v)
    return out
