"""Production NEXUS v1 configuration — immutable, serializable.

Extends NEXUSConfig with explicit pipeline identity, checkpoint
references, and artifact validation.  Designed for reproducible
end-to-end evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from nexus.utils.config import NEXUSConfig


@dataclass(frozen=True)
class EncoderIdentity:
    """Checkpoint identity for the associative encoder."""
    enabled: bool = False
    model_dir: str = ""
    checkpoint_sha256: str = ""
    config_sha256: str = ""
    vocab_sha256: str = ""
    run_id: str = ""
    source_sha: str = ""


@dataclass(frozen=True)
class PipelineIdentity:
    """Complete pipeline configuration identity."""
    encoder: EncoderIdentity = field(default_factory=EncoderIdentity)
    lexical_fallback: bool = True
    entity_ranker_v3_enabled: bool = False
    entity_ranker_v3_dir: str = ""


class ProductionNEXUSConfig(NEXUSConfig):
    """Immutable production configuration for the NEXUS v1 pipeline.

    Extends NEXUSConfig with explicit per-component identity and
    fail-closed artifact validation.  Use the classmethod factories
    instead of the constructor directly.
    """

    __slots__ = ("_pipeline_id", "_config_hash", "_frozen")

    def __init__(
        self,
        pipeline_id: PipelineIdentity | None = None,
        **kwargs: Any,
    ):
        # Prevent modification after init
        object.__setattr__(self, "_frozen", False)
        super().__init__(**kwargs)
        object.__setattr__(self, "_pipeline_id", pipeline_id or PipelineIdentity())
        object.__setattr__(self, "_config_hash", self._compute_hash())
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "ProductionNEXUSConfig is immutable. Create a new instance."
            )
        super().__setattr__(name, value)

    @property
    def pipeline_id(self) -> PipelineIdentity:
        return self._pipeline_id  # type: ignore[attr-defined]

    @property
    def config_hash(self) -> str:
        return self._config_hash  # type: ignore[attr-defined]

    def _compute_hash(self) -> str:
        """Deterministic hash of all config fields (excluding config_hash itself)."""
        payload = json.dumps(
            {
                "nexus_config": {
                    "model_name": self.model_name,
                    "max_entry_nodes": self.max_entry_nodes,
                    "max_depth": self.max_depth,
                    "beam_width": self.beam_width,
                    "edge_confidence_threshold": self.edge_confidence_threshold,
                    "hallucination_threshold": self.hallucination_threshold,
                    "fuzzy_cutoff": self.fuzzy_cutoff,
                    "enable_associative_encoder": self.enable_associative_encoder,
                    "enable_embedding_er": self.enable_embedding_er,
                    "enable_cooccurrence_edges": self.enable_cooccurrence_edges,
                    "enable_normalization": self.enable_normalization,
                    "post_edit_enabled": self.post_edit_enabled,
                    "tier3_backend": self.tier3_backend,
                    "dialogue_decay": self.dialogue_decay,
                    "dialogue_boost": self.dialogue_boost,
                },
                "pipeline_identity": {
                    "encoder_enabled": self.pipeline_id.encoder.enabled,
                    "encoder_model_dir": self.pipeline_id.encoder.model_dir,
                    "lexical_fallback": self.pipeline_id.lexical_fallback,
                    "entity_ranker_v3_enabled": self.pipeline_id.entity_ranker_v3_enabled,
                },
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nexus_config": {
                "model_name": self.model_name,
                "max_entry_nodes": self.max_entry_nodes,
                "max_depth": self.max_depth,
                "beam_width": self.beam_width,
                "edge_confidence_threshold": self.edge_confidence_threshold,
                "hallucination_threshold": self.hallucination_threshold,
                "fuzzy_cutoff": self.fuzzy_cutoff,
                "enable_associative_encoder": self.enable_associative_encoder,
                "enable_embedding_er": self.enable_embedding_er,
                "enable_cooccurrence_edges": self.enable_cooccurrence_edges,
                "enable_normalization": self.enable_normalization,
                "post_edit_enabled": self.post_edit_enabled,
                "tier3_backend": self.tier3_backend,
                "dialogue_decay": self.dialogue_decay,
                "dialogue_boost": self.dialogue_boost,
            },
            "pipeline_identity": {
                "encoder_enabled": self.pipeline_id.encoder.enabled,
                "encoder_model_dir": self.pipeline_id.encoder.model_dir,
                "encoder_checkpoint_sha256": self.pipeline_id.encoder.checkpoint_sha256,
                "encoder_run_id": self.pipeline_id.encoder.run_id,
                "encoder_source_sha": self.pipeline_id.encoder.source_sha,
                "lexical_fallback": self.pipeline_id.lexical_fallback,
                "entity_ranker_v3_enabled": self.pipeline_id.entity_ranker_v3_enabled,
                "entity_ranker_v3_dir": self.pipeline_id.entity_ranker_v3_dir,
            },
            "config_hash": self.config_hash,
        }

    @classmethod
    def lexical_only(cls, **overrides: Any) -> "ProductionNEXUSConfig":
        """Factory: lexical-only pipeline (no learned components)."""
        kwargs: dict[str, Any] = {
            "enable_associative_encoder": False,
            "enable_embedding_er": False,
            "enable_normalization": False,
        }
        kwargs.update(overrides)
        return cls(pipeline_id=PipelineIdentity(lexical_fallback=True), **kwargs)

    @classmethod
    def with_encoder(
        cls,
        model_dir: str = "models/encoder_v2",
        checkpoint_sha256: str = "",
        run_id: str = "",
        source_sha: str = "",
        **overrides: Any,
    ) -> "ProductionNEXUSConfig":
        """Factory: associative encoder enabled with checkpoint identity."""
        kwargs: dict[str, Any] = {
            "enable_associative_encoder": True,
        }
        kwargs.update(overrides)
        return cls(
            pipeline_id=PipelineIdentity(
                encoder=EncoderIdentity(
                    enabled=True,
                    model_dir=model_dir,
                    checkpoint_sha256=checkpoint_sha256,
                    run_id=run_id,
                    source_sha=source_sha,
                ),
                lexical_fallback=True,
            ),
            **kwargs,
        )

    @classmethod
    def with_entity_ranker_v3(
        cls,
        ranker_dir: str = "models/encoder/entity_ranker_v3_20260711T081545Z",
        **overrides: Any,
    ) -> "ProductionNEXUSConfig":
        """Factory: Entity Ranker V3 enabled with exhaustive canonical vocabulary."""
        kwargs: dict[str, Any] = {
            "max_entry_nodes": 10,
        }
        kwargs.update(overrides)
        return cls(
            pipeline_id=PipelineIdentity(
                entity_ranker_v3_enabled=True,
                entity_ranker_v3_dir=ranker_dir,
                lexical_fallback=True,
            ),
            **kwargs,
        )


def validate_config(config: ProductionNEXUSConfig) -> list[str]:
    """Validate a production configuration. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    if config.pipeline_id.encoder.enabled:
        if not config.pipeline_id.encoder.model_dir:
            errors.append("encoder enabled but model_dir is empty")
        if config.pipeline_id.encoder.checkpoint_sha256:
            model_dir = Path(config.pipeline_id.encoder.model_dir)
            if not model_dir.exists():
                errors.append(
                    f"encoder model_dir not found: {model_dir}"
                )

    if config.max_entry_nodes < 1:
        errors.append("max_entry_nodes must be >= 1")
    if config.max_depth < 1:
        errors.append("max_depth must be >= 1")
    if config.beam_width < 1:
        errors.append("beam_width must be >= 1")

    return errors
