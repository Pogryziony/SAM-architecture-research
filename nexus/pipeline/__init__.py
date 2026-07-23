"""NEXUS v1 pipeline — canonical end-to-end runner."""
from nexus.pipeline.config import (
    CONFIG_IDENTITY_SCHEMA,
    ProductionNEXUSConfig,
    PipelineIdentity,
    EncoderIdentity,
    validate_config,
)
from nexus.pipeline.runner import (
    NEXUSRunner,
    QuestionResult,
    PipelineResult,
)
from nexus.pipeline.entity_resolver import EntityResolver

__all__ = [
    "CONFIG_IDENTITY_SCHEMA",
    "ProductionNEXUSConfig",
    "PipelineIdentity",
    "EncoderIdentity",
    "validate_config",
    "NEXUSRunner",
    "QuestionResult",
    "PipelineResult",
    "EntityResolver",
]
