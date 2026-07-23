"""Fair comparison baseline interfaces for NEXUS evaluation.

These modules define provider-neutral arms. When credentials or local models
are unavailable, runners must emit ``NOT_RUN`` rather than fabricating scores.
Deterministic placeholders (``SynthesizingModel``, ``EvidenceBlindModel``) are
**not** real LLM or modern RAG baselines.
"""

from nexus.baselines.interface import (
    BaselineArm,
    BaselineRequest,
    BaselineResult,
    BaselineStatus,
    run_baseline_or_not_run,
)
from nexus.baselines.registry import BASELINE_ARMS, get_arm, list_arms

__all__ = [
    "BASELINE_ARMS",
    "BaselineArm",
    "BaselineRequest",
    "BaselineResult",
    "BaselineStatus",
    "get_arm",
    "list_arms",
    "run_baseline_or_not_run",
]
