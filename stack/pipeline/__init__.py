"""stack/pipeline — NEXUS pipeline orchestration.

This package depends on both nexus/ (graph engine) and stack/ (learned components).
It implements the EntityResolver protocol from nexus/pipeline/ so that the
lower-level nexus/ package never directly imports stack/ modules.
"""
from stack.pipeline.resolver import (
    DialogueAwareResolver,
    ER3Resolver,
    LexicalFallbackResolver,
    LexicalResolver,
    UnionResolver,
    mention_score,
)

__all__ = [
    "DialogueAwareResolver",
    "ER3Resolver",
    "LexicalFallbackResolver",
    "LexicalResolver",
    "UnionResolver",
    "mention_score",
]
