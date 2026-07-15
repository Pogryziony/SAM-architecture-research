"""stack/pipeline — ER3 entity resolver and high-level pipeline orchestration.

This package implements the EntityResolver protocol from nexus/pipeline/
so that nexus/ never directly imports stack/ modules.

ER3Resolver wraps the Entity Ranker V3 checkpoint for use in the canonical
NEXUS pipeline. It implements exhaustive canonical-vocabulary ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.entity_resolver import EntityResolver


class ER3Resolver:
    """Entity Ranker V3 resolver implementing the EntityResolver protocol.

    Loads the ER3 checkpoint and tokenizer from a model directory.
    Performs exhaustive canonical-vocabulary ranking: all canonical-pattern
    nodes (Exp_*, Concept_*, Decision_*) are scored and the top-K are
    returned after canonical deduplication.

    Usage:
        resolver = ER3Resolver.from_directory("models/encoder/entity_ranker_v3_<TS>/")
        entities = resolver.resolve("What is NEXUS?", graph)
    """

    def __init__(self, model, tokenizer, graph: InMemoryGraphStore):
        import torch
        self._model = model
        self._tokenizer = tokenizer
        self._graph = graph
        # Pre-compute canonical entities and text
        from stack.encoder.canonical_mapping import _is_canonical_id, build_canonical_mapping
        from stack.encoder.entity_text import build_entity_text

        self._canonical_ids = sorted(
            nid for nid in graph._nodes
            if _is_canonical_id(str(nid)) and graph.get_node(nid) is not None
        )
        self._canonical_texts = [build_entity_text(cid, graph) for cid in self._canonical_ids]
        self._mapping = build_canonical_mapping(graph)

    @classmethod
    def from_directory(cls, model_dir: str, graph: InMemoryGraphStore) -> "ER3Resolver":
        """Load ER3 from a model directory containing weights.pt, vocab.json, config.json."""
        from stack.encoder.entity_ranker_v3 import load_ranker_v3
        model, tokenizer, config = load_ranker_v3(model_dir)
        model.eval()
        return cls(model, tokenizer, graph)

    def resolve(self, question: str, graph: InMemoryGraphStore) -> list[str]:
        """Resolve entities using ER3 exhaustive canonical-vocabulary ranking."""
        import torch
        from stack.encoder.canonical_mapping import apply_canonical_mapping

        offsets, indices = self._tokenizer.tokenize_batch([question])
        with torch.no_grad():
            scores = self._model(
                torch.tensor(indices),
                torch.tensor(offsets[:-1]),
                self._canonical_texts,
                self._tokenizer,
            )
        ranked_indices = torch.argsort(scores[0], descending=True).tolist()
        ranked_ids = [self._canonical_ids[i] for i in ranked_indices]
        return apply_canonical_mapping(ranked_ids, self._mapping, top_k=10)


class LexicalFallbackResolver:
    """Trivial lexical resolver — returns empty list (lexical parser handles it).

    Used when the caller wants explicit fallback tracking.
    """

    def resolve(self, question: str, graph: InMemoryGraphStore) -> list[str]:
        return []
