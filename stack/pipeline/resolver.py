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
    def from_directory(cls, model_dir: str, graph: InMemoryGraphStore,
                       *, weights_path: str | None = None,
                       verify_sha256: bool = True) -> "ER3Resolver":
        """Load ER3 from a model directory or explicit weights path.

        Args:
            model_dir: Directory containing manifest.json and vocab.json.
            graph: The populated graph.
            weights_path: Optional explicit path to weights.pt. If not given,
                looks for $ER3_WEIGHTS_PATH env var, then falls back to
                model_dir/weights.pt.
            verify_sha256: If True, verify weights match manifest before loading.

        Raises:
            FileNotFoundError: If weights cannot be located.
            ValueError: If SHA-256 verification fails.
        """
        import hashlib, json, os
        from pathlib import Path
        from stack.encoder.entity_ranker_v3 import load_ranker_v3

        manifest_path = Path(model_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"ER3 manifest not found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha256 = manifest["files"]["weights.pt"]["sha256"]

        # Resolve weights path
        if weights_path is None:
            weights_path = os.environ.get("ER3_WEIGHTS_PATH", "")
        if not weights_path:
            weights_path = str(Path(model_dir) / "weights.pt")

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(
                f"ER3 weights not found at {weights_file}. "
                "Set ER3_WEIGHTS_PATH environment variable or pass --weights-path."
            )

        if verify_sha256:
            actual = hashlib.sha256(weights_file.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"ER3 weights SHA-256 mismatch:\n"
                    f"  expected: {expected_sha256}\n"
                    f"  actual:   {actual}\n"
                    f"  path:     {weights_file}"
                )

        model, tokenizer, config = load_ranker_v3(str(model_dir))
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
