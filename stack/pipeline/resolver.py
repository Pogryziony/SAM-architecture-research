"""stack/pipeline — ER3 entity resolver and high-level pipeline orchestration.

This package implements the EntityResolver protocol from nexus/pipeline/
so that nexus/ never directly imports stack/ modules.

ER3Resolver wraps the Entity Ranker V3 checkpoint for use in the canonical
NEXUS pipeline. It implements exhaustive canonical-vocabulary ranking.
UnionResolver merges lexical mentions with ER3 ranks and prunes hub noise
before traversal handoff (no retraining).
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Callable

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.entity_resolver import (
    EntityResolver,
    ResolutionCandidate,
    ResolutionResult,
    coerce_resolution_result,
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_MENTION_STOP_PARTS = frozenset({
    "exp",
    "concept",
    "decision",
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "a",
    "an",
})


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
        # Entity descriptions are static for the lifetime of the resolver.
        # Precomputing their projections removes repeated neural work from
        # every question without changing scores or rankings.
        self._entity_projections = None
        if all(
            hasattr(self._model, name)
            for name in ("encode_entities", "project_entities")
        ):
            with torch.no_grad():
                entity_embeddings = self._model.encode_entities(
                    self._canonical_texts, self._tokenizer
                )
                self._entity_projections = self._model.project_entities(
                    entity_embeddings
                )

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
        weights_manifest = manifest["files"]["weights.pt"]
        expected_sha256 = weights_manifest["sha256"]
        expected_size = int(weights_manifest.get("size", 0))

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
            actual_size = weights_file.stat().st_size
            if expected_size and actual_size != expected_size:
                raise ValueError(
                    f"ER3 weights size mismatch: expected {expected_size}, "
                    f"actual {actual_size}, path: {weights_file}"
                )
            actual = hashlib.sha256(weights_file.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"ER3 weights SHA-256 mismatch:\n"
                    f"  expected: {expected_sha256}\n"
                    f"  actual:   {actual}\n"
                    f"  path:     {weights_file}"
                )

        model, tokenizer, config = load_ranker_v3(
            str(model_dir), weights_path=str(weights_file)
        )
        model.eval()
        return cls(model, tokenizer, graph)

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        """Resolve entities using ER3 exhaustive canonical-vocabulary ranking."""
        import torch
        from stack.encoder.canonical_mapping import apply_canonical_mapping

        started = time.perf_counter()
        offsets, indices = self._tokenizer.tokenize_batch([question])
        with torch.no_grad():
            if self._entity_projections is not None and all(
                hasattr(self._model, name)
                for name in ("encode_question", "project_question", "score")
            ):
                question_encoding = self._model.encode_question(
                    torch.tensor(indices), torch.tensor(offsets[:-1])
                )
                question_projection = self._model.project_question(
                    question_encoding
                )
                scores = self._model.score(
                    question_projection,
                    self._entity_projections.unsqueeze(0),
                )
            else:
                scores = self._model(
                    torch.tensor(indices),
                    torch.tensor(offsets[:-1]),
                    self._canonical_texts,
                    self._tokenizer,
                )
        scored = sorted(
            (
                (self._canonical_ids[index], float(scores[0][index].detach()))
                for index in range(len(self._canonical_ids))
            ),
            key=lambda item: (-item[1], item[0]),
        )
        ranked_ids = [entity_id for entity_id, _ in scored]
        selected = apply_canonical_mapping(ranked_ids, self._mapping, top_k=10)
        return ResolutionResult(
            selected_entity_ids=selected,
            candidates=[
                ResolutionCandidate(entity_id=entity_id, score=score)
                for entity_id, score in scored
            ],
            candidate_pool_size=len(scored),
            resolver_name="entity_ranker_v3",
            resolver_version="3",
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rejection_reason="" if selected else "no_candidate_selected",
        )


class LexicalResolver:
    """Composition-layer adapter for the rule-based NEXUS parser."""

    def __init__(
        self,
        *,
        dialogue_state=None,
        normalizer: Callable[[str], str] | None = None,
        config=None,
    ):
        self.dialogue_state = dialogue_state
        self.normalizer = normalizer
        self.config = config

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        from nexus.query.parser import parse_question

        started = time.perf_counter()
        kwargs = {
            "dialogue_state": self.dialogue_state,
            "normalizer": self.normalizer,
        }
        if self.config is not None:
            kwargs["config"] = self.config
        parsed = parse_question(question, graph, **kwargs)
        candidates = [
            ResolutionCandidate(entity_id=entity_id)
            for entity_id in parsed.entity_ids
        ]
        return ResolutionResult(
            selected_entity_ids=list(parsed.entity_ids),
            candidates=candidates,
            candidate_pool_size=len(candidates),
            resolver_name="lexical_parser",
            resolver_version="1",
            fallback_used=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rejection_reason="" if candidates else "no_candidate_selected",
        )


def _question_tokens(question: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(question) if len(token) > 1}


def mention_score(entity_id: str, question: str, graph: InMemoryGraphStore) -> float:
    """Higher means the entity is explicitly grounded in the question text."""
    q_lower = question.casefold()
    q_tokens = _question_tokens(question)
    eid = str(entity_id)
    score = 0.0
    if eid.casefold() in q_lower:
        score += 100.0
    parts = {
        part.casefold()
        for part in eid.split("_")
        if len(part) > 2 and part.casefold() not in _MENTION_STOP_PARTS and not part.isdigit()
    }
    # Prefer multi-token ID overlap (Exp_0_12_Selection → selection, etc.).
    overlap = parts & q_tokens
    if overlap:
        score += 10.0 * len(overlap)
    node = graph.get_node(eid)
    if node is None:
        return score
    for alias in node.aliases:
        alias_text = str(alias).strip()
        if len(alias_text) >= 3 and alias_text.casefold() in q_lower:
            score += 40.0
    for key in ("name", "title", "display_name", "description"):
        value = str((node.properties or {}).get(key, "")).strip()
        if len(value) >= 4 and value.casefold() in q_lower:
            score += 20.0
            break
    return score


def _soft_question_overlap(entity_id: str, question: str, graph: InMemoryGraphStore) -> float:
    """Weaker overlap used only to diversify near-tied ER3 hub fillers."""
    q_tokens = _question_tokens(question)
    if not q_tokens:
        return 0.0
    node = graph.get_node(entity_id)
    chunks = [str(entity_id)]
    if node is not None:
        chunks.extend(str(alias) for alias in node.aliases)
        props = node.properties or {}
        for key in ("name", "title", "display_name", "description", "key_finding"):
            chunks.append(str(props.get(key, "")))
    entity_tokens = _question_tokens(" ".join(chunks))
    return float(len(q_tokens & entity_tokens))


def _diversity_tiebreak(question: str, entity_id: str) -> str:
    return hashlib.sha256(f"{question}\0{entity_id}".encode("utf-8")).hexdigest()


class UnionResolver:
    """Question-gated re-rank of the ER3 pool, with lexical mention boosts.

    Keeps the full ER3 candidate pool for pool-recall diagnostics. Entry
    handoff selects only from that pool (plus rare canonical lexical IDs),
    preferring question-grounded entities over static hub top-K. Non-canonical
    lexical parser hits are ignored for handoff so they cannot displace ER3.
    No retraining.
    """

    def __init__(
        self,
        er3: ER3Resolver,
        lexical: LexicalResolver | None = None,
        *,
        top_k: int = 12,
        er3_pool_bonus: float = 1.0,
        max_ungrounded_fillers: int | None = None,
    ):
        self.er3 = er3
        self.lexical = lexical or LexicalResolver()
        self.top_k = max(1, int(top_k))
        self.er3_pool_bonus = float(er3_pool_bonus)
        # Optional cap on ungrounded ER3 fillers when mentions exist. Default
        # None keeps full top_k padding so entry_recall stays high; path
        # ranking focus (NEXUSConfig.path_score_focus) handles hub dilution.
        self.max_ungrounded_fillers = (
            None if max_ungrounded_fillers is None else max(0, int(max_ungrounded_fillers))
        )

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        from stack.encoder.canonical_mapping import _is_canonical_id

        started = time.perf_counter()
        lexical_result = coerce_resolution_result(
            self.lexical.resolve(question, graph),
            resolver_name="lexical_parser",
        )
        er3_result = coerce_resolution_result(
            self.er3.resolve(question, graph),
            resolver_name="entity_ranker_v3",
        )

        lexical_selected = {
            entity_id
            for entity_id in lexical_result.selected_entity_ids
            if _is_canonical_id(str(entity_id))
        }
        er3_rank = {
            item.entity_id: index
            for index, item in enumerate(er3_result.candidates)
        }
        er3_score = {
            item.entity_id: float(item.score or 0.0)
            for item in er3_result.candidates
        }

        # Diagnostic pool stays ER3-first; handoff universe is ER3 + canonical
        # lexical extras only (never free-form parser nodes).
        pool_ids: list[str] = []
        seen: set[str] = set()
        for item in er3_result.candidates:
            if item.entity_id in seen:
                continue
            seen.add(item.entity_id)
            pool_ids.append(item.entity_id)
        for entity_id in lexical_selected:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            pool_ids.append(entity_id)

        handoff_ids = [
            entity_id
            for entity_id in pool_ids
            if entity_id in er3_rank or entity_id in lexical_selected
        ]

        def sort_key(entity_id: str) -> tuple[float, float, str]:
            mention = mention_score(entity_id, question, graph)
            lexical_bonus = 50.0 if entity_id in lexical_selected else 0.0
            neural = er3_score.get(entity_id, 0.0) * self.er3_pool_bonus
            soft = _soft_question_overlap(entity_id, question, graph)
            # Ascending sort: more negative primary key first ⇒ higher score first.
            # Soft overlap + question-hash break static hub ties across questions.
            return (
                -(mention + lexical_bonus + neural),
                -soft,
                _diversity_tiebreak(question, entity_id),
            )

        ranked = sorted(handoff_ids, key=sort_key)
        grounded = [
            entity_id
            for entity_id in ranked
            if mention_score(entity_id, question, graph) > 0.0
            or entity_id in lexical_selected
        ]
        fillers = [entity_id for entity_id in ranked if entity_id not in set(grounded)]

        selected: list[str] = []
        if not grounded:
            # Ungrounded handoff: keep a few ER3 quality anchors, then diversify
            # the remaining slots from a wider window so questions do not share
            # one static hub 12-pack.
            by_neural = sorted(
                handoff_ids,
                key=lambda entity_id: (
                    -er3_score.get(entity_id, 0.0),
                    er3_rank.get(entity_id, 10_000_000),
                    entity_id,
                ),
            )
            # Keep enough ER3 anchors for entry_recall, diversify the tail.
            anchor_k = min(5, self.top_k)
            window = by_neural[: max(self.top_k * 3, 30)]
            anchors = window[:anchor_k]
            rest = sorted(
                window[anchor_k:],
                key=lambda entity_id: (
                    -_soft_question_overlap(entity_id, question, graph),
                    _diversity_tiebreak(question, entity_id),
                ),
            )
            selected = list(anchors)
            for entity_id in rest:
                if entity_id in selected:
                    continue
                selected.append(entity_id)
                if len(selected) >= self.top_k:
                    break
        else:
            for entity_id in grounded:
                if entity_id not in selected:
                    selected.append(entity_id)
                if len(selected) >= self.top_k:
                    break
            filler_budget = self.top_k
            if self.max_ungrounded_fillers is not None:
                filler_budget = self.max_ungrounded_fillers
            fillers_added = 0
            diversified_fillers = sorted(
                fillers,
                key=lambda entity_id: (
                    -_soft_question_overlap(entity_id, question, graph),
                    -er3_score.get(entity_id, 0.0),
                    _diversity_tiebreak(question, entity_id),
                ),
            )
            if len(selected) < self.top_k and filler_budget > 0:
                for entity_id in diversified_fillers:
                    if entity_id in selected:
                        continue
                    selected.append(entity_id)
                    fillers_added += 1
                    if len(selected) >= self.top_k or fillers_added >= filler_budget:
                        break

        if not selected:
            selected = list(er3_result.selected_entity_ids[: self.top_k])

        candidates = [
            ResolutionCandidate(
                entity_id=entity_id,
                score=er3_score.get(entity_id),
            )
            for entity_id in pool_ids
        ]
        return ResolutionResult(
            selected_entity_ids=selected[: self.top_k],
            candidates=candidates,
            candidate_pool_size=len(candidates),
            resolver_name="union_lexical_er3",
            resolver_version="1",
            fallback_used=bool(lexical_selected),
            rejection_reason="" if selected else "no_candidate_selected",
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class DialogueAwareResolver:
    """Combine dialogue references with any injected base resolver."""

    def __init__(self, base: EntityResolver, dialogue_state, *, top_k: int = 10):
        self.base = base
        self.dialogue_state = dialogue_state
        self.top_k = top_k

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        started = time.perf_counter()
        base_result = coerce_resolution_result(
            self.base.resolve(question, graph),
            resolver_name=self.base.__class__.__name__,
        )
        context_started = time.perf_counter()
        contextual = self.dialogue_state.resolve_references(question, graph)
        context_latency_ms = round(
            (time.perf_counter() - context_started) * 1000, 3
        )
        selected: list[str] = []
        for entity_id in [*contextual, *base_result.selected_entity_ids]:
            if entity_id not in selected:
                selected.append(entity_id)
            if len(selected) >= self.top_k:
                break

        candidates = list(base_result.candidates)
        known = {item.entity_id for item in candidates}
        for entity_id in reversed(contextual):
            if entity_id not in known:
                candidates.insert(0, ResolutionCandidate(entity_id=entity_id, score=1.0))

        return ResolutionResult(
            selected_entity_ids=selected,
            candidates=candidates,
            candidate_pool_size=max(base_result.candidate_pool_size, len(candidates)),
            resolver_name=f"dialogue+{base_result.resolver_name}",
            resolver_version="1",
            threshold=base_result.threshold,
            fallback_used=base_result.fallback_used,
            rejection_reason="" if selected else "no_candidate_selected",
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            context_latency_ms=context_latency_ms,
        )


class LexicalFallbackResolver:
    """Trivial lexical resolver — returns empty list (lexical parser handles it).

    Used when the caller wants explicit fallback tracking.
    """

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        return ResolutionResult(
            resolver_name="lexical_fallback",
            resolver_version="1",
            fallback_used=True,
            rejection_reason="delegated_to_internal_parser",
        )
