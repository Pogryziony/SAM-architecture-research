"""Stage 1C deterministic candidate expansion and graph-only pair generation.

This module deliberately consumes only graph nodes and their metadata.  It does
not load benchmark splits, so generated supervision cannot inspect frozen test
questions or answers.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "did", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "what", "when", "which", "why", "with", "this", "these", "were", "about",
}
_TOKEN_RE = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+")


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _node_text(node: Any) -> str:
    properties = node.properties if isinstance(node.properties, dict) else {}
    values = [node.id, *getattr(node, "aliases", [])]
    values.extend(str(properties.get(key, "")) for key in ("key_finding", "description"))
    return " ".join(values)


def stage1c_property_candidates(question: str, graph: Any, limit: int = 30) -> list[str]:
    """Return graph nodes whose aliases/facts share informative question terms.

    Exact alias/name matches are ranked first.  Remaining nodes are ranked by
    weighted token overlap with key findings and descriptions.  The operation
    is bounded and deterministic; it never uses labels from a benchmark split.
    """
    if limit <= 0:
        return []
    query_tokens = _tokens(question)
    scored: list[tuple[int, int, str]] = []
    for node_id, node in graph._nodes.items():
        aliases = [str(alias) for alias in getattr(node, "aliases", [])]
        name_tokens = _tokens(node_id)
        alias_tokens = _tokens(" ".join(aliases))
        properties = node.properties if isinstance(node.properties, dict) else {}
        finding_tokens = _tokens(str(properties.get("key_finding", "")))
        description_tokens = _tokens(str(properties.get("description", "")))
        exact_phrase = any(
            len(_tokens(alias)) >= 2 and alias.casefold() in question.casefold()
            for alias in aliases
        )
        overlap = (
            8 * len(query_tokens & alias_tokens)
            + 5 * len(query_tokens & name_tokens)
            + 4 * len(query_tokens & finding_tokens)
            + 2 * len(query_tokens & description_tokens)
        )
        if exact_phrase:
            overlap += 100
        if overlap > 0:
            scored.append((overlap, len(finding_tokens), node_id))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [node_id for _score, _finding_size, node_id in scored[:limit]]


def _pair_id(question: str, entity_ids: list[str]) -> str:
    payload = question + "\0" + "\0".join(sorted(entity_ids))
    return "gen_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def generate_stage1c_pairs(graph: Any) -> list[dict[str, Any]]:
    """Generate deterministic alias, key-finding, and weak relation pairs.

    All sources are graph-derived identifiers.  Relation pairs are weak labels
    only when both graph endpoints are known, which keeps provenance explicit.
    """
    pairs: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for node_id, node in sorted(graph._nodes.items()):
        aliases = [str(alias).strip() for alias in getattr(node, "aliases", []) if str(alias).strip()]
        properties = node.properties if isinstance(node.properties, dict) else {}
        finding = str(properties.get("key_finding", "")).strip()
        if aliases:
            question = f"What is {aliases[0]}?"
            key = (question, (node_id,))
            pairs[key] = {
                "id": _pair_id(question, [node_id]), "question": question,
                "answer": finding or node_id, "question_type": "factual",
                "entities": [node_id], "intent": "factual_lookup", "category": "factual",
                "source_id": f"graph:{node_id}:alias", "label_source": "exact_alias",
                "confidence": 1.0,
            }
        if finding:
            terms = sorted(_tokens(finding))[:8]
            question = "What is the finding about " + " ".join(terms) + "?"
            key = (question, (node_id,))
            pairs[key] = {
                "id": _pair_id(question, [node_id]), "question": question,
                "answer": finding, "question_type": "factual", "entities": [node_id],
                "intent": "factual_lookup", "category": "factual",
                "source_id": f"graph:{node_id}:key_finding", "label_source": "key_finding",
                "confidence": 0.95,
            }
        for edge in graph.get_outgoing(node_id):
            if edge.target not in graph._nodes:
                continue
            question = f"What is the {edge.type} relation between {node_id} and {edge.target}?"
            entity_ids = [node_id, edge.target]
            pairs[(question, tuple(sorted(entity_ids)))] = {
                "id": _pair_id(question, entity_ids), "question": question,
                "answer": edge.type, "question_type": "multi-hop", "entities": entity_ids,
                "intent": "multi_hop", "category": "multi-hop",
                "source_id": f"graph:{node_id}:edge:{edge.type}:{edge.target}",
                "label_source": "weak", "confidence": 0.9,
            }
    return list(pairs.values())


def write_stage1c_pairs(graph: Any, output_path: str | Path) -> int:
    """Write graph-only generated pairs and return their count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = generate_stage1c_pairs(graph)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
    return len(pairs)
