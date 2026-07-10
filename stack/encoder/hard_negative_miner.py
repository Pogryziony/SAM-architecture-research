"""Score-derived hard negative mining for the Entity Ranker V3.

Hard negatives are the highest-scoring incorrect candidates, not just the
first N pipeline outputs. Two-phase mining: lexical baseline then encoder pass.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence


def _tokens(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens."""
    import re
    return {
        token.casefold()
        for token in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+", text)
        if len(token) >= 2
    }


def score_lexical(question: str, node_id: str, graph: Any) -> float:
    """Compute lexical overlap score between question and entity.

    Same formula as trivial_baseline._lexical_score but returns float.
    """
    node = graph.get_node(node_id) if graph is not None else None
    if node is None:
        return 0.0

    properties = getattr(node, "properties", {}) or {}
    query = _tokens(question)
    aliases = _tokens(" ".join(str(a) for a in getattr(node, "aliases", [])))
    name = _tokens(str(node.id))
    finding = _tokens(str(properties.get("key_finding", "")))
    description = _tokens(str(properties.get("description", "")))

    exact_phrase = any(
        len(_tokens(str(a))) >= 2 and str(a).casefold() in question.casefold()
        for a in getattr(node, "aliases", [])
    )

    return float(
        100 * int(exact_phrase)
        + 8 * len(query & aliases)
        + 5 * len(query & name)
        + 4 * len(query & finding)
        + 2 * len(query & description)
    )


def mine_hard_negatives_lexical(
    question: str,
    candidate_ids: list[str],
    positive_ids: set[str],
    graph: Any,
    top_k: int = 10,
) -> list[str]:
    """Phase 1: Score all non-gold candidates lexically, retain highest-scoring.

    Args:
        question: The question text.
        candidate_ids: All candidate entity IDs.
        positive_ids: Gold entity IDs (excluded from negatives).
        graph: InMemoryGraphStore instance.
        top_k: Number of hard negatives to retain.

    Returns:
        List of top_k highest-scoring non-gold candidate IDs.
    """
    non_gold = [cid for cid in candidate_ids if cid not in positive_ids]
    scored = [(cid, score_lexical(question, cid, graph)) for cid in non_gold]
    scored.sort(key=lambda x: -x[1])
    return [cid for cid, _ in scored[:top_k]]


def mine_hard_negatives_encoder(
    question: str,
    candidate_ids: list[str],
    positive_ids: set[str],
    graph: Any,
    score_fn: Callable[[str, str], float],
    top_k: int = 10,
) -> list[str]:
    """Phase 2: Score non-gold candidates with encoder, retain highest.

    Args:
        question: The question text.
        candidate_ids: All candidate entity IDs.
        positive_ids: Gold entity IDs.
        graph: InMemoryGraphStore instance.
        score_fn: Function(question, node_id) → float score.
        top_k: Number of hard negatives to retain.

    Returns:
        List of top_k highest-scoring non-gold candidate IDs (by encoder).
    """
    non_gold = [cid for cid in candidate_ids if cid not in positive_ids]
    scored = [(cid, score_fn(question, cid)) for cid in non_gold]
    scored.sort(key=lambda x: -x[1])
    return [cid for cid, _ in scored[:top_k]]


def mine_diverse_hard_negatives(
    question: str,
    candidate_ids: list[str],
    positive_ids: set[str],
    graph: Any,
    top_k: int = 15,
    lexical_top: int = 8,
    same_type_k: int = 3,
    same_source_k: int = 2,
    high_degree_k: int = 2,
    alias_confusable_k: int = 2,
    graph_neighbor_k: int = 3,
) -> list[str]:
    """Mine diverse hard negatives covering multiple confusability categories.

    Categories:
    - Lexical: highest lexical-overlap non-gold (score-derived)
    - Same-type: entities with same node type as any gold entity
    - Same-source: entities sharing source/provenance with gold
    - High-degree: entities with top-10% node degree
    - Alias-confusable: entities with overlapping aliases
    - Graph-neighbor: 1-hop graph neighbors of gold entities

    Args:
        question: Question text.
        candidate_ids: All candidate IDs.
        positive_ids: Gold entity IDs.
        graph: InMemoryGraphStore instance.
        top_k: Total hard negatives to return (max).
        lexical_top, same_type_k, etc.: Per-category allocations.

    Returns:
        Deduplicated list of up to top_k diverse hard negatives.
    """
    non_gold = [cid for cid in candidate_ids if cid not in positive_ids]
    if not non_gold:
        return []

    positives = [graph.get_node(pid) for pid in positive_ids]
    positives = [n for n in positives if n is not None]

    # Gold properties for category-based mining
    gold_types = {getattr(n, "type", "") for n in positives}
    gold_sources: set[str] = set()
    for n in positives:
        props = getattr(n, "properties", {}) or {}
        src = str(props.get("source", props.get("source_snippet", "")))
        if src:
            gold_sources.add(src)
    gold_aliases: set[str] = set()
    for n in positives:
        for a in getattr(n, "aliases", []):
            gold_aliases.add(str(a).casefold())
    gold_neighbors: set[str] = set()
    for pid in positive_ids:
        for edge in graph.get_outgoing(pid):
            if edge.target in graph._nodes and edge.target not in positive_ids:
                gold_neighbors.add(edge.target)
        for edge in graph.get_incoming(pid):
            if edge.source in graph._nodes and edge.source not in positive_ids:
                gold_neighbors.add(edge.source)

    # Compute degrees for all non-gold candidates
    degrees: dict[str, int] = {}
    for cid in non_gold:
        degrees[cid] = len(graph.get_outgoing(cid)) + len(graph.get_incoming(cid))
    if degrees:
        degree_cutoff = sorted(degrees.values())[max(0, int(len(degrees) * 0.9))]
    else:
        degree_cutoff = 0

    result: list[str] = []
    seen: set[str] = set()

    def add(cids: list[str], limit: int):
        for cid in cids:
            if len(result) >= top_k:
                return
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
                if len([x for x in result if x in cids]) >= limit:
                    return

    # 1. Lexical (score-derived)
    lexical = mine_hard_negatives_lexical(question, candidate_ids, positive_ids, graph, top_k=20)
    add(lexical, lexical_top)

    # 2. Same-type
    same_type = [cid for cid in non_gold if getattr(graph.get_node(cid), "type", "") in gold_types]
    add(same_type, same_type_k)

    # 3. Same-source
    same_source = []
    for cid in non_gold:
        node = graph.get_node(cid)
        if node:
            props = getattr(node, "properties", {}) or {}
            src = str(props.get("source", props.get("source_snippet", "")))
            if src in gold_sources:
                same_source.append(cid)
    add(same_source, same_source_k)

    # 4. High-degree
    high_deg = [cid for cid in non_gold if degrees.get(cid, 0) >= degree_cutoff]
    add(sorted(high_deg, key=lambda cid: -degrees.get(cid, 0)), high_degree_k)

    # 5. Alias-confusable
    alias_confusable = []
    for cid in non_gold:
        node = graph.get_node(cid)
        if node:
            for a in getattr(node, "aliases", []):
                if str(a).casefold() in gold_aliases:
                    alias_confusable.append(cid)
                    break
    add(alias_confusable, alias_confusable_k)

    # 6. Graph neighbors
    add(list(gold_neighbors & set(non_gold)), graph_neighbor_k)

    return result[:top_k]


def mine_hard_negatives_group(
    question: str,
    candidate_ids: list[str],
    positive_ids: list[str],
    graph: Any,
    hard_negative_k: int = 15,
) -> tuple[list[str], dict[str, Any]]:
    """Mine diverse hard negatives for one training group.

    Returns:
        (hard_negative_ids, metadata_dict) where metadata includes
        category counts for provenance.
    """
    positives_set = set(positive_ids)
    negatives = mine_diverse_hard_negatives(
        question=question,
        candidate_ids=candidate_ids,
        positive_ids=positives_set,
        graph=graph,
        top_k=hard_negative_k,
    )

    # Count per category
    lexical = set(mine_hard_negatives_lexical(question, candidate_ids, positives_set, graph, top_k=20))
    cat_counts: dict[str, int] = {
        "lexical": len(set(negatives) & lexical),
        "total": len(negatives),
    }

    return negatives, {"hard_negative_categories": cat_counts, "hard_negative_count": len(negatives)}
