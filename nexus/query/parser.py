"""
NEXUS query parser — rule-based entity spotting and intent detection.

Converts a natural language question into a ParsedQuery with resolved
entity IDs, intent classification, and traversal direction hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional

from nexus.graph.store import InMemoryGraphStore


# ── Intent keyword mapping ──

INTENT_KEYWORDS: list[tuple[str, str, str]] = [
    # (regex pattern, intent, default_direction)
    (r"\b(why|cause|reason|led to)\b",       "causal_explanation", "in"),
    (r"\bwhat\s+depends\b",                   "dependency_chain",  "both"),
    (r"\bwhat\s+affects?\b",                  "impact_analysis",   "out"),
    (r"\b(compare|vs\.?|versus|difference|diff)", "comparison",    "both"),
    (r"\b(what\s+is|what\s+are|list|who)\b", "factual_lookup",    "both"),
    (r"\b(how\b(?:\s+do|\s+does|\s+to)?|diagnose|debug|fix|broken|wrong|error|bug|issue)", "diagnostic", "in"),
]

# Words that are never entity candidates (common English words that happen to
# match graph names through fuzzy matching)
STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "because", "but", "and", "or", "if", "while",
    "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "them", "their", "we", "you", "i", "me", "my",
    "what", "which", "who", "whom", "about", "also",
}

# ── Entity type priority for ranking entry nodes ──
# Higher-priority types get preferred as beam-search entry points.
_TYPE_PRIORITY: dict[str, int] = {
    "Experiment":  10,
    "Decision":    9,
    "Concept":     8,
    "Bug":         7,
    "Requirement": 6,
    "TestCase":    5,
    "Metric":      4,
    "Entity":      3,
    "Document":    2,
    "CodeFile":    1,
    "Function":    0,
}

_MAX_ENTRY_NODES: int = 5


@dataclass
class ParsedQuery:
    """A structured representation of a natural language question."""
    question: str
    entity_ids: list[str] = field(default_factory=list)
    intent: str = "factual_lookup"
    direction: str = "both"
    entity_spans: list[tuple[int, int, str]] = field(default_factory=list)
    # (start, end, matched_text) — character offsets of matched entities


def detect_intent(question: str) -> tuple[str, str]:
    """
    Detect the query intent and recommended traversal direction
    from keyword patterns in the question.

    Returns (intent, direction).
    """
    lowered = question.lower()
    for pattern, intent, direction in INTENT_KEYWORDS:
        if re.search(pattern, lowered):
            return intent, direction
    return "factual_lookup", "both"


def spot_entities(
    question: str,
    graph: InMemoryGraphStore,
    cutoff: float = 0.6,
) -> list[tuple[int, int, str, str]]:
    """
    Scan the question text for substrings matching known graph node names.

    Uses sliding window with fuzzy matching against the graph name index.
    Returns list of (start, end, matched_substring, node_id) tuples.
    """
    lowered = question.lower()
    words = lowered.split()
    results: list[tuple[int, int, str, str]] = []
    matched_node_ids: set[str] = set()

    # Build a list of all candidate substrings (n-grams of various sizes)
    # Start with longer n-grams for greedy matching
    max_ngram = min(len(words), 8)

    for ngram_size in range(max_ngram, 0, -1):
        for i in range(len(words) - ngram_size + 1):
            # Compute character offsets from word positions
            chunk = " ".join(words[i:i + ngram_size])
            chunk_stripped = chunk.strip(".,;:?!\"'()[]{}")

            # Skip stop words or very short chunks
            if len(chunk_stripped) < 2:
                continue

            # Try exact match via the graph's find_entity
            node_id = graph.find_entity(chunk_stripped, cutoff=cutoff)
            if node_id and node_id not in matched_node_ids:
                # Find character offsets in the original question
                # We use case-insensitive search in the lowered text
                start = lowered.find(chunk_stripped)
                if start >= 0:
                    end = start + len(chunk_stripped)
                    results.append((start, end, chunk_stripped, node_id))
                    matched_node_ids.add(node_id)

            # Also try individual words if ngram_size is 1
            if ngram_size == 1 and chunk_stripped not in STOP_WORDS:
                node_id = graph.find_entity(chunk_stripped, cutoff=cutoff)
                if node_id and node_id not in matched_node_ids:
                    start = lowered.find(chunk_stripped)
                    if start >= 0:
                        end = start + len(chunk_stripped)
                        results.append((start, end, chunk_stripped, node_id))
                        matched_node_ids.add(node_id)

    # Sort by start position
    results.sort(key=lambda x: x[0])
    return results


def parse_question(
    question: str,
    graph: InMemoryGraphStore,
    cutoff: float = 0.6,
) -> ParsedQuery:
    """
    Parse a natural language question into structured query intent.

    Args:
        question: The natural language question
        graph: The graph store to resolve entities against
        cutoff: Fuzzy matching cutoff for entity resolution

    Returns:
        ParsedQuery with resolved entity IDs, intent, and direction
    """
    # Detect intent
    intent, direction = detect_intent(question)

    # Spot entities
    entity_spots = spot_entities(question, graph, cutoff=cutoff)

    entity_ids = [node_id for _, _, _, node_id in entity_spots]
    entity_spans = [(start, end, text) for start, end, text, _ in entity_spots]

    # ── Rank and cap entry nodes ──
    entity_ids = _rank_entities(graph, entity_ids)

    return ParsedQuery(
        question=question,
        entity_ids=entity_ids,
        intent=intent,
        direction=direction,
        entity_spans=entity_spans,
    )


# ── Convenience: scan all node names for substring matches ──


def _rank_entities(graph: InMemoryGraphStore, entity_ids: list[str]) -> list[str]:
    """
    Rank entity IDs by quality and return the top candidates (capped).

    Ranking criteria (in order):
      1. **Type priority**: Experiment > Decision > Concept > Bug > ...
      2. **Name length**: longer-span matches beat shorter ones (e.g.
         "Exp_0_6_Validation" beats "Exp")
      3. **Cap at ~5 entry nodes** — beam search with 5 entries already
         explores plenty of the graph.

    Returns the top-ranked entity IDs as a list (deduplicated, order preserved
    by rank).
    """
    if not entity_ids:
        return []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for eid in entity_ids:
        if eid not in seen:
            seen.add(eid)
            unique.append(eid)

    # Rank each entity by (type_priority DESC, name_length DESC)
    def _score(eid: str) -> tuple[int, int]:
        node = graph.get_node(eid)
        type_prio = _TYPE_PRIORITY.get(node.type, 0) if node else 0
        # Use the node ID as the name for length scoring (canonical ID)
        name_len = len(eid) if eid else 0
        return (type_prio, name_len)

    ranked = sorted(unique, key=_score, reverse=True)
    return ranked[:_MAX_ENTRY_NODES]


# ── Convenience: scan all node names for substring matches ──

def find_entities_by_substring(
    question: str,
    graph: InMemoryGraphStore,
    cutoff: float = 0.6,
) -> list[str]:
    """
    Scan the question for any substring that fuzzy-matches a graph node name.

    Simpler alternative: does not compute offsets, just returns node IDs.
    """
    entity_spots = spot_entities(question, graph, cutoff=cutoff)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for _, _, _, node_id in entity_spots:
        if node_id not in seen:
            seen.add(node_id)
            result.append(node_id)
    return result
