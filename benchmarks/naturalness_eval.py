"""
Naturalness evaluation metric for Stage 2 — Realization L1.

Composite score 0-100 measuring grammatical naturalness of synthesized answers.

Components (per EXPERIMENT_SAM_NEXUS_STACK.md):
  - Aggregation rate (25 pts): facts merged per sentence
  - Connector presence (20 pts): edge-type-matched discourse connectors
  - Referring expressions (20 pts): full name first mention, short form after
  - Repetition penalty (20 pts): lower score for repeated identical phrases
  - Mean sentence length (15 pts): in natural band (10-25 words)

Usage:
    from benchmarks.naturalness_eval import score_naturalness
    result = score_naturalness(answer_text, facts_list, edge_types=None)
"""

from __future__ import annotations

import math
import re
from typing import Any

# ── Edge-type → connector mapping ─────────────────────────────────────────
# Must match the discourse connector mapping defined for SynthesizingModel.
EDGE_CONNECTORS: dict[str, list[str]] = {
    "caused_by":    ["because", "since", "as a result of", "due to",
                     "ponieważ", "gdyż", "bo", "dlatego że"],
    "depends_on":   ["which depends on", "depending on", "relies on",
                     "zależy od", "który zależy od"],
    "validates":    ["validating", "which validates", "confirming",
                     "potwierdzając", "co potwierdza"],
    "contradicts":  ["however", "but", "although", "on the other hand",
                     "contradicting", "jednak", "ale", "natomiast",
                     "mimo to", "wbrew"],
    "blocked_by":   ["but is blocked by", "blocked by", "prevented by",
                     "ale jest blokowane przez", "blokowane przez"],
    "derived_from": ["derived from", "originating from", "based on",
                     "wywodzący się z", "oparty na", "pochodzący z"],
    "implements":   ["which implements", "implementing",
                     "który implementuje", "implementując"],
    "replaces":     ["which replaces", "replacing", "superseding",
                     "który zastępuje", "zastępując"],
    "related_to":   ["related to", "connected with", "associated with",
                     "związany z", "powiązany z"],
    "mentioned_in": ["mentioned in", "referenced in", "described in",
                     "wspomniany w", "opisany w"],
}

# Aggregated English → Polish connector shortcuts for morphology
PL_CONNECTORS: dict[str, str] = {
    "because": "ponieważ",
    "however": "jednak",
    "validating": "potwierdzając",
    "derived from": "wywodzący się z",
}


# ── Sentence splitting ────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'[.!?]+(?:\s+|$)')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, returning stripped non-empty sentences."""
    sentences: list[str] = []
    # Split on sentence-ending punctuation followed by whitespace or end.
    parts = _SENTENCE_RE.split(text)
    for part in parts:
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)
    if not sentences:
        # Fallback: treat entire text as one sentence
        stripped = text.strip()
        if stripped:
            sentences.append(stripped)
    return sentences


# ── Component 1: Aggregation rate (0-25) ───────────────────────────────────

def _score_aggregation(answer: str, num_facts: int) -> float:
    """Score how well facts are merged per sentence (fewer robot-like
    one-fact-per-sentence patterns).

    If no facts are provided, returns 0.
    A perfect score means 2+ facts per sentence on average.
    """
    if num_facts <= 0:
        return 0.0

    sentences = _split_sentences(answer)
    n_sentences = max(len(sentences), 1)
    facts_per_sentence = num_facts / n_sentences

    # Clamp: 0 = 1 fact/sentence (robot), 25 = 3+ facts/sentence (excellent)
    # Linear interpolation between 1.0 and 3.0
    if facts_per_sentence <= 1.0:
        return 0.0
    if facts_per_sentence >= 3.0:
        return 25.0
    return round(25.0 * (facts_per_sentence - 1.0) / 2.0, 1)


# ── Component 2: Connector presence (0-20) ─────────────────────────────────

def _score_connectors(answer: str, edge_types: list[str] | None) -> float:
    """Score presence of discourse connectors matching edge types.

    For each edge type found in the evidence, check if the answer
    contains a matching connector. Score is proportional to
    (matched connectors) / (max expected connectors).
    """
    if not edge_types:
        return 20.0  # No edges → no connectors expected → full credit

    answer_lower = answer.lower()
    matched = 0
    for etype in edge_types:
        connectors = EDGE_CONNECTORS.get(etype, [])
        if not connectors:
            continue
        for conn in connectors:
            if conn.lower() in answer_lower:
                matched += 1
                break  # One match per edge type is enough

    # Score: up to min(5, len(edge_types)) expected connectors
    expected = min(len(edge_types), 5)
    if expected == 0:
        return 20.0
    return round(20.0 * min(matched, expected) / expected, 1)


# ── Component 3: Referring expressions (0-20) ─────────────────────────────

# Patterns for entity names (handles "Exp_0_11_ChainRetrieval" style IDs
# and capitalized multi-word names). Case-insensitive matching on lower().
_ENTITY_ID_RE_CI = re.compile(
    r'\b(exp_\d+_\d+[a-z]?_\w+|concept_\w+|decision_\w+)\b'
)
# Multi-word capitalized phrase: at least 2 sequential capitalized words
# (handles "Eksperyment ChainRetrieval", "Oracle Memory", "BCE Retriever")
_CAPITALIZED_PHRASE_RE = re.compile(
    r'\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]*(?:[A-Z][a-ząćęłńóśźż]*)*'
    r'(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]*(?:[A-Z][a-ząćęłńóśźż]*)*)+)\b'
)


def _score_referring(answer: str) -> float:
    """Score referring expression usage: full name on first mention,
    short form (pronoun or abbreviated) after.

    Checks for presence of both long-form entity names and
    abbreviated/pronoun references (it, this, that, the experiment, etc.).
    Partial credit for having any referring variety.
    """
    sentences = _split_sentences(answer)
    if len(sentences) <= 1:
        return 12.5  # Single sentence: neutral score

    answer_lower = answer.lower()

    # Check for pronoun/anaphor usage anywhere in answer
    anaphor_patterns = [
        r'\b(it|this|that|these|those)\b',
        r'\b(the experiment|the model|the system|the result|the concept)\b',
        r'\b(ten|ta|to|te|oni|one|ono|ony)\b',  # Polish demonstratives
        r'\b(ten eksperyment|ten model|ten system|ten wynik)\b',
    ]

    has_anaphor = any(
        re.search(pat, answer_lower) for pat in anaphor_patterns
    )

    # Check for entity IDs or long names in first sentence
    first_sent = sentences[0] if sentences else ""
    first_sent_lower = first_sent.lower()
    has_long_form = bool(
        _ENTITY_ID_RE_CI.search(first_sent_lower)
        or _CAPITALIZED_PHRASE_RE.search(first_sent)
        or (len(first_sent_lower.split()) >= 10)
    )

    # Scoring:
    # - Full marks (20): clear long-form introduction + later short forms
    # - Partial (12-16): some referencing variety
    # - Low (0-8): monolithic without referring variety
    if has_long_form and has_anaphor:
        return 20.0
    elif has_anaphor:
        return 16.0
    elif has_long_form:
        return 12.0
    else:
        return 4.0


# ── Component 4: Repetition penalty (0-20) ────────────────────────────────

def _score_repetition(answer: str) -> float:
    """Penalize repeated identical phrases. Higher score = less repetition.

    Uses both trigram and bigram analysis to catch short repeating patterns.
    Score decreases as repetition increases.
    """
    words = re.findall(r'[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+', answer.lower())
    if len(words) < 4:
        return 20.0  # Too short to have meaningful repetition

    # Build bigram and trigram sets for better detection
    bigrams: list[str] = []
    trigrams: list[str] = []
    for i in range(len(words) - 1):
        bigrams.append(" ".join(words[i:i+2]))
    for i in range(len(words) - 2):
        trigrams.append(" ".join(words[i:i+3]))

    # Combined repetition score: weight trigrams more heavily
    bi_unique = len(set(bigrams))
    bi_total = len(bigrams)
    tri_unique = len(set(trigrams))
    tri_total = len(trigrams)

    bi_rep = 0.0
    tri_rep = 0.0

    if bi_total > 0:
        bi_rep = 1.0 - (bi_unique / bi_total)
    if tri_total > 0:
        tri_rep = 1.0 - (tri_unique / tri_total)

    # Weighted combination: 40% bigram, 60% trigram
    combined_rep = 0.4 * max(bi_rep, 0) + 0.6 * max(tri_rep, 0)

    # 0% repetition → 20 pts, 100% repetition → 0 pts
    return round(20.0 * (1.0 - combined_rep), 1)


# ── Component 5: Mean sentence length (0-15) ──────────────────────────────

def _score_sentence_length(answer: str) -> float:
    """Score mean sentence length. Target band: 10-25 words.

    Sentences outside this band get reduced scores.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)

    # Ideal band: 10-25 words → full 15 points
    # Below 5 or above 40 → 0 points
    # Linear interpolation between edges
    if 10 <= mean_len <= 25:
        return 15.0
    elif mean_len < 5:
        return 0.0
    elif mean_len > 40:
        return 0.0
    elif mean_len < 10:
        return round(15.0 * (mean_len - 5) / 5.0, 1)
    else:  # 25 < mean_len <= 40
        return round(15.0 * (40 - mean_len) / 15.0, 1)


# ── Composite scoring ────────────────────────────────────────────────────

def score_naturalness(
    answer: str,
    facts: list[str] | None = None,
    edge_types: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the composite naturalness score (0-100).

    Parameters:
        answer:     The synthesized answer text.
        facts:      List of fact strings used to build the answer
                    (for computing aggregation rate).
        edge_types: List of edge relation types (caused_by, depends_on,
                    etc.) that connect the facts in evidence.

    Returns:
        dict with keys:
            total           — composite score (0-100)
            aggregation     — aggregation rate sub-score (max 25)
            connectors      — connector presence sub-score (max 20)
            referring       — referring expressions sub-score (max 20)
            repetition      — repetition penalty sub-score (max 20)
            sentence_length — mean sentence length sub-score (max 15)
            detail          — raw metrics used for scoring
    """
    num_facts = len(facts) if facts else 0
    sentences = _split_sentences(answer)

    agg = _score_aggregation(answer, num_facts)
    conn = _score_connectors(answer, edge_types)
    ref = _score_referring(answer)
    rep = _score_repetition(answer)
    slen = _score_sentence_length(answer)

    total = round(agg + conn + ref + rep + slen, 1)

    # Cap at 100
    total = min(total, 100.0)

    word_counts = [len(s.split()) for s in sentences]
    mean_words = sum(word_counts) / len(word_counts) if word_counts else 0.0

    return {
        "total": total,
        "aggregation": agg,
        "connectors": conn,
        "referring": ref,
        "repetition": rep,
        "sentence_length": slen,
        "detail": {
            "num_sentences": len(sentences),
            "num_facts": num_facts,
            "mean_sentence_length": round(mean_words, 1),
            "sentence_lengths": word_counts,
            "edge_types_provided": edge_types or [],
        },
    }


def score_naturalness_batch(
    answers: list[str],
    facts_lists: list[list[str]] | None = None,
    edge_types_lists: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Compute naturalness scores for a batch of answers.

    Returns aggregate stats: mean total, per-component means, stddev.
    """
    n = len(answers)
    if n == 0:
        return {"mean_total": 0.0, "n": 0}

    facts_lists = facts_lists or [[] for _ in range(n)]
    edge_types_lists = edge_types_lists or [None for _ in range(n)]

    scores: list[dict[str, Any]] = []
    totals: list[float] = []

    for i in range(n):
        s = score_naturalness(
            answers[i],
            facts_lists[i] if i < len(facts_lists) else [],
            edge_types_lists[i] if i < len(edge_types_lists) else None,
        )
        scores.append(s)
        totals.append(s["total"])

    mean_total = round(sum(totals) / n, 1)
    # Std dev
    if n > 1:
        variance = sum((t - mean_total) ** 2 for t in totals) / (n - 1)
        stddev = round(math.sqrt(variance), 1)
    else:
        stddev = 0.0

    components = ["aggregation", "connectors", "referring", "repetition", "sentence_length"]
    component_means: dict[str, float] = {}
    for comp in components:
        vals = [s[comp] for s in scores]
        component_means[comp] = round(sum(vals) / n, 1)

    return {
        "mean_total": mean_total,
        "stddev_total": stddev,
        "n": n,
        "component_means": component_means,
        "individual_scores": scores,
    }
