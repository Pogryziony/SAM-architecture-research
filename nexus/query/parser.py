"""
NEXUS query parser — rule-based entity spotting and intent detection.

Converts a natural language question into a ParsedQuery with resolved
entity IDs, intent classification, and traversal direction hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from nexus.graph.store import InMemoryGraphStore
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


# ── Intent keyword mapping ──

INTENT_KEYWORDS: list[tuple[str, str, str]] = [
    # (regex pattern, intent, default_direction)
    (r"\b(why|cause|reason|led to)\b",       "causal_explanation", "in"),
    (r"\bwhat\s+depends\b",                   "dependency_chain",  "both"),
    (r"\bwhat\s+affects?\b",                  "impact_analysis",   "out"),
    (r"\b(compare|vs\.?|versus|difference|diff)", "comparison",    "both"),
    (r"\b(what\s+is|what\s+are|how\s+many|how\s+much|list|who)\b", "factual_lookup",    "both"),
    (r"\b(how\b(?:\s+(?:do|does|to|can|should|would|could|did|is|are))?|diagnose|debug|fix|broken|wrong|error|bug|issue)", "diagnostic", "in"),
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


# ── Sub-run node marker patterns ──
# Node IDs matching these patterns are likely sub-experiment runs
# (top-K variants, weighted variants, baselines) rather than curated top-level nodes.
# Numeric patterns require 2+ digits to avoid false-positives like _6 in `Exp_0_6_Validation`.
_SUB_RUN_PATTERNS: list[re.Pattern] = [
    re.compile(r"_(top|weighted|baseline)\d*(_|$)", re.IGNORECASE),
    re.compile(r"_(\d{2,})(_|$)", re.IGNORECASE),       # _005, _16, _64, _128
    re.compile(r"_(\d+k)(_|$)", re.IGNORECASE),           # _16k, _64k
]

# ── Known acronyms for entity resolution expansion ──

_ACRONYMS: dict[str, str] = {
    "sam": "sparse_associative_memory",
    "pkm": "product_key_memory",
    "rag": "retrieval_augmented_generation",
    "nexus": "non_parametric_execution",
}


@dataclass
class ParsedQuery:
    """A structured representation of a natural language question."""
    question: str
    entity_ids: list[str] = field(default_factory=list)
    intent: str = "factual_lookup"
    direction: str = "both"
    entity_spans: list[tuple[int, int, str]] = field(default_factory=list)
    # (start, end, matched_text) — character offsets of matched entities
    alias_matched_ids: set[str] = field(default_factory=set)
    # entity IDs that were resolved via exact alias match (high-confidence)
    # "alias" | "fuzzy" | "none" — primary resolution method for this query
    resolution_method: str = "none"


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
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> tuple[list[tuple[int, int, str, str]], set[str]]:
    """
    Scan the question text for substrings matching known graph node names.

    Uses sliding window with fuzzy matching against the graph name index,
    plus word-boundary matching for higher precision. Optimized with
    word-index pruning and n-gram pre-filtering.

    Returns:
        (entity_spots, wb_matched)
        - entity_spots: list of (start, end, matched_substring, node_id) tuples
        - wb_matched: set of node_ids that were matched via word-boundary (not just fuzzy)
    """
    lowered = question.lower()
    words = lowered.split()
    results: list[tuple[int, int, str, str]] = []
    matched_node_ids: set[str] = set()
    wb_matched: set[str] = set()

    max_ngram = min(len(words), 8)
    content_words = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    target_entities = config.max_entry_nodes * 2  # early termination threshold

    # ── N-gram scanning with pre-filtering ──
    # Only generate and check n-grams whose words appear in the graph word index,
    # and deduplicate overlapping n-grams (keep the longest match).
    seen_chunks: set[str] = set()

    for ngram_size in range(max_ngram, 0, -1):
        if len(matched_node_ids) >= target_entities:
            break  # early termination: enough entities found

        for i in range(len(words) - ngram_size + 1):
            if len(matched_node_ids) >= target_entities:
                break

            chunk = " ".join(words[i:i + ngram_size])
            chunk_stripped = chunk.strip(".,;:?!\"'()[]{}")

            if len(chunk_stripped) < 2 or chunk_stripped in seen_chunks:
                continue
            seen_chunks.add(chunk_stripped)

            # ── OPTIMIZATION: skip n-grams that are all stop words or very short ──
            if chunk_stripped in STOP_WORDS:
                continue

            node_id = _try_match(chunk_stripped, graph, cutoff)
            if node_id and node_id not in matched_node_ids:
                start = lowered.find(chunk_stripped)
                if start >= 0:
                    end = start + len(chunk_stripped)
                    results.append((start, end, chunk_stripped, node_id))
                    matched_node_ids.add(node_id)
                    if _word_boundary_match(chunk_stripped, node_id):
                        wb_matched.add(node_id)

    # ── Second pass: word-boundary search using word index ──
    for word in content_words:
        word_lower = word.lower()
        if len(word_lower) < 3:
            continue
        candidates = graph._word_index.get(word_lower, set())
        for nid in candidates:
            if nid in matched_node_ids:
                continue
            normalized_name = graph._norm_name_by_id.get(nid)
            if normalized_name is None:
                continue
            segments = _split_into_segments(normalized_name)
            if any(word_lower == seg for seg in segments):
                start = lowered.find(word)
                if start >= 0:
                    end = start + len(word)
                    results.append((start, end, word, nid))
                    matched_node_ids.add(nid)
                    wb_matched.add(nid)

    results.sort(key=lambda x: x[0])
    return results, wb_matched


def _try_match(
    chunk: str,
    graph: InMemoryGraphStore,
    cutoff: float = 0.6,
) -> Optional[str]:
    """Try to match a chunk against the graph.
    Uses word-indexed search first (prunes candidates), falls back to full search.
    Both use trigram-based scoring — no expensive SequenceMatcher."""
    candidates = graph.get_word_index_candidates(chunk)
    node_id = graph.find_entity_fast(chunk, cutoff=cutoff, candidate_ids=candidates or None)
    return node_id


def parse_question(
    question: str,
    graph: InMemoryGraphStore,
    cutoff: float | None = None,
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> ParsedQuery:
    """
    Parse a natural language question into structured query intent.

    Args:
        question: The natural language question
        graph: The graph store to resolve entities against
        cutoff: Fuzzy matching cutoff for entity resolution (default from config)
        config: NEXUSConfig with tunable parameters

    Returns:
        ParsedQuery with resolved entity IDs, intent, and direction
    """
    if cutoff is None:
        cutoff = config.fuzzy_cutoff
    # Detect intent
    intent, direction = detect_intent(question)

    # Expand acronyms — add expanded forms as additional search terms
    # that help fuzzy matching find the right entities
    expanded_question = question
    lowered_q = question.lower()
    for acronym, expansion in _ACRONYMS.items():
        if acronym in lowered_q:
            # Append the expansion at the end so spot_entities can match it
            expanded_question = f"{expanded_question} ({expansion})"

    # Spot entities from substring matching
    entity_spots, wb_matched = spot_entities(expanded_question, graph, cutoff=cutoff)

    entity_ids = [node_id for _, _, _, node_id in entity_spots]
    entity_spans = [(start, end, text) for start, end, text, _ in entity_spots]

    # Also try keyword-based property matching — these have higher confidence
    # than substring matches, so prepend them for ranking priority
    keyword_matches = graph.find_entity_by_keywords(question, cutoff=cutoff)
    keyword_scores: dict[str, int] = {}
    existing_set = set(entity_ids)
    for kid, score in keyword_matches:
        keyword_scores[kid] = score
        if kid not in existing_set:
            entity_ids.insert(0, kid)
            existing_set.add(kid)

    # ── Track alias-matched entities for ranking boost ──
    # Alias matching is very precise: the question literally contains a phrase
    # mapped to this entity. Give these a strong ranking boost.
    alias_matched: set[str] = _find_alias_matches(question, graph)

    # Ensure alias-matched entities are in the list (they may not be found by
    # fuzzy substring matching if the alias is a multi-word phrase)
    for amid in alias_matched:
        if amid not in existing_set:
            entity_ids.insert(0, amid)
            existing_set.add(amid)

    # ── Rank and cap entry nodes ──
    entity_ids = _rank_entities(graph, entity_ids, question=question,
                                keyword_scores=keyword_scores, wb_matched=wb_matched,
                                alias_matched=alias_matched, config=config)

    # ── Determine resolution method ──
    if not entity_ids:
        resolution_method = "none"
    elif alias_matched:
        resolution_method = "alias"
    else:
        resolution_method = "fuzzy"

    return ParsedQuery(
        question=question,
        entity_ids=entity_ids,
        intent=intent,
        direction=direction,
        entity_spans=entity_spans,
        alias_matched_ids=alias_matched,
        resolution_method=resolution_method,
    )


def extract_metric_term(question: str) -> str | None:
    """Extract the metric term a factual question is asking about.

    Detects patterns like:
        "What was the overall accuracy of ..."  → "accuracy"
        "What was the precision of ..."          → "precision"
        "What was the recall of ..."             → "recall"
        "How many random distractors ..."         → "distractors"
        "What was the all_required@64 result ..." → "all_required"
        "What Rec@8 did ..."                     → "recall@8"

    Returns the normalized metric key or None if no metric term detected.
    """
    q_lower = question.lower()

    # ── @-notation metrics (all_required@64, Rec@8, etc.) ──
    at_match = re.search(
        r'(all_required|recall|rec|precision|prec|coverage|cov|f1)@(\d+)',
        q_lower, re.IGNORECASE,
    )
    if at_match:
        prefix = at_match.group(1).lower()
        if prefix == "rec":
            prefix = "recall"
        elif prefix == "prec":
            prefix = "precision"
        elif prefix == "cov":
            prefix = "coverage"
        num = at_match.group(2)
        return f"{prefix}@{num}"

    # ── Explicit metric names ──
    direct_metrics = [
        "accuracy", "precision", "recall", "f1", "coverage",
        "loss", "perplexity", "bleu", "rouge",
    ]
    for metric in direct_metrics:
        if metric in q_lower:
            return metric

    # ── "how many" → count-related ──
    if re.search(r'how\s+many', q_lower):
        return "count"

    # ── "number of" → count-related ──
    if re.search(r'number\s+of\s+(\w+)', q_lower):
        num_match = re.search(r'number\s+of\s+(\w+)', q_lower)
        if num_match:
            return num_match.group(1).rstrip('s')

    return None


# ── Convenience: scan all node names for substring matches ──


def _contextual_type_boost(entity_name: str, question: str, node_type: str) -> float:
    """
    Boost entity priority based on contextual keywords in the question.

    Returns a boost (0.0-0.15) to add to the type priority score during ranking.
    """
    lowered = question.lower()
    boost = 0.0

    # Experiment-related questions
    if node_type == "Experiment" and any(
        kw in lowered for kw in (
            "experiment", "result", "finding", "accuracy", "showed",
            "proved", "measured", "demonstrated", "found that",
        )
    ):
        boost = 0.15

    # Concept-related questions
    if node_type == "Concept" and any(
        kw in lowered for kw in ("concept", "idea", "theory", "principle")
    ):
        boost = 0.10

    # Causal/diagnostic questions → boost Bug and Decision
    if node_type in ("Bug", "Decision") and any(
        kw in lowered for kw in ("why", "caused", "reason", "led to", "pivot")
    ):
        boost = 0.10

    return boost


# ── Word-boundary and entity disambiguation helpers ──


def _split_into_segments(node_id: str) -> list[str]:
    """Split a node ID into word-like segments on _, space, and camelCase boundaries.

    Examples:
      ThisIsCamelCase   → ["This", "Is", "Camel", "Case"]
      Exp_0_9_OracleFilter_top32 → ["Exp", "0", "9", "Oracle", "Filter", "top", "32"]
      oracle_memory     → ["oracle", "memory"]
    """
    # First split on underscore and space
    parts = re.split(r"[_ ]+", node_id)
    segments: list[str] = []
    for part in parts:
        if not part:
            continue
        # Split camelCase: "OracleFilter" → ["Oracle", "Filter"]
        # Insert a space before uppercase letters that follow lowercase or digit,
        # and between letters and digits.
        sub = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        sub = re.sub(r"([A-Za-z])(\d)", r"\1 \2", sub)
        sub = re.sub(r"(\d)([A-Za-z])", r"\1 \2", sub)
        segments.extend(sub.split())
    return [s.lower() for s in segments if s]


def _word_boundary_match(query_term: str, node_id: str) -> bool:
    """Check if the query term appears as a whole word/segment in the node ID.

    "oracle memory" matches a node containing both "oracle" and "memory" as
    whole segments (e.g. "Exp_Oracle_Memory" or "oracle_memory_benchmark").
    This is stricter than fuzzy substring matching and prevents false matches
    like "oracle" matching "OracleFilter" (partial within a segment).
    """
    query_words = query_term.lower().split()
    if not query_words:
        return False
    segments = _split_into_segments(node_id)
    return all(any(qw == seg for seg in segments) for qw in query_words)


def _is_sub_run_node(node_id: str) -> bool:
    """Return True if the node ID matches sub-experiment run naming patterns.

    Detects sub-run markers like _top32, _weighted, _baseline, _005, _16k, etc.
    """
    for pattern in _SUB_RUN_PATTERNS:
        if pattern.search(node_id):
            return True
    return False


def _question_has_ranking_keywords(question: str) -> bool:
    """Check if the question explicitly mentions ranking/topK/weights concepts."""
    lowered = question.lower()
    return any(
        kw in lowered
        for kw in ("top", "topk", "top-k", "ranking", "ranked", "weighted",
                   "weights", "best k", "top k")
    )


def _property_keyword_boost(node, question: str, config: NEXUSConfig) -> float:
    """Check if question keywords appear in node's key_finding or description.

    Returns a boost value: 0.0 (no match) up to property_keyword_boost.
    """
    if node is None:
        return 0.0
    question_tokens = set(re.findall(r"[a-z]{3,}", question.lower()))
    if not question_tokens:
        return 0.0

    # Collect searchable property values
    searchable: list[str] = []
    for prop_name in ("key_finding", "description"):
        val = node.properties.get(prop_name, "")
        if isinstance(val, str) and val:
            searchable.append(val.lower())

    if not searchable:
        return 0.0

    # Count how many question tokens appear in property text
    combined = " ".join(searchable)
    match_count = sum(1 for t in question_tokens if t in combined)
    if match_count == 0:
        return 0.0

    # Scale boost: higher for Experiment nodes (they are the curated info-rich ones)
    boost = config.property_keyword_boost
    base = boost if node.type == "Experiment" else boost * 0.6
    # More matching tokens = higher confidence
    return min(base, match_count * (boost / 5.0))


def _find_alias_matches(
    question: str,
    graph: InMemoryGraphStore,
) -> set[str]:
    """Find entity IDs that match via the alias index for any phrase in the question.

    Checks all n-grams (2–8 words) against the alias index. Returns the set of
    entity IDs that have at least one alias phrase directly present in the question.
    """
    lowered = question.lower()
    words = lowered.split()
    alias_matched: set[str] = set()
    alias_index: dict[str, str] = getattr(graph, "_alias_index", {})

    max_ngram = min(len(words), 8)
    for ngram_size in range(max_ngram, 1, -1):  # Skip 1-word (too noisy, causes first_30 regression)
        for i in range(len(words) - ngram_size + 1):
            chunk = " ".join(words[i:i + ngram_size])
            chunk_stripped = chunk.strip(".,;:?!\"'()[]{}")
            if len(chunk_stripped) < 2:
                continue
            normalized = chunk_stripped.replace(" ", "_").replace("-", "_")
            if normalized in alias_index:
                alias_matched.add(alias_index[normalized])

    return alias_matched


# ── Intent-conditioned type prior detection ──

_METRIC_TERMS: set[str] = {
    "accuracy", "precision", "recall", "@k", "%", "f1", "accuracy@",
    "recall@", "precision@", "latency", "throughput", "tokens",
    "parameters", "slots", "loss", "perplexity", "bleu", "rouge",
    "coverage", "speed", "time", "ms", "seconds",
}

_CONCEPT_TERMS: set[str] = {
    "why", "what concept", "how does", "role", "purpose",
    "relationship", "concept", "idea", "theory", "principle",
    "what is the role", "what is the purpose",
    "what are the concept", "definition", "define",
}


def _is_metric_question(question: str) -> bool:
    """Detect if the question is asking about metrics/measurements."""
    lowered = question.lower()
    return any(term in lowered for term in _METRIC_TERMS)


def _is_concept_question(question: str) -> bool:
    """Detect if the question is asking about concepts/explanations."""
    lowered = question.lower()
    return any(term in lowered for term in _CONCEPT_TERMS)


def _rank_entities(
    graph: InMemoryGraphStore,
    entity_ids: list[str],
    question: str = "",
    keyword_scores: dict[str, int] | None = None,
    wb_matched: set[str] | None = None,
    alias_matched: set[str] | None = None,
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> list[str]:
    """
    Rank entity IDs by quality and return the top candidates (capped).

    Two-tier ranking ensures the type-prior boost affects ordering but never
    causes an entity to displace another past the max_entry_nodes cap:

    **Tier 1 — base_score determines acceptance (inclusion under cap):**
      1. **Alias match**: question contains a phrase mapped to this entity (+config.alias_match_boost)
      2. **Keyword match count** (from property token index): +5.0 per match (≥3 tokens)
         or +0.5 per match (<3 tokens)
      3. **Type priority**: from config.type_priority (lower = higher priority)
      4. **Contextual type boost**: keywords in question boost relevant types (+0.10–0.15)
      5. **Key-finding text match**: question words in key_finding/description (+config.property_keyword_boost)
      6. **Curated node boost**: nodes with key_finding property (+config.curated_node_boost)
      7. **Word-boundary match**: exact segment match beats fuzzy (+config.word_boundary_boost)
      8. **Sub-run penalty**: deprioritize _top, _weighted, _baseline nodes (config.sub_run_penalty)
         unless the question mentions ranking/topK/weights

    **Tier 2 — tie-breakers (affect ordering within same base_score):**
      9. **Intent-conditioned type prior**: metric questions → +boost for Experiment/Metric;
         concept questions → +boost for Concept/Decision (config.type_prior_boost)
         — TIE-BREAKER ONLY: cannot push an entity above the cap over one with higher base_score
     10. **Name length**: longer-span matches beat shorter ones

    **Final: Cap at config.max_entry_nodes entry nodes**

    Returns the top-ranked entity IDs as a list (deduplicated, order preserved).
    """
    if keyword_scores is None:
        keyword_scores = {}
    if wb_matched is None:
        wb_matched = set()
    if alias_matched is None:
        alias_matched = set()
    if not entity_ids:
        return []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for eid in entity_ids:
        if eid not in seen:
            seen.add(eid)
            unique.append(eid)

    # Pre-compute contextual flags
    has_ranking_kw = _question_has_ranking_keywords(question) if question else False
    metric_q = _is_metric_question(question) if question else False
    concept_q = _is_concept_question(question) if question else False

    def _score(eid: str) -> tuple[float, float, int]:
        node = graph.get_node(eid)
        node_type = node.type if node else ""

        # Compute type priority: config uses lower=higher priority,
        # invert so that higher score = higher priority.
        _type_prio_map = config.type_priority
        _max_prio = max(_type_prio_map.values()) if _type_prio_map else 10
        base_type_prio = float(_max_prio - _type_prio_map.get(node_type, _max_prio))

        ctx_boost = (
            _contextual_type_boost(eid, question, node_type) if node and question
            else 0.0
        )

        # Intent-conditioned type prior: additive bonus when question intent
        # aligns with the node's type (e.g., metric questions prefer Experiment/Metric nodes).
        # APPLIED AS A TIE-BREAKER ONLY — affects ordering among entities with the
        # same base_score; never causes an entity to be accepted or rejected from the
        # max_entry_nodes cap.
        type_prior = 0.0
        if metric_q and node_type in ("Experiment", "Metric"):
            type_prior = config.type_prior_boost
        elif concept_q and node_type in ("Concept", "Decision"):
            type_prior = config.type_prior_boost

        # Alias match: question literally contains a phrase mapped to this entity.
        # This is the strongest signal — override all other scores.
        alias_boost = config.alias_match_boost if eid in alias_matched else 0.0

        # Keyword match boost: proportional to token match count.
        kw_count = keyword_scores.get(eid, 0)
        kw_boost = kw_count * 5.0 if kw_count >= 3 else kw_count * 0.5

        # Key-finding / description text match boost — use precomputed property text
        prop_boost = _property_keyword_boost_from_text(
            graph._property_text.get(eid, ""), question, config, node_type
        ) if question else 0.0

        # Curated node boost: nodes from populate_from_experiments have key_finding
        curated_boost = config.curated_node_boost if node and "key_finding" in node.properties else 0.0

        # Word-boundary match boost
        wb_boost = config.word_boundary_boost if eid in wb_matched else 0.0

        # Sub-run penalty: push down sub-experiment noise unless question is ranking-relevant
        sub_run_penalty = 0.0
        if _is_sub_run_node(eid) and not has_ranking_kw:
            sub_run_penalty = config.sub_run_penalty

        # Base score: all ranking signals *except* type_prior.
        # This determines acceptance into the max_entry_nodes cap.
        base_score = (
            alias_boost + kw_boost + base_type_prio + ctx_boost + prop_boost
            + curated_boost + wb_boost + sub_run_penalty
        )
        name_len = (len(eid) if eid else 0) * 2
        # Sort key: (base_score, type_prior, name_len)
        #   - base_score: primary — determines inclusion under cap
        #   - type_prior: secondary tie-breaker — order among same-base entities
        #   - name_len:   tertiary tie-breaker
        return (base_score, type_prior, name_len)

    ranked = sorted(unique, key=_score, reverse=True)
    return ranked[:config.max_entry_nodes]


def _property_keyword_boost_from_text(
    property_text: str,
    question: str,
    config: NEXUSConfig,
    node_type: str = "",
) -> float:
    """Check if question keywords appear in precomputed property text.
    Returns a boost value: 0.0 (no match) up to property_keyword_boost."""
    if not property_text:
        return 0.0
    question_tokens = set(re.findall(r"[a-z]{3,}", question.lower()))
    if not question_tokens:
        return 0.0

    match_count = sum(1 for t in question_tokens if t in property_text)
    if match_count == 0:
        return 0.0

    boost = config.property_keyword_boost
    base = boost if node_type == "Experiment" else boost * 0.6
    return min(base, match_count * (boost / 5.0))


# ── Convenience: scan all node names for substring matches ──

def find_entities_by_substring(
    question: str,
    graph: InMemoryGraphStore,
    cutoff: float = 0.6,
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> list[str]:
    """
    Scan the question for any substring that fuzzy-matches a graph node name.

    Simpler alternative: does not compute offsets, just returns node IDs.
    """
    entity_spots, _wb = spot_entities(question, graph, cutoff=cutoff, config=config)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for _, _, _, node_id in entity_spots:
        if node_id not in seen:
            seen.add(node_id)
            result.append(node_id)
    return result
