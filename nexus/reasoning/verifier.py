"""
NEXUS rule-based verifier — deterministic hallucination detection.

Checks whether the model's answer is grounded in the evidence pack by:
1. Splitting the answer into claim sentences
2. Filtering to factual claims only (skipping filler/transitional sentences)
3. Checking each factual claim's entities against evidence nodes (with fuzzy matching)
4. Checking each factual claim's relation types against evidence edges (with normalization)
5. Computing a hallucination rate

The verifier is designed to be semantic, not verbosity-penalizing:
verbose answers with filler sentences don't get penalized for filler —
only actual factual assertions are checked.

No ML — pure regex and string matching. Fast and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Transitional / filler sentence patterns ─────────────────────────
# Sentences starting with these patterns are considered non-factual.
_TRANSITIONAL_PREFIXES = (
    "let me explain", "let me elaborate", "let me summarize",
    "let me break", "let me start", "let me begin", "let me be",
    "let me first", "let me note", "let me add", "let me clarify",
    "based on the evidence", "based on the above", "based on this",
    "looking at the evidence", "looking at the data",
    "from the evidence", "from the data", "from what i can see",
    "i can see that", "i can see how", "we can see that",
    "it is worth noting", "it is important to note",
    "it should be noted", "it is clear that",
    "it appears that", "it seems that", "it is evident that",
    "in summary", "to summarize", "in conclusion", "to conclude",
    "in short", "in brief", "to put it simply",
    "first", "second", "third", "fourth", "fifth",
    "finally", "lastly",
    "additionally", "furthermore", "moreover",
    "this means that", "this indicates that", "this suggests that",
    "the evidence shows", "the evidence suggests", "the data shows",
    "according to the evidence", "according to the data",
    "moving on", "next", "now let", "now i",
    "as mentioned", "as noted", "as discussed",
    "to answer", "to address", "here is", "here are",
    "what we have", "what can be", "what the",
    "so",  # standalone "So." / "So," transitions
)

# ── Pure qualifier / editorial sentence patterns ────────────────────
_QUALIFIER_PATTERNS = (
    "this is important", "this is significant", "this is notable",
    "this is interesting", "this is quite", "this is noteworthy",
    "the findings are clear", "the findings are compelling",
    "the results are clear", "the results are compelling",
    "the implications are", "the significance is",
    "these results", "these findings",
    "it is worth", "it is also worth",
    "one thing to note", "something to note",
    "it is important", "it is crucial",
    "it is interesting", "it is remarkable",
    "what stands out", "what is striking",
    "what is notable", "what is interesting",
    "what is important", "what is significant",
    "note that", "please note",
    "keep in mind", "bear in mind",
    "it goes without saying",
    "needless to say",
    "as a side note", "as an aside",
    "as a matter of fact",
)

# ── Relation normalization map ──────────────────────────────────────
# Maps human-readable relation phrases to underscore forms.
_RELATION_NORMALIZE: dict[str, str] = {
    "is caused by": "caused_by",
    "are caused by": "caused_by",
    "was caused by": "caused_by",
    "were caused by": "caused_by",
    "depends on": "depends_on",
    "is dependent on": "depends_on",
    "are dependent on": "depends_on",
    "is blocked by": "blocked_by",
    "are blocked by": "blocked_by",
    "was blocked by": "blocked_by",
    "blocks": "blocked_by",
    "is derived from": "derived_from",
    "are derived from": "derived_from",
    "was derived from": "derived_from",
    "validates": "validated_by",
    "is validated by": "validated_by",
    "are validated by": "validated_by",
    "was validated by": "validated_by",
    "is replaced by": "replaced_by",
    "are replaced by": "replaced_by",
    "was replaced by": "replaced_by",
    "replaces": "replaced_by",
    "is related to": "related_to",
    "are related to": "related_to",
    "relates to": "related_to",
    "is contradicted by": "contradicted_by",
    "are contradicted by": "contradicted_by",
    "contradicts": "contradicted_by",
    "implements": "implemented_by",
    "is implemented by": "implemented_by",
    "are implemented by": "implemented_by",
    "supports": "supported_by",
    "is supported by": "supported_by",
    "are supported by": "supported_by",
    "is mentioned in": "mentioned_in",
    "mentions": "mentioned_in",
    "causes": "caused_by",
}

# Reverse lookup: underscore form → all surface forms including itself
_RELATION_REVERSE_MAP: dict[str, set[str]] = {}
for _surface, _canonical in _RELATION_NORMALIZE.items():
    _RELATION_REVERSE_MAP.setdefault(_canonical, set()).add(_surface)

# We also add direct underscore forms (e.g., "caused_by" → {"caused_by"})
for _canonical in set(_RELATION_NORMALIZE.values()):
    _RELATION_REVERSE_MAP.setdefault(_canonical, set()).add(_canonical)


@dataclass
class VerificationResult:
    """Result of verifying an answer against evidence."""

    supported_count: int
    unsupported_claims: list[str] = field(default_factory=list)
    hallucination_rate: float = 0.0
    passed: bool = True

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"VerificationResult({status}, "
            f"supported={self.supported_count}, "
            f"unsupported={len(self.unsupported_claims)}, "
            f"hallucination_rate={self.hallucination_rate:.2f})"
        )


# ── Factual claim detection ─────────────────────────────────────────

def _has_entity_like_token(sentence: str) -> bool:
    """
    Check if a sentence contains at least one entity-like token.

    Entity-like tokens are:
    - Capitalized words (>= 3 chars): e.g., "Alpha", "Oracle"
    - Underscore-separated identifiers: e.g., "oracle_memory", "Exp_0_11"
    - Experiment/ID-like patterns: e.g., "Exp_0_11", "X-42"
    - Numeric/percentage patterns: e.g., "99.87%", "42ms"
    """
    # Capitalized multi-word phrases (e.g., "Chain Aware Retrieval")
    if re.search(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b', sentence):
        return True
    # Single capitalized words (>= 3 letters) at word boundaries
    if re.search(r'\b[A-Z][a-z]{2,}\b', sentence):
        return True
    # Underscore-separated identifiers
    if re.search(r'\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)+\b', sentence):
        return True
    # Percentages and numbers with units
    if re.search(r'\b\d+(?:\.\d+)?%\b', sentence):
        return True
    # Experiment/machine-like IDs (alphanumeric with hyphens)
    if re.search(r'\b[A-Z][A-Z0-9]*[-_]\d+\b', sentence):
        return True
    return False


def _is_factual_claim(sentence: str) -> bool:
    """
    Determine whether a sentence is a factual claim worth verifying.

    Returns False for:
    - Transitional sentences ("Let me explain...", "Based on the evidence...")
    - Pure editorial/qualifier sentences ("This is important", "The findings are clear")
    - Sentences without any entity-like tokens

    Returns True for sentences that contain factual assertions with entities.
    """
    if not sentence or not sentence.strip():
        return False

    cleaned = sentence.strip().rstrip(".;:!?, ")
    lower = cleaned.lower().lstrip()

    # Skip transitional prefixes
    for prefix in _TRANSITIONAL_PREFIXES:
        if lower.startswith(prefix):
            # Allow if the sentence ALSO has an entity AND makes a substantive claim
            # e.g., "Based on the evidence, Alpha validates Beta." — this IS factual
            # We check: if the sentence has a comma AND an entity after the prefix
            if "," in cleaned[:len(prefix) + 10] or " that " in cleaned[:len(prefix) + 20]:
                # Check portion after the prefix for entity-like tokens
                after = cleaned[len(prefix):].lstrip(",;: ")
                if _has_entity_like_token(after):
                    return True
            # Short transitional sentence without entity content → skip
            if len(cleaned) < 40:
                return False

    # Skip pure qualifier sentences
    for pattern in _QUALIFIER_PATTERNS:
        if lower.startswith(pattern):
            # Short qualifier → skip
            if len(cleaned) < 50:
                return False
            # Longer qualifier that also has entities? Check the rest
            after = cleaned[len(pattern):].lstrip(",;: ")
            if not _has_entity_like_token(after):
                return False

    # Must have at least one entity-like token
    if not _has_entity_like_token(cleaned):
        return False

    # Skip very short sentences that are likely editorial
    if len(cleaned) < 10:
        return False

    return True


def extract_claims(text: str) -> list[str]:
    """
    Split answer text into individual claim sentences.

    Handles both prose answers and bullet-list answers by:
    1. Splitting on newlines to separate bullet points
    2. Then splitting on sentence boundaries (. ! ?)
    3. Filtering boilerplate/metadata lines
    """
    if not text:
        return []

    # First split on newlines to separate bullet points
    # Then for each line, split on sentence boundaries
    lines = text.split("\n")
    raw_sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split on sentence boundaries within each line
        parts = re.split(r'(?<=[.!?])\s+', line)
        raw_sentences.extend(parts)

    claims = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        # Skip empty
        if not sentence:
            continue
        # Clean up bullet/list markers
        cleaned = re.sub(r'^[-*•]\s*', '', sentence).strip()
        cleaned = re.sub(r'^\d+[.)]\s*', '', cleaned).strip()
        # Skip pure metadata / boilerplate
        lower = cleaned.lower().strip(".,;: ")
        if lower in ("insufficient evidence to answer.", "insufficient evidence to answer"):
            continue
        if lower.startswith("based on the evidence"):
            continue
        if lower.startswith("sources:"):
            continue
        if lower.startswith("additional "):
            continue
        if len(cleaned) < 10:
            continue
        claims.append(cleaned)

    return claims


def extract_factual_claims(text: str) -> list[str]:
    """
    Extract only factual claims from an answer — semantic, not verbosity-penalizing.

    First splits into sentences via ``extract_claims()``, then filters to
    keep only sentences that contain factual assertions using ``_is_factual_claim()``.

    This means verbose models with lots of filler sentences (transitions,
    editorial commentary, qualifiers) don't get penalized for those —
    only actual factual claims are checked against the evidence pack.
    """
    all_claims = extract_claims(text)
    return [c for c in all_claims if _is_factual_claim(c)]


# ── Relation normalization ──────────────────────────────────────────

def _normalize_relation(rel_phrase: str) -> str:
    """
    Normalize a relation phrase to its canonical underscore form.

    Handles human-readable forms like "is caused by", "depends on", etc.
    and maps them to "caused_by", "depends_on", etc.

    Also handles already-normalized forms (returns them unchanged if
    they're already canonical).
    """
    rel_lower = rel_phrase.strip().lower()
    if rel_lower in _RELATION_NORMALIZE:
        return _RELATION_NORMALIZE[rel_lower]
    # Check if it's already a canonical form
    if rel_lower in _RELATION_REVERSE_MAP:
        return rel_lower
    return rel_lower


def _all_relation_forms(rel: str) -> set[str]:
    """
    Return all surface forms (human-readable + canonical) for a relation.

    E.g., for "caused_by" returns {"caused by", "is caused by", "causes", "caused_by"}.
    """
    rel_lower = rel.strip().lower()
    canonical = _normalize_relation(rel_lower)
    return _RELATION_REVERSE_MAP.get(canonical, {rel_lower})


# ── Evidence collection ─────────────────────────────────────────────

def _collect_evidence_entities(evidence_pack: dict[str, Any]) -> set[str]:
    """
    Collect all entity identifiers from the evidence pack.

    Returns a set of lowercase entity names/IDs found in:
    - node IDs (plus individual underscore-separated parts)
    - node names/display_names/titles (plus individual words/parts)
    - node aliases (list of alternative names)
    - edge from/to IDs (plus individual underscore-separated parts)
    - fact strings (extracted key terms)
    """
    entities: set[str] = set()

    for path_data in evidence_pack.get("paths", []):
        for node in path_data.get("nodes", []):
            nid = node.get("id", "")
            if nid:
                entities.add(nid.lower())
                # Also add individual words from IDs that use underscores
                for part in nid.replace("_", " ").split():
                    if len(part) > 2:
                        entities.add(part.lower())
            for key in ("name", "display_name", "title"):
                val = node.get(key, "")
                if val:
                    entities.add(val.lower())
                    for part in re.split(r'[\s_]+', val):
                        if len(part) > 2:
                            entities.add(part.lower())
            # Collect aliases
            aliases = node.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        entities.add(alias.lower())
                        for part in re.split(r'[\s_]+', alias):
                            if len(part) > 2:
                                entities.add(part.lower())

        for edge in path_data.get("edges", []):
            for field in ("from", "to"):
                val = edge.get(field, "")
                if val:
                    entities.add(val.lower())
                    for part in val.replace("_", " ").split():
                        if len(part) > 2:
                            entities.add(part.lower())

    # Also collect from fact strings
    for fact in evidence_pack.get("facts", []):
        # Extract key capitalized or technical terms
        for word in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', fact):
            entities.add(word.lower())
        # Extract underscore identifiers
        for word in re.findall(r'\b([a-z]+_[a-z_]+)\b', fact):
            entities.add(word.lower())

    return entities


def _collect_evidence_relations(evidence_pack: dict[str, Any]) -> set[str]:
    """
    Collect all relation types present in the evidence edges.

    Returns a set of canonical relation strings (e.g., 'caused_by', 'depends_on')
    plus all human-readable surface forms.
    """
    relations: set[str] = set()

    for path_data in evidence_pack.get("paths", []):
        for edge in path_data.get("edges", []):
            etype = edge.get("type", "")
            if etype:
                canonical = _normalize_relation(etype)
                relations.add(canonical)
                # Also add all human-readable surface forms
                relations |= _all_relation_forms(etype)
                relations |= _all_relation_forms(canonical)

    # Also from fact strings
    relation_words = {"depends on", "is a dependency of",
                      "caused by", "causes", "validates", "validated by",
                      "supports", "supported by",
                      "blocks", "blocked by", "is blocked by",
                      "implements", "implemented by",
                      "related to", "is related to",
                      "contradicts", "contradicted by", "is contradicted by",
                      "derived from", "is derived from",
                      "replaces", "replaced by", "is replaced by",
                      "mentioned in", "mentions"}
    for fact in evidence_pack.get("facts", []):
        for rw in relation_words:
            if rw in fact.lower():
                canonical = _normalize_relation(rw)
                relations.add(canonical)
                relations |= _all_relation_forms(canonical)

    return relations


def _entity_present(entities: set[str], entity: str) -> bool:
    """
    Check if an entity (or its parts) appears in the evidence entity set.

    Uses fuzzy matching:
    1. Exact match (case-insensitive)
    2. Individual part match (splitting on whitespace and underscores)
    3. Partial match — if the claim entity contains evidence entity tokens
       or vice-versa (e.g., "oracle memory" matches evidence with "oracle_memory")
    4. Compound match — if ALL parts of a multi-word entity exist in evidence
    """
    entity_lower = entity.lower().strip()

    # 1. Exact match
    if entity_lower in entities:
        return True

    # 2. Individual part match (existing behavior)
    parts = re.split(r'[\s_]+', entity_lower)
    # If any part (>= 4 chars) exists in entities
    for part in parts:
        if len(part) > 3 and part in entities:
            return True

    # 3. Partial / fuzzy match — check if claim entity parts overlap with
    #    evidence entity tokens (handles "oracle memory" vs "oracle_memory")
    #    Normalize both: join multi-word claim entities to underscore form
    claim_underscore = "_".join(parts)
    if claim_underscore in entities:
        return True

    # Check each evidence entity against the claim for overlap
    for ev_ent in entities:
        # Skip very short evidence entities (too noisy)
        if len(ev_ent) < 4:
            continue
        # If evidence entity is a substring of claim entity (e.g., "oracle_memory" in "oracle memory thing")
        if ev_ent in entity_lower.replace("_", " "):
            return True
        # If claim entity (normalized) is a substring of evidence entity
        if entity_lower in ev_ent.replace("_", " "):
            return True

    # 4. Compound match: if all parts of a multi-word entity are found
    #    individually in evidence (e.g., "oracle" + "memory" both exist)
    if len(parts) >= 2 and all(
        p in entities
        for p in parts
        if len(p) > 2
    ):
        return True

    return False


def _claim_mentions_relation(claim_lower: str, relations: set[str]) -> bool:
    """
    Check if a claim mentions any relation from the evidence set.

    Uses normalized matching:
    - Claim text is checked against all surface forms of each relation
    - "is caused by" matches evidence edge "caused_by"
    - "depends on" matches evidence edge "depends_on"
    """
    for rel in relations:
        if rel in claim_lower:
            return True
        # Also check reversed / directional variants
        # e.g., if evidence has "caused_by", also check "causes" in claim
        canonical = _normalize_relation(rel)
        for form in _all_relation_forms(canonical):
            if form in claim_lower:
                return True
    return False


class Verifier:
    """
    Rule-based answer verifier.

    Checks whether the factual claims in an answer are supported by the
    evidence pack using entity presence and relation presence checks.

    The verifier is semantic, not verbosity-penalizing: it filters out
    filler/transitional/editorial sentences before checking claims,
    so verbose answers don't get penalized for non-factual filler text.
    """

    def __init__(self, hallucination_threshold: float = 0.2):
        """
        Args:
            hallucination_threshold: Maximum allowed hallucination rate.
                Answers with rate > threshold are flagged as FAIL.
        """
        self._threshold = hallucination_threshold

    def verify(self, answer: str, evidence_pack: dict[str, Any]) -> VerificationResult:
        """
        Verify an answer against an evidence pack.

        Only factual claims are checked — filler sentences (transitions,
        qualifiers, editorial commentary) are filtered out so verbose
        answers aren't unfairly penalized.

        Args:
            answer: The model-generated answer text
            evidence_pack: The evidence pack dict from build_evidence_pack()

        Returns:
            VerificationResult with supported/unsupported counts and pass/fail
        """
        # Extract only factual claims (skip filler/transitional sentences)
        claims = extract_factual_claims(answer)

        # Edge case: "insufficient evidence" answer
        if not claims and "insufficient evidence" in answer.lower():
            return VerificationResult(
                supported_count=0,
                unsupported_claims=[],
                hallucination_rate=0.0,
                passed=True,
            )

        if not claims:
            return VerificationResult(
                supported_count=0,
                unsupported_claims=[],
                hallucination_rate=0.0,
                passed=True,
            )

        # Collect evidence features
        entities = _collect_evidence_entities(evidence_pack)
        relations = _collect_evidence_relations(evidence_pack)

        # Check each factual claim
        unsupported: list[str] = []
        for claim in claims:
            if not self._claim_supported(claim, entities, relations):
                unsupported.append(claim)

        supported = len(claims) - len(unsupported)
        total = len(claims)
        rate = len(unsupported) / total if total > 0 else 0.0

        return VerificationResult(
            supported_count=supported,
            unsupported_claims=unsupported,
            hallucination_rate=rate,
            passed=(rate <= self._threshold),
        )

    def _claim_supported(
        self,
        claim: str,
        entities: set[str],
        relations: set[str],
    ) -> bool:
        """
        Check if a single claim is supported by the evidence.

        A claim is considered supported if it references entities and/or
        relations found in the evidence pack. Uses fuzzy entity matching
        and normalized relation matching for lenient, semantic checking.

        This is a lenient check: we look for at least one entity match
        OR one relation match.
        """
        claim_lower = claim.lower()

        # Extract potential entity mentions from the claim
        # Look for capitalized words, underscore identifiers, and technical terms
        potential_entities: set[str] = set()

        # Capitalized multi-word phrases (e.g., "Chain Aware Retrieval")
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', claim):
            potential_entities.add(m.group(1).lower())

        # Single capitalized words at word boundaries (handles leading Caps in snake_case)
        for m in re.finditer(r'(?:^|[_\s])([A-Z][a-z]{3,})(?:$|[_\s])', claim):
            potential_entities.add(m.group(1).lower())

        # Underscore-separated identifiers (e.g., concept_pivottonexus)
        for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)+)\b', claim):
            full = m.group(1).lower()
            potential_entities.add(full)
            # Also add individual snake_case parts
            for part in full.split("_"):
                if len(part) >= 3:
                    potential_entities.add(part)

        # Simple words >= 4 chars (only from non-snake_case spans)
        # Replace underscores with spaces so we get word boundaries at snake_case breaks
        spaced = claim_lower.replace("_", " ")
        for word in re.findall(r'\b([a-z]{4,})\b', spaced):
            potential_entities.add(word)

        # Check if any entity from the claim exists in evidence (fuzzy)
        entity_found = any(
            _entity_present(entities, ent)
            for ent in potential_entities
        )

        # Check if claim mentions any relation from evidence (normalized)
        relation_found = _claim_mentions_relation(claim_lower, relations)

        # A claim needs at least some grounding — either an entity or a relation match
        return entity_found or relation_found
