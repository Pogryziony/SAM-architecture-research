"""
NEXUS rule-based verifier — deterministic hallucination detection.

Checks whether the model's answer is grounded in the evidence pack by:
1. Splitting the answer into claim sentences
2. Checking each claim's entities against evidence nodes
3. Checking each claim's relation types against evidence edges
4. Computing a hallucination rate

No ML — pure regex and string matching. Fast and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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


def _collect_evidence_entities(evidence_pack: dict[str, Any]) -> set[str]:
    """
    Collect all entity identifiers from the evidence pack.

    Returns a set of lowercase entity names/IDs found in:
    - node IDs
    - node names/display_names/titles
    - edge from/to IDs
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

    Returns a set of edge type strings (e.g., 'caused_by', 'depends_on').
    """
    relations: set[str] = set()

    for path_data in evidence_pack.get("paths", []):
        for edge in path_data.get("edges", []):
            etype = edge.get("type", "")
            if etype:
                relations.add(etype.lower())
                # Also add human-readable forms
                relations.add(etype.replace("_", " ").lower())

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
                relations.add(rw)

    return relations


def _entity_present(entities: set[str], entity: str) -> bool:
    """Check if an entity (or its parts) appears in the evidence entity set."""
    if entity.lower() in entities:
        return True
    # Check parts (for compound names)
    for part in re.split(r'[\s_]+', entity.lower()):
        if len(part) > 3 and part in entities:
            return True
    return False


class Verifier:
    """
    Rule-based answer verifier.

    Checks whether the claims in an answer are supported by the evidence
    pack using entity presence and relation presence checks.
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

        Args:
            answer: The model-generated answer text
            evidence_pack: The evidence pack dict from build_evidence_pack()

        Returns:
            VerificationResult with supported/unsupported counts and pass/fail
        """
        claims = extract_claims(answer)

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

        # Check each claim
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
        relations found in the evidence pack. This is a lenient check:
        we look for at least one entity match OR one relation match.
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

        # Check if any entity from the claim exists in evidence
        entity_found = any(
            _entity_present(entities, ent)
            for ent in potential_entities
        )

        # Check if claim mentions any relation from evidence
        relation_found = any(
            rel in claim_lower
            for rel in relations
        )

        # A claim needs at least some grounding — either an entity or a relation match
        return entity_found or relation_found
