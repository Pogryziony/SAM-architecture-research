"""Canonical intent labels for the SAM+NEXUS QA dataset.

The dataset, rule classifier, encoder, and eval script use inconsistent
label strings for the same intent (e.g. "factual" vs "factual_lookup",
"multihop" vs "multi_hop"). This module provides a single canonical
mapping used by ALL components — a metric bug fix, not a model change.

Rules:
- dataset loader: normalize intent/QT labels on load
- rule classifier: output canonical labels (already does)
- eval script: compare canonical labels, not raw strings
"""

from __future__ import annotations

# Canonical label set — the four QA intents used throughout the stack
CANONICAL_INTENT_LABELS: dict[str, str] = {
    # factual_lookup variants
    "factual_lookup": "factual_lookup",
    "factual": "factual_lookup",

    # multi_hop variants
    "multi_hop": "multi_hop",
    "multihop": "multi_hop",
    "multi-hop": "multi_hop",

    # comparison variants
    "comparison": "comparison",
    "comparative": "comparison",

    # diagnostic variants
    "diagnostic": "diagnostic",
    "causal_explanation": "diagnostic",
    "dependency_chain": "diagnostic",
    "impact_analysis": "diagnostic",
}

# The set of canonical label strings that all components MUST use
CANONICAL_LABEL_SET: frozenset[str] = frozenset({
    "factual_lookup", "comparison", "multi_hop", "diagnostic",
})


def canonicalize_intent(raw: str) -> str:
    """Map any raw intent string to its canonical form.

    If the raw string is not recognized, it is returned unchanged
    (conspicuous for debugging) rather than silently defaulting.
    """
    return CANONICAL_INTENT_LABELS.get(raw, raw)


def canonicalize_question_type(question_type: str) -> str:
    """Map question_type / questionType to canonical intent."""
    return CANONICAL_INTENT_LABELS.get(question_type, question_type)
