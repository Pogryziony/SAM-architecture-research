"""Rule-based intent classifier for the SAM+NEXUS QA dataset.

Handles templated QA dataset patterns that account for ~43% of questions
at 96% accuracy. Falls back to the encoder model for unmatched cases.

The rules distinguish intent types that an EmbeddingBag alone cannot,
because token averaging collapses "What was the significance of X" (diagnostic)
into the same representation as "What was the accuracy of X" (factual).
"""

from __future__ import annotations

import re


class RuleIntentClassifier:
    """Rule-based intent classification using prefix/pattern matching.

    Designed for the SAM+NEXUS QA dataset. Each rule targets a specific
    templated pattern with >90% accuracy on matched cases.
    """

    RULES: list[tuple[str, str, bool]] = [
        # ── COMPARISON ──
        # "Compare X vs Y", "Compare the 3-hop...", etc.
        (r"^compare\b", "comparison", True),
        # "How does SAM differ from RAG?" / "How does NEXUS handle X differently from Y?" / "How does NEXUS store knowledge compared to Z?"
        (r"^how does .+ (differ|compared)", "comparison", True),

        # ── MULTI-HOP ──
        # "How does the X experiment relate to the Y experiment?" (78/78 multi-hop)
        (r"^how does the .+ experiment relate to", "multi_hop", True),
        # "What experiment directly preceded/followed X?" (13/14 multi-hop)
        (r"^what experiment directly", "multi_hop", True),
        # "Which experiments provide evidence for the concept that X?" (7/7 multi-hop)
        (r"^which experiments provide evidence", "multi_hop", True),
        # "Walk through the evolution from X to Y" (2/2 multi-hop)
        (r"^walk through the evolution", "multi_hop", True),
        # "What was the chronological order / transition point" (5/5 multi-hop)
        (
            r"^what was the (chronological order|transition point)",
            "multi_hop",
            True,
        ),

        # ── DIAGNOSTIC ──
        # "What was/is the significance of X" (26/27 diagnostic)
        (r"^what (was|is) the significance", "diagnostic", True),
        # "What was the goal/key challenge/breakthrough of the X phase?" (20/20 diagnostic)
        (
            r"^what was the (goal|key challenge|breakthrough moment|biggest surprise|lesson for)",
            "diagnostic",
            True,
        ),
        # "What if X?", "What problem or limitation..." (24/24 diagnostic)
        (r"^what (if|problem or)\b", "diagnostic", True),
        # "Why is it important that X?" (7/7 diagnostic)
        (r"^why is it important", "diagnostic", True),
        # "Why ..." (34/41 diagnostic — 82.9% accuracy, acceptable as fallback rule)
        (
            r"^why\b",
            "diagnostic",
            True,
        ),
        # "Walk through why ..." / "Walk through the NEXUS ..." → diagnostic
        (r"^walk through (why|the nexus)", "diagnostic", True),

        # ── FACTUAL LOOKUP ──
        # "How many/much X?" (27/29 factual)
        (r"^(how many|how much)\b", "factual_lookup", True),
        # "Which research phase does X belong to?" (13/13 factual)
        (r"^which research phase", "factual_lookup", True),
        # "Where would you find X?" / "Where is X?" (10/11 factual)
        (r"^where\b", "factual_lookup", True),
        # "What was the main finding of X experiment?" (13/13 factual)
        (r"^what was the main finding", "factual_lookup", True),
        # "What is the role of X?" (3/3 factual)
        (r"^what is the role\b", "factual_lookup", True),
        # "When was the ..." (3/3 factual)
        (r"^when was the\b", "factual_lookup", True),
    ]

    def classify(self, question: str) -> str | None:
        """Return intent string if a rule matches, None if the model should decide.

        Args:
            question: Natural language question text.

        Returns:
            One of "factual_lookup", "comparison", "multi_hop", "diagnostic",
            or None if no rule matched.
        """
        q_lower = question.lower().strip()
        for pattern, intent, is_regex in self.RULES:
            if is_regex:
                if re.match(pattern, q_lower):
                    return intent
            elif pattern in q_lower:
                return intent
        return None

    def classify_with_confidence(self, question: str) -> tuple[str | None, float]:
        """Classify with confidence score.

        Rule-based predictions have high confidence (0.95) reflecting
        the empirically measured accuracy.
        """
        result = self.classify(question)
        if result is not None:
            return result, 0.95
        return None, 0.0


# Singleton instance
_global_rule_classifier: RuleIntentClassifier | None = None


def get_rule_classifier() -> RuleIntentClassifier:
    """Get the global rule-based intent classifier instance."""
    global _global_rule_classifier
    if _global_rule_classifier is None:
        _global_rule_classifier = RuleIntentClassifier()
    return _global_rule_classifier
