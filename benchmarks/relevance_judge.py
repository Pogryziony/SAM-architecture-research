"""
Relevance Judge for SynthesizingModel answers.

Checklist-based, transparent heuristic that assesses whether
a synthesized answer is actually relevant to the asked question.

Rubric:
  1. Focus entity present
  2. Asked metric present
  3. No more than N unrelated facts
  4. Directiveness

Scoring:
  - All 4 conditions → "yes"
  - 2–3 conditions    → "partial"
  - 0–1 conditions    → "no"
"""

from __future__ import annotations

import re
import json
from typing import Literal


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _extract_numeric_value(text: str) -> set[str]:
    """Extract percentage-like or numeric tokens from text."""
    values = set()
    # percentages
    for m in re.finditer(r'\b\d+(?:\.\d+)?%', text):
        values.add(m.group())
    # decimal numbers (catch things like 99.87 without %)
    for m in re.finditer(r'\b\d+\.\d+\b', text):
        values.add(m.group())
    # integers >= 1 (not years like 2026)
    for m in re.finditer(r'\b(?<!\d)([1-9]\d{0,3})(?![\d%])', text):
        num = int(m.group())
        if 1 <= num <= 9999 and num not in (2024, 2025, 2026, 2027):
            values.add(m.group())
    return values


def _keyword_match(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


class RelevanceJudge:
    """Heuristic relevance judge for synthesized answers."""

    # ── per-question-type entity & metric extractors ──

    # Words that signal the answer is dodging / preamble
    PREAMBLE_PATTERN = re.compile(
        r'^(Regarding|Based on|The evidence (reveals|forms|clearly indicates)|'
        r'According to|These findings)',
        re.IGNORECASE
    )

    # Common metric keywords that questions tend to ask about
    METRIC_KEYWORDS = [
        "accuracy", "precision", "recall", "all_required",
        "Rec@", "percentage", "tolerate", "distractors",
        "baseline", "breakthrough", "compare", "differ",
        "fail", "enable", "relationship", "evolution",
        "chain", "pivot",
    ]

    # ── public API ──

    def judge(
        self,
        question: str,
        answer: str,
        question_type: str,
    ) -> dict:
        """Run the 4-point rubric on a question-answer pair.

        Returns:
            dict with keys: verdict (yes|partial|no), reasons (list[str]),
            checks (dict of check_name -> bool), score (int)
        """
        checks = {
            "focus_entity": self._check_focus_entity(question, answer),
            "asked_metric": self._check_asked_metric(question, answer, question_type),
            "max_unrelated": self._check_unrelated_facts(question, answer, question_type),
            "directiveness": self._check_directiveness(question, answer, question_type),
        }
        score = sum(1 for v in checks.values() if v)
        reasons = [
            f"[{'PASS' if v else 'FAIL'}] {self._check_label(k)}"
            for k, v in checks.items()
        ]

        if score == 4:
            verdict = "yes"
        elif score >= 2:
            verdict = "partial"
        else:
            verdict = "no"

        return {
            "verdict": verdict,
            "reasons": reasons,
            "checks": checks,
            "score": score,
        }

    # ── private helpers ──

    @staticmethod
    def _check_label(key: str) -> str:
        labels = {
            "focus_entity": "Focus entity present",
            "asked_metric": "Asked metric present",
            "max_unrelated": "No more than N unrelated facts",
            "directiveness": "Answer is direct (no preamble dodge)",
        }
        return labels.get(key, key)

    # ── check 1: focus entity ──

    @staticmethod
    def _check_focus_entity(question: str, answer: str) -> bool:
        """Does the answer mention the entity/experiment from the question?"""
        q_lower = question.lower()

        # Extract the likely focus entity from the question
        # Try experiment numbers first
        exp_match = re.search(r'experiment\s+([\d.]+[A-Za-z]?)', q_lower)
        if exp_match:
            exp_ref = exp_match.group(0)
            if exp_ref in answer.lower():
                return True

        # Look for key noun phrases
        key_phrases = [
            "oracle memory", "core-only", "chain-set bce",
            "dual encoder", "learned slot selector", "learned selector",
            "dense dataset fix", "sam oracle",
            "nexus", "chain-set retrieval",
            "oracle text memory", "oracle latent memory",
            "selector bottleneck", "rag",
            "architecture pivot", "experiment 0.6",
            "experiment 0.12", "experiment 0.10",
        ]
        for phrase in key_phrases:
            if phrase in q_lower and phrase in answer.lower():
                return True

        # General fallback: does the answer contain at least one
        # significant content word from the question's presumed entity?
        q_words = set(re.findall(r'\b[a-z]{4,}\b', q_lower))
        # filter out question words
        question_stopwords = {"what", "that", "this", "with", "from", "which",
                              "does", "many", "while", "over", "through",
                              "there", "their", "about", "between", "these",
                              "differ", "compare", "compared", "experiment"}
        q_content = q_words - question_stopwords
        a_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
        overlap = q_content & a_words
        return len(overlap) >= 2

    # ── check 2: asked metric ──

    @staticmethod
    def _check_asked_metric(
        question: str, answer: str, question_type: str
    ) -> bool:
        """Does the answer contain the kind of metric/number asked for?"""
        q_lower = question.lower()
        a_lower = answer.lower()

        # If the question asks about a specific metric (accuracy, precision, recall, etc.)
        metric_map = {
            "accuracy": ["%", "percent", "accuracy"],
            "precision": ["%", "precision"],
            "recall": ["%", "recall"],
            "all_required": ["%", "all_required", "100"],
        }

        # Check which metrics the question asks about
        asked_metrics = []
        for metric, indicators in metric_map.items():
            if metric in q_lower:
                asked_metrics.append((metric, indicators))

        if not asked_metrics:
            # Generic check: does question ask for a number and answer has one?
            asks_number = any(p in q_lower for p in [
                "what was", "what accuracy", "how many",
                "what precision", "what recall", "how does",
            ])
            if asks_number:
                answer_values = _extract_numeric_value(answer)
                return len(answer_values) >= 1
            return True  # can't determine metric, pass by default

        # Check if answer contains appropriate metric indicators
        for metric_name, indicators in asked_metrics:
            for ind in indicators:
                if ind in a_lower:
                    return True

        # Fallback: does answer have any numeric values?
        return len(_extract_numeric_value(answer)) >= 1

    # ── check 3: unrelated facts ──

    @staticmethod
    def _check_unrelated_facts(
        question: str, answer: str, question_type: str
    ) -> bool:
        """Count facts not about the focus entity. Too many = fail."""
        sentences = _split_sentences(answer)
        q_lower = question.lower()

        # N threshold
        max_unrelated = 1 if question_type == "factual" else 2

        # Build focus terms from the question
        focus_terms = set()
        for m in re.finditer(r'\b[a-z]{4,}\b', q_lower):
            t = m.group()
            if t not in {"what", "that", "this", "with", "from", "does",
                         "many", "while", "over", "through", "their",
                         "about", "between", "these", "differ", "compare",
                         "compared", "experiment"}:
                focus_terms.add(t)

        # Count sentences that are clearly about a DIFFERENT experiment or topic
        unrelated = 0
        exp_pattern = re.compile(r'experiment\s+([\d.]+[A-Za-z]?)', re.IGNORECASE)

        # Find which experiments are mentioned in the question
        q_exps = set(m.group(0).lower() for m in exp_pattern.finditer(question))

        for sent in sentences:
            sent_lower = sent.lower()

            # Is this sentence about a specific experiment NOT in the question?
            sent_exps = set(m.group(0).lower() for m in exp_pattern.finditer(sent))
            if sent_exps and not (sent_exps & q_exps) and q_exps:
                unrelated += 1
                continue

            # Is this sentence about dependency chains (boilerplate)?
            if re.match(r'^\s*- Experiment.*depends on', sent):
                unrelated += 1
                continue

            # Is this sentence "Additional evidence suggests"?
            if sent_lower.startswith("additional evidence"):
                unrelated += 1
                continue

            # Does this sentence mention a completely different concept?
            if len(sent_lower) > 30:
                overlap = focus_terms & set(re.findall(r'\b[a-z]{4,}\b', sent_lower))
                if len(overlap) < 2:
                    unrelated += 1

        return unrelated <= max_unrelated

    # ── check 4: directiveness ──

    @staticmethod
    def _check_directiveness(
        question: str, answer: str, question_type: str
    ) -> bool:
        """Does the answer directly address the question or start with preamble?"""
        q_lower = question.lower()

        # If the answer starts with "Regarding", "Based on", etc.
        if RelevanceJudge.PREAMBLE_PATTERN.match(answer.strip()):
            return False

        # For "what was X?" / "what accuracy?" questions, answer should
        # start with the value or the entity, not meta-commentary
        if re.search(r'what (was|is|did|accuracy|precision)', q_lower):
            first_sent = _split_sentences(answer)[0] if _split_sentences(answer) else answer
            # Acceptable: starts with a number, the entity name, or direct statement
            if re.match(r'^\d+', first_sent.strip()):
                return True
            if _keyword_match(first_sent, [
                "oracle", "core-only", "chain-set", "sam", "learned",
                "dual encoder", "dense dataset",
            ]):
                return True
            if RelevanceJudge.PREAMBLE_PATTERN.match(first_sent.strip()):
                return False

        return True


# ── CLI entrypoint ──

def main():
    import sys
    import os

    sample_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "results", "relevance_sample.json"
    )
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "relevance_audit.md"
    )

    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)

    judge = RelevanceJudge()

    lines = [
        "# SynthesizingModel Relevance Audit",
        "",
        f"**Sample size**: {len(samples)} stratified by question_type",
        f"**Source**: `{os.path.basename(sample_path)}`",
        f"**Judge**: heuristic checklist (4-point rubric)",
        "",
        "---",
        "",
        "## Per-Case Analysis",
        "",
    ]

    verdict_counts = {"yes": 0, "partial": 0, "no": 0}
    type_verdicts = {}

    for i, case in enumerate(samples, 1):
        result = judge.judge(
            case["question"],
            case["answer"],
            case["question_type"],
        )
        verdict = result["verdict"]
        verdict_counts[verdict] += 1
        qt = case["question_type"]
        if qt not in type_verdicts:
            type_verdicts[qt] = {"yes": 0, "partial": 0, "no": 0}
        type_verdicts[qt][verdict] += 1

        lines.append(f"### Case {i}: {case['question_id']} ({case['question_type']})")
        lines.append("")
        lines.append(f"**Question**: {case['question']}")
        lines.append("")
        lines.append(f"**Answer**:")
        # indent answer for readability
        for ans_line in case["answer"].split("\n"):
            lines.append(f"> {ans_line}")
        lines.append("")
        lines.append(f"**Heuristic verdict**: `{verdict}` (score: {result['score']}/4)")
        lines.append("")
        lines.append("**Reasons**:")
        for reason in result["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append("**Manual review note**: (leave blank)")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Aggregate
    total = sum(verdict_counts.values())
    relevance_rate = (verdict_counts["yes"] + 0.5 * verdict_counts["partial"]) / total * 100 if total > 0 else 0

    lines.append("## Aggregate Results")
    lines.append("")
    lines.append(f"| Verdict | Count | % |")
    lines.append(f"|---------|-------|---|")
    for v in ["yes", "partial", "no"]:
        lines.append(f"| {v} | {verdict_counts[v]} | {verdict_counts[v]/total*100:.1f}% |")
    lines.append("")
    lines.append(f"**Relevance rate**: {relevance_rate:.1f}%")
    lines.append(f"  (Formula: % yes + 0.5 × % partial)")
    lines.append("")

    lines.append("### Per Question Type")
    lines.append("")
    lines.append(f"| Type | Yes | Partial | No | Rate |")
    lines.append(f"|------|-----|---------|----|------|")
    for qt in ["factual", "comparative", "diagnostic", "multi-hop"]:
        vt = type_verdicts.get(qt, {"yes": 0, "partial": 0, "no": 0})
        qt_total = sum(vt.values())
        if qt_total == 0:
            continue
        qt_rate = (vt["yes"] + 0.5 * vt["partial"]) / qt_total * 100
        lines.append(f"| {qt} | {vt['yes']} | {vt['partial']} | {vt['no']} | {qt_rate:.1f}% |")
    lines.append("")

    if relevance_rate < 70:
        lines.append("## ⚠️ Metric Caveat")
        lines.append("")
        lines.append(
            "The zero-LLM SynthesizingModel accuracy claim (39–44%) is **unvalidated** "
            "by this relevance audit. The key-fact-overlap metric in the verifier rewards "
            "evidence dumping even when the answer does not directly address the question. "
            "Heuristic relevance is below 70%, meaning fewer than 7 out of 10 answers "
            "are actually relevant to the asked question."
        )
        lines.append("")
        lines.append(
            "**Recommendation**: Replace or augment the accuracy metric with a "
            "relevance-gated accuracy score. A simple weighted score could be: "
            "`accuracy × relevance_rate`. This would place the true actionable "
            f"SynthesizingModel accuracy at approximately {0.3933 * relevance_rate / 100:.1%} "
            "instead of the reported 39.33%."
        )
        lines.append("")
    else:
        lines.append("## ✅ Metric Validation")
        lines.append("")
        lines.append(
            f"Relevance rate ({relevance_rate:.1f}%) exceeds the 70% threshold. "
            "The key-fact-overlap accuracy metric is reasonably well-aligned with "
            "actual answer relevance."
        )
        lines.append("")

    report = "\n".join(lines) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {output_path}")
    print(f"Verdicts: {verdict_counts}")
    print(f"Relevance rate: {relevance_rate:.1f}%")
    for qt, vt in sorted(type_verdicts.items()):
        qt_total = sum(vt.values())
        qt_rate = (vt["yes"] + 0.5 * vt["partial"]) / qt_total * 100 if qt_total > 0 else 0
        print(f"  {qt}: {vt['yes']}y/{vt['partial']}p/{vt['no']}n = {qt_rate:.1f}%")


if __name__ == "__main__":
    main()
