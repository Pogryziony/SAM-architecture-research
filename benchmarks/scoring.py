"""
Unified key-fact accuracy scoring for SAM-architecture benchmarks.

This module is the SINGLE SOURCE OF TRUTH for computing fuzzing/eact
fact-match scores. Both NEXUS and RAG arms must import and use
``compute_fact_score`` from here — never each maintain their own copy.

Returns:
    {"fuzzy_accuracy": float | None,
     "exact_accuracy": float | None,
     "scoring_detail": dict}

Rules (matching the Phase 1 specification):
    * "Insufficient evidence" or empty / whitespace-only answer → 0.0
    * Ground truth with *no* extractable facts → None (arm-independent)
    * Otherwise: fuzzy numeric match if GT contains numbers;
      fall back to exact regex-based overlap.
"""

from __future__ import annotations

import re
from typing import Any

# ── Regex patterns for extracting key facts ──────────────────────────────
# These are identical to the patterns in:
#   benchmarks/run_benchmark.py  (original)
#   benchmarks/rag_baseline.py   (duplicate — now obsoleted)
#   benchmarks/model_swarm.py    (duplicate — now obsoleted)
_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Percentages: 99.87%, 100%, 50%, 96.6%
    (re.compile(r'\b(\d+\.?\d*\s*%)(?=\s|$|[,.);])'), "percentage"),
    # Numbers with "million": 15.7 million, 19,000
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million\b', re.IGNORECASE), "number+million"),
    # Numbers with common technical units: 1,650 slots, 19,000 examples
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:live\s+)?(slots?|examples?|tokens?|parameters?|params?|subkeys?|distractors?|vocabulary|hops?)\b', re.IGNORECASE), "number+unit"),
    # Standalone large numbers (>=100) — exclude digits part of percentages
    (re.compile(r'\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b(?!\s*%)'), "large_number"),
    # @ notation: all_required@32, Rec@8
    (re.compile(r'\b(\w+@\d+)\b'), "at_notation"),
    # K= notation: K=32
    (re.compile(r'\b([Kk]=\d+)\b'), "k_notation"),
    # Named experiment IDs: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
    (re.compile(r'\b(Exp_\d+_\d+[A-Z]?_\w+)\b'), "experiment_id"),
    # Named concept IDs: Concept_SelectorBottleneck
    (re.compile(r'\b(Concept_\w+)\b'), "concept_id"),
    # Named decision IDs: Decision_PivotToNEXUS
    (re.compile(r'\b(Decision_\w+)\b'), "decision_id"),
    # Relation words
    (re.compile(r'\b(depends_on|validates|caused_by|contradicts|implements|mentioned_in|derived_from|related_to|replaces|blocked_by)\b', re.IGNORECASE), "relation"),
    # Key named SAM modes
    (re.compile(r'\b(core_only|oracle_memory|retrieved_memory|random_memory|oracle_text_memory|oracle_filter|oracle_text_memory|retrieved_memory_external_text_query)\b', re.IGNORECASE), "sam_mode"),
    # Gate references: Gate 1, Gate 2
    (re.compile(r'\b(Gate\s+\d+)\b', re.IGNORECASE), "gate_ref"),
    # Narrow qualitative markers for remaining comparative golds only.
    # Avoid generic phrases (e.g. bare KB sizes / cosine similarity) that would
    # newly score unrelated questions without a matching L1 surface.
    (re.compile(r'(O\(depth\s*\*\s*branching\))', re.IGNORECASE), "big_o"),
    (re.compile(r'\b(3 interdependent components)\b', re.IGNORECASE), "training_trio"),
    (re.compile(r'\b(source pointers)\b', re.IGNORECASE), "debug_phrase"),
    (re.compile(r'\b(narrative/textual)\b', re.IGNORECASE), "rag_wins_phrase"),
    (re.compile(r'\b(multi-hop causal)\b', re.IGNORECASE), "rag_hard_phrase"),
    (re.compile(
        r'\b(5-10x less context)\b',
        re.IGNORECASE,
    ), "context_advantage"),
]


# ── Public helpers ───────────────────────────────────────────────────────

def _extract_key_facts(text: str) -> set[str]:
    """Extract key facts from text using the canonical regex patterns.

    Returns a set of normalized fact strings suitable for set intersection.
    """
    facts: set[str] = set()
    for pattern, _fact_type in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact_str = match.group(0).strip().lower()
            # Normalize comma-separated numbers: 1,650 → 1650
            fact_str = re.sub(r'(\d),(\d)', r'\1\2', fact_str)
            facts.add(fact_str)
    return facts


def _extract_numbers(text: str) -> set[float]:
    """Extract all numeric values from text.

    Percentages are converted to decimals: "99.87%" → 0.9987.
    Comma-separated numbers are normalized: "1,650" → 1650.0.
    Avoids false positives like extracting "1" from "1-hop".
    """
    numbers: set[float] = set()
    tokens: list[str] = re.findall(r'[^\s]+', text)
    percent_re = re.compile(r'^([\d,]+\.?\d*)\s*%$')

    for token in tokens:
        # Strip trailing punctuation that is not %
        while token and token[-1] in ',.;:)!?' and token[-1] != '%':
            token = token[:-1]
        if not token:
            continue

        pm = percent_re.match(token)
        if pm:
            try:
                val = float(pm.group(1).replace(',', ''))
                numbers.add(round(val / 100.0, 10))
            except ValueError:
                pass
            continue

        stripped = token.replace(',', '')
        if stripped.replace('.', '', 1).isdigit():
            try:
                numbers.add(float(stripped))
            except ValueError:
                pass

    return numbers


def _fuzzy_number_match(pred_nums: set[float], gt_nums: set[float]) -> tuple[int, int]:
    """Match predicted numbers against ground truth with 5 % relative tolerance.

    Each GT number is matched at most once (greedy best-match).
    Returns (matches, total_gt_nums).
    """
    if not gt_nums:
        return 0, 0

    pred_list: list[float | None] = list(pred_nums)
    gt_list = sorted(gt_nums, reverse=True)
    matches = 0

    for gt in gt_list:
        best_idx: int = -1
        best_err: float = float('inf')

        for i, pred in enumerate(pred_list):
            if pred is None:
                continue
            denom = max(abs(gt), 0.001)
            rel_err = abs(pred - gt) / denom

            if rel_err < 0.05 or abs(pred - gt) < 0.001:
                if rel_err < best_err:
                    best_err = rel_err
                    best_idx = i

        if best_idx >= 0:
            matches += 1
            pred_list[best_idx] = None  # mark as consumed

    return matches, len(gt_nums)


# ── Primary scoring function ─────────────────────────────────────────────

def compute_fact_score(predicted: str, ground_truth: str) -> dict[str, Any]:
    """Compute key-fact match score with fuzzy numeric scoring.

    This is the **canonical** scoring function for all SAM benchmarks.
    Both NEXUS and RAG arms must call this — never maintain a copy elsewhere.

    Returns a dict with:
        fuzzy_accuracy  — primary score (fuzzy numeric, OR exact regex fallback)
        exact_accuracy  — legacy exact-regex score (for comparison)
        scoring_detail  — breakdown of numbers, matches, entity overlap

    ``fuzzy_accuracy`` is ``None`` **only** when the ground truth has no
    extractable facts.  It never depends on the predicted answer alone.
    """
    # Common detail skeleton
    empty_detail: dict[str, Any] = {
        "gt_numbers": [],
        "pred_numbers": [],
        "fuzzy_matches": 0,
        "total_gt": 0,
        "fuzzy_score": 0.0,
        "exact_score": 0.0,
        "entity_overlap": [],
    }

    # ── Ground-truth fact extraction (MUST come first) ─────────────
    # None is answer-independent: if GT has no extractable facts,
    # both arms get None regardless of what they answered.
    gt_facts = _extract_key_facts(ground_truth)

    if not gt_facts:
        empty_detail["fuzzy_score"] = None
        empty_detail["exact_score"] = None
        return {
            "fuzzy_accuracy": None,
            "exact_accuracy": None,
            "scoring_detail": empty_detail,
        }

    # ── Arm-independent early exits ────────────────────────────────
    # "Insufficient evidence" or empty / whitespace answer → 0.0
    # Only applies when GT *has* extractable facts (checked above).
    answer_lower = (predicted or "").strip().lower()
    if not answer_lower or "insufficient evidence" in answer_lower:
        return {
            "fuzzy_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "scoring_detail": empty_detail,
        }

    pred_facts = _extract_key_facts(predicted)

    # ── Exact regex score (legacy) ─────────────────────────────────
    intersection = gt_facts & pred_facts
    exact_score: float = round(len(intersection) / len(gt_facts), 4)

    # ── Fuzzy numeric score ────────────────────────────────────────
    gt_nums = _extract_numbers(ground_truth)
    pred_nums = _extract_numbers(predicted)

    if gt_nums:
        fuzzy_matches, total_gt = _fuzzy_number_match(pred_nums, gt_nums)
        fuzzy_score: float | None = (
            round(fuzzy_matches / total_gt, 4) if total_gt > 0 else None
        )
        primary_accuracy: float = fuzzy_score if fuzzy_score is not None else exact_score
    else:
        fuzzy_matches = 0
        total_gt = 0
        fuzzy_score = None
        primary_accuracy = exact_score

    return {
        "fuzzy_accuracy": primary_accuracy,
        "exact_accuracy": exact_score,
        "scoring_detail": {
            "gt_numbers": sorted(list(gt_nums)),
            "pred_numbers": sorted(list(pred_nums)),
            "fuzzy_matches": fuzzy_matches,
            "total_gt": total_gt,
            "fuzzy_score": fuzzy_score,
            "exact_score": exact_score,
            "entity_overlap": sorted(list(intersection)),
        },
    }
