"""
Post-edit module — corrects hallucinated numbers in LLM answers.

After the LLM generates an answer, this module checks every sentence
for numeric assertions and grounds them against the evidence pack.
Uses the same 5%-tolerance numeric matching as the verifier and
benchmark scorer.

Strategy:
  1. Split the answer into sentences.
  2. For each sentence, extract all numbers via _extract_numbers.
  3. If ALL numbers in a sentence match evidence (5% tolerance) → keep it.
  4. If ANY number doesn't match:
     a. Try to find a replacement number from evidence with similar context
        (surrounding words suggesting the same metric type).
     b. If a replacement is found: substitute the number in the sentence.
     c. If not: remove the entire sentence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Import scoring utils for 5%-tolerance numeric matching
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.scoring import _extract_numbers, _fuzzy_number_match  # noqa: E402


# ── Metric-context words for number disambiguation ─────────────────────
# When we need to replace a hallucinated number, we look for the closest
# evidence number that appears near similar context words.
_METRIC_CONTEXT_WORDS: set[str] = {
    "accuracy", "precision", "recall", "f1", "f1-score", "loss",
    "perplexity", "ppl", "bleu", "rouge", "exact", "fuzzy",
    "top-1", "top-5", "all-required", "all_required",
    "slots", "examples", "tokens", "parameters", "params",
    "million", "billion", "percent", "percentage",
    "score", "rate", "latency", "ms", "seconds", "throughput",
    "improvement", "degradation", "drop", "gain", "increase",
    "decrease", "baseline", "oracle", "retrieved", "random",
    "core_only", "oracle_memory", "retrieved_memory",
    "retrieval", "generation", "embedding", "hidden",
    "layers", "heads", "dimension", "batch", "epoch",
    "step", "training", "validation", "test",
    "gate", "distractors", "subkeys", "vocabulary", "hops",
}


def _context_words_around(text: str, number_str: str, window: int = 4) -> set[str]:
    """Extract metric-context words within `window` tokens of a number string."""
    # Find the number in the text
    idx = text.find(number_str)
    if idx == -1:
        return set()

    # Get surrounding tokens
    before = text[:idx].split()
    after = text[idx + len(number_str):].split()

    nearby = before[-window:] + after[:window]
    context = set()
    for token in nearby:
        token_clean = token.strip(",.()[]{}:;\"'!?").lower()
        if token_clean in _METRIC_CONTEXT_WORDS:
            context.add(token_clean)
        # Also partial matches (e.g., "retrieved_memory" in "retrieved_memory_...")
        for ctx_word in _METRIC_CONTEXT_WORDS:
            if len(ctx_word) >= 4 and (
                ctx_word in token_clean or token_clean in ctx_word
            ):
                context.add(ctx_word)
                break

    return context


def _find_matching_evidence_number(
    number: float,
    context_words: set[str],
    evidence_facts: list[str],
    evidence_numbers: set[float],
) -> float | None:
    """
    Find the best replacement number from evidence.

    Strategy:
      1. Find evidence facts with overlapping context words.
      2. Extract numbers from those context-matched facts.
      3. If any numbers are found, return the one closest to the original.
      4. If NO context words exist (meaning we can't tell what the number
         represents), return None — the sentence should be removed, not
         blindly replaced.
    """
    if not evidence_numbers:
        return None

    # Without context words, we can't reliably find a replacement
    if not context_words:
        return None

    # Collect numbers from context-matching evidence facts
    context_numbers: list[float] = []
    for fact in evidence_facts:
        fact_lower = fact.lower()
        if any(cw in fact_lower for cw in context_words):
            context_numbers.extend(_extract_numbers(fact))

    if not context_numbers:
        return None

    # Closest context-matched number
    return min(context_numbers, key=lambda x: abs(x - number))


def _format_number_in_context(text: str, old_num_str: str, new_value: float) -> str:
    """
    Replace a number occurrence in text, preserving percent formatting.

    Returns the modified text with the first occurrence of old_num_str
    replaced by the new_value, formatted appropriately.
    """
    # Determine if this was a percentage (check if the original token has %)
    is_pct = old_num_str.endswith('%')

    if is_pct:
        if new_value < 1.0:
            # Already in decimal form (0.9987) → convert to percent
            pct_str = f"{new_value * 100:.2f}"
            if '.' in pct_str:
                pct_str = pct_str.rstrip('0').rstrip('.')
            new_str = f"{pct_str}%"
        else:
            # Already a percentage value (99.87) → use directly
            pct_str = f"{new_value:.2f}"
            if '.' in pct_str:
                pct_str = pct_str.rstrip('0').rstrip('.')
            new_str = f"{pct_str}%"
    else:
        if new_value >= 1000 and new_value == int(new_value):
            new_str = f"{int(new_value):,}"
        elif new_value == int(new_value):
            new_str = f"{int(new_value)}"
        else:
            new_str = f"{new_value:.4g}"

    # Replace only the first occurrence of old_num_str in text
    return text.replace(old_num_str, new_str, 1)


def edit_answer(raw_answer: str, evidence_pack: dict[str, Any]) -> dict[str, Any]:
    """
    Post-edit an LLM answer to fix or remove hallucinated numbers.

    Args:
        raw_answer: The raw answer text from the model.
        evidence_pack: The evidence pack dict (from build_evidence_pack).

    Returns:
        Dict with:
            - answer: the edited answer string
            - changes: list of dicts describing each fix/removal
            - numbers_fixed: count of numbers replaced with evidence values
            - numbers_removed: count of numbers that couldn't be fixed
    """
    # Collect evidence numbers (same logic as verifier._collect_evidence_numbers)
    evidence_numbers = _collect_evidence_numbers(evidence_pack)

    # Collect all evidence text for context matching
    evidence_facts: list[str] = []
    for fact in evidence_pack.get("facts", []):
        if isinstance(fact, str):
            evidence_facts.append(fact)
    for nf in evidence_pack.get("node_facts", []):
        text = nf.get("text", "") if isinstance(nf, dict) else str(nf)
        if text:
            evidence_facts.append(text)
    for path_data in evidence_pack.get("paths", []):
        for node in path_data.get("nodes", []):
            for key in ("id", "name", "display_name", "title", "key_finding", "description"):
                val = node.get(key, "")
                if isinstance(val, str) and val:
                    evidence_facts.append(val)
    for nf in evidence_pack.get("neighbor_facts", []):
        text = nf.get("text", "") if isinstance(nf, dict) else str(nf)
        if text:
            evidence_facts.append(text)

    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', raw_answer)

    changes: list[dict[str, Any]] = []
    numbers_fixed = 0
    numbers_removed = 0
    kept_sentences: list[str] = []

    for sentence in sentences:
        if not sentence.strip():
            kept_sentences.append(sentence)
            continue

        numbers = _extract_numbers(sentence)
        if not numbers:
            # No numbers in this sentence — keep it as-is
            kept_sentences.append(sentence)
            continue

        # Check each number in this sentence against evidence
        unsupported: list[tuple[float, str, set[str]]] = []
        # We need to track which actual text substrings are the numbers
        # to be able to replace them later
        number_occurrences: list[tuple[float, str]] = []

        # Extract number occurrences with their string representations
        tokens = re.findall(r'[^\s]+', sentence)
        percent_re = re.compile(r'^([\d,]+\.?\d*)\s*%$')

        for token in tokens:
            token_clean = token.rstrip(',.;:)!?')
            pm = percent_re.match(token_clean)
            if pm:
                try:
                    val = float(pm.group(1).replace(',', '')) / 100.0
                    number_occurrences.append((round(val, 10), pm.group(0)))
                except ValueError:
                    pass
                continue

            stripped = token_clean.replace(',', '')
            if stripped.replace('.', '', 1).isdigit():
                try:
                    val = float(stripped)
                    number_occurrences.append((val, token_clean))
                except ValueError:
                    pass

        # Check each number occurrence against evidence
        has_unsupported = False
        for num_val, num_str in number_occurrences:
            if evidence_numbers:
                matches, _total = _fuzzy_number_match({num_val}, evidence_numbers)
                if matches == 0:
                    # Unsupported — try to find context
                    ctx_words = _context_words_around(sentence, num_str, window=6)
                    unsupported.append((num_val, num_str, ctx_words))
                    has_unsupported = True
            else:
                # No evidence numbers at all — all numbers are unsupported
                ctx_words = _context_words_around(sentence, num_str, window=6)
                unsupported.append((num_val, num_str, ctx_words))
                has_unsupported = True

        if not has_unsupported:
            # All numbers supported — keep sentence
            kept_sentences.append(sentence)
            continue

        # Try to fix unsupported numbers
        edited_sentence = sentence
        sentence_repairable = True

        for num_val, num_str, ctx_words in unsupported:
            replacement = _find_matching_evidence_number(
                num_val, ctx_words, evidence_facts, evidence_numbers,
            )

            # Sanity check: if replacement is too different (>5x), it's likely
            # a unit mismatch (count vs percentage) — remove instead of fix.
            if replacement is not None:
                larger = max(abs(num_val), abs(replacement), 0.001)
                ratio = abs(num_val - replacement) / larger
                if ratio > 0.8:  # More than 80% different = order-of-magnitude mismatch
                    replacement = None

            if replacement is not None:
                edited_sentence = _format_number_in_context(
                    edited_sentence, num_str, replacement,
                )
                numbers_fixed += 1
                changes.append({
                    "action": "fixed",
                    "sentence": sentence.strip(),
                    "old_number": num_str,
                    "new_number": replacement,
                    "context": sorted(ctx_words),
                })
            else:
                numbers_removed += 1
                changes.append({
                    "action": "removed",
                    "sentence": sentence.strip(),
                    "unsupported_number": num_str,
                    "context": sorted(ctx_words),
                })
                sentence_repairable = False
                break  # Can't fix this sentence — remove it entirely

        # Only keep the sentence if ALL unsupported numbers were fixed
        if sentence_repairable:
            kept_sentences.append(edited_sentence)
        # Otherwise the sentence is dropped (removed)

    edited_answer = " ".join(kept_sentences)
    # Clean up extra whitespace
    edited_answer = re.sub(r'\s+', ' ', edited_answer).strip()

    return {
        "answer": edited_answer,
        "changes": changes,
        "numbers_fixed": numbers_fixed,
        "numbers_removed": numbers_removed,
    }


def _collect_evidence_numbers(evidence_pack: dict[str, Any]) -> set[float]:
    """
    Collect all numeric values from evidence facts and node metadata.

    Mirrors verifier._collect_evidence_numbers to ensure consistent
    5%-tolerance verification.
    """
    numbers: set[float] = set()

    for fact in evidence_pack.get("facts", []):
        numbers.update(_extract_numbers(fact))

    for nf in evidence_pack.get("node_facts", []):
        text = nf.get("text", "")
        if isinstance(text, str) and text:
            numbers.update(_extract_numbers(text))

    for path_data in evidence_pack.get("paths", []):
        for node in path_data.get("nodes", []):
            for key in ("id", "name", "display_name", "title", "key_finding", "description"):
                val = node.get(key, "")
                if isinstance(val, str) and val:
                    numbers.update(_extract_numbers(val))

    # Also from neighbor_facts
    for nf in evidence_pack.get("neighbor_facts", []):
        text = nf.get("text", "")
        if isinstance(text, str) and text:
            numbers.update(_extract_numbers(text))

    # Also from the numbers section (flat, machine-readable)
    for n in evidence_pack.get("numbers", []):
        if isinstance(n, (int, float)):
            numbers.add(float(n))
        elif isinstance(n, dict):
            for k, v in n.items():
                if k != "entity":
                    numbers.update(_extract_numbers(str(v)))

    # Also from numbers_by_metric (grouped)
    for metric_name, entries in evidence_pack.get("numbers_by_metric", {}).items():
        for entry in entries:
            value = entry.get("value", "")
            if value:
                numbers.update(_extract_numbers(str(value)))

    return numbers
