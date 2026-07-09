"""Augmentation module for encoder training (TRAIN ONLY).

Generates synthetic variants of training questions to improve
robustness of the associative encoder. Never applied to val/test.

Augmentation strategies:
1. Word-order variants (swap adjective-noun pairs)
2. Colloquial fillers ("tell me", "I want to know", "can you find")
3. Synonym substitution from mined graph aliases
4. Polish phrasings for selected train questions
"""

from __future__ import annotations

import json
import random
import re
from typing import Any


# ── Colloquial filler templates ──

_FILLER_PREFIXES: list[str] = [
    "Tell me, ",
    "I want to know, ",
    "Can you tell me ",
    "Can you find out ",
    "I'd like to understand: ",
    "I wonder, ",
    "Could you explain ",
    "Let me ask you, ",
    "I was wondering, ",
    "Please tell me, ",
    "Help me understand: ",
]

_FILLER_SUFFIXES: list[str] = [
    ", please.",
    " — can you help?",
    ", if you could.",
    " — I need this info.",
    " — what do you think?",
    ", any ideas?",
]

# ── Alias synonym map (from graph aliases) ──
# Extracted from nexus/ingestion/populate_from_experiments.py

_ALIAS_SYNONYMS: dict[str, list[str]] = {
    "pipeline diagnosis": ["diagnosis experiment", "initial experiment", "pipeline setup"],
    "oracle memory": ["oracle memory experiment", "full validation", "validation experiment"],
    "chain retrieval": ["chain-set", "chain bce", "chain-set bce"],
    "selector": ["slot selection", "candidate selection", "learned selector"],
    "noise tolerance": ["noisy memory", "controlled noise", "noise handling"],
    "compact pkm": ["product-key memory", "product key memory", "pkm retrieval"],
    "dense dataset": ["dataset fix", "dense synthetic"],
    "required set": ["required-set", "required-set diagnostics"],
    "oracle filter": ["filter", "candidate filter"],
    "aggregation": ["aggregation variants"],
    "pivot to nexus": ["architecture pivot", "nexus pivot", "nexus approach"],
    "retrieval mismatch": ["projection mismatch", "query projection"],
    "selector bottleneck": ["selection bottleneck"],
    "architecture works": ["architecture validated", "architecture confirmed"],
    "external text": ["text query", "external query"],
    "pkm candidates": ["candidate generation", "candidate search"],
    "realistic distractors": ["realistic noise", "realistic noise inject", "real-world distractors"],
}

# ── Adjective-noun swap patterns ──
# Simple adjective-noun pairs where reordering is grammatically valid

_ADJ_NOUN_PAIRS: list[tuple[str, str]] = [
    ("overall accuracy", "accuracy overall"),
    ("controlled noise", "noise controlled"),
    ("synthetic dataset", "dataset synthetic"),
    ("external memory", "memory external"),
    ("random distractors", "distractors random"),
    ("key finding", "finding key"),
    ("learned selector", "selector learned"),
    ("oracle memory", "memory oracle"),
    ("chain retrieval", "retrieval chain"),
    ("dense dataset", "dataset dense"),
    ("noisy memory", "memory noisy"),
    ("compact PKM", "PKM compact"),
    ("oracle filter", "filter oracle"),
    ("realistic distractors", "distractors realistic"),
    ("rule-based verifier", "verifier rule-based"),
    ("retrieval revolution", "revolution retrieval"),
]


# ── Polish phrasing templates for train questions ──

_POLISH_PHRASINGS: list[tuple[str, str]] = [
    # (train_question_substring, polish_phrasing)
    # These match specific train questions by keyword
    ("What was the overall accuracy of the SAM oracle memory experiment",
     "Jaka byla ogolna dokladnosc eksperymentu SAM z pamiecia oracle?"),
    ("What accuracy did the SAM core-only baseline achieve",
     "Jaka dokladnosc osiagnal wariant bazowy SAM core-only?"),
    ("What was the breakthrough result of the chain-set BCE retriever",
     "Jaki byl przelomowy wynik retrievera chain-set BCE?"),
    ("What was the precision of the learned slot selector",
     "Jaka byla precyzja nauczonego selektora slotow?"),
    ("How many random distractors can SAM tolerate",
     "Ile losowych dystraktorow toleruje SAM?"),
    ("At what noise level does 3-hop reasoning collapse",
     "Przy jakim poziomie szumu zalamuje sie rozumowanie 3-skokowe?"),
    ("What accuracy did SAM achieve with the oracle-filter from chain candidates",
     "Jaka dokladnosc osiagnal SAM z filtrem oracle na kandydatach lancuchowych?"),
    ("What was the all_required@64 result for the dual encoder retriever",
     "Jaki byl wynik all_required@64 dla retrievera dual encoder?"),
    ("How many parameters does the SAM core model have",
     "Ile parametrow ma model rdzenia SAM?"),
    ("What is the 3-hop accuracy of SAM without memory",
     "Jaka jest dokladnosc 3-skokowa SAM bez pamieci?"),
]


def _apply_synonym_substitution(question: str) -> str | None:
    """Replace known entity mentions with synonym aliases."""
    lowered = question.lower()
    for original, synonyms in _ALIAS_SYNONYMS.items():
        if original in lowered:
            replacement = random.choice(synonyms)
            # Only substitute the first occurrence
            result = question.lower().replace(original, replacement, 1)
            # Restore capitalization of first letter
            if question[0].isupper():
                result = result[0].upper() + result[1:]
            return result
    return None


def _apply_word_order_variant(question: str) -> str | None:
    """Swap adjective-noun pairs where valid."""
    lowered = question.lower()
    for phrase, swapped in _ADJ_NOUN_PAIRS:
        if phrase in lowered:
            result = lowered.replace(phrase, swapped, 1)
            if question[0].isupper():
                result = result[0].upper() + result[1:]
            return result
    return None


def _apply_colloquial_filler(question: str) -> str:
    """Add colloquial filler prefix or suffix."""
    if random.random() < 0.7:  # prefix more common
        prefix = random.choice(_FILLER_PREFIXES)
        result = prefix + question[0].lower() + question[1:]
    else:
        suffix = random.choice(_FILLER_SUFFIXES)
        result = question.rstrip(".") + suffix
    return result


def _apply_polish_phrasing(question: str) -> str | None:
    """Replace an English question with its Polish equivalent."""
    for english_substr, polish in _POLISH_PHRASINGS:
        if question.startswith(english_substr) or english_substr in question:
            return polish
    return None


def augment_question(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate augmented variants of a single question.

    Returns list of augmented question dicts (original not included).
    Each variant preserves GT entities, intent, category.
    """
    text = question["question"]
    variants: list[dict[str, Any]] = []

    # Strategy 1: Word-order variant (1 variant)
    swapped = _apply_word_order_variant(text)
    if swapped and swapped != text.lower():
        v = dict(question)
        v["question"] = swapped
        v["augmentation"] = "word_order"
        variants.append(v)

    # Strategy 2: Colloquial filler (1 variant)
    filler = _apply_colloquial_filler(text)
    v = dict(question)
    v["question"] = filler
    v["augmentation"] = "filler"
    variants.append(v)

    # Strategy 3: Synonym substitution (up to 2 variants)
    for _ in range(2):
        sub = _apply_synonym_substitution(text)
        if sub and sub.lower() != text.lower():
            v = dict(question)
            v["question"] = sub
            v["augmentation"] = "synonym"
            variants.append(v)

    # Strategy 4: Polish phrasing (if available, 1 variant)
    polish = _apply_polish_phrasing(text)
    if polish:
        v = dict(question)
        v["question"] = polish
        v["augmentation"] = "polish"
        variants.append(v)

    return variants


def augment_dataset(
    questions: list[dict[str, Any]], seed: int = 42
) -> list[dict[str, Any]]:
    """Apply augmentation to a list of questions (TRAIN ONLY).

    Generates variants for each question and returns augmented dataset.
    Original questions are always included alongside their variants.

    Args:
        questions: List of question dicts from the training split
        seed: Random seed for reproducibility

    Returns:
        Augmented list (originals + variants)
    """
    random.seed(seed)

    augmented: list[dict[str, Any]] = list(questions)  # Include all originals

    for q in questions:
        variants = augment_question(q)
        augmented.extend(variants)

    random.shuffle(augmented)  # Shuffle for batch diversity
    return augmented


def load_augmented_train(train_path: str, seed: int = 42) -> list[dict[str, Any]]:
    """Load training data and apply augmentation."""
    with open(train_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    return augment_dataset(questions, seed=seed)
