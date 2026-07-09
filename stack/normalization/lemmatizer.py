"""Normalization sublayer for entity spotting input.

Dictionary-based PL + EN lemmatization using simple stemming rules
and stopword removal. Controlled by NEXUSConfig.enable_normalization.
"""

from __future__ import annotations

import re

# ── English stopwords ──
EN_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "their", "we", "you", "i", "me", "my", "what", "which",
    "who", "whom", "about", "also", "tell", "find", "want", "know",
    "show", "give", "let", "please",
}

# ── Polish stopwords ──
PL_STOPWORDS: set[str] = {
    "i", "w", "na", "z", "do", "się", "nie", "to", "że", "a", "o",
    "po", "za", "co", "jak", "jest", "tym", "tego", "tej", "ten",
    "ta", "te", "być", "był", "była", "było", "byli", "były", "będzie",
    "są", "ma", "mają", "może", "można", "dla", "od", "przez", "przy",
    "ale", "lub", "czy", "gdy", "aby", "więc", "już", "tak", "oraz",
    "jako", "jego", "jej", "ich", "mnie", "mi", "nas", "nam", "ci",
    "tobie", "ciebie", "sobie", "nią", "nim", "tam", "tu", "teraz",
    "jeszcze", "bardzo", "by", "bo", "gdzie", "kiedy", "który", "która",
    "które", "jaki", "jaka", "jakie", "tylko", "także", "również",
    "ponieważ", "dlatego", "jednak", "natomiast",
}

# ── English suffixes for light stemming ──
EN_SUFFIXES: list[tuple[str, str]] = [
    ("ization", "ize"),
    ("isation", "ise"),
    ("tional", "tion"),
    ("ational", "ate"),
    ("fulness", "ful"),
    ("nesses", "ness"),
    ("ments", "ment"),
    ("ables", "able"),
    ("ingly", "ing"),
    ("ingly", "ing"),
    ("ement", "e"),
    ("eness", "ene"),
    ("ities", "ity"),
    ("ively", "ive"),
    ("sness", "s"),
    ("ingly", "ing"),
    ("fully", "ful"),
    # Plural / verb forms
    ("ies", "y"),
    ("ves", "f"),
    ("ses", "s"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("ing", ""),
    ("ed", "e"),
    ("ed", ""),
    ("er", ""),
    ("est", ""),
    ("ly", ""),
    ("s", ""),
    ("'s", ""),
]

# ── Polish suffixes for light stemming ──
PL_SUFFIXES: list[tuple[str, str]] = [
    # Noun endings
    ("ami", ""),
    ("ach", ""),
    ("om", ""),
    ("owi", ""),
    ("em", ""),
    ("ie", ""),
    ("u", ""),
    ("a", ""),
    ("y", ""),
    ("e", ""),
    ("i", ""),
    # Verb endings
    ("amy", "ać"),
    ("amy", "eć"),
    ("amy", "ić"),
    ("asz", "ać"),
    ("asz", "eć"),
    ("asz", "ić"),
    ("ał", "ać"),
    ("ał", "eć"),
    ("ał", "ić"),
    ("ali", "ać"),
    ("ali", "eć"),
    ("ali", "ić"),
    ("ają", "ać"),
    ("ają", "eć"),
    ("ają", "ić"),
    ("ano", "ać"),
    ("ano", "eć"),
    ("ano", "ić"),
    # Adjective endings
    ("ego", "y"),
    ("ego", "i"),
    ("emu", "y"),
    ("emu", "i"),
    ("ymi", "y"),
    ("ymi", "i"),
    ("ym", "y"),
    ("ym", "i"),
    ("ej", "y"),
    ("ej", "i"),
    # General PL morphology
    ("ów", ""),
    ("owie", ""),
    ("ów", ""),
    ("ami", ""),
    ("ami", "a"),
    # Diminutive
    ("eczek", "ek"),
    ("eczk", "ek"),
    ("eczek", ""),
]


def _stem_en(word: str) -> str:
    """Apply simple English suffix-stripping to produce a pseudo-stem."""
    word_lower = word.lower()
    if len(word_lower) <= 3:
        return word_lower
    for suffix, replacement in EN_SUFFIXES:
        if word_lower.endswith(suffix):
            stemmed = word_lower[: -len(suffix)] + replacement
            if len(stemmed) >= 3:
                return stemmed
            return word_lower
    return word_lower


def _stem_pl(word: str) -> str:
    """Apply simple Polish suffix-stripping to produce a pseudo-stem."""
    word_lower = word.lower()
    if len(word_lower) <= 3:
        return word_lower
    for suffix, replacement in PL_SUFFIXES:
        if word_lower.endswith(suffix):
            stemmed = word_lower[: -len(suffix)] + replacement
            if len(stemmed) >= 3:
                return stemmed
            return word_lower
    return word_lower


# ── Language detection (simple character-based) ──
_PL_SPECIFIC: set[str] = set("ąćęłńóśźż")


def _is_polish(text: str) -> bool:
    """Heuristic: if text contains Polish-specific characters, treat as Polish."""
    return any(ch in _PL_SPECIFIC for ch in text.lower())


def normalize(text: str) -> str:
    """Normalize question text by stemming and removing stopwords.

    Detects language (PL/EN) and applies appropriate stemming.
    Returns space-separated normalized tokens.
    """
    is_pl = _is_polish(text)
    stopwords = PL_STOPWORDS if is_pl else EN_STOPWORDS
    stem_fn = _stem_pl if is_pl else _stem_en

    tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", text.lower())
    normalized: list[str] = []
    for token in tokens:
        if token in stopwords or len(token) <= 1:
            continue
        stemmed = stem_fn(token)
        if stemmed and len(stemmed) >= 2:
            normalized.append(stemmed)

    return " ".join(normalized)


def normalize_tokens(text: str) -> list[str]:
    """Normalize and return token list instead of joined string."""
    result = normalize(text)
    return result.split() if result else []
