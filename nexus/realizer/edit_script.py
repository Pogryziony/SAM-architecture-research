"""Deterministic edit-script computation for non-autoregressive realization.

Aligns a canonical answer string against a natural-language target at the
token level and produces per-token edit operations.  The operations are
non-autoregressive: every canonical token position receives exactly one
prediction, and all predictions can be made independently from the encoder
hidden states.
"""

from __future__ import annotations

import re
from typing import Any

_PIECE_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

# ── Edit operations ─────────────────────────────────────────────────
KEEP = 0
DELETE = 1
REPLACE = 2

# Special token IDs (reserved in the output vocabulary).
DELETE_ID = 0   # model predicts this to remove a token
KEEP_ID = 1     # model predicts the same token (implicit)
# PAD = 2, BOS = 3, EOS = 4 are reserved by the tokenizer convention.
# Actual vocabulary tokens start at OFFSET = 5.
OUTPUT_OFFSET = 5


def tokenize(text: str) -> list[str]:
    """Split *text* into lexical, whitespace and punctuation pieces.

    Uses the same regex as ``TrainOnlySubwordTokenizer`` so the edit
    script operates on the same token granularity as the neural model.
    """
    return _PIECE_RE.findall(text)


def _levenshtein_align(
    source: list[str],
    target: list[str],
) -> list[tuple[str, str]]:
    """Compute the optimal edit alignment between *source* and *target*.

    Returns a list of ``(source_op, target_op)`` pairs where *source_op*
    is a source token or ``"-"`` for insertions and *target_op* is a
    target token or ``"-"`` for deletions.
    """
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1].casefold() == target[j - 1].casefold() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack — produce alignment from right to left.
    alignment: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1].casefold() == target[j - 1].casefold():
            alignment.append((source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(("-", target[j - 1]))
            j -= 1
        else:
            alignment.append((source[i - 1], "-"))
            i -= 1
    alignment.reverse()
    return alignment


def compute_edit_target(canonical: str, target: str) -> list[str]:
    """Map each canonical-answer token to its surface-realization output.

    Every canonical token receives exactly one label:

    * The token itself (KEEP — identity mapping).
    * ``"[DELETE]"`` (the token is dropped).
    * A replacement token (REPLACE — the canonical token is substituted).

    Inserted tokens (present in *target* but absent from *canonical*)
    are merged into the preceding non-delete position.  When an insert
    precedes the first canonical token it is prepended to the first token's
    output via concatenation.
    """
    canonical_tokens = tokenize(canonical)
    target_tokens = tokenize(target)
    if not canonical_tokens:
        return []

    alignment = _levenshtein_align(canonical_tokens, target_tokens)
    per_source: list[list[str]] = [[] for _ in range(len(canonical_tokens))]
    source_index = 0

    for src, tgt in alignment:
        if src == "-":
            # Insertion — attach to nearest preceding source token.
            if source_index == 0:
                per_source[0].insert(max(0, len(per_source[0]) - 1), tgt)
            else:
                per_source[source_index - 1].append(tgt)
        elif tgt == "-":
            # Deletion.
            per_source[source_index] = ["[DELETE]"]
            source_index += 1
        else:
            # Match or substitution.
            per_source[source_index].append(tgt)
            source_index += 1

    # Flatten: keep only the last token at each position (substitutions
    # with merged inserts collapse to a single token).
    result: list[str] = []
    for tokens in per_source:
        if not tokens:
            result.append("[DELETE]")
        elif tokens == ["[DELETE]"]:
            result.append("[DELETE]")
        else:
            result.append(tokens[-1])

    # Post-condition: every canonical position has exactly one label.
    assert len(result) == len(canonical_tokens), (
        f"edit-target length mismatch: {len(result)} != {len(canonical_tokens)}"
    )
    return result


def apply_edit_target(canonical: str, labels: list[str]) -> str:
    """Apply predicted edit labels to the canonical answer.

    Tokens labeled ``"[DELETE]"`` are dropped; all other labels replace
    the corresponding canonical token verbatim.
    """
    canonical_tokens = tokenize(canonical)
    output: list[str] = []
    for index, label in enumerate(labels):
        if label == "[DELETE]":
            continue
        output.append(label)
    return "".join(output)


def edit_accuracy(labels: list[str], predicted: list[str]) -> dict[str, Any]:
    """Compute per-position accuracy of predicted edit labels."""
    if len(labels) != len(predicted):
        return {"exact_match": 0.0, "position_accuracy": 0.0, "total": 0, "correct": 0}
    total = len(labels)
    correct = sum(1 for g, p in zip(labels, predicted) if g == p)
    return {
        "exact_match": 1.0 if correct == total else 0.0,
        "position_accuracy": correct / total,
        "total": total,
        "correct": correct,
    }


__all__ = [
    "DELETE", "DELETE_ID", "KEEP", "KEEP_ID", "OUTPUT_OFFSET",
    "REPLACE", "apply_edit_target", "compute_edit_target",
    "edit_accuracy", "tokenize",
]
