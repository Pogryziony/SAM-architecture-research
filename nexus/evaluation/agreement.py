"""Inter-annotator agreement helpers (Cohen's κ and exact agreement)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def cohen_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> dict[str, Any]:
    """Cohen's κ for two equal-length categorical label sequences."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have equal length")
    n = len(labels_a)
    if n == 0:
        return {"n": 0, "kappa": None, "status": "NOT_RUN", "po": None, "pe": None}

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    cats = sorted(set(labels_a) | set(labels_b), key=lambda x: str(x))
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return {
        "n": n,
        "kappa": round(kappa, 6),
        "po": round(po, 6),
        "pe": round(pe, 6),
        "status": "COMPUTED",
    }
