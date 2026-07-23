"""Paired statistical evaluation helpers.

Systems answer the same questions, so comparisons use paired methods.
These helpers are deterministic given a seed and operate on binary or
continuous paired scores.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Paired bootstrap CI for mean(left - right)."""
    if len(left) != len(right):
        raise ValueError("paired sequences must have equal length")
    n = len(left)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": None,
            "ci_low": None,
            "ci_high": None,
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
        }
    diffs = [float(left[i]) - float(right[i]) for i in range(n)]
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_bootstrap))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return {
        "n": n,
        "mean_diff": round(mean_diff, 6),
        "ci_low": round(samples[lo_idx], 6),
        "ci_high": round(samples[hi_idx], 6),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "seed": seed,
    }


def mcnemar_exact(
    left_correct: Sequence[bool],
    right_correct: Sequence[bool],
) -> dict[str, float | int]:
    """Exact McNemar test on paired binary outcomes (b vs c discordant pairs)."""
    if len(left_correct) != len(right_correct):
        raise ValueError("paired sequences must have equal length")
    b = 0  # left correct, right wrong
    c = 0  # left wrong, right correct
    for l, r in zip(left_correct, right_correct):
        if l and not r:
            b += 1
        elif r and not l:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    # Two-sided exact binomial test under p=0.5
    # P(X<=min(b,c)) * 2, capped at 1
    k = min(b, c)
    # Sum C(n,i) / 2^n for i=0..k
    # Use iterative product to avoid huge factorials.
    prob = 0.0
    for i in range(k + 1):
        prob += _binom_pmf(n, i)
    p = min(1.0, 2.0 * prob)
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": round(p, 6),
    }


def paired_effect_size(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, float | int | None]:
    """Cohen's dz for paired differences."""
    if len(left) != len(right):
        raise ValueError("paired sequences must have equal length")
    n = len(left)
    if n < 2:
        return {"n": n, "cohens_dz": None, "mean_diff": None}
    diffs = [float(left[i]) - float(right[i]) for i in range(n)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    dz = (mean / sd) if sd > 0 else None
    return {
        "n": n,
        "mean_diff": round(mean, 6),
        "cohens_dz": None if dz is None else round(dz, 6),
    }


def _binom_pmf(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    # C(n,k) / 2^n
    coeff = 1.0
    for i in range(k):
        coeff *= (n - i) / (i + 1)
    return coeff / (2.0 ** n)
