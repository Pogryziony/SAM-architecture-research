"""
Paired comparison aggregation for NEXUS vs RAG (or any two arms).

PRIMARY section — paired-only metrics:
    Only questions where BOTH arms have a non-None score are compared.
    Reports: paired_n, mean accuracy per arm, win/loss/tie counts,
    and a two-sided sign test p-value.

SECONDARY section — per-arm unpaired stats (clearly labelled).
"""

from __future__ import annotations

import math
from typing import Any


# ── Sign test (two-sided exact binomial) ─────────────────────────────────

def _exact_binomial_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial test p-value (no scipy needed).

    Computes the probability of observing |k - n/2| or more extreme
    under the null hypothesis H0: p(win) = 0.5.
    """
    if n == 0:
        return 1.0

    # Use exact combinatoric computation with early bail-out for
    # large n (approximate with normal when feasible, but prefer exact).
    # For small sample sizes this is cheap and exact.
    prob_sum = 0.0
    expected = n * p
    # Determine the "more extreme" tail
    if k <= expected:
        # Left tail up to k, plus symmetric right tail
        for i in range(0, k + 1):
            prob_sum += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        # Add right tail (symmetric)
        right_start = int(2 * expected - k)
        if right_start <= n:
            for i in range(max(right_start, 0), n + 1):
                prob_sum += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    else:
        # Right tail from k, plus symmetric left tail
        for i in range(k, n + 1):
            prob_sum += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        left_end = int(2 * expected - k)
        if left_end >= 0:
            for i in range(0, min(left_end, n) + 1):
                prob_sum += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return min(prob_sum, 1.0)


def sign_test(arm_a_wins: int, arm_b_wins: int, ties: int = 0) -> dict[str, Any]:
    """Two-sided sign test for paired comparisons.

    Parameters:
        arm_a_wins: Number of questions where arm A beats arm B.
        arm_b_wins: Number of questions where arm B beats arm A.
        ties:       Number of ties (ignored in p-value computation).

    Returns:
        dict with: test, n_discordant, arm_a_wins, arm_b_wins, ties, p_value
    """
    n = arm_a_wins + arm_b_wins
    if n == 0:
        return {
            "test": "two-sided sign test (exact binomial)",
            "n_discordant": 0,
            "arm_a_wins": arm_a_wins,
            "arm_b_wins": arm_b_wins,
            "ties": ties,
            "p_value": 1.0,
            "note": "No discordant pairs — test cannot be performed.",
        }

    # Use the smaller of the two for the tail probability
    k = min(arm_a_wins, arm_b_wins)
    p_value = _exact_binomial_p(k, n, 0.5)

    return {
        "test": "two-sided sign test (exact binomial)",
        "n_discordant": n,
        "arm_a_wins": arm_a_wins,
        "arm_b_wins": arm_b_wins,
        "ties": ties,
        "p_value": round(p_value, 6),
    }


# ── Paired comparison ───────────────────────────────────────────────────

def compare_paired(
    arm_a_scores: list[float | None],
    arm_b_scores: list[float | None],
    arm_a_label: str = "NEXUS",
    arm_b_label: str = "RAG",
) -> dict[str, Any]:
    """Paired comparison of two arms.

    Only questions where BOTH arms have a non-None score are included
    in the *primary* comparison.  Unpaired (missing-on-one-side) stats
    are reported in a clearly-labelled *secondary* section.

    Parameters:
        arm_a_scores: List of accuracy scores for arm A (aligned 1:1 with arm B).
        arm_b_scores: List of accuracy scores for arm B.
        arm_a_label:  Display label for arm A.
        arm_b_label:  Display label for arm B.

    Returns:
        dict with keys: paired_n, arm_a, arm_b, win_loss_tie, sign_test_p,
                        secondary_unpaired
    """
    assert len(arm_a_scores) == len(arm_b_scores), \
        "Score lists must have equal length (aligned per question)"

    # ── Paired subset ───────────────────────────────────────────────
    paired_a: list[float] = []
    paired_b: list[float] = []
    wins_a = 0
    wins_b = 0
    ties = 0

    # Track unpaired
    unpaired_a_only = 0   # A has score, B is None
    unpaired_b_only = 0   # B has score, A is None
    unpaired_a_scores: list[float] = []
    unpaired_b_scores: list[float] = []

    for sa, sb in zip(arm_a_scores, arm_b_scores):
        if sa is None and sb is None:
            continue
        elif sa is not None and sb is not None:
            # Paired
            paired_a.append(sa)
            paired_b.append(sb)
            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            else:
                ties += 1
        elif sa is not None:
            # Only arm A has a score
            unpaired_a_only += 1
            unpaired_a_scores.append(sa)
        else:
            # Only arm B has a score
            unpaired_b_only += 1
            unpaired_b_scores.append(sb)

    paired_n = len(paired_a)

    # ── Pairwise stats ──────────────────────────────────────────────
    if paired_n > 0:
        mean_a = round(sum(paired_a) / paired_n, 4)
        mean_b = round(sum(paired_b) / paired_n, 4)
    else:
        mean_a = 0.0
        mean_b = 0.0

    sign_result = sign_test(wins_a, wins_b, ties)

    # ── Build result ────────────────────────────────────────────────
    result: dict[str, Any] = {
        "comparison_type": "PAIRED (both arms scorable)",
        "paired_n": paired_n,
        arm_a_label.lower(): {
            "mean_accuracy": mean_a,
            "n": paired_n,
            "label": arm_a_label,
        },
        arm_b_label.lower(): {
            "mean_accuracy": mean_b,
            "n": paired_n,
            "label": arm_b_label,
        },
        "win_loss_tie": {
            f"{arm_a_label}_wins": wins_a,
            f"{arm_b_label}_wins": wins_b,
            "ties": ties,
        },
        "sign_test_p": sign_result["p_value"],
        "sign_test_detail": sign_result,
        "secondary_unpaired": {
            "label": "SECONDARY -- unpaired stats (excluded from primary comparison)",
            "total_questions": len(arm_a_scores),
            "both_none": len(arm_a_scores) - paired_n - unpaired_a_only - unpaired_b_only,
            arm_a_label.lower() + "_only_scored": {
                "n": unpaired_a_only,
                "mean_accuracy": (
                    round(sum(unpaired_a_scores) / unpaired_a_only, 4)
                    if unpaired_a_only > 0 else None
                ),
            },
            arm_b_label.lower() + "_only_scored": {
                "n": unpaired_b_only,
                "mean_accuracy": (
                    round(sum(unpaired_b_scores) / unpaired_b_only, 4)
                    if unpaired_b_only > 0 else None
                ),
            },
        },
    }

    return result


def pretty_print_comparison(comp: dict[str, Any]) -> None:
    """Print a human-readable summary of the comparison."""
    primary = comp["comparison_type"]
    paired_n = comp["paired_n"]
    wlt = comp["win_loss_tie"]
    p_val = comp["sign_test_p"]

    # Find labels
    arm_keys = [k for k in comp if isinstance(comp[k], dict) and "mean_accuracy" in comp[k] and comp[k].get("n") == paired_n]
    # Simpler approach: just iterate known keys
    arm_entries = []
    for key in ("nexus", "rag"):
        if key in comp:
            arm_entries.append((key.upper(), comp[key]["mean_accuracy"]))

    if not arm_entries:
        # Try to detect labels
        for k, v in comp.items():
            if isinstance(v, dict) and "label" in v and "mean_accuracy" in v and v.get("n") == paired_n:
                arm_entries.append((v["label"], v["mean_accuracy"]))

    print(f"\n{'='*60}")
    print(f"  {primary}")
    print(f"{'='*60}")
    print(f"  Paired N (both scorable): {paired_n}")
    for label, acc in arm_entries:
        print(f"  {label} mean accuracy:       {acc:.4f} ({acc:.2%})")
    print(f"\n  Win / Loss / Tie:")
    for k, v in wlt.items():
        print(f"    {k}: {v}")
    print(f"\n  Sign test p-value: {p_val}")
    if p_val < 0.05:
        print(f"  => Statistically significant at alpha=0.05")
    else:
        print(f"  => Not statistically significant (p >= 0.05)")

    # Secondary
    sec = comp.get("secondary_unpaired", {})
    if sec:
        print(f"\n  {sec.get('label', '')}")
        for k, v in sec.items():
            if k == "label":
                continue
            if isinstance(v, dict):
                print(f"    {k}: n={v.get('n')}, acc={v.get('mean_accuracy')}")
            else:
                print(f"    {k}: {v}")
    print()
