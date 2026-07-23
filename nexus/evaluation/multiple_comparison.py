"""Family-wide multiple-comparison correction (Holm–Bonferroni)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values preserving input order.

    For sorted p_(1) ≤ … ≤ p_(m):
      adj_(i) = max_{j≤i} min(1, (m − j + 1) · p_(j))
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda t: t[1])
    adj_by_rank: list[float] = []
    running = 0.0
    for rank, (_orig_i, p) in enumerate(indexed):
        # rank 0 → multiplier m; rank m-1 → multiplier 1
        cand = min(1.0, (m - rank) * p)
        running = max(running, cand)
        adj_by_rank.append(running)
    out = [0.0] * m
    for rank, (orig_i, _p) in enumerate(indexed):
        out[orig_i] = adj_by_rank[rank]
    return out


def apply_holm_to_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    p_key: str = "mcnemar.p_value",
) -> list[dict[str, Any]]:
    """Attach Holm-adjusted p-values across a comparison family.

    ``p_key`` is a dotted path into each comparison dict.
    """

    def _get(d: Mapping[str, Any], path: str) -> float:
        cur: Any = d
        for part in path.split("."):
            cur = cur[part]
        return float(cur)

    raw = [_get(c, p_key) for c in comparisons]
    adjusted = holm_adjust(raw)
    out: list[dict[str, Any]] = []
    for comp, adj, raw_p in zip(comparisons, adjusted, raw):
        row = dict(comp)
        mc = dict(row.get("multiple_comparison") or {})
        mc.update(
            {
                "method": "holm",
                "n_tests": len(comparisons),
                "raw_p_value": raw_p,
                "adjusted_p_value": adj,
                "family_size": len(comparisons),
            }
        )
        row["multiple_comparison"] = mc
        boot = row.get("bootstrap") or {}
        ci_low = boot.get("ci_low")
        ci_high = boot.get("ci_high")
        if ci_low is not None and float(ci_low) > 0 and adj < 0.05:
            row["superiority_verdict"] = "LEFT_BETTER"
        elif ci_high is not None and float(ci_high) < 0 and adj < 0.05:
            row["superiority_verdict"] = "RIGHT_BETTER"
        else:
            row["superiority_verdict"] = "INCONCLUSIVE"
        out.append(row)
    return out
