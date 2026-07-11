"""Explicit evaluation contract — PASS / FAIL / INVALID.

Torch-independent.  Produces a mechanical verdict from a gate set
and measured metrics.  Never returns PASS for empty, missing or
malformed evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


@dataclass
class GateResult:
    name: str
    value: float
    threshold: float
    operator: str = ">="
    passed: bool = False
    note: str = ""


@dataclass
class ContractResult:
    verdict: Verdict
    gates: list[GateResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "gates": [
                {
                    "name": g.name,
                    "value": g.value,
                    "threshold": g.threshold,
                    "operator": g.operator,
                    "passed": g.passed,
                    "note": g.note,
                }
                for g in self.gates
            ],
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
        }


def _is_valid_float(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return True


def evaluate_contract(
    metrics: Mapping[str, Any],
    gates: list[dict[str, Any]],
    required_meta: dict[str, Any] | None = None,
) -> ContractResult:
    """Evaluate a mechanical contract from metrics and gate definitions.

    Args:
        metrics: Measured values (e.g. {"recall@10": 0.7747, "latency_p50": 32.7}).
        gates: Gate definitions, each with name, threshold, and optional operator.
        required_meta: Optional metadata constraints (run_id, source_sha, etc.).

    Returns:
        ContractResult with verdict and per-gate details.
    """
    errors: list[str] = []

    # ── Empty gate set is INVALID ──
    if not gates:
        return ContractResult(
            verdict=Verdict.INVALID,
            errors=["empty gate set: at least one gate is required"],
        )

    # ── Validate each gate ──
    results: list[GateResult] = []
    for gdef in gates:
        name = gdef.get("name", "")
        threshold = gdef.get("threshold", 0.0)
        operator = gdef.get("operator", ">=")

        if not name:
            errors.append(f"gate missing name: {gdef}")
            continue

        if name not in metrics:
            errors.append(f"required gate '{name}' not found in metrics")
            continue

        value = metrics[name]
        if not _is_valid_float(value):
            errors.append(f"gate '{name}' has invalid value: {value}")
            continue

        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        elif operator == ">":
            passed = value > threshold
        elif operator == "<":
            passed = value < threshold
        else:
            errors.append(f"gate '{name}' has unknown operator: {operator}")
            continue

        results.append(GateResult(
            name=name, value=float(value), threshold=float(threshold),
            operator=operator, passed=passed,
            note="" if passed else f"{value:.4f} {operator} {threshold}",
        ))

    # ── Validate meta constraints ──
    if required_meta:
        for key, expected in required_meta.items():
            actual = metrics.get(key)
            if actual is None:
                errors.append(f"required metadata '{key}' missing")
            elif actual != expected:
                errors.append(
                    f"metadata '{key}' mismatch: expected {expected!r}, got {actual!r}"
                )

    # ── Any validation error → INVALID ──
    if errors:
        return ContractResult(
            verdict=Verdict.INVALID, gates=results, errors=errors, metrics=dict(metrics)
        )

    # ── All gates must be present ──
    if not results:
        return ContractResult(
            verdict=Verdict.INVALID,
            errors=["no valid gates could be evaluated"],
            metrics=dict(metrics),
        )

    # ── All pass → PASS; any fail → FAIL ──
    all_pass = all(g.passed for g in results)
    return ContractResult(
        verdict=Verdict.PASS if all_pass else Verdict.FAIL,
        gates=results,
        metrics=dict(metrics),
    )
