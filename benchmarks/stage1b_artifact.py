"""Serialized-artifact validation for the Stage 1B frozen evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_REQUIRED_META = {
    "evaluation_commit_sha", "model_checkpoint", "calibration_split",
    "calibration_sample_count", "selected_threshold", "threshold_search_metrics",
    "frozen_split", "question_count",
}
_REQUIRED_METRICS = {
    "entity_precision", "entity_recall", "entity_f1", "exact_accuracy",
    "candidate_pool_recall", "reranker_recall", "parser_failures", "latency_ms", "rss_mb",
}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_stage1b_artifact(path: str | Path) -> list[str]:
    """Return errors found in a serialized Stage 1B artifact.

    The file is always read from disk.  No caller-supplied in-memory result is
    accepted as evidence of validity.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        return [f"artifact missing: {artifact_path}"]
    if artifact_path.stat().st_size == 0:
        return [f"artifact is zero-byte: {artifact_path}"]
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"artifact invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return ["artifact root must be an object"]

    errors: list[str] = []
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        errors.append("missing meta")
        meta = {}
    errors.extend(f"missing metadata: {key}" for key in sorted(_REQUIRED_META - set(meta)))
    config = meta.get("configuration")
    if not isinstance(config, dict) or not config:
        errors.append("missing configuration")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("missing metrics")
        metrics = {}
    errors.extend(f"missing metric: {key}" for key in sorted(_REQUIRED_METRICS - set(metrics)))
    for key in _REQUIRED_METRICS & set(metrics):
        if not _number(metrics[key]):
            errors.append(f"metric is not finite numeric: {key}")

    denominators = payload.get("metric_denominators")
    if not isinstance(denominators, dict) or not denominators:
        errors.append("missing metric denominators")
    else:
        gold = denominators.get("gold_entities")
        correct = denominators.get("correct_entities")
        predicted = denominators.get("predicted_entities")
        if not all(isinstance(value, int) and value >= 0 for value in (gold, correct, predicted)):
            errors.append("inconsistent metric denominators")
        else:
            if correct > gold or correct > predicted:
                errors.append("inconsistent metric denominators")
            if gold == 0 or predicted == 0:
                errors.append("inconsistent metric denominators")
            if _number(metrics.get("entity_recall")) and gold and abs(metrics["entity_recall"] - correct / gold) > 1e-9:
                errors.append("inconsistent entity recall denominator")
            if _number(metrics.get("entity_precision")) and predicted and abs(metrics["entity_precision"] - correct / predicted) > 1e-9:
                errors.append("inconsistent entity precision denominator")

    gates = payload.get("gates")
    if not isinstance(gates, dict) or not gates:
        errors.append("missing gate results")
    if payload.get("decision") not in {"HONEST PASS", "HONEST FAIL"}:
        errors.append("missing final decision")
    if isinstance(gates, dict) and gates:
        derived = all(bool(value.get("passed")) for value in gates.values() if isinstance(value, dict))
        expected = "HONEST PASS" if derived else "HONEST FAIL"
        if payload.get("decision") != expected:
            errors.append("final decision does not match gate results")
    return errors


def assert_valid_stage1b_artifact(path: str | Path) -> dict[str, Any]:
    errors = validate_stage1b_artifact(path)
    if errors:
        raise ValueError("serialized Stage 1B artifact rejected: " + "; ".join(errors))
    return json.loads(Path(path).read_text(encoding="utf-8"))
