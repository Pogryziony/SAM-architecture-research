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
    else:
        threshold = meta.get("selected_threshold")
        if not _number(threshold) or not 0.0 <= threshold <= 1.0:
            errors.append("inconsistent selected threshold")
        if _number(config.get("entity_threshold")) and _number(threshold):
            if abs(config["entity_threshold"] - threshold) > 1e-12:
                errors.append("metadata/configuration threshold mismatch")
        cap = meta.get("selected_parser_handoff_cap")
        configured_cap = config.get("max_entry_nodes")
        if cap is not None:
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                errors.append("invalid parser handoff cap")
            if configured_cap != cap:
                errors.append("parser handoff cap metadata/configuration mismatch")
        elif configured_cap is not None:
            errors.append("missing selected parser handoff cap")
    if not isinstance(meta.get("model_checkpoint"), str) or not meta.get("model_checkpoint"):
        errors.append("missing model/checkpoint identifier")
    if not isinstance(meta.get("calibration_split"), str) or not meta.get("calibration_split"):
        errors.append("missing calibration split identifier")
    if not isinstance(meta.get("frozen_split"), str) or not meta.get("frozen_split"):
        errors.append("missing frozen split identifier")
    if not isinstance(meta.get("calibration_sample_count"), int) or meta.get("calibration_sample_count") <= 0:
        errors.append("inconsistent calibration sample count")
    if not isinstance(meta.get("question_count"), int) or meta.get("question_count") <= 0:
        errors.append("inconsistent question count")
    curve = meta.get("threshold_search_metrics")
    if not isinstance(curve, list) or not curve:
        errors.append("missing threshold search metrics")
    elif _number(meta.get("selected_threshold")) and not any(
        isinstance(row, dict) and _number(row.get("threshold"))
        and abs(row["threshold"] - meta["selected_threshold"]) <= 1e-12
        for row in curve
    ):
        errors.append("selected threshold absent from threshold search metrics")

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
        required_gates = {"entity_recall", "resolution_rate", "paraphrase_drop", "intent_accuracy", "rss_delta", "inference_p50"}
        missing_gates = sorted(required_gates - set(gates))
        if missing_gates:
            errors.append(f"missing gate results: {missing_gates}")
        derived = all(bool(value.get("passed")) for value in gates.values() if isinstance(value, dict))
        expected = "HONEST PASS" if derived else "HONEST FAIL"
        if payload.get("decision") != expected:
            errors.append("final decision does not match gate results")
        if payload.get("all_pass") is not derived:
            errors.append("all_pass does not match gate results")

    details = payload.get("question_details")
    if not isinstance(details, list) or len(details) != meta.get("question_count"):
        errors.append("question details/count mismatch")
    if meta.get("validated_ids_match") is not True:
        errors.append("frozen split IDs were not validated")
    pipeline = payload.get("pipeline_with_fallback")
    if isinstance(pipeline, dict) and isinstance(metrics, dict):
        aliases = {
            "entity_precision": "entity_precision",
            "entity_recall": "entity_recall",
            "entity_f1": "entity_f1",
            "exact_entity_accuracy": "exact_accuracy",
        }
        for pipeline_key, metric_key in aliases.items():
            if _number(pipeline.get(pipeline_key)) and _number(metrics.get(metric_key)) and abs(pipeline[pipeline_key] - metrics[metric_key]) > 1e-12:
                errors.append(f"metric mismatch: {metric_key}")
    latency = payload.get("latency")
    encoder_info = payload.get("encoder_info")
    if isinstance(latency, dict) and _number(metrics.get("latency_ms")) and _number(latency.get("inference_p50_ms")) and abs(metrics["latency_ms"] - latency["inference_p50_ms"]) > 1e-12:
        errors.append("latency metric mismatch")
    if isinstance(encoder_info, dict) and _number(metrics.get("rss_mb")) and _number(encoder_info.get("rss_delta_mb")) and abs(metrics["rss_mb"] - encoder_info["rss_delta_mb"]) > 1e-12:
        errors.append("RSS metric mismatch")
    return errors


def assert_valid_stage1b_artifact(path: str | Path) -> dict[str, Any]:
    errors = validate_stage1b_artifact(path)
    if errors:
        raise ValueError("serialized Stage 1B artifact rejected: " + "; ".join(errors))
    return json.loads(Path(path).read_text(encoding="utf-8"))
