from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.stage1b_artifact import validate_stage1b_artifact
from stack.encoder.eval_gates import make_encoder_eval_config


def test_eval_config_propagates_parser_handoff_cap():
    config = make_encoder_eval_config(30)
    assert config.max_entry_nodes == 30
    assert config.enable_associative_encoder is True


def test_artifact_rejects_parser_cap_metadata_configuration_mismatch(tmp_path):
    payload = {
        "meta": {
            "evaluation_commit_sha": "abc",
            "model_checkpoint": "models/encoder_v2",
            "calibration_split": "stack/encoder/data/val.jsonl",
            "calibration_sample_count": 1,
            "selected_threshold": 0.1,
            "threshold_search_metrics": [{"threshold": 0.1}],
            "frozen_split": "stack/encoder/data/test.jsonl",
            "question_count": 1,
            "validated_ids_match": True,
            "selected_parser_handoff_cap": 30,
            "configuration": {"entity_threshold": 0.1, "max_entry_nodes": 5},
        },
        "metrics": {
            "entity_precision": 1.0,
            "entity_recall": 1.0,
            "entity_f1": 1.0,
            "exact_accuracy": 1.0,
            "candidate_pool_recall": 1.0,
            "reranker_recall": 1.0,
            "parser_failures": 0,
            "latency_ms": 1.0,
            "rss_mb": 1.0,
        },
        "metric_denominators": {"gold_entities": 1, "correct_entities": 1, "predicted_entities": 1},
        "question_details": [{}],
        "gates": {name: {"passed": True} for name in (
            "entity_recall", "resolution_rate", "paraphrase_drop",
            "intent_accuracy", "rss_delta", "inference_p50",
        )},
        "decision": "HONEST PASS",
        "all_pass": True,
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    errors = validate_stage1b_artifact(path)
    assert any("parser handoff cap" in error for error in errors)
