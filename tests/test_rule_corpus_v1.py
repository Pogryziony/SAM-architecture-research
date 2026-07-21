"""Stage 4 rule corpus development + frozen gates."""

from __future__ import annotations

from pathlib import Path

from benchmarks.eval_rule_engine import (
    FROZEN_FILE_SHA256,
    evaluate_development,
    evaluate_frozen,
    load_corpus,
    sha256_file,
)

_CORPUS = Path("benchmarks/qa-dataset/rule_corpus_v1.json")
_FROZEN = Path("benchmarks/qa-dataset/rule_corpus_v1_frozen.json")


def test_rule_corpus_development_meets_prereg_f1():
    report = evaluate_development(load_corpus(_CORPUS))
    assert report["status"] == "PASS", report.get("errors")
    assert report["rule_count"] >= 12
    assert report["metrics"]["f1"] >= 0.90


def test_frozen_rule_eval_opened_with_published_hash():
    assert sha256_file(_FROZEN) == FROZEN_FILE_SHA256
    report = evaluate_frozen(
        load_corpus(_FROZEN),
        rules_corpus=load_corpus(_CORPUS),
        frozen_path=_FROZEN,
    )
    assert report["status"] == "PASS", report.get("errors")
    assert report["preregistration_id"] == "rule-engine-v2"
    assert report["metrics"]["f1"] >= 0.90
