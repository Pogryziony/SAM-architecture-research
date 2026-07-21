"""Stage 4 rule corpus development gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.eval_rule_engine import evaluate_development, load_corpus, main

_CORPUS = Path("benchmarks/qa-dataset/rule_corpus_v1.json")


def test_rule_corpus_development_meets_prereg_f1():
    report = evaluate_development(load_corpus(_CORPUS))
    assert report["status"] == "PASS", report.get("errors")
    assert report["rule_count"] >= 6
    assert report["metrics"]["f1"] >= 0.90


def test_frozen_rule_eval_is_sealed():
    with pytest.raises(RuntimeError, match="sealed"):
        main(["--mode", "frozen"])
