"""Stage 5 contradiction F1 / calibration campaign."""

from __future__ import annotations

from pathlib import Path

from benchmarks.eval_contradiction_policy import evaluate, _read_jsonl

_GOLD = Path("benchmarks/qa-dataset/contradiction_gold_v1.jsonl")


def test_contradiction_campaign_meets_prereg_gates():
    report = evaluate(_read_jsonl(_GOLD))
    assert report["status"] == "PASS", report.get("errors")
    assert report["classification"]["macro_f1"] >= 0.90
    assert report["policy"]["unconditional_leaks"] == 0
    assert report["calibration"]["n"] >= 4
