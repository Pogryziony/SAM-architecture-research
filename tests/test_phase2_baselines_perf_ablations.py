"""Phase 2 baselines, performance, ablations, robustness."""

from __future__ import annotations

import json
from pathlib import Path

from nexus.baselines.adapters import run_baseline_eval
from nexus.domain import load_domain_pack
from nexus.evaluation.ablations import (
    ROBUSTNESS_TRANSFORMS,
    apply_ablation,
    list_ablations,
    transform_question,
)
from nexus.evaluation.performance import measure_grounded_e2e
from nexus.evaluation.validate import validate_result_artifact
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.model_interface import DummyModel


def test_baseline_closed_book_emits_not_run_without_credentials(monkeypatch):
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    pack = load_domain_pack("mini")
    artifact = run_baseline_eval(
        "closed_book_llm",
        pack.evaluation_tasks()[:1],
        dataset_id="mini-tasks",
        comparison_mode="controlled",
        source_commit="test",
    )
    assert artifact["status"] == "NOT_RUN"
    assert validate_result_artifact(artifact) == []
    assert artifact["per_question"][0]["terminal_outcome"] == "not_run"
    assert artifact["arm_metadata"]["is_placeholder"] is False


def test_placeholder_arm_cannot_be_modern_rag_or_llm():
    pack = load_domain_pack("mini")
    artifact = run_baseline_eval(
        "placeholder_synthesizing_model",
        pack.evaluation_tasks()[:1],
        dataset_id="mini-tasks",
        source_commit="test",
    )
    assert artifact["status"] == "NOT_RUN"
    assert artifact["arm_metadata"]["is_placeholder"] is True
    assert artifact["arm_metadata"]["modern_rag"] is False


def test_performance_grounded_mini_records_raw_samples(tmp_path: Path):
    pack = load_domain_pack("mini")
    runner = NEXUSRunner(
        pack.build_graph(),
        ProductionNEXUSConfig.grounded(),
        model=DummyModel(),
    )
    artifact = measure_grounded_e2e(
        runner,
        pack.evaluation_tasks()[:2],
        warmup=1,
        repeats=2,
        profile_name="grounded",
    )
    assert artifact["schema_version"] == "nexus-performance-v1"
    assert artifact["profile"] == "grounded"
    assert artifact["config_hash"] == runner.config.config_hash
    assert len(artifact["samples"]) >= 3
    assert "warm_p50_ms" in artifact["summary"]
    assert artifact["budgets"]["latency_p50_gate"] in {"PASS", "FAIL", "NOT_RUN"}
    assert artifact["budgets"]["peak_rss_gate"] in {"PASS", "FAIL", "NOT_RUN"}
    out = tmp_path / "perf.json"
    out.write_text(json.dumps(artifact), encoding="utf-8")
    assert out.exists()


def test_ablations_change_config_identity():
    base = ProductionNEXUSConfig.grounded()
    names = list_ablations()
    assert "no_er3" in names
    assert "no_multi_hop" in names
    changed = apply_ablation("no_multi_hop", base)
    assert changed.max_depth == 1
    assert changed.config_hash != base.config_hash


def test_robustness_transforms_are_seeded_and_linked():
    record = {
        "id": "mini_q1",
        "question": "What is the capital of Poland?",
        "domain": "mini",
    }
    for name in ROBUSTNESS_TRANSFORMS:
        a = transform_question(record, name, seed=7)
        b = transform_question(record, name, seed=7)
        assert a["question"] == b["question"]
        assert a["parent_question_id"] == "mini_q1"
        assert a["id"].startswith("mini_q1__rob_")
        assert a["robustness_transform"] == name
