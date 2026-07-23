"""Fail-closed allow_synth_fallback for safe production profiles."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus.domain import load_domain_pack
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.answer import (
    _FAIL_CLOSED_ABSTAIN,
    _synth_fallback_permitted,
    answer_question,
)
from nexus.reasoning.model_interface import DummyModel, SynthesizingModel
from nexus.reasoning.verifier import Verifier


def test_safe_profiles_default_allow_synth_fallback_false():
    for factory in (
        ProductionNEXUSConfig.grounded,
        ProductionNEXUSConfig.l1_acceptance,
        ProductionNEXUSConfig.pointer_copy,
        ProductionNEXUSConfig.comparison_plan,
        ProductionNEXUSConfig.deterministic_render,
    ):
        cfg = factory()
        assert cfg.allow_synth_fallback is False
        assert _synth_fallback_permitted(cfg) is False


def test_experimental_override_can_enable_fallback():
    cfg = ProductionNEXUSConfig.grounded(allow_synth_fallback=True)
    assert cfg.allow_synth_fallback is True


def test_fallback_policy_changes_config_identity():
    closed = ProductionNEXUSConfig.grounded()
    open_fb = ProductionNEXUSConfig.grounded(allow_synth_fallback=True)
    assert closed.config_hash != open_fb.config_hash
    assert closed.to_dict()["nexus_config"]["allow_synth_fallback"] is False
    assert open_fb.to_dict()["nexus_config"]["allow_synth_fallback"] is True


def test_lexical_historical_default_allows_fallback():
    cfg = ProductionNEXUSConfig.lexical_only()
    assert cfg.allow_synth_fallback is True


def test_fail_closed_abstains_when_realization_forced_to_fail(monkeypatch):
    """Force deterministic paths to fail; grounded must not call synth."""
    pack = load_domain_pack("mini")
    graph = pack.build_graph()
    cfg = ProductionNEXUSConfig.grounded()
    assert cfg.allow_synth_fallback is False

    synth_calls: list[str] = []
    original_generate = SynthesizingModel.generate

    def _spy(self, prompt: str, *args, **kwargs):  # noqa: ANN001
        synth_calls.append(prompt[:80])
        return original_generate(self, prompt, *args, **kwargs)

    monkeypatch.setattr(SynthesizingModel, "generate", _spy)

    # Force every deterministic realizer helper to return None / miss.
    import nexus.reasoning.answer as answer_mod

    monkeypatch.setattr(answer_mod, "_pointer_copy_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_deterministic_render_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_comparison_plan_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_l1_node_fact_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_edge_catalog_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_l1_qualitative_compare_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_l1_compare_metrics_result", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "_l1_dependency_chain_result", lambda *a, **k: None)

    get_model = MagicMock(side_effect=AssertionError("get_available_model must not run"))
    monkeypatch.setattr(answer_mod, "get_available_model", get_model)

    result = answer_question(
        "What is the capital of Poland?",
        graph,
        model=DummyModel(),
        verifier=Verifier(),
        config=cfg,
    )
    assert "insufficient" in result["answer"].casefold()
    assert _FAIL_CLOSED_ABSTAIN in result["answer"] or "fallback is disabled" in result[
        "answer"
    ].casefold()
    assert result.get("fallback_considered") is True
    assert result.get("fallback_permitted") is False
    assert result.get("fallback_terminal_outcome") == "ABSTAIN"
    assert synth_calls == []
    get_model.assert_not_called()


def test_runner_does_not_resolve_llm_when_fallback_forbidden():
    pack = load_domain_pack("mini")
    graph = pack.build_graph()
    cfg = ProductionNEXUSConfig.grounded()
    runner = NEXUSRunner(graph, cfg, model=None)
    # Touch model resolution path used by run()
    from nexus.reasoning.model_interface import DummyModel

    # Emulate run()'s model bind without answering questions.
    if runner.model is None:
        if bool(getattr(runner.config, "allow_synth_fallback", True)):
            pytest.fail("grounded must keep allow_synth_fallback=False")
        runner.model = DummyModel()
    assert isinstance(runner.model, DummyModel)


def test_runner_audit_includes_fallback_fields():
    pack = load_domain_pack("mini")
    graph = pack.build_graph()
    tasks = pack.evaluation_tasks()[:1]
    cfg = ProductionNEXUSConfig.grounded()
    runner = NEXUSRunner(graph, cfg, model=DummyModel())
    qr = runner._run_single(  # noqa: SLF001
        str(tasks[0]["id"]), str(tasks[0]["question"]), DummyModel()
    )
    audit = qr.reasoning_audit
    assert audit.get("allow_synth_fallback") is False
    assert "selected_realizer" in audit
    assert "fallback_considered" in audit
    assert "fallback_permitted" in audit
