"""Domain pack interface — mini domain loads without SAM core edits."""

from __future__ import annotations

from nexus.domain import load_domain_pack
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.model_interface import DummyModel


def test_mini_domain_pack_loads_independently():
    pack = load_domain_pack("mini")
    assert pack.meta.domain_id == "mini"
    assert pack.meta.version == "mini-v1"
    graph = pack.build_graph()
    assert graph.node_count == 3
    assert graph.edge_count == 2
    tasks = pack.evaluation_tasks()
    assert len(tasks) == 3
    assert all(t.get("domain") == "mini" for t in tasks)


def test_mini_domain_runs_through_nexus_runner():
    pack = load_domain_pack("mini")
    graph = pack.build_graph()
    config = ProductionNEXUSConfig.lexical_only()
    runner = NEXUSRunner(graph, config, model=DummyModel())
    result = runner.run(pack.evaluation_tasks()[:1])
    assert result.questions_total == 1
    assert result.per_question[0].question_id == "mini_q1"


def test_sam_domain_pack_identity():
    pack = load_domain_pack("sam")
    assert pack.meta.version == "sam-v1"
    prov = pack.provenance()
    assert prov["domain_pack_version"] == "sam-v1"


def test_unknown_domain_pack_fails_closed():
    try:
        load_domain_pack("does-not-exist")
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "unknown domain pack" in str(exc)
