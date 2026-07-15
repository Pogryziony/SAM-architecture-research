"""Regression tests for the pilot-integrity fixes."""
from __future__ import annotations

import json
import hashlib

import pytest


def test_preset_normalizes_lr_and_drops_metadata():
    from stack.encoder.training_presets import apply_preset

    params = apply_preset(
        "smoke",
        model_type="er3",
        cli_overrides={"lr": 0.002},
    )
    assert params["learning_rate"] == 0.002
    assert params["epochs"] == 1
    assert "note" not in params
    assert "lr" not in params


def test_er3_cli_maps_every_effective_parameter():
    from benchmarks.run_er3_training import _training_config_kwargs

    params = {
        "epochs": 1,
        "patience": 2,
        "batch_size": 3,
        "learning_rate": 0.004,
        "weight_decay": 0.01,
        "hard_negative_k": 7,
        "embed_dim": 32,
        "hidden_dim": 64,
        "proj_dim": 16,
        "dropout": 0.2,
    }
    mapped = _training_config_kwargs(params, seed=123)
    assert mapped == {**params, "seed": 123}


def test_realizer_training_overrides_are_applied_and_hashed():
    from benchmarks.train_nexus_realizer import (
        apply_training_overrides,
        effective_config_sha256,
    )

    config = {
        "training": {
            "epochs": 50,
            "early_stopping_patience": 10,
            "batch_size": 16,
            "learning_rate": 0.0001,
        }
    }
    before = effective_config_sha256(config)
    apply_training_overrides(config, {
        "epochs": 3,
        "patience": 2,
        "batch_size": 4,
        "learning_rate": 0.002,
        "decoder_strategy": "greedy",
    })
    assert config["training"] == {
        "epochs": 3,
        "early_stopping_patience": 2,
        "batch_size": 4,
        "learning_rate": 0.002,
    }
    assert effective_config_sha256(config) != before


def test_resolution_result_reports_raw_pool_without_private_attributes():
    from nexus.pipeline.entity_resolver import ResolutionCandidate, ResolutionResult

    result = ResolutionResult(
        selected_entity_ids=["Exp_A"],
        candidates=[
            ResolutionCandidate("Exp_A", 0.9),
            ResolutionCandidate("Exp_B", 0.8),
        ],
        candidate_pool_size=2,
        resolver_name="test",
    )
    assert result.candidate_pool_size == 2
    assert len(result.selected_entity_ids) == 1
    assert result.to_dict()["candidates"][1]["score"] == 0.8


def test_runner_uses_structured_injected_resolver():
    from nexus.graph import Node
    from nexus.graph.store import InMemoryGraphStore
    from nexus.pipeline.config import ProductionNEXUSConfig
    from nexus.pipeline.entity_resolver import ResolutionCandidate, ResolutionResult
    from nexus.pipeline.runner import NEXUSRunner

    class Resolver:
        def resolve(self, question, graph):
            return ResolutionResult(
                selected_entity_ids=["Exp_A"],
                candidates=[
                    ResolutionCandidate("Exp_A", 0.9),
                    ResolutionCandidate("Exp_B", 0.1),
                ],
                candidate_pool_size=2,
                resolver_name="fixture",
                resolver_version="1",
                latency_ms=0.25,
            )

    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Exp_A", type="Experiment", aliases=["alpha"],
        properties={"key_finding": "Alpha worked."},
    ))
    result = NEXUSRunner(
        graph,
        ProductionNEXUSConfig.lexical_only(),
        entity_resolver=Resolver(),
    ).run([{"id": "q1", "question": "What happened?"}])
    qr = result.per_question[0]
    assert qr.selected_entry_nodes == ["Exp_A"]
    assert qr.candidate_pool_size == 2
    assert qr.resolver_name == "fixture"
    assert qr.resolution_candidates[0] == {"entity_id": "Exp_A", "score": 0.9}


def test_er3_enabled_without_injected_resolver_fails_closed():
    from nexus.graph.store import InMemoryGraphStore
    from nexus.pipeline.config import ProductionNEXUSConfig
    from nexus.pipeline.runner import NEXUSRunner

    result = NEXUSRunner(
        InMemoryGraphStore(), ProductionNEXUSConfig.with_entity_ranker_v3()
    ).run([])
    assert any("no EntityResolver was injected" in error for error in result.errors)


def test_stage2_protocol_names_are_unambiguous():
    from benchmarks.run_stage2_stage3 import stage2_protocol_for_limit

    assert stage2_protocol_for_limit(30) == "registered_stage2_v1"
    assert stage2_protocol_for_limit(5) == "smoke_stage2_5"
    with pytest.raises(ValueError):
        stage2_protocol_for_limit(0)


def test_invalid_stage0_artifact_is_registered():
    registry = json.loads(
        open("benchmarks/results/artifact_status.json", encoding="utf-8").read()
    )
    assert registry["artifacts"]["stage0_v2_30.json"]["status"] == "INVALID"


def test_er3_loads_the_exact_verified_external_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from nexus.graph.store import InMemoryGraphStore
    from stack.pipeline.resolver import ER3Resolver
    import stack.encoder.entity_ranker_v3 as ranker_module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    external_weights = tmp_path / "external.pt"
    external_weights.write_bytes(b"verified-external-checkpoint")
    digest = hashlib.sha256(external_weights.read_bytes()).hexdigest()
    (model_dir / "manifest.json").write_text(json.dumps({
        "files": {
            "weights.pt": {
                "sha256": digest,
                "size": external_weights.stat().st_size,
            }
        }
    }), encoding="utf-8")

    captured = {}

    class FakeModel:
        def eval(self):
            return self

    class FakeTokenizer:
        pass

    def fake_load(model_dir_arg, *, weights_path=None):
        captured["model_dir"] = model_dir_arg
        captured["weights_path"] = weights_path
        return FakeModel(), FakeTokenizer(), {}

    monkeypatch.setattr(ranker_module, "load_ranker_v3", fake_load)
    ER3Resolver.from_directory(
        str(model_dir),
        InMemoryGraphStore(),
        weights_path=str(external_weights),
    )
    assert captured["weights_path"] == str(external_weights)
