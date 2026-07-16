"""Regression tests for the pilot-integrity fixes."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

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


def test_offline_lexical_rag_is_deterministic_and_question_conditioned():
    from benchmarks.rag_baseline import LexicalRAGEmbedder, cosine_similarity

    embedder = LexicalRAGEmbedder(dimensions=256)
    query = embedder.embed_query("oracle memory accuracy")
    matching = embedder.embed_query("oracle memory reached high accuracy")
    unrelated = embedder.embed_query("dialogue pronoun resolution latency")
    assert query == embedder.embed_query("oracle memory accuracy")
    assert cosine_similarity(query, matching) > cosine_similarity(query, unrelated)


def test_synthesizer_can_answer_from_rag_document_excerpts():
    from nexus.reasoning.model_interface import SynthesizingModel

    answer = SynthesizingModel().generate(
        "QUESTION: What accuracy did oracle memory reach?\n"
        "DOCUMENT EXCERPTS:\n"
        "[1] report.md:\n"
        "The oracle memory experiment reached 99.87% overall accuracy.\n"
        "Sources: 1 document excerpt(s)\nANSWER:"
    )
    assert "99.87%" in answer


def test_repository_weight_output_is_restricted_to_configured_model_root():
    from benchmarks.train_nexus_realizer import _assert_external_output

    _assert_external_output(
        Path("models/realizer/pilot-v1"),
        allow_in_repository=True,
        repository_output_root="models/realizer",
    )
    with pytest.raises(ValueError, match="must be under"):
        _assert_external_output(
            Path("training/output"),
            allow_in_repository=True,
            repository_output_root="models/realizer",
        )


def test_registered_er3_checkpoint_is_committed_and_manifest_verified():
    model_dir = Path("models/encoder/entity_ranker_v3_20260711T081545Z")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    weights = model_dir / "weights.pt"
    assert weights.is_file()
    assert weights.stat().st_size == manifest["files"]["weights.pt"]["size"]
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == (
        manifest["files"]["weights.pt"]["sha256"]
    )


def test_stage2_canonical_hash_excludes_runtime_hash_seed():
    from benchmarks.run_stage2_stage3 import _canonical_stage2_payload

    base = {
        "protocol": "registered_stage2_v1",
        "python_hash_seed": "0",
        "per_question": [],
    }
    changed = dict(base, python_hash_seed="42")
    assert _canonical_stage2_payload(base) == _canonical_stage2_payload(changed)


def test_phase4_contract_requires_every_gate(monkeypatch, tmp_path):
    import benchmarks.check_phase4_readiness as phase4

    monkeypatch.setattr(phase4, "validate_dataset_manifest", lambda *_: [])
    monkeypatch.setattr(phase4, "validate_oracle_artifact", lambda *_: [])
    monkeypatch.setattr(phase4, "validate_readiness_for_training", lambda *_: [])
    monkeypatch.setattr(
        phase4, "_verify_er3_bundle", lambda *_: (True, {"hashes": {}})
    )
    dataset_sha = "dataset-sha"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    common_sha = "a" * 40
    stage2_runs = [
        {
            "source_sha": common_sha,
            "python_hash_seed": seed,
            "canonical_content_sha256": "canonical",
            "protocol": "registered_stage2_v1",
            "registered_gate_status": "PASS",
            "status": "PASS",
            "questions_total": 30,
        }
        for seed in ("0", "1", "42")
    ]
    result = phase4.evaluate_phase4(
        config={
            "training": {"epochs": 5, "early_stopping_patience": 3},
        },
        config_path=config_path,
        dataset_manifest={
            "pairs_accepted": 7127,
            "target_met": True,
            "dataset_sha256": dataset_sha,
        },
        dataset_manifest_path=manifest_path,
        oracle={"source_sha": common_sha},
        stage0={
            "source_sha": common_sha,
            "status": "VALID",
            "publication_guard": {"status": "PASS"},
            "questions_total": 30,
            "nexus": {"answered": 25},
            "rag": {"answered": 25},
            "paired_comparison": {"paired_n": 25},
        },
        stage2_runs=stage2_runs,
        stage3={
            "source_sha": common_sha,
            "status": "PASS",
            "total_turns": 110,
            "metrics": {
                "reference_resolution": 0.875,
                "single_turn_regression": 0.0,
                "dialogue_state_latency_p50_ms": 0.04,
            },
        },
        readiness={"status": "READY_FOR_TRAINING", "blocking_checks": []},
        preflight={
            "status": "PREFLIGHT_PASS",
            "weights_written": False,
            "dataset_sha256": dataset_sha,
            "config_sha256": config_sha,
            "parameter_count": 2_770_752,
        },
        overfit_smoke={
            "status": "OVERFIT_PASS",
            "weights_written": False,
            "dataset_sha256": dataset_sha,
            "config_sha256": config_sha,
            "initial_loss": 190.0,
            "final_loss": 140.0,
        },
        er3_dir=tmp_path,
    )
    assert result["status"] == "GO_FOR_REALIZER_TRAINING"
    stage2_runs[0]["status"] = "FAIL"
    blocked = phase4.evaluate_phase4(
        config={"training": {"epochs": 5, "early_stopping_patience": 3}},
        config_path=config_path,
        dataset_manifest={"pairs_accepted": 7127, "target_met": True, "dataset_sha256": dataset_sha},
        dataset_manifest_path=manifest_path,
        oracle={"source_sha": common_sha},
        stage0={"source_sha": common_sha, "status": "VALID", "publication_guard": {"status": "PASS"}, "questions_total": 30, "nexus": {"answered": 25}, "rag": {"answered": 25}, "paired_comparison": {"paired_n": 25}},
        stage2_runs=stage2_runs,
        stage3={"source_sha": common_sha, "status": "PASS", "total_turns": 110, "metrics": {"reference_resolution": 0.875, "single_turn_regression": 0.0, "dialogue_state_latency_p50_ms": 0.04}},
        readiness={"status": "READY_FOR_TRAINING", "blocking_checks": []},
        preflight={"status": "PREFLIGHT_PASS", "weights_written": False, "dataset_sha256": dataset_sha, "config_sha256": config_sha, "parameter_count": 2_770_752},
        overfit_smoke={"status": "OVERFIT_PASS", "weights_written": False, "dataset_sha256": dataset_sha, "config_sha256": config_sha, "initial_loss": 190.0, "final_loss": 140.0},
        er3_dir=tmp_path,
    )
    assert blocked["status"] == "BLOCKED"
    assert "stage2" in blocked["blocking_checks"]
