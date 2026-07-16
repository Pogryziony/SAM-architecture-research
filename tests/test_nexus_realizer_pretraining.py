"""Pre-training safety, reproducibility, and leakage regression tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.acquire_realizer_train_data import (
    acquire_claim_records,
    discover_sources,
    load_verified_acquisition,
    write_acquisition,
)
from benchmarks.build_distillation_dataset import build_distillation_dataset
from benchmarks.check_realizer_readiness import evaluate_readiness, estimate_parameter_count
from benchmarks.realizer_contracts import (
    assert_no_split_leakage,
    normalize_question,
    split_by_entity_family,
    stable_example_id,
    validate_dataset_manifest,
    validate_distillation_record,
)
from benchmarks.run_nexus_oracle import (
    _path_recall,
    build_oracle_records,
    token_f1,
    validate_oracle_artifact,
    validate_oracle_records,
)
from benchmarks.run_stage2_stage3 import run_stage2
from benchmarks.train_nexus_realizer import (
    _assert_external_output, load_training_inputs, serialize_source,
    serialization_coverage, validate_readiness_for_training,
)
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.realizer.model import validate_model_config
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.reasoning.answer import answer_question


def _safe_record(record_id: str, question: str, entities: list[str]) -> dict:
    normalized = normalize_question(question)
    import hashlib
    return {
        "id": stable_example_id(question, entities),
        "question": question,
        "answer": "Alpha achieved 90% accuracy.",
        "source_sha": "a" * 40,
        "config_hash": "config",
        "source_split": "train",
        "canonical_entities": entities,
        "normalized_question_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "evidence_pack": {"node_facts": [{"text": "Alpha achieved 90% accuracy."}]},
        "target_verification": {"passed": True},
        "reasoning_audit": {
            "proof_valid": True,
            "recommended_action": "answer",
            "provenance_coverage": 1.0,
        },
        "source_question_id": record_id,
    }


def test_normalization_and_stable_id_ignore_punctuation_and_entity_order():
    assert normalize_question("  What's Alpha? ") == normalize_question("WHAT’S alpha")
    assert stable_example_id("Alpha?", ["B", "A"]) == stable_example_id("alpha", ["A", "B"])


def test_train_only_acquisition_is_large_unique_and_excludes_evaluation_sources(tmp_path: Path):
    root = Path.cwd()
    sources = discover_sources(root)
    relative = [path.relative_to(root).as_posix() for path in sources]
    assert relative
    assert not any(path.startswith("benchmarks/") for path in relative)
    assert not any(Path(path).name.casefold() in {"test.jsonl", "val.jsonl", "validation.jsonl"} for path in relative)
    assert not any("/results/" in f"/{path}/" for path in relative)

    records, manifest = acquire_claim_records(root)
    assert len(records) >= 5000
    assert manifest["semantic_targets_unique"] == len(records)
    assert manifest["normalized_questions_unique"] == len(records)
    assert manifest["normalized_answers_unique"] == len(records)
    assert manifest["source_families"] >= 100
    assert {"config_value", "table_cell", "markdown_claim", "api_contract"} <= set(manifest["counts_by_kind"])

    output = tmp_path / "acquisition"
    write_acquisition(output, records, manifest)
    loaded, loaded_manifest = load_verified_acquisition(output / "manifest.json", root)
    assert loaded == records
    assert loaded_manifest["records_sha256"] == manifest["records_sha256"]
    assert loaded_manifest["records_file_sha256"] != ""

    records_path = output / "source_claims.jsonl.xz"
    records_path.write_bytes(records_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_acquisition(output / "manifest.json", root)


def test_distillation_contract_rejects_legacy_and_tampered_records():
    legacy = {"question": "What happened?", "answer": "Something happened."}
    errors = validate_distillation_record(legacy)
    assert "no_evidence_pack" in errors
    assert "invalid_proof" in errors
    record = _safe_record("q1", "What accuracy did Alpha achieve?", ["Alpha"])
    assert validate_distillation_record(record) == []
    record["reasoning_audit"]["provenance_coverage"] = 0.5
    assert "incomplete_provenance_for_answer" in validate_distillation_record(record)


def test_entity_family_split_is_transitive_and_leakage_free():
    records = [
        _safe_record("q1", "Question one about Alpha?", ["Alpha"]),
        _safe_record("q2", "Question two joins Alpha Beta?", ["Alpha", "Beta"]),
        _safe_record("q3", "Question three about Beta?", ["Beta"]),
        _safe_record("q4", "Question four about Gamma?", ["Gamma"]),
    ]
    splits = split_by_entity_family(records, validation_fraction=0.25, seed=7)
    assert_no_split_leakage(splits)
    locations = {
        entity: split
        for split, rows in splits.items()
        for row in rows
        for entity in row["canonical_entities"]
    }
    assert locations["Alpha"] == locations["Beta"]
    assert locations["Gamma"] != locations["Alpha"]


def test_explicit_cross_split_entity_leakage_is_rejected():
    left = _safe_record("q1", "First Alpha question?", ["Alpha"])
    right = _safe_record("q2", "Second Alpha question?", ["Alpha"])
    with pytest.raises(ValueError, match="entity leakage"):
        assert_no_split_leakage({"train": [left], "validation": [right]})


def _training_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"):
        graph.add_node(Node(
            name, "Concept",
            properties={"description": f"{name} achieved 90% accuracy."},
            sources=[f"docs/{name}.md"],
        ))
    graph.add_edge(Edge("related_to", "Alpha", "Beta", evidence="docs/alpha-beta.md"))
    graph.add_edge(Edge("related_to", "Gamma", "Delta", evidence="docs/gamma-delta.md"))
    graph.add_edge(Edge("related_to", "Epsilon", "Zeta", evidence="docs/epsilon-zeta.md"))
    graph.add_edge(Edge("related_to", "Eta", "Theta", evidence="docs/eta-theta.md"))
    graph.add_edge(Edge("related_to", "Iota", "Kappa", evidence="docs/iota-kappa.md"))
    return graph


def test_builder_writes_full_evidence_grouped_splits_and_hash_manifest(tmp_path: Path):
    questions = [
        {"id": "q1", "question": "What accuracy did Alpha achieve?", "answer": "Alpha achieved 90% accuracy.", "entities": ["Alpha"], "source_split": "train"},
        {"id": "q2", "question": "What accuracy did Gamma achieve?", "answer": "Gamma achieved 90% accuracy.", "entities": ["Gamma"], "source_split": "train"},
    ]
    output = tmp_path / "dataset"
    manifest = build_distillation_dataset(
        questions, _training_graph(), str(output), "a" * 40,
        min_pairs=2, source_path="",
    )
    assert manifest["target_met"] is True
    assert validate_dataset_manifest(manifest, output) == []
    records = [
        json.loads(line)
        for split in ("train", "validation")
        for line in (output / f"{split}.jsonl").read_text().splitlines()
    ]
    assert all(record["evidence_pack"]["paths"] for record in records)
    assert all(record["reasoning_audit"]["proof_valid"] for record in records)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_distillation_dataset(questions, _training_graph(), str(output), "b" * 40, min_pairs=2)

    (output / "train.jsonl").write_text("{}\n", encoding="utf-8")
    errors = validate_dataset_manifest(manifest, output)
    assert "train sha256 mismatch" in errors


def test_oracle_contract_includes_exact_relation_paths_and_negative_abstention():
    questions = [{"id": "v1", "question": "What is Alpha?", "answer": "Alpha is a concept.", "entities": ["Alpha"], "category": "factual"}]
    relations = [
        {"source": "Alpha", "target": "Beta", "edge_type": "depends_on", "evidence": "x"},
        {"source": "Alpha", "target": "Gamma", "edge_type": "depends_on", "evidence": "negative", "is_negative": True},
    ]
    records = build_oracle_records(questions, relations)
    assert validate_oracle_records(records) == []
    positive = next(row for row in records if row["category"] == "relation")
    negative = next(row for row in records if row["category"] == "negative_relation")
    assert positive["path_required"] is True and positive["gold_path"]
    assert negative["should_abstain"] is True and not negative["gold_path"]


def test_oracle_path_recall_honors_reverse_traversal():
    gold = [{"source": "A", "relation": "depends_on", "target": "B"}]
    audit = {"proof_steps": [{"from_node": "B", "relation": "depends_on", "to_node": "A", "reversed": True}]}
    assert _path_recall(gold, audit) == 1.0
    assert token_f1("Alpha achieved 90 percent", "Alpha reached 90 percent") > 0.5


def test_absent_explicit_relation_fails_closed_instead_of_answering_endpoint_fact():
    graph = _training_graph()
    result = answer_question(
        "Does Alpha have the depends_on relation to Gamma?",
        graph,
        entry_nodes_override=["Alpha", "Gamma"],
    )
    assert "insufficient evidence" in result["answer"].casefold()
    assert result["reasoning_audit"]["recommended_action"] == "abstain"


def _valid_oracle_artifact() -> dict:
    row = {
        "question_id": "q",
        "reasoning_audit": {"proof_steps": [{"from_node": "A", "to_node": "B"}]},
    }
    return {
        "schema_version": "nexus-oracle-v1",
        "evaluation_mode": "oracle",
        "source_sha": "a",
        "config_hash": "b",
        "questions_total": 150,
        "dataset": {"sha256": "c", "sources": {"val": "d"}},
        "metrics": {
            "fact_accuracy_mean": 0.8,
            "token_f1_mean": 0.8,
            "gold_path_recall_mean": 0.9,
            "gold_entity_coverage_mean": 0.9,
            "proof_valid_rate": 0.98,
            "provenance_coverage_mean": 0.95,
            "abstention_precision": 0.9,
            "abstention_recall": 0.9,
            "abstention_f1": 0.9,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
            "latency_p99_ms": 3.0,
        },
        "per_question": [dict(row, question_id=f"q{i}") for i in range(150)],
        "errors": [],
    }


def test_oracle_publication_guard_fails_closed():
    artifact = _valid_oracle_artifact()
    assert validate_oracle_artifact(artifact) == []
    artifact["evaluation_mode"] = "predicted"
    artifact["per_question"].pop()
    errors = validate_oracle_artifact(artifact)
    assert "evaluation_mode must be oracle" in errors
    assert "per_question count mismatch" in errors


def test_readiness_aggregates_all_gates_and_blocks_runtime_or_data(tmp_path: Path):
    questions = [
        {"id": "q1", "question": "What accuracy did Alpha achieve?", "answer": "Alpha achieved 90% accuracy.", "entities": ["Alpha"], "source_split": "train"},
        {"id": "q2", "question": "What accuracy did Gamma achieve?", "answer": "Gamma achieved 90% accuracy.", "entities": ["Gamma"], "source_split": "train"},
        {"id": "q3", "question": "What accuracy did Epsilon achieve?", "answer": "Epsilon achieved 90% accuracy.", "entities": ["Epsilon"], "source_split": "train"},
        {"id": "q4", "question": "What accuracy did Eta achieve?", "answer": "Eta achieved 90% accuracy.", "entities": ["Eta"], "source_split": "train"},
        {"id": "q5", "question": "What accuracy did Iota achieve?", "answer": "Iota achieved 90% accuracy.", "entities": ["Iota"], "source_split": "train"},
    ]
    root = tmp_path / "dataset"
    manifest = build_distillation_dataset(questions, _training_graph(), str(root), "a" * 40, min_pairs=5)
    config = json.loads(Path("training/nexus_realizer_v1.json").read_text())
    config["data"]["minimum_pairs"] = 5
    stage2 = {
        "schema_version": "nexus-stage2-v1", "source_sha": "a", "source_tree_sha": "t", "config_hash": "b",
        "registered_baseline_sha256": "c", "questions_total": 30,
        "question_set_sha256": "d", "canonical_content_sha256": "e",
        "protocol": "registered_stage2_v1", "protocol_kind": "registered",
        "registered_gate_status": "PASS",
        "case_order": [f"q{i}" for i in range(30)],
        "per_question": [{"evidence": {}} for _ in range(30)], "status": "PASS",
        "metrics": {"relevance_rate": 0.8, "naturalness_improvement": 6.0, "hallucination_delta_vs_baseline": 0.0, "accuracy_delta_vs_baseline": -0.01},
    }
    result = evaluate_readiness(config, manifest, root, _valid_oracle_artifact(), stage2, torch_available=True)
    assert result["status"] == "READY_FOR_TRAINING"
    blocked = evaluate_readiness(config, dict(manifest, target_met=False), root, _valid_oracle_artifact(), stage2, torch_available=False)
    assert blocked["status"] == "BLOCKED"
    assert {"dataset_target_met", "pytorch_runtime"} <= set(blocked["blocking_checks"])


def test_training_config_budget_tokenizer_and_output_policy(tmp_path: Path):
    config = json.loads(Path("training/nexus_realizer_v1.json").read_text())
    assert validate_model_config(config["model"]) == []
    assert estimate_parameter_count(config["model"]) < 50_000_000
    tokenizer = ByteTokenizer()
    text = "Zażółć gęślą"
    assert tokenizer.decode(tokenizer.encode(text, 100)) == text
    with pytest.raises(ValueError, match="outside"):
        _assert_external_output(Path("training/output"))
    _assert_external_output(tmp_path.parent / "external-model-output")
    record = _safe_record("q1", "What accuracy did Alpha achieve?", ["Alpha"])
    compact = serialize_source(record, 128)
    assert len(compact.encode("utf-8")) <= 128
    assert 0.0 <= serialization_coverage(record, 128) <= 1.0


def test_training_loader_detects_dataset_tampering(tmp_path: Path):
    questions = [
        {"id": "q1", "question": "What accuracy did Alpha achieve?", "answer": "Alpha achieved 90% accuracy.", "entities": ["Alpha"], "source_split": "train"},
        {"id": "q2", "question": "What accuracy did Gamma achieve?", "answer": "Gamma achieved 90% accuracy.", "entities": ["Gamma"], "source_split": "train"},
    ]
    root = tmp_path / "dataset"
    build_distillation_dataset(questions, _training_graph(), str(root), "a" * 40, min_pairs=2)
    config_path = Path("training/nexus_realizer_v1.json")
    _, _, splits = load_training_inputs(root / "manifest.json", config_path)
    assert len(splits["train"]) + len(splits["validation"]) == 2
    (root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_training_inputs(root / "manifest.json", config_path)


def test_stage2_uses_scalar_judge_scores_and_registered_deltas(tmp_path: Path):
    output_path = tmp_path / "stage2.json"
    artifact = run_stage2(
        [{
            "id": "q1",
            "question": "What accuracy did Alpha achieve?",
            "answer": "Alpha achieved 90% accuracy.",
            "question_type": "factual",
        }],
        _training_graph(),
        ProductionNEXUSConfig.lexical_only(),
        "a" * 40,
        str(output_path),
        {"schema_version": "nexus-stage2-baseline-v1", "naturalness_mean": 35.0, "accuracy_mean": 0.1, "hallucination_mean": 0.5},
        protocol="smoke_test_stage2",
    )
    assert isinstance(artifact["metrics"]["naturalness_mean"], float)
    assert isinstance(artifact["metrics"]["relevance_rate"], float)
    assert "accuracy_delta_vs_baseline" in artifact["metrics"]
    assert artifact["registered_baseline_sha256"]
    assert artifact["case_order"] == ["q1"]
    assert artifact["per_question"][0]["evidence"]
    assert artifact["canonical_content_sha256"]
    assert artifact["serialized_file_sha256"]
    assert artifact["protocol_kind"] == "smoke_or_adhoc"
    assert artifact["registered_gate_status"] == "NOT_APPLICABLE"
    assert artifact["status"].startswith("SMOKE_")
    import hashlib
    actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert actual == artifact["serialized_file_sha256"]
    sidecar = output_path.with_suffix(".json.sha256")
    assert sidecar.read_text(encoding="ascii").split()[0] == actual


def test_registered_stage2_relevance_gate_passes_on_canonical_graph(tmp_path: Path):
    from benchmarks.run_benchmark import build_benchmark_graph

    questions = [
        json.loads(line)
        for line in Path("benchmarks/qa-dataset/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ][:30]
    baseline = json.loads(Path("training/stage2_baseline_v1.json").read_text(encoding="utf-8"))
    graph, _ = build_benchmark_graph()
    artifact = run_stage2(
        questions,
        graph,
        ProductionNEXUSConfig.lexical_only(),
        "a" * 40,
        str(tmp_path / "registered-stage2.json"),
        baseline,
    )
    assert artifact["metrics"]["relevance_rate"] >= 0.77
    assert artifact["status"] == "PASS"


def test_training_rejects_forged_ready_status(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")
    checks = [{"name": "data", "passed": False}]
    import hashlib
    readiness = {
        "schema_version": "nexus-realizer-readiness-v1",
        "status": "READY_FOR_TRAINING",
        "blocking_checks": [],
        "checks": checks,
        "inputs": {
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "readiness_sha256": hashlib.sha256(
            json.dumps(checks, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    errors = validate_readiness_for_training(readiness, manifest_path, config_path)
    assert "readiness contains failed checks" in errors
    assert "readiness status is inconsistent with checks" in errors
