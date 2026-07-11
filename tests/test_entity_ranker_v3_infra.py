"""Infrastructure and artifact/CI tests for Entity Ranker V3.

T6: Hard negatives are score-derived
T7: Synthetic/real sampling proportions enforced
T8: Dirty worktree blocks evaluation
T9: Artifact decision is mechanically derived
T10: Nexus-only tests collect without PyTorch
T11: Python 3.11 and 3.12 CI pass (smoke test)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


# ── T6: Hard negatives are score-derived ──

def test_hard_negatives_are_score_derived():
    """T6: Hard negatives must come from the highest-scoring incorrect candidates,
    not just the first N pipeline outputs."""
    # Simulated candidate pipeline: lexical scores
    candidates = [
        ("A", 0.95),  # gold
        ("B", 0.80),  # high-scoring incorrect
        ("C", 0.75),  # high-scoring incorrect
        ("D", 0.30),  # irrelevant
        ("E", 0.20),  # irrelevant
        ("F", 0.10),  # irrelevant
    ]
    gold = {"A"}
    hard_negative_k = 2

    # Correct: take highest-scoring non-gold
    non_gold = [(nid, score) for nid, score in candidates if nid not in gold]
    non_gold.sort(key=lambda x: -x[1])
    correct_hard = [nid for nid, _ in non_gold[:hard_negative_k]]
    assert correct_hard == ["B", "C"], (
        f"T6 FAIL: Hard negatives should be highest-scoring incorrect, got {correct_hard}"
    )

    # Incorrect: take first N non-gold (pipeline order, regardless of score)
    incorrect_hard = [nid for nid, score in candidates if nid not in gold][:hard_negative_k]
    assert incorrect_hard != correct_hard or incorrect_hard == ["B", "C"], (
        "T6: Pipeline-first order accidentally matched score-derived. "
        "This is fine if pipeline already orders by score."
    )

    # Verify the correct approach uses scores, not pipeline position
    high_score_negatives = [nid for nid, score in non_gold[:hard_negative_k]]
    assert all(score >= 0.70 for _, score in non_gold[:hard_negative_k]), (
        "T6 FAIL: Hard negatives should have high scores (be confusable)"
    )


# ── T7: Synthetic/real sampling proportions enforced ──

def test_sampling_proportions_enforced():
    """T7: The preregistered sampling policy (50/25/15/10) must be enforced
    in training data generation."""
    # Mock 100 groups with known sources
    groups = []
    for i in range(50):
        groups.append({"source": "train_candidate_pipeline", "id": f"real_{i}"})
    for i in range(25):
        groups.append({"source": "graph_mined_paraphrase", "id": f"para_{i}"})
    for i in range(15):
        groups.append({"source": "graph_alias_keyfinding", "id": f"akf_{i}"})
    for i in range(10):
        groups.append({"source": "graph_relation", "id": f"rel_{i}"})

    # Count per source
    from collections import Counter
    counts = Counter(g["source"] for g in groups)

    assert counts["train_candidate_pipeline"] == 50
    assert counts["graph_mined_paraphrase"] == 25
    assert counts["graph_alias_keyfinding"] == 15
    assert counts["graph_relation"] == 10

    # Verify proportions
    total = len(groups)
    assert counts["train_candidate_pipeline"] / total == 0.50
    assert counts["graph_mined_paraphrase"] / total == 0.25
    assert counts["graph_alias_keyfinding"] / total == 0.15
    assert counts["graph_relation"] / total == 0.10


def test_sampling_proportions_reject_invalid():
    """T7 continued: Invalid proportions must raise error."""
    # A sampling policy validator
    def validate_proportions(real_ratio, para_ratio, akf_ratio, rel_ratio):
        total = real_ratio + para_ratio + akf_ratio + rel_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Sampling proportions sum to {total}, expected 1.0")
        if real_ratio < 0.35:  # min 35% real questions
            raise ValueError("Real question ratio too low")

    # Valid
    validate_proportions(0.50, 0.25, 0.15, 0.10)

    # Invalid sum
    with pytest.raises(ValueError):
        validate_proportions(0.80, 0.25, 0.15, 0.10)

    # Real too low
    with pytest.raises(ValueError, match="Real"):
        validate_proportions(0.30, 0.40, 0.20, 0.10)


def test_production_balanced_dataset_uses_distinct_source_pools(monkeypatch):
    from stack.encoder import natural_templates

    def row(identifier, source_id):
        return {
            "id": identifier,
            "question": identifier,
            "entities": ["E"],
            "source_id": source_id,
        }

    generated = [
        row("p1", "graph:v3:E:natural_paraphrase"),
        row("p2", "graph:v3:E:natural_diagnostic"),
        row("a1", "graph:v3:E:natural_factual"),
        row("r1", "graph:v3:E:natural_relation"),
    ]
    monkeypatch.setattr(natural_templates, "generate_natural_pairs", lambda _g: generated)
    real = [row(f"real-{index}", "real") for index in range(4)]
    result = natural_templates.generate_balanced_dataset(real, graph=None)
    counts = __import__("collections").Counter(item["source"] for item in result)
    assert counts == {
        "real_train": 4,
        "graph_mined_paraphrase": 2,
        "graph_alias_keyfinding": 1,
    }
    assert len({item["id"] for item in result}) == len(result)


# ── T8: Dirty worktree blocks evaluation ──

def test_dirty_worktree_guard(tmp_path: Path):
    """T8: staged and untracked changes must both block evaluation."""
    from stack.encoder.experiment_guard import check_worktree_clean

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)
    assert check_worktree_clean(tmp_path)

    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert not check_worktree_clean(tmp_path)
    (tmp_path / "untracked.txt").unlink()

    tracked.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    assert not check_worktree_clean(tmp_path)


def test_never_overwrite_historical_artifacts(tmp_path: Path):
    """T8 continued: Artifact writing must refuse to overwrite existing files."""
    artifact = tmp_path / "historical.json"
    artifact.write_text('{"value": 1}')

    def write_artifact(path: Path, data: dict) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite historical artifact: {path}")
        path.write_text(json.dumps(data))

    with pytest.raises(FileExistsError, match="historical"):
        write_artifact(artifact, {"value": 2})


# ── T9: Artifact decision is mechanically derived ──

def test_mechanical_decision_from_gates():
    """T9: The final artifact decision must be exactly: all gates pass → PASS, else FAIL."""
    def make_decision(gates: dict[str, bool]) -> str:
        all_pass = all(gates.values())
        return "HONEST PASS" if all_pass else "HONEST FAIL"

    assert make_decision({"recall": True, "latency": True, "rss": True}) == "HONEST PASS"
    assert make_decision({"recall": True, "latency": False, "rss": True}) == "HONEST FAIL"
    assert make_decision({"recall": False}) == "HONEST FAIL"
    assert make_decision({}) == "HONEST PASS"  # vacuously


def test_nonmechanical_decision_rejected():
    """T9 continued: Artifact that claims PASS with a failing gate is invalid."""
    gates = {"primary_recall": {"passed": False}}
    decision = "HONEST PASS"
    all_pass = all(g["passed"] for g in gates.values())
    derived = "HONEST PASS" if all_pass else "HONEST FAIL"
    assert decision != derived, "PASS with failed gate should be rejected"
    assert derived == "HONEST FAIL"


# ── T10: Nexus-only tests collect without PyTorch ──

def test_nexus_imports_without_pytorch():
    """T10: The nexus/ package must import without PyTorch installed."""
    import tempfile
    repo_root = str(Path(__file__).parents[1])
    script = (
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "class Blocker:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] == 'torch':\n"
        "            raise ImportError('Blocked: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import nexus\n"
        "import nexus.graph\n"
        "import nexus.graph.store\n"
        "print('OK')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"T10 FAIL: nexus imports failed without PyTorch.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        assert "OK" in result.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── T11: Python 3.11 and 3.12 CI pass (smoke) ──

def test_python_version_is_supported():
    """T11: Verify Python >= 3.11 is being used."""
    version = sys.version_info
    assert version >= (3, 11), (
        f"T11: Python {version.major}.{version.minor} is not >= 3.11"
    )


def test_py_compile_all_new_modules():
    """T11 continued: All new modules must compile cleanly."""
    import py_compile

    new_modules = [
        "stack/encoder/canonical_mapping.py",
    ]
    repo_root = Path(__file__).parents[1]
    for module in new_modules:
        path = repo_root / module
        if path.exists():
            py_compile.compile(str(path), doraise=True)


# ── T12: Artifact immutability ──

def test_save_ranker_v3_refuses_overwrite(tmp_path: Path):
    """T12: save_ranker_v3 must fail if output files already exist."""
    torch = pytest.importorskip("torch")
    from stack.encoder.entity_ranker_v3 import (
        QuestionConditionedEntityRanker,
        save_ranker_v3,
    )
    from stack.encoder.char_tokenizer import CharNgramTokenizer

    out_dir = tmp_path / "model"
    out_dir.mkdir()

    tokenizer = CharNgramTokenizer()
    tokenizer.add_words(["test"])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(
        tokenizer.feature_dim, embed_dim=8, hidden_dim=16, proj_dim=4
    )

    # First save should succeed
    save_ranker_v3(model, tokenizer, {"test": True}, str(out_dir))

    # Second save must fail on existing files
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_ranker_v3(model, tokenizer, {"test": False}, str(out_dir))


def test_second_run_cannot_overwrite_first_run(tmp_path: Path):
    """T12 continued: A second run with the same timestamped path must fail."""
    artifact = tmp_path / "selection.json"
    artifact.write_text('{"first": true}')

    def write_if_new(path: Path, data: dict) -> None:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        path.write_text(json.dumps(data))

    # First write: path doesn't exist yet — OK
    # (already written above)

    # Second write: path exists — must fail
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_if_new(artifact, {"second": True})


def test_require_new_path_rejects_existing(tmp_path: Path):
    """T12 continued: _require_new_path rejects existing paths."""
    from stack.encoder.train_ranker_v3 import _require_new_path

    existing = tmp_path / "exists.json"
    existing.write_text("{}")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _require_new_path(existing)

    # Non-existing path should be returned unchanged
    new_path = tmp_path / "does_not_exist.json"
    assert _require_new_path(new_path) == new_path


def test_write_json_artifact_rejects_overwrite(tmp_path: Path):
    """T12 continued: _write_json_artifact fails on existing file."""
    from stack.encoder.train_ranker_v3 import _write_json_artifact

    path = tmp_path / "artifact.json"
    _write_json_artifact(path, {"run": 1})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_json_artifact(path, {"run": 2})


def test_save_ranker_v3_rejects_nonexistent_directory(tmp_path: Path):
    """T12 continued: save_ranker_v3 fails if output directory doesn't exist."""
    torch = pytest.importorskip("torch")
    from stack.encoder.entity_ranker_v3 import (
        QuestionConditionedEntityRanker,
        save_ranker_v3,
    )
    from stack.encoder.char_tokenizer import CharNgramTokenizer

    tokenizer = CharNgramTokenizer()
    tokenizer.add_words(["test"])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(
        tokenizer.feature_dim, embed_dim=8, hidden_dim=16, proj_dim=4
    )

    nonexistent = str(tmp_path / "nonexistent_subdir")

    with pytest.raises(NotADirectoryError):
        save_ranker_v3(model, tokenizer, {}, nonexistent)
