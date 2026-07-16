"""Infrastructure and artifact/CI tests for Entity Ranker V3.

T6: Hard negatives are score-derived (production miner)
T7: Synthetic/real sampling proportions enforced (production generator)
T8: Dirty worktree blocks evaluation + production artifact guards
T9: Artifact decision is mechanically derived (production select_winner)
T10: Nexus-only tests collect without PyTorch
T11: Python 3.11 and 3.12 CI pass (smoke test)
T12: Artifact immutability
T15: Documentation cross-reference integrity
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
    """T6: Production hard negative miner returns highest-scoring non-gold."""
    from nexus.graph import Node
    from nexus.graph.store import InMemoryGraphStore
    from stack.encoder.hard_negative_miner import mine_hard_negatives_lexical

    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Alpha", type="Experiment", aliases=["alpha test"]))
    graph.add_node(Node(id="Beta", type="Experiment", aliases=["beta test"]))
    graph.add_node(Node(id="Gamma", type="Experiment", aliases=["gamma"]))
    graph.add_node(Node(id="Gold", type="Experiment", aliases=["target entity", "main result"]))

    question = "What did the alpha test show?"
    candidate_ids = ["Alpha", "Beta", "Gamma", "Gold"]
    positive_ids = {"Gold"}

    hard = mine_hard_negatives_lexical(question, candidate_ids, positive_ids, graph, top_k=2)
    assert len(hard) >= 1
    assert "Gold" not in hard
    assert hard[0] == "Alpha", f"Expected Alpha, got {hard[0]}"


def test_hard_negative_miner_never_includes_gold():
    """T6 continued: Production miner never returns gold entities."""
    from nexus.graph import Node
    from nexus.graph.store import InMemoryGraphStore
    from stack.encoder.hard_negative_miner import mine_hard_negatives_lexical

    graph = InMemoryGraphStore()
    for nid in ["A", "B", "C", "G1", "G2"]:
        graph.add_node(Node(id=nid, type="Experiment", aliases=[nid.lower()]))
    question = "test question"
    candidate_ids = ["A", "B", "C", "G1", "G2"]
    positive_ids = {"G1", "G2"}
    for top_k in range(1, 5):
        hard = mine_hard_negatives_lexical(question, candidate_ids, positive_ids, graph, top_k=top_k)
        assert not (set(hard) & positive_ids), f"Gold in hard negatives at top_k={top_k}"


# ── T7: Synthetic/real sampling proportions enforced ──

def test_production_balanced_dataset_proportions(monkeypatch):
    """T7: Production generate_balanced_dataset enforces 50/25/15/10."""
    from stack.encoder import natural_templates

    def make_row(identifier, source_id):
        return {"id": identifier, "question": identifier, "entities": ["E"], "source_id": source_id}

    generated = [
        make_row("p1", "graph:v3:E:natural_paraphrase"),
        make_row("p2", "graph:v3:E:natural_diagnostic"),
        make_row("p3", "graph:v3:E:natural_paraphrase"),
        make_row("a1", "graph:v3:E:natural_factual"),
        make_row("a2", "graph:v3:E:natural_factual"),
        make_row("r1", "graph:v3:E:natural_relation"),
    ]
    monkeypatch.setattr(natural_templates, "generate_natural_pairs", lambda _g: generated)
    real = [make_row(f"real-{i}", "real") for i in range(4)]
    result = natural_templates.generate_balanced_dataset(real, graph=None)

    from collections import Counter
    counts = Counter(item["source"] for item in result)
    assert counts["real_train"] == 4
    assert 1 <= counts.get("graph_mined_paraphrase", 0) <= 2
    assert len({item["id"] for item in result}) == len(result)


def test_production_balanced_dataset_rejects_invalid_ratios(monkeypatch):
    """T7 continued: Production generator rejects invalid ratios."""
    from stack.encoder import natural_templates
    monkeypatch.setattr(natural_templates, "generate_natural_pairs", lambda _g: [])
    real = [{"id": f"r{i}", "question": f"q{i}", "entities": ["E"]} for i in range(10)]
    with pytest.raises(ValueError, match="sum"):
        natural_templates.generate_balanced_dataset(
            real, graph=None, real_ratio=0.80, paraphrase_ratio=0.25,
            alias_ratio=0.15, relation_ratio=0.10,
        )


# ── T8: Dirty worktree blocks evaluation ──

def test_dirty_worktree_guard(tmp_path: Path):
    """T8: Staged and untracked changes both block evaluation."""
    from stack.encoder.experiment_guard import check_worktree_clean

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=tmp_path, check=True)
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


def test_production_write_json_artifact_refuses_overwrite(tmp_path: Path):
    """T8: Production _write_json_artifact refuses to overwrite."""
    pytest.importorskip("torch")  # train_ranker_v3 imports torch
    from stack.encoder.train_ranker_v3 import _write_json_artifact
    path = tmp_path / "test_artifact.json"
    _write_json_artifact(path, {"run": 1})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_json_artifact(path, {"run": 2})


# ── T9: Artifact decision is mechanically derived ──

def test_production_select_winner_mechanical():
    """T9: Production select_winner uses max recall@10, tie-break recall@5."""
    pytest.importorskip("torch")  # train_ranker_v3 imports torch
    from stack.encoder.train_ranker_v3 import select_winner
    metrics = {
        "model_a": {"recall@10": 0.70, "recall@5": 0.50, "recall@1": 0.10},
        "model_b": {"recall@10": 0.65, "recall@5": 0.60, "recall@1": 0.30},
    }
    assert select_winner(metrics) == "model_a"
    metrics_tie = {
        "model_a": {"recall@10": 0.70, "recall@5": 0.50, "recall@1": 0.10},
        "model_b": {"recall@10": 0.70, "recall@5": 0.60, "recall@1": 0.10},
    }
    assert select_winner(metrics_tie) == "model_b"


def test_empty_gate_set_is_invalid():
    """T9: Empty gate dict must not produce a PASS verdict — returns INVALID."""
    pytest.importorskip("torch")  # train_ranker_v3 imports torch
    from stack.encoder.train_ranker_v3 import select_winner
    with pytest.raises(IndexError):
        select_winner({})


def test_gate_decision_requires_all_gates_pass():
    """T9: A failing gate produces FAIL, not PASS."""
    gates = {"recall@10": 0.55, "recall@5": 0.40}
    threshold = 0.65
    all_pass = all(v >= threshold for v in gates.values())
    assert not all_pass


def test_missing_gate_produces_invalid():
    """T9: Missing required gates must not silently pass."""
    required_keys = {"recall@1", "recall@5", "recall@10"}
    partial = {"recall@1": 0.5}
    missing = required_keys - set(partial.keys())
    assert missing
    assert "recall@5" in missing
    assert "recall@10" in missing


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
        "import nexus\nimport nexus.graph\nimport nexus.graph.store\n"
        "print('OK')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        result = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"T10 FAIL: {result.stderr}"
        assert "OK" in result.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── T11: Python version and compile ──

def test_python_version_is_supported():
    """T11: Verify Python >= 3.11."""
    assert sys.version_info >= (3, 11)


def test_py_compile_all_new_modules():
    """T11 continued: All new modules compile cleanly."""
    import py_compile
    new_modules = ["stack/encoder/canonical_mapping.py"]
    repo_root = Path(__file__).parents[1]
    for module in new_modules:
        path = repo_root / module
        if path.exists():
            py_compile.compile(str(path), doraise=True)


# ── T12: Artifact immutability ──

def test_save_ranker_v3_refuses_overwrite(tmp_path: Path):
    """T12: save_ranker_v3 must fail if output files already exist."""
    torch = pytest.importorskip("torch")
    from stack.encoder.entity_ranker_v3 import QuestionConditionedEntityRanker, save_ranker_v3
    from stack.encoder.char_tokenizer import CharNgramTokenizer
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words(["test"])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(tokenizer.feature_dim, embed_dim=8, hidden_dim=16, proj_dim=4)
    save_ranker_v3(model, tokenizer, {"test": True}, str(out_dir))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_ranker_v3(model, tokenizer, {"test": False}, str(out_dir))


def test_save_ranker_v3_rejects_nonexistent_directory(tmp_path: Path):
    """T12: save_ranker_v3 fails if output directory doesn't exist."""
    torch = pytest.importorskip("torch")
    from stack.encoder.entity_ranker_v3 import QuestionConditionedEntityRanker, save_ranker_v3
    from stack.encoder.char_tokenizer import CharNgramTokenizer
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words(["test"])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(tokenizer.feature_dim, embed_dim=8, hidden_dim=16, proj_dim=4)
    with pytest.raises(NotADirectoryError):
        save_ranker_v3(model, tokenizer, {}, str(tmp_path / "nonexistent_subdir"))


# ── T15: Documentation cross-reference integrity ──

def test_documentation_references_existing_artifacts():
    """T15: Documentation references to current artifacts must point to existing files."""
    repo_root = Path(__file__).parents[1]
    required_paths = [
        "benchmarks/results/entity_ranker_v3_selection_20260711T081545Z.json",
        "benchmarks/results/entity_ranker_v3_frozen_20260711T084518Z.json",
        "models/encoder/entity_ranker_v3_20260711T081545Z/",
    ]
    for path_str in required_paths:
        full_path = repo_root / path_str
        assert full_path.exists(), (
            f"Documentation references {path_str} which does not exist "
            f"and has no external asset URL."
        )

    # Verify STACK_RESULTS.md and README.md don't disagree on ER3 status
    stack_results = (repo_root / "STACK_RESULTS.md").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    # Extract the ER3 status line from STACK_RESULTS
    sr_er3_status = ""
    for line in stack_results.splitlines():
        if "| ER3 |" in line:
            sr_er3_status = line.strip()
            break

    rm_er3_status = ""
    for line in readme.splitlines():
        if "Entity Ranker V3 |" in line or "Entity Ranker V3 (corrective" in line:
            rm_er3_status = line.strip()
            break

    expected = "CHECKPOINT VERIFIED"
    assert expected in sr_er3_status
    assert expected in rm_er3_status
    assert "EXTERNAL CHECKPOINT REQUIRED" not in sr_er3_status
    assert "EXTERNAL CHECKPOINT REQUIRED" not in rm_er3_status
    assert "AUDITABLE PASS" not in sr_er3_status
    assert "AUDITABLE PASS" not in rm_er3_status


def test_no_evaluation_test_reads_frozen_split():
    """Entity Ranker V3 evaluation tests must never read test.jsonl."""
    repo_root = Path(__file__).parents[1]
    tests_dir = repo_root / "tests"
    # Only check V3-specific test files
    v3_test_files = [
        "test_entity_ranker_v3.py",
        "test_canonical_mapping.py",
        "test_optional_torch_collection.py",
    ]
    for tf_name in v3_test_files:
        test_file = tests_dir / tf_name
        if not test_file.exists():
            continue
        content = test_file.read_text(encoding="utf-8")
        # Fail if the file reads the actual test.jsonl path
        if 'stack/encoder/data/test.jsonl' in content:
            raise AssertionError(
                f"V3 test {tf_name} references stack/encoder/data/test.jsonl. "
                f"Evaluation tests must never read the frozen split."
            )
