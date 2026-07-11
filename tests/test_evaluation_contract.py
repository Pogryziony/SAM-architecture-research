"""Tests for the evaluation contract module (torch-independent)."""
from __future__ import annotations

import pytest

from pathlib import Path

from stack.encoder.evaluation_contract import (
    Verdict,
    evaluate_contract,
)


def test_all_gates_pass_returns_pass():
    result = evaluate_contract(
        {"recall@10": 0.80, "latency": 25.0},
        [
            {"name": "recall@10", "threshold": 0.70},
            {"name": "latency", "threshold": 50.0, "operator": "<="},
        ],
    )
    assert result.verdict == Verdict.PASS
    assert len(result.gates) == 2
    assert all(g.passed for g in result.gates)
    assert not result.errors


def test_one_gate_fails_returns_fail():
    result = evaluate_contract(
        {"recall@10": 0.55, "baseline_gap": 0.10},
        [
            {"name": "recall@10", "threshold": 0.70},
            {"name": "baseline_gap", "threshold": 0.15},
        ],
    )
    assert result.verdict == Verdict.FAIL
    assert len(result.gates) == 2
    assert not all(g.passed for g in result.gates)


def test_empty_gate_set_returns_invalid():
    result = evaluate_contract({"recall@10": 0.80}, [])
    assert result.verdict == Verdict.INVALID
    assert "empty gate set" in " ".join(result.errors)


def test_missing_required_gate_returns_invalid():
    result = evaluate_contract(
        {"recall@10": 0.80},
        [{"name": "recall@10", "threshold": 0.70}, {"name": "latency", "threshold": 50.0}],
    )
    assert result.verdict == Verdict.INVALID
    assert any("latency" in e for e in result.errors)


def test_nan_value_returns_invalid():
    result = evaluate_contract(
        {"recall@10": float("nan")},
        [{"name": "recall@10", "threshold": 0.70}],
    )
    assert result.verdict == Verdict.INVALID


def test_inf_value_returns_invalid():
    result = evaluate_contract(
        {"recall@10": float("inf")},
        [{"name": "recall@10", "threshold": 0.70}],
    )
    assert result.verdict == Verdict.INVALID


def test_malformed_metric_returns_invalid():
    result = evaluate_contract(
        {"recall@10": "not_a_number"},
        [{"name": "recall@10", "threshold": 0.70}],
    )
    assert result.verdict == Verdict.INVALID


def test_meta_mismatch_returns_invalid():
    result = evaluate_contract(
        {"recall@10": 0.80, "run_id": "wrong_id"},
        [{"name": "recall@10", "threshold": 0.70}],
        required_meta={"run_id": "correct_id"},
    )
    assert result.verdict == Verdict.INVALID
    assert any("run_id" in e for e in result.errors)


def test_meta_match_passes():
    result = evaluate_contract(
        {"recall@10": 0.80, "run_id": "entity_ranker_v3_20260711T081545Z"},
        [{"name": "recall@10", "threshold": 0.70}],
        required_meta={"run_id": "entity_ranker_v3_20260711T081545Z"},
    )
    assert result.verdict == Verdict.PASS


def test_unknown_operator_returns_invalid():
    result = evaluate_contract(
        {"recall@10": 0.80},
        [{"name": "recall@10", "threshold": 0.70, "operator": "??"}],
    )
    assert result.verdict == Verdict.INVALID


def test_no_valid_gates_returns_invalid():
    result = evaluate_contract(
        {},
        [
            {"name": "missing_gate", "threshold": 0.70},
        ],
    )
    assert result.verdict == Verdict.INVALID


def test_to_dict_serializable():
    result = evaluate_contract(
        {"recall@10": 0.80},
        [{"name": "recall@10", "threshold": 0.70}],
    )
    d = result.to_dict()
    assert d["verdict"] == "PASS"
    assert len(d["gates"]) == 1
    assert d["gates"][0]["passed"] is True
    assert d["gates"][0]["value"] == 0.80


def test_module_does_not_import_torch():
    """Verify the evaluation contract module is torch-independent."""
    import subprocess, sys
    repo_root = str(Path(__file__).resolve().parents[1])

    script = (
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "class Blocker:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch':\n"
        "            raise ImportError('torch not allowed')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from stack.encoder.evaluation_contract import evaluate_contract, Verdict\n"
        "result = evaluate_contract({'r': 0.80}, [{'name': 'r', 'threshold': 0.70}])\n"
        "assert result.verdict.value == 'PASS'\n"
        "print('OK')\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        result = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert "OK" in result.stdout
    finally:
        Path(tmp).unlink(missing_ok=True)
