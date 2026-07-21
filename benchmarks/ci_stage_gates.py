"""Fail-closed Stage-gate checks for CI (cheap, deterministic, no training).

Gates:
1. Traversal budget units + small campaign (reference-CPU prereg fields)
2. Paired oracle publication smoke (DummyModel, lexical predicted arm)
3. AnswerPlan status recorder (honest BLOCKED / deferred, no full training)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_project_root))


def gate_traversal_units() -> None:
    _run([
        sys.executable, "-m", "pytest",
        "tests/test_traversal_budgets.py",
        "tests/test_traversal_budget_campaign.py",
        "tests/test_rule_engine.py",
        "tests/test_rule_corpus_v1.py",
        "tests/test_conflict_policy.py",
        "tests/test_bitemporal_replay.py",
        "tests/test_deterministic_realization_gates.py",
        "tests/test_union_resolver.py",
        "-q", "--tb=short",
    ])


def gate_rule_corpus() -> None:
    _run([
        sys.executable,
        "benchmarks/eval_rule_engine.py",
        "--mode", "development",
    ])


def gate_traversal_small_campaign() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "traversal_small.json"
        _run([
            sys.executable,
            "benchmarks/run_traversal_budget_campaign.py",
            "--sizes", "small",
            "--output", str(output),
        ])
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("status") != "PASS":
            raise RuntimeError("traversal small campaign did not PASS")
        if artifact.get("preregistration_id") != "traversal-budgets-v1":
            raise RuntimeError("missing preregistration_id")
        print(json.dumps({
            "gate": "traversal_small",
            "status": "PASS",
            "sha_source": artifact.get("source_sha", "")[:12],
        }, sort_keys=True))


def gate_oracle_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "oracle_smoke.json"
        _run([
            sys.executable,
            "benchmarks/run_oracle_vs_predicted.py",
            "--predicted-resolver", "lexical",
            "--limit", "8",
            "--dummy-model",
            "--output", str(output),
        ])
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != "nexus-oracle-vs-predicted-v2":
            raise RuntimeError("oracle smoke: bad schema")
        metrics = artifact["predicted"]["metrics"]
        if "entry_recall_mean" not in metrics or "pool_recall_mean" not in metrics:
            raise RuntimeError("oracle smoke: missing ER metrics")
        print(json.dumps({
            "gate": "oracle_smoke",
            "status": "PASS",
            "questions": artifact.get("questions_total"),
        }, sort_keys=True))


def gate_answer_plan_status() -> None:
    _run([sys.executable, "benchmarks/record_answer_plan_status.py", "--ci"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=("all", "traversal", "oracle", "answer_plan", "rules"),
        default="all",
    )
    args = parser.parse_args(argv)
    if args.gate in ("all", "traversal"):
        gate_traversal_units()
        gate_traversal_small_campaign()
    if args.gate in ("all", "rules"):
        gate_rule_corpus()
    if args.gate in ("all", "oracle"):
        gate_oracle_smoke()
    if args.gate in ("all", "answer_plan"):
        gate_answer_plan_status()
    print(json.dumps({"status": "PASS", "gate": args.gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
