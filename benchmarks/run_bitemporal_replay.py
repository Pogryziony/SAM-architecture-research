"""Stage 6 bi-temporal replay campaign against oracle gold facts.

Filters stamped facts with as_valid_at / as_known_at, rejects look-ahead, and
checks gold path presence after filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph import Edge, Node
from nexus.graph.bitemporal import (
    assert_no_lookahead,
    filter_edges_bitemporal,
    filter_facts_bitemporal,
)
from nexus.graph.store import InMemoryGraphStore

PREREGISTRATION_ID = "bitemporal-replay-v1"
DEFAULT_GOLD = (
    _project_root / "benchmarks" / "qa-dataset" / "bitemporal_oracle_v1.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _graph_from_facts(facts: list[dict[str, Any]]) -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    for fact in facts:
        source = str(fact["source"])
        target = str(fact["target"])
        if not graph.has_node(source):
            graph.add_node(Node(id=source, type="Entity", sources=["bitemporal_oracle_v1"]))
        if not graph.has_node(target):
            graph.add_node(Node(id=target, type="Entity", sources=["bitemporal_oracle_v1"]))
        graph.add_edge(
            Edge(
                type=str(fact["relation"]),
                source=source,
                target=target,
                confidence=1.0,
                evidence="bitemporal_oracle_v1",
                valid_from=str(fact.get("valid_from") or ""),
                valid_to=str(fact.get("valid_to") or ""),
                observed_at=str(fact.get("observed_at") or ""),
                retracted_at=str(fact.get("retracted_at") or ""),
            )
        )
    return graph


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    facts = list(record.get("facts") or [])
    as_known_at = str(record.get("as_known_at") or "")
    as_valid_at = str(record.get("as_valid_at") or "")
    lookahead_errors = assert_no_lookahead(facts, as_known_at=as_known_at) if as_known_at else []
    kept_facts = filter_facts_bitemporal(
        facts, as_valid_at=as_valid_at, as_known_at=as_known_at
    )
    graph = _graph_from_facts(facts)
    edges = [edge for node in graph._nodes for edge in graph.get_outgoing(node)]
    kept_edges = filter_edges_bitemporal(
        edges, as_valid_at=as_valid_at, as_known_at=as_known_at
    )
    kept_triples = {
        (edge.source, edge.type, edge.target) for edge in kept_edges
    }
    gold_path = [
        (str(step["source"]), str(step["relation"]), str(step["target"]))
        for step in record.get("gold_path") or []
    ]
    path_ok = all(triple in kept_triples for triple in gold_path)
    should_abstain = bool(record.get("should_abstain"))
    predicted_abstain = len(kept_facts) == 0
    errors: list[str] = []
    if lookahead_errors:
        # Look-ahead facts must be excluded by the filter, not used as answers.
        leaked = [
            fact
            for fact in kept_facts
            if fact in facts
            and any(
                err.startswith("look-ahead") and str(fact.get("source")) in err
                for err in lookahead_errors
            )
        ]
        if leaked:
            errors.append("look-ahead leakage in kept facts")
    if should_abstain and not predicted_abstain and not gold_path:
        # Abstain gold with empty path expects empty kept set.
        if kept_facts:
            errors.append("expected abstain but facts remained after filter")
    if not should_abstain and not path_ok:
        errors.append("gold_path missing after bi-temporal filter")
    if should_abstain and gold_path and path_ok and kept_facts:
        errors.append("abstain gold unexpectedly retained gold_path")
    return {
        "id": record["id"],
        "kept_fact_count": len(kept_facts),
        "kept_edge_count": len(kept_edges),
        "predicted_abstain": predicted_abstain,
        "should_abstain": should_abstain,
        "path_ok": path_ok,
        "lookahead_error_count": len(lookahead_errors),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [evaluate_record(record) for record in records]
    failures = [row for row in rows if row["status"] != "PASS"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "preregistration_id": PREREGISTRATION_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "pass_count": len(rows) - len(failures),
        "fail_count": len(failures),
        "schema_version": "nexus-bitemporal-replay-v1",
        "rows": rows,
        "errors": [err for row in failures for err in row["errors"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate(_read_jsonl(args.gold))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_records": report["n_records"],
                "pass_count": report["pass_count"],
                "fail_count": report["fail_count"],
                "errors": report["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
