"""Deterministic evidence-to-answer distillation dataset builder.

Builds verifier-passed training pairs from the canonical NEXUS pipeline
using only train-split questions.  Each accepted record contains the
structured evidence pack, graph paths, sources, and verifier result.

Does not read the consumed frozen split.
Does not use validation or holdout labels.

Usage:
    python benchmarks/build_distillation_dataset.py --limit 200 --output data/distillation/pairs_v2.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner


# ═══════════════════════════════════════════════════════════════════════
# Acceptance rules
# ═══════════════════════════════════════════════════════════════════════

def accept_pair(
    record: dict,
    question: dict,
    graph: InMemoryGraphStore,
) -> tuple[bool, str]:
    """Check if a generated pair meets all acceptance criteria.

    Returns (accepted, reason).
    """
    answer = record.get("answer", "")

    # 1. Non-empty answer
    if not answer or len(answer.strip()) < 10:
        return False, "empty_or_too_short_answer"

    # 2. No generic refusal
    refusal_patterns = [
        "insufficient evidence", "cannot answer", "not enough",
        "unable to determine", "i don't know",
    ]
    for pat in refusal_patterns:
        if pat in answer.lower():
            return False, "refusal_detected"

    # 3. Verifier must pass
    if not record.get("verifier_passed", False):
        return False, "verifier_failed"

    # 4. Must have evidence pack
    ep = record.get("evidence_pack_keys", [])
    if not ep:
        return False, "no_evidence"

    # 5. No duplicate question (checked by caller via hashing)
    # 6. No validation/holdout labels (enforced by caller — train only)

    return True, "accepted"


def stable_id(question: str, entities: list[str]) -> str:
    """Deterministic example ID from question + entities."""
    payload = question + "\0" + "\0".join(sorted(entities))
    return "distill_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_distillation_dataset(
    questions: list[dict],
    graph: InMemoryGraphStore,
    output_path: str,
    source_sha: str = "",
    min_pairs: int = 5000,
) -> dict[str, Any]:
    """Build a deterministic distillation dataset from train questions.

    Args:
        questions: Training questions (never validation or holdout).
        graph: Populated NEXUS graph.
        output_path: Where to write the JSONL output.
        source_sha: Current commit SHA.
        min_pairs: Target minimum pair count (reports shortfall).

    Returns:
        Summary dict with counts and acceptance statistics.
    """
    config = ProductionNEXUSConfig.with_entity_ranker_v3()
    runner = NEXUSRunner(graph, config)
    utc_now = datetime.now(timezone.utc)

    accepted = []
    rejected: dict[str, int] = {}
    seen_hashes: set[str] = set()

    for i, q in enumerate(questions):
        result = runner.run([q], source_sha=source_sha)
        qr = result.per_question[0]

        record = {
            "id": stable_id(q["question"], qr.predicted_entities),
            "question": q["question"],
            "question_type": q.get("question_type", qr.parsed_intent),
            "canonical_entities": qr.predicted_entities[:10],
            "intent": qr.parsed_intent,
            "evidence_pack_keys": qr.evidence_pack_keys,
            "graph_paths_count": qr.graph_paths_count,
            "answer": qr.answer,
            "verifier_passed": qr.verifier_passed,
            "hallucination_rate": qr.hallucination_rate,
            "entity_resolution_method": qr.entity_resolution_method,
            "generator_identity": "nexus_v1_er3",
            "source_sha": source_sha,
            "created_utc": utc_now.isoformat(),
        }

        ok, reason = accept_pair(record, q, graph)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        # Deduplicate by question hash
        qhash = hashlib.sha256(q["question"].encode("utf-8")).hexdigest()[:16]
        if qhash in seen_hashes:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        seen_hashes.add(qhash)

        accepted.append(record)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)}: {len(accepted)} accepted")

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "pipeline": "distillation_v1",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "questions_processed": len(questions),
        "pairs_accepted": len(accepted),
        "pairs_rejected": sum(rejected.values()),
        "rejection_reasons": dict(rejected),
        "min_pairs_target": min_pairs,
        "target_met": len(accepted) >= min_pairs,
        "output_path": output_path,
    }

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build distillation dataset")
    parser.add_argument("--limit", type=int, default=200, help="Questions to process")
    parser.add_argument("--output", default="data/distillation/pairs_v2.jsonl")
    parser.add_argument("--min-pairs", type=int, default=5000)
    args = parser.parse_args()

    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()

    train_path = _project_root / "stack" / "encoder" / "data" / "train.jsonl"
    questions = [
        json.loads(line)
        for line in Path(train_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:args.limit]

    print(f"Building distillation dataset from {len(questions)} train questions")
    summary = build_distillation_dataset(questions, graph, args.output, min_pairs=args.min_pairs)
    print(f"\nSummary:")
    print(f"  Accepted: {summary['pairs_accepted']}")
    print(f"  Rejected: {summary['pairs_rejected']}")
    print(f"  Target ({args.min_pairs}) met: {summary['target_met']}")
    for reason, count in summary["rejection_reasons"].items():
        print(f"    {reason}: {count}")
