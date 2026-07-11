"""Entity Ranker V3 — single-read frozen evaluation.

Reads the frozen test.jsonl exactly once, evaluates with a
committed model checkpoint, and writes an immutable artifact.

Usage:
    python -m benchmarks.entity_ranker_v3_final --model-dir models/encoder/entity_ranker_v3_<TIMESTAMP>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.entity_ranker_v3 import (
    QuestionConditionedEntityRanker,
    load_ranker_v3,
)
from stack.encoder.entity_text import build_entity_text
from stack.encoder.canonical_mapping import (
    build_canonical_mapping,
    apply_canonical_mapping,
    export_canonical_mapping_metadata,
)
from stack.encoder.trivial_baseline import candidate_pool
from stack.encoder.train_ranker_v3 import (
    build_evaluation_group,
    evaluate_trivial_baseline,
    K_MAX,
    SEED,
)
from stack.encoder.experiment_guard import check_worktree_clean


def _require_new_path(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {path}"
        )
    return path


def _write_json_artifact(path: Path, data: dict[str, Any]) -> None:
    _require_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(path)


def run_frozen_evaluation(model_dir: str, root: str | Path = ".") -> dict[str, Any]:
    """Run the single-read frozen evaluation.

    1. Verify clean worktree
    2. Load frozen model, canonical mapping, graph
    3. Load test.jsonl (read exactly once)
    4. Build evaluation groups (never inject gold)
    5. Evaluate
    6. Write immutable artifact
    """
    root = Path(root)

    # Guard: clean worktree
    if not check_worktree_clean(root):
        raise RuntimeError(
            "Dirty worktree detected. Commit or stash changes before frozen evaluation."
        )

    # Timestamp
    utc_now = datetime.now(timezone.utc)
    run_ts = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"entity_ranker_v3_frozen_{run_ts}"

    # Source SHA
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    # Load model
    model, tokenizer, model_config = load_ranker_v3(model_dir)

    # Load frozen test split (read exactly once)
    test_path = root / "stack/encoder/data/test.jsonl"
    test = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()

    # Build graph and canonical mapping
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, graph_meta = build_benchmark_graph()
    canonical_mapping = build_canonical_mapping(graph)
    canonical_meta = export_canonical_mapping_metadata(canonical_mapping, graph)

    # Build evaluation groups (ALL 225 questions preserved, no gold injection)
    test_groups = []
    for record in test:
        group = build_evaluation_group(
            str(record.get("id", "")),
            str(record["question"]),
            [str(e) for e in record.get("entities", [])],
            [str(item["node_id"]) for item in candidate_pool(record["question"], graph)],
            "test",
            graph,
        )
        if group is not None:
            test_groups.append(group)
        else:
            test_groups.append({
                "question_id": str(record.get("id", "")),
                "question": str(record["question"]),
                "candidate_ids": [
                    str(item["node_id"])
                    for item in candidate_pool(record["question"], graph)
                ],
                "positive_ids": sorted(set(str(e) for e in record.get("entities", []))),
                "source": "test",
                "gold_present_in_candidates": False,
            })

    total_questions = len(test_groups)
    total_gold = sum(len(set(g["positive_ids"])) for g in test_groups)

    # Compute candidate ceilings
    raw_ceiling = sum(
        len(set(g["positive_ids"]) & set(g["candidate_ids"]))
        for g in test_groups
    )

    model.eval()
    entity_text_map = {
        str(nid): build_entity_text(str(nid), graph)
        for nid in graph._nodes
    }

    # Evaluate
    hits = {k: 0 for k in (1, 5, 10)}
    predicted = {k: 0 for k in (1, 5, 10)}

    with torch.no_grad():
        for g in test_groups:
            gold = set(g["positive_ids"])
            cand_ids = g["candidate_ids"]

            # Score candidates
            offsets, indices = tokenizer.tokenize_batch([g["question"]])
            q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
            q_indices = torch.tensor(indices, dtype=torch.long)
            cand_texts = [
                entity_text_map.get(cid, build_entity_text(cid, graph))
                for cid in cand_ids
            ]
            scores = model(q_indices, q_offsets, cand_texts, tokenizer)
            ranked_indices = torch.argsort(scores[0], descending=True).tolist()
            ranked_ids = [cand_ids[i] for i in ranked_indices]

            # Apply canonical mapping
            canonical_ranked = apply_canonical_mapping(
                ranked_ids, canonical_mapping, top_k=K_MAX
            )

            for k in (1, 5, 10):
                top_k_canonical = set(canonical_ranked[:k])
                hits[k] += len(top_k_canonical & gold)
                predicted[k] += len(canonical_ranked[:k])

    metrics = {}
    for k in (1, 5, 10):
        metrics[f"canonical_recall@{k}"] = hits[k] / total_gold if total_gold else 0.0
        metrics[f"canonical_precision@{k}"] = (
            hits[k] / predicted[k] if predicted[k] else 0.0
        )
    metrics["total_questions"] = float(total_questions)
    metrics["total_gold_entities"] = float(total_gold)
    metrics["raw_candidate_recall_ceiling"] = (
        raw_ceiling / total_gold if total_gold else 0.0
    )

    # Gate check
    frozen_recall10 = metrics["canonical_recall@10"]
    gate_pass = frozen_recall10 >= 0.65

    # Write artifact
    artifact_path = root / "benchmarks" / "results" / f"{run_id}.json"
    artifact = {
        "run_id": run_id,
        "run_timestamp_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "model_dir": model_dir,
        "model_config": model_config,
        "split": "stack/encoder/data/test.jsonl",
        "split_sha256": test_hash,
        "graph": graph_meta,
        "canonical_mapping": canonical_meta,
        "total_questions": total_questions,
        "total_gold_entities": total_gold,
        "metrics": metrics,
        "gate": {
            "frozen_recall10": frozen_recall10,
            "threshold": 0.65,
            "passed": gate_pass,
        },
        "k_max": K_MAX,
        "seed": SEED,
        "test_split_read": True,
        "frozen": True,
    }
    _write_json_artifact(artifact_path, artifact)

    return {
        "artifact_path": str(artifact_path),
        "frozen_recall10": frozen_recall10,
        "gate_passed": gate_pass,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Single-read frozen evaluation for Entity Ranker V3"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the timestamped model directory (e.g., models/encoder/entity_ranker_v3_<TS>)",
    )
    args = parser.parse_args()

    result = run_frozen_evaluation(args.model_dir)
    print(json.dumps(result, indent=2))
    verdict = "HONEST PASS" if result["gate_passed"] else "HONEST FAIL"
    print(f"\n{verdict}: frozen canonical recall@10 = {result['frozen_recall10']:.4f}")


if __name__ == "__main__":
    main()
