"""Entity Ranker V3 — single-read frozen evaluation.

Reads a new, unconsumed holdout split exactly once, evaluates with a
committed model checkpoint, and writes an immutable artifact.

The consumed frozen split (test.jsonl, SHA-256
ac7877084f2384d2e80ef3ce43d48c842eb4d404936d3139a1c7b06d41616c6a)
is permanently rejected.  Future evaluations MUST use a new holdout.

Usage:
    python -m benchmarks.entity_ranker_v3_final \
        --model-dir models/encoder/entity_ranker_v3_<TIMESTAMP> \
        --validation-artifact benchmarks/results/entity_ranker_v3_selection_<TS>.json \
        --split path/to/new_holdout.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
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
    K_MAX,
    SEED,
)
from stack.encoder.experiment_guard import check_worktree_clean
from stack.encoder.frozen_split_guard import validate_new_holdout, ConsumedSplitError
from stack.encoder.loader import get_peak_rss_mb


REQUIRED_VAL_RECALL10 = 0.70
REQUIRED_BASELINE_GAP = 0.15


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


def _compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_frozen_evaluation(
    model_dir: str,
    validation_artifact: str,
    split_path: str | None = None,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Run the single-read frozen evaluation on a new, unconsumed holdout.

    1. Verify clean worktree
    2. Validate new holdout split (reject consumed split)
    3. Load and verify validation artifact
    4. Load model, verify checkpoint integrity
    5. Build evaluation groups (never inject gold)
    6. Evaluate with per-question diagnostics
    7. Write immutable artifact
    """
    root = Path(root)

    # Guard: clean worktree
    if not check_worktree_clean(root):
        raise RuntimeError(
            "Dirty worktree detected. Commit or stash changes before frozen evaluation."
        )

    # ── Validate split ──
    if split_path is None:
        raise ValueError(
            "A new holdout split path is required. "
            "The consumed frozen split must not be reused. "
            "Use --split path/to/new_holdout.jsonl"
        )
    split_path_p = Path(split_path)
    split_hash = validate_new_holdout(split_path_p)

    # ── Timestamp ──
    utc_now = datetime.now(timezone.utc)
    run_ts = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"entity_ranker_v3_frozen_{run_ts}"

    # Source SHA
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    # ── Load and verify validation artifact ──
    val_path = Path(validation_artifact)
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation artifact not found: {val_path}"
        )
    val_data = json.loads(val_path.read_text(encoding="utf-8"))
    val_winner = val_data["selection"]["winner"]
    val_r10 = val_data["selection"]["winner_recall@10"]
    val_baseline_r10 = val_data["selection"]["baseline_recall@10"]
    val_source_sha = val_data["source_sha"]
    val_run_id = val_data["run_id"]

    # Verify validation gates
    if val_r10 < REQUIRED_VAL_RECALL10:
        raise ValueError(
            f"Validation recall@10 {val_r10:.4f} is below "
            f"required {REQUIRED_VAL_RECALL10}. Frozen evaluation blocked."
        )
    if val_r10 - val_baseline_r10 < REQUIRED_BASELINE_GAP:
        raise ValueError(
            f"Validation baseline gap {val_r10 - val_baseline_r10:.4f} "
            f"is below required {REQUIRED_BASELINE_GAP}. Frozen evaluation blocked."
        )

    # ── Load and verify model checkpoint ──
    model_dir_p = Path(model_dir)
    if not model_dir_p.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir_p}")

    checkpoint_path = model_dir_p / "weights.pt"
    config_path = model_dir_p / "config.json"
    vocab_path = model_dir_p / "vocab.json"

    for p, name in [(checkpoint_path, "weights"), (config_path, "config"),
                     (vocab_path, "tokenizer")]:
        if not p.exists():
            raise FileNotFoundError(f"Model {name} not found: {p}")

    checkpoint_sha256 = _compute_file_sha256(checkpoint_path)
    config_sha256 = _compute_file_sha256(config_path)
    vocab_sha256 = _compute_file_sha256(vocab_path)

    model, tokenizer, model_config = load_ranker_v3(str(model_dir_p))

    # Verify model source SHA matches validation
    model_source_sha = model_config.get("source_sha", "")
    if model_source_sha != val_source_sha:
        raise ValueError(
            f"Model source SHA {model_source_sha} does not match "
            f"validation source SHA {val_source_sha}"
        )

    # Verify model run ID
    model_run_id = model_config.get("run_id", "")
    if model_run_id != val_run_id:
        raise ValueError(
            f"Model run ID {model_run_id} does not match "
            f"validation run ID {val_run_id}"
        )

    # ── Load split ──
    split_bytes = split_path_p.read_bytes()
    split_data = [
        json.loads(line)
        for line in split_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    # ── Build graph and canonical mapping ──
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, graph_meta = build_benchmark_graph()
    canonical_mapping = build_canonical_mapping(graph)
    canonical_meta = export_canonical_mapping_metadata(canonical_mapping, graph)

    # ── Build evaluation groups ──
    test_groups = []
    for record in split_data:
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

    # Candidate ceiling
    raw_ceiling = sum(
        len(set(g["positive_ids"]) & set(g["candidate_ids"]))
        for g in test_groups
    )

    rss_before = get_peak_rss_mb()
    latencies: list[float] = []

    model.eval()
    entity_text_map = {
        str(nid): build_entity_text(str(nid), graph)
        for nid in graph._nodes
    }

    # Evaluate with per-question diagnostics
    hits = {k: 0 for k in (1, 5, 10)}
    predicted = {k: 0 for k in (1, 5, 10)}
    per_question: list[dict[str, Any]] = []

    with torch.no_grad():
        for g in test_groups:
            gold = set(g["positive_ids"])
            cand_ids = g["candidate_ids"]

            t_start = time.perf_counter()
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
            ranked_scores = [float(scores[0][i]) for i in ranked_indices]
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

            canonical_ranked = apply_canonical_mapping(
                ranked_ids, canonical_mapping, top_k=K_MAX
            )

            for k in (1, 5, 10):
                top_k_set = set(canonical_ranked[:k])
                hits[k] += len(top_k_set & gold)
                predicted[k] += len(canonical_ranked[:k])

            per_question.append({
                "question_id": g["question_id"],
                "question": g["question"][:200],
                "gold_ids": sorted(gold),
                "candidate_ids": cand_ids[:50],
                "raw_ranked_ids": ranked_ids[:20],
                "raw_ranked_scores": ranked_scores[:20],
                "canonical_top10": canonical_ranked[:10],
                "gold_hit": bool(set(canonical_ranked[:10]) & gold),
                "gold_in_candidates": g.get("gold_present_in_candidates", True),
            })

    rss_after = get_peak_rss_mb()
    import statistics
    latencies.sort()

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
    metrics["latency_p50_ms"] = statistics.median(latencies)
    metrics["latency_p95_ms"] = (
        latencies[int(len(latencies) * 0.95)] if len(latencies) > 1
        else latencies[0]
    )
    metrics["peak_rss_mb"] = rss_after - rss_before

    frozen_recall10 = metrics["canonical_recall@10"]
    gate_pass = frozen_recall10 >= 0.65

    # ── Write artifact ──
    artifact_path = root / "benchmarks" / "results" / f"{run_id}.json"
    artifact = {
        "run_id": run_id,
        "run_timestamp_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "model_dir": str(model_dir_p),
        "model_config": model_config,
        "model_checkpoint_sha256": checkpoint_sha256,
        "model_config_sha256": config_sha256,
        "model_vocab_sha256": vocab_sha256,
        "validation_artifact": str(val_path),
        "validation_winner": val_winner,
        "validation_recall10": val_r10,
        "validation_source_sha": val_source_sha,
        "validation_run_id": val_run_id,
        "split": str(split_path_p),
        "split_sha256": split_hash,
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
        "per_question_predictions": per_question,
    }
    _write_json_artifact(artifact_path, artifact)

    return {
        "artifact_path": str(artifact_path),
        "frozen_recall10": frozen_recall10,
        "gate_passed": gate_pass,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Single-read frozen evaluation for Entity Ranker V3 "
                    "(new holdouts only; consumed split permanently rejected)"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the timestamped model directory",
    )
    parser.add_argument(
        "--validation-artifact",
        required=True,
        help="Path to the committed validation selection artifact",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Path to the new, unconsumed holdout split (test.jsonl is consumed and rejected)",
    )
    args = parser.parse_args()

    result = run_frozen_evaluation(args.model_dir, args.validation_artifact, args.split)
    print(json.dumps(result, indent=2))
    verdict = "HONEST PASS" if result["gate_passed"] else "HONEST FAIL"
    print(f"\n{verdict}: frozen canonical recall@10 = {result['frozen_recall10']:.4f}")


if __name__ == "__main__":
    main()
