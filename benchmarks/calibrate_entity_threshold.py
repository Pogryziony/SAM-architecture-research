"""Calibrate the entity reranker threshold on the non-frozen validation split.

This script never reads stack/encoder/data/test.jsonl. It reports a threshold
curve on val.jsonl and writes a timestamped artifact for provenance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import build_benchmark_graph
from nexus.query.embedding_resolver import NodeEmbeddingIndex
from stack.encoder.eval_gates import eval_encoder, load_questions
from stack.encoder.loader import get_encoder
from nexus.utils.config import NEXUSConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    val_path = root / "stack" / "encoder" / "data" / "val.jsonl"
    questions = load_questions(str(val_path))
    graph, _ = build_benchmark_graph()
    index = NodeEmbeddingIndex()
    index.build_index(graph)
    encoder = get_encoder()
    if not encoder.load():
        raise RuntimeError("failed to load encoder")

    curve = []
    for threshold in args.thresholds:
        config = NEXUSConfig(enable_associative_encoder=True)
        result = eval_encoder(graph, questions, encoder, threshold, index, config)
        curve.append({
            "threshold": threshold,
            "precision": result["entity_precision"],
            "recall": result["entity_recall"],
            "f1": result["entity_f1"],
            "candidate_pool_recall": result["stage_candidate_recall"]["candidate_pool"],
            "reranker_top1_recall": result["stage_candidate_recall"]["reranker_top1"],
            "final_accepted_recall": result["entity_recall"],
            "parser_success_rate": result["parser_success_rate"],
        })

    best = max(curve, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output = args.output or root / "benchmarks" / "results" / f"entity_threshold_calibration_{ts}.json"
    payload = {
        "meta": {
            "calibration_split": "stack/encoder/data/val.jsonl",
            "calibration_sample_count": len(questions),
            "model_checkpoint": "models/encoder_v2",
            "frozen_split_used": False,
            "timestamp_utc": ts,
            "source_commit_sha": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "thresholds": args.thresholds,
            "metric_denominators": {
                "precision": "correct predicted entity IDs / predicted entity IDs",
                "recall": "correct predicted entity IDs / gold entity IDs",
                "f1": "harmonic mean of precision and recall",
            },
        },
        "curve": curve,
        "selected_threshold": best["threshold"],
        "selection_rule": "maximum validation F1, then recall, then lowest threshold",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "best": best}, indent=2))


if __name__ == "__main__":
    main()
