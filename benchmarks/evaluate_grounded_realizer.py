"""Evaluate the fail-closed Grounded Realizer on an untouched dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import sha256_file, validate_dataset_manifest
from nexus.realizer.grounded import answer_similarity, realize_grounded, token_f1


def _normalized(text: str) -> str:
    return " ".join(str(text).casefold().split())


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Realize from evidence first, then use labels only for scoring."""
    exact = 0
    similarity = 0.0
    f1 = 0.0
    grounded = 0
    insufficient = 0
    fallback = 0
    outputs: set[str] = set()
    per_example: list[dict[str, Any]] = []
    latencies: list[float] = []

    for record in records:
        started = time.perf_counter()
        result = realize_grounded(record)
        latencies.append((time.perf_counter() - started) * 1000)
        reference = str(record.get("answer", ""))
        is_exact = _normalized(result.answer) == _normalized(reference)
        current_similarity = answer_similarity(result.answer, reference)
        current_f1 = token_f1(result.answer, reference)
        exact += is_exact
        similarity += current_similarity
        f1 += current_f1
        grounded += result.grounding_score >= 0.99
        insufficient += result.strategy == "insufficient_evidence"
        fallback += result.fallback_used
        outputs.add(result.answer)
        per_example.append({
            "id": record.get("id"),
            "question": record.get("question"),
            "answer": result.answer,
            "reference": reference,
            "exact_match": is_exact,
            "similarity": round(current_similarity, 6),
            "token_f1": round(current_f1, 6),
            "realization": result.to_dict(),
        })

    n = max(len(records), 1)
    sorted_latency = sorted(latencies)
    p50 = sorted_latency[len(sorted_latency) // 2] if sorted_latency else 0.0
    metrics = {
        "samples": len(records),
        "exact_match_rate": round(exact / n, 6),
        "similarity_mean": round(similarity / n, 6),
        "token_f1_mean": round(f1 / n, 6),
        "grounded_rate": round(grounded / n, 6),
        "hallucination_rate": round(1.0 - grounded / n, 6),
        "fallback_rate": round(fallback / n, 6),
        "insufficient_evidence_rate": round(insufficient / n, 6),
        "unique_outputs": len(outputs),
        "uniqueness_ratio": round(len(outputs) / n, 6),
        "latency_p50_ms": round(p50, 6),
    }
    passed = (
        metrics["exact_match_rate"] >= 0.95
        and metrics["token_f1_mean"] >= 0.95
        and metrics["grounded_rate"] >= 0.99
        and metrics["hallucination_rate"] <= 0.01
        and metrics["uniqueness_ratio"] >= 0.80
    )
    return {
        "status": "GROUNDED_REALIZER_PASS" if passed else "GROUNDED_REALIZER_FAIL",
        "metrics": metrics,
        "per_example": per_example,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/distillation/realizer_v1/manifest.json"),
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_dataset_manifest(manifest, args.manifest.parent)
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(errors))
    split_meta = manifest["splits"][args.split]
    split_path = args.manifest.parent / split_meta["path"]
    records = [
        json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.limit > 0:
        records = records[:args.limit]

    artifact = evaluate_records(records)
    artifact.update({
        "schema_version": "nexus-grounded-realizer-eval-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_tree_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "split": args.split,
        "split_sha256": split_meta.get("sha256"),
        "label_use": "scoring_only",
        "realization_input": "question_and_evidence_only",
    })

    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({
        "status": artifact["status"],
        "metrics": artifact["metrics"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if artifact["status"] == "GROUNDED_REALIZER_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
