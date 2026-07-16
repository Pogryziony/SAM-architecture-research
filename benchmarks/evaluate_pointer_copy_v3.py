"""Evaluate deterministic Pointer/Copy Realizer v3 without label leakage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import sha256_file, validate_dataset_manifest
from nexus.realizer.grounded import (
    evidence_candidates, grounding_diagnostics, token_f1,
)
from nexus.realizer.pointer_copy import (
    PointerCopyConfig, pointer_copy_config_from_dict, realize_pointer_copy,
)


ORDER_SEED = 20260716


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _permuted_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reorder candidates once per record without creating extra records."""
    result = copy.deepcopy(record)
    evidence = result.get("evidence_pack", {})
    digest = hashlib.sha256(
        f"{ORDER_SEED}:{record.get('id', '')}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    for key in ("node_facts", "snippets", "facts", "paths"):
        values = evidence.get(key)
        if isinstance(values, list):
            rng.shuffle(values)
    for path in evidence.get("paths", []):
        if isinstance(path, dict) and isinstance(path.get("nodes"), list):
            rng.shuffle(path["nodes"])
    return result


def _classify(record: dict[str, Any]) -> tuple[str, int | None]:
    answer = _norm(record.get("answer", ""))
    candidates = evidence_candidates(record)
    for index, candidate in enumerate(candidates):
        if _norm(candidate.text) == answer:
            return "extractive_full_candidate", index
    for index, candidate in enumerate(candidates):
        text = _norm(candidate.text)
        if answer and (answer in text or text in answer):
            return "extractive_span", index
    return "unsupported_or_invalid", None


def _synthetic_record(
    question: str, candidates: list[tuple[str, str, float]]
) -> dict[str, Any]:
    return {
        "question": question,
        "evidence_pack": {
            "node_facts": [
                {"text": text, "source": source, "confidence": confidence}
                for text, source, confidence in candidates
            ],
            "snippets": [], "paths": [], "facts": [],
        },
    }


def evaluate_adversarial(pointer_config: PointerCopyConfig) -> dict[str, Any]:
    """Exercise failure modes that aggregate validation metrics can hide."""
    question = "What is model.max_seq_len in configs/nexus.yaml?"
    correct = (
        "In configs/nexus.yaml, model.max_seq_len is set to 128.",
        "model.max_seq_len", 1.0,
    )
    distractors = [
        ("In configs/other.yaml, model.max_seq_len is set to 256.", "other", 0.7),
        ("In configs/nexus.yaml, train.epochs is set to 3.", "train.epochs", 0.7),
        ("In README.md, model.max_seq_len is set to 512.", "README", 0.7),
    ]
    checks: dict[str, bool] = {}
    selected = realize_pointer_copy(
        _synthetic_record(question, distractors + [correct]), config=pointer_config,
    )
    checks["correct_candidate_not_first"] = selected.answer == correct[0]
    checks["wrong_number_rejected"] = "256" not in selected.answer
    checks["wrong_identifier_rejected"] = "configs/other.yaml" not in selected.answer
    checks["wrong_key_rejected"] = "train.epochs" not in selected.answer

    missing = realize_pointer_copy(
        {"question": question, "evidence_pack": {}}, config=pointer_config,
    )
    checks["missing_evidence_fails_closed"] = (
        missing.strategy == "insufficient_evidence"
        and missing.rejection_reason == "no_evidence_candidate"
    )
    conflicting = realize_pointer_copy(
        _synthetic_record(question, [
            correct,
            ("In configs/nexus.yaml, model.max_seq_len is set to 256.",
             "model.max_seq_len", 1.0),
        ]),
        config=pointer_config,
    )
    checks["conflicting_evidence_fails_closed"] = (
        conflicting.strategy == "insufficient_evidence"
        and conflicting.rejection_reason == "ambiguous_evidence_candidates"
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
    }


def evaluate(
    records: list[dict[str, Any]], pointer_config: PointerCopyConfig
) -> dict[str, Any]:
    classification = Counter()
    ranked_positions = Counter()
    available = top1 = top3 = exact = 0
    shuffled_exact = 0
    hallucinated = unsupported_number = unsupported_identifier = 0
    f1_sum = 0.0
    outputs: set[str] = set()
    margins: list[float] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    for record in records:
        category, target_position = _classify(record)
        classification[category] += 1
        if target_position is not None:
            available += 1
            ranked_positions[target_position] += 1
            top1 += target_position == 0
            top3 += target_position < 3

        started = time.perf_counter()
        result = realize_pointer_copy(record, config=pointer_config)
        latencies.append((time.perf_counter() - started) * 1000)
        reference = str(record.get("answer", ""))
        diagnostics = grounding_diagnostics(result.answer, evidence_candidates(record))
        unsupported_number += bool(diagnostics.unsupported_numbers)
        unsupported_identifier += bool(diagnostics.unsupported_identifiers)
        hallucinated += bool(diagnostics.rejection_reason)
        is_exact = _norm(result.answer) == _norm(reference)
        exact += is_exact
        f1_sum += token_f1(result.answer, reference)
        outputs.add(result.answer)
        margins.append(result.selection_margin)

        shuffled = realize_pointer_copy(
            _permuted_record(record), config=pointer_config,
        )
        shuffled_exact += _norm(shuffled.answer) == _norm(reference)
        if not is_exact:
            failures.append({
                "id": record.get("id"),
                "question": record.get("question"),
                "reference": reference,
                "answer": result.answer,
                "realization": result.to_dict(),
            })

    n = max(len(records), 1)
    ordered_latency = sorted(latencies)
    ordered_margins = sorted(margins)

    def quantile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        return values[min(int((len(values) - 1) * fraction), len(values) - 1)]

    exact_rate = exact / n
    shuffled_rate = shuffled_exact / n
    metrics = {
        "samples": len(records),
        "candidate_availability": round(available / n, 6),
        "top1_accuracy": round(top1 / n, 6),
        "top3_recall": round(top3 / n, 6),
        "exact_match_rate": round(exact_rate, 6),
        "token_f1_mean": round(f1_sum / n, 6),
        "wrong_candidate_rate": round(1.0 - exact_rate, 6),
        "unsupported_number_rate": round(unsupported_number / n, 6),
        "unsupported_identifier_rate": round(unsupported_identifier / n, 6),
        "hallucination_rate": round(hallucinated / n, 6),
        "insufficient_evidence_rate": round(
            sum(item["realization"]["strategy"] == "insufficient_evidence" for item in failures) / n,
            6,
        ),
        "unique_outputs": len(outputs),
        "uniqueness_ratio": round(len(outputs) / n, 6),
        "position_shuffle_exact_match_rate": round(shuffled_rate, 6),
        "position_shuffle_drop": round(exact_rate - shuffled_rate, 6),
        "latency_p50_ms": round(quantile(ordered_latency, 0.5), 6),
        "selection_margin": {
            "min": round(quantile(ordered_margins, 0.0), 6),
            "p10": round(quantile(ordered_margins, 0.1), 6),
            "p50": round(quantile(ordered_margins, 0.5), 6),
            "p90": round(quantile(ordered_margins, 0.9), 6),
        },
    }
    adversarial = evaluate_adversarial(pointer_config)
    passed = (
        metrics["candidate_availability"] >= 0.99
        and metrics["exact_match_rate"] >= 0.98
        and metrics["token_f1_mean"] >= 0.99
        and metrics["unsupported_number_rate"] == 0.0
        and metrics["unsupported_identifier_rate"] == 0.0
        and metrics["hallucination_rate"] <= 0.01
        and metrics["uniqueness_ratio"] >= 0.80
        and metrics["position_shuffle_drop"] <= 0.01
        and metrics["latency_p50_ms"] <= 5.0
        and adversarial["passed"]
    )
    return {
        "status": (
            "POINTER_COPY_REALIZER_V3_ACCEPTED"
            if passed else "POINTER_COPY_REALIZER_V3_REJECTED"
        ),
        "dataset_classification": dict(sorted(classification.items())),
        "target_ranked_positions": {
            str(key): value for key, value in sorted(ranked_positions.items())
        },
        "baselines": {
            "current_evidence_ranking": {
                "top1_accuracy": metrics["top1_accuracy"],
            },
            "lexical_pointer_copy": {
                "exact_match_rate": metrics["exact_match_rate"],
            },
            "oracle_candidate_availability": {
                "candidate_availability": metrics["candidate_availability"],
                "label_use": "offline_scoring_only",
            },
        },
        "metrics": metrics,
        "position_ordering": {
            "algorithm": "sha256(seed:record_id) deterministic in-place permutation",
            "seed": ORDER_SEED,
            "records_duplicated": 0,
        },
        "adversarial": adversarial,
        "failures": failures[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/distillation/realizer_v1/manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("training/pointer_copy_realizer_v3.json"),
    )
    parser.add_argument(
        "--source-commit",
        help="Recorded source commit; allowed only with --source-tree matching HEAD.",
    )
    parser.add_argument(
        "--source-tree",
        help="Recorded source tree; must exactly match the checked-out HEAD tree.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_dataset_manifest(manifest, args.manifest.parent)
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(errors))
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        path = args.manifest.parent / manifest["splits"][split]["path"]
        splits[split] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    pointer_config = pointer_copy_config_from_dict(config_payload)
    validation = evaluate(splits["validation"], pointer_config)
    train_classification = Counter(_classify(record)[0] for record in splits["train"])
    local_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    local_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True,
    ).strip()
    if bool(args.source_commit) != bool(args.source_tree):
        raise ValueError("--source-commit and --source-tree must be provided together")
    if args.source_tree and args.source_tree != local_tree:
        raise ValueError(
            f"recorded source tree {args.source_tree} does not match HEAD tree {local_tree}"
        )
    artifact = {
        "schema_version": "nexus-pointer-copy-eval-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit or local_commit,
        "source_tree_sha": args.source_tree or local_tree,
        "dataset_manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "effective_config": config_payload,
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": {
            split: manifest["splits"][split]["sha256"]
            for split in ("train", "validation")
        },
        "label_use": "classification_and_scoring_only",
        "realization_input": "question_and_evidence_only",
        "train_dataset_classification": dict(sorted(train_classification.items())),
        "validation": validation,
    }
    canonical = copy.deepcopy(artifact)
    canonical.pop("created_utc")
    artifact["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({
        "status": validation["status"],
        "metrics": validation["metrics"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if validation["status"].endswith("ACCEPTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
