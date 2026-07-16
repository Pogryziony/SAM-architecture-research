"""Evaluate a constrained-plan Realizer checkpoint on full validation."""

from __future__ import annotations

import argparse
import copy
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

from benchmarks.realizer_contracts import canonical_json, sha256_file
from benchmarks.train_nexus_realizer import (
    load_training_inputs, serialize_source_for_config,
)
from benchmarks.train_nexus_realizer_v2 import _generate_and_score
from nexus.realizer.decoder import DecoderConfig
from nexus.realizer.model import build_model, parameter_count


def _git_identity() -> dict[str, str]:
    def read(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=_project_root, text=True,
        ).strip()

    return {"commit": read("rev-parse", "HEAD"), "tree": read("rev-parse", "HEAD^{tree}")}


def evaluate_checkpoint(
    manifest_path: Path, config_path: Path, weights_path: Path,
    *, source_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch unavailable") from exc

    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    if config.get("data", {}).get("target_format") != "relation_label_v2":
        raise ValueError("full evaluation requires relation_label_v2")
    model = build_model(config["model"])
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    started = time.perf_counter()
    metrics = _generate_and_score(
        model, list(splits["validation"]), config,
        DecoderConfig(
            strategy="constrained_relation_v2", repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            max_length=int(config["training"]["generation_max_tokens"]),
        ),
        len(splits["validation"]),
    )

    # A verifier flag alone is insufficient.  Runtime serialization must also
    # reject a plan whose relation contradicts the immutable slot values.
    tampered = copy.deepcopy(splits["validation"][0])
    relation = tampered["composition"]["relation"]
    tampered["composition"]["relation"] = (
        "different" if relation == "the same" else "the same"
    )
    try:
        serialize_source_for_config(tampered, config)
    except ValueError:
        tampered_plan_rejected = True
    else:
        tampered_plan_rejected = False

    promotion = config.get("gates", {}).get("promotion", {})
    hallucination_rate = round(1.0 - metrics["neural_grounded_rate"], 6)
    checks = [
        {
            "name": "materialized_exact_match",
            "passed": metrics["answer_exact_match_rate"] >= float(
                promotion["materialized_exact_match_min"]
            ),
            "value": metrics["answer_exact_match_rate"],
            "threshold": promotion["materialized_exact_match_min"],
        },
        {
            "name": "slot_placeholder_exact",
            "passed": metrics["slot_placeholder_exact_rate"] >= float(
                promotion["slot_placeholder_exact_min"]
            ),
            "value": metrics["slot_placeholder_exact_rate"],
            "threshold": promotion["slot_placeholder_exact_min"],
        },
        {
            "name": "relation_accuracy",
            "passed": metrics["relation_accuracy"] >= float(
                promotion["relation_accuracy_min"]
            ),
            "value": metrics["relation_accuracy"],
            "threshold": promotion["relation_accuracy_min"],
        },
        {
            "name": "relation_accuracy_each",
            "passed": metrics["relation_min_class_accuracy"] >= float(
                promotion["relation_accuracy_each_min"]
            ),
            "value": metrics["relation_accuracy_by_class"],
            "threshold": promotion["relation_accuracy_each_min"],
        },
        {
            "name": "hallucination",
            "passed": hallucination_rate <= float(promotion["hallucination_max"]),
            "value": hallucination_rate,
            "threshold": promotion["hallucination_max"],
        },
        {
            "name": "tampered_plan_fails_closed",
            "passed": tampered_plan_rejected,
            "value": tampered_plan_rejected,
            "threshold": True,
        },
    ]
    blocking = [check["name"] for check in checks if not check["passed"]]
    canonical_payload = {
        "schema_version": "nexus-realizer-abstractive-evaluation-v1",
        "source": source_identity or _git_identity(),
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "parameter_count": parameter_count(model),
        "validation_count": len(splits["validation"]),
        "decoder_strategy": "constrained_relation_v2",
        "metrics": metrics,
        "checks": checks,
        "blocking_checks": blocking,
        "status": "PILOT_CHECKPOINT_ACCEPTED" if not blocking else "PILOT_CHECKPOINT_REJECTED",
    }
    return {
        **canonical_payload,
        "canonical_sha256": hashlib.sha256(
            canonical_json(canonical_payload).encode("utf-8")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args()
    if bool(args.source_commit) is not bool(args.source_tree):
        raise ValueError("--source-commit and --source-tree must be provided together")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    source_identity = (
        {"commit": args.source_commit, "tree": args.source_tree}
        if args.source_commit else None
    )
    result = evaluate_checkpoint(
        args.manifest, args.config, args.weights,
        source_identity=source_identity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii",
    )
    print(json.dumps({
        "status": result["status"],
        "canonical_sha256": result["canonical_sha256"],
        "blocking_checks": result["blocking_checks"],
    }, sort_keys=True))
    return 0 if result["status"] == "PILOT_CHECKPOINT_ACCEPTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
