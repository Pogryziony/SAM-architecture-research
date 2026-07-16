"""Combine preparation and full pilot evaluation without launching training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import canonical_json, sha256_file


SCHEMA_VERSION = "nexus-realizer-abstractive-full-readiness-v1"


def _git_identity() -> dict[str, str]:
    def read(ref: str) -> str:
        return subprocess.check_output(
            ["git", "rev-parse", ref], cwd=_project_root, text=True,
        ).strip()

    return {"commit": read("HEAD"), "tree": read("HEAD^{tree}")}


def build_readiness(
    preparation_path: Path,
    evaluation_path: Path,
    manifest_path: Path,
    config_path: Path,
    weights_path: Path,
) -> dict[str, Any]:
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = _git_identity()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append({
            "name": name, "passed": bool(passed), "value": value,
            "requirement": requirement,
        })

    check(
        "preparation", preparation.get("status") == "READY_FOR_BOUNDED_PILOT"
        and not preparation.get("blocking_checks"), preparation.get("status"),
        "bounded-pilot preparation passes with zero blocking checks",
    )
    check(
        "pilot_evaluation", evaluation.get("status") == "PILOT_CHECKPOINT_ACCEPTED"
        and not evaluation.get("blocking_checks"), evaluation.get("status"),
        "full validation accepts the bounded pilot checkpoint",
    )
    check(
        "dataset_identity",
        preparation.get("dataset_sha256") == evaluation.get("dataset_sha256")
        == manifest.get("dataset_sha256"), manifest.get("dataset_sha256"),
        "preparation, pilot and manifest use one dataset",
    )
    config_sha = sha256_file(config_path)
    check(
        "config_identity",
        preparation.get("config_sha256") == evaluation.get("config_sha256")
        == config_sha, config_sha,
        "preparation, pilot and launch use one config",
    )
    weights_sha = sha256_file(weights_path)
    check(
        "weights_identity", evaluation.get("weights_sha256") == weights_sha,
        weights_sha, "pilot evaluation identifies the exact candidate weights",
    )
    check(
        "source_identity", evaluation.get("source") == identity,
        {"evaluation": evaluation.get("source"), "current": identity},
        "pilot evaluation was generated from the current committed source",
    )
    check(
        "promotion_gates", all(
            item.get("passed") is True for item in evaluation.get("checks", [])
        ), evaluation.get("checks", []),
        "all full-validation promotion and fail-closed gates pass",
    )
    blocking = [item["name"] for item in checks if not item["passed"]]
    canonical_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_FULL_TRAINING" if not blocking else "BLOCKED",
        "blocking_checks": blocking,
        "checks": checks,
        "source": identity,
        "dataset_sha256": manifest.get("dataset_sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha,
        "weights_sha256": weights_sha,
        "preparation_sha256": sha256_file(preparation_path),
        "pilot_evaluation_sha256": sha256_file(evaluation_path),
        "full_training_launched": False,
        "authorization_scope": "one explicitly requested full run only",
    }
    return {
        **canonical_payload,
        "canonical_sha256": hashlib.sha256(
            canonical_json(canonical_payload).encode("utf-8")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    result = build_readiness(
        args.preparation, args.evaluation, args.manifest, args.config, args.weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    sidecar.write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="ascii",
    )
    print(json.dumps({
        "status": result["status"],
        "canonical_sha256": result["canonical_sha256"],
        "blocking_checks": result["blocking_checks"],
        "full_training_launched": False,
    }, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_FULL_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
