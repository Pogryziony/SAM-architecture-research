"""Fail-closed verification for externally distributed Entity Ranker bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BundleVerificationError(ValueError):
    """Raised when a model bundle or its validation evidence is incomplete."""


@dataclass(frozen=True)
class VerifiedBundle:
    manifest: dict[str, Any]
    validation: dict[str, Any]
    model_config: dict[str, Any]
    hashes: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_file(path: Path, spec: dict[str, Any], label: str) -> str:
    if not path.is_file():
        raise BundleVerificationError(f"{label} is missing: {path}")
    expected_size = int(spec["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise BundleVerificationError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = _sha256(path)
    expected_hash = str(spec["sha256"])
    if actual_hash != expected_hash:
        raise BundleVerificationError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return actual_hash


def verify_model_bundle(
    root: str | Path,
    model_dir: str | Path,
    validation_artifact: str | Path,
) -> VerifiedBundle:
    """Verify manifest, local model files, and committed validation evidence.

    Weights are intentionally not stored in git.  A caller must place the exact
    externally supplied ``weights.pt`` in *model_dir*.  Its size and SHA-256 are
    verified before PyTorch is allowed to load it.
    """
    root = Path(root).resolve()
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = (root / model_dir).resolve()
    validation_path = Path(validation_artifact)
    if not validation_path.is_absolute():
        validation_path = (root / validation_path).resolve()

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleVerificationError(f"model manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != "2.0":
        raise BundleVerificationError("unsupported model manifest schema")

    hashes: dict[str, str] = {}
    for filename in ("config.json", "vocab.json"):
        spec = manifest["files"][filename]
        hashes[filename] = _verify_file(
            model_dir / filename,
            {"sha256": spec["sha256"], "size_bytes": spec["size_bytes"]},
            filename,
        )

    weights_spec = manifest["external_weights"]
    if weights_spec.get("storage") != "external":
        raise BundleVerificationError("weights storage must be declared external")
    hashes["weights.pt"] = _verify_file(
        model_dir / str(weights_spec["local_filename"]),
        weights_spec,
        "external weights",
    )

    val_spec = manifest["validation_artifact"]
    expected_validation = (root / str(val_spec["path"])).resolve()
    if validation_path != expected_validation:
        raise BundleVerificationError(
            f"unexpected validation artifact: expected {expected_validation}, got {validation_path}"
        )
    hashes["validation_artifact"] = _verify_file(
        validation_path,
        {"sha256": val_spec["sha256"], "size_bytes": val_spec["size_bytes"]},
        "validation artifact",
    )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    source_sha = str(manifest["training_source_sha"])
    if config.get("run_id") != run_id or validation.get("run_id") != run_id:
        raise BundleVerificationError("model/validation run ID does not match manifest")
    if config.get("source_sha") != source_sha or validation.get("source_sha") != source_sha:
        raise BundleVerificationError("model/validation source SHA does not match manifest")
    if config.get("winner") != validation.get("selection", {}).get("winner"):
        raise BundleVerificationError("model winner does not match validation winner")

    selection = validation.get("selection", {})
    for flag in ("val_gate_70pct", "baseline_gate_15pp", "proceed_to_frozen"):
        if selection.get(flag) is not True:
            raise BundleVerificationError(f"validation flag {flag} is not true")
    if validation.get("dataset_stats", {}).get("validation_groups") != 150:
        raise BundleVerificationError("validation question denominator is not 150")
    evaluations = validation.get("evaluations", [])
    if not evaluations or any(
        item.get("metrics", {}).get("total_gold_entities") != 182.0
        for item in evaluations
    ):
        raise BundleVerificationError("validation gold denominator is not 182")

    return VerifiedBundle(manifest, validation, config, hashes)
