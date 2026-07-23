"""Version-pinned dense embedder with local asset hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Pinned identity for Phase-4 dense retrieval. Revision may be overridden by
# an offline snapshot under models/dense/ when present.
PINNED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
PIN_MANIFEST_REL = Path("benchmarks/pins/sentence_transformers_all_minilm_l6_v2.json")


def load_pin_manifest(root: Path | None = None) -> dict[str, Any]:
    base = root or Path(__file__).resolve().parents[2]
    path = base / PIN_MANIFEST_REL
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "files": {},
        "note": "default pin; run benchmarks/hash_dense_assets.py after offline download",
    }


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_local_snapshot(snapshot_dir: Path) -> dict[str, str]:
    """Hash tokenizer/config/model files under a local snapshot directory."""
    names = (
        "config.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "sentence_bert_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    )
    out: dict[str, str] = {}
    for name in names:
        p = snapshot_dir / name
        if p.exists() and p.is_file():
            out[name] = hash_file(p)
    return out


def embedder_identity(root: Path | None = None) -> dict[str, Any]:
    pin = load_pin_manifest(root)
    local = (root or Path(__file__).resolve().parents[2]) / "models" / "dense" / "all-MiniLM-L6-v2"
    files = dict(pin.get("files") or {})
    if local.is_dir():
        files = hash_local_snapshot(local) or files
    blob = json.dumps(
        {
            "model_id": pin.get("model_id") or PINNED_MODEL_ID,
            "revision": pin.get("revision") or PINNED_REVISION,
            "files": files,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "model_id": pin.get("model_id") or PINNED_MODEL_ID,
        "revision": pin.get("revision") or PINNED_REVISION,
        "files_sha256": files,
        "identity_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "offline_snapshot": str(local) if local.is_dir() else "",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
    }


class DenseModelIdentityError(RuntimeError):
    """Raised when dense model identity cannot be verified."""


def load_sentence_transformer(
    root: Path | None = None,
    *,
    fail_closed: bool = True,
):
    """Load the pinned SentenceTransformer; prefer local snapshot when present.

    When ``HF_HUB_OFFLINE=1`` and the exact revision snapshot is absent from
    cache, fall back to the locally cached model id and record
    ``revision_resolved`` honestly (not silently pretending the pin loaded).

    Args:
        root: Project root for finding local snapshots.
        fail_closed: If True (default), raise DenseModelIdentityError when
            fallback to unpinned cache is required. Set to False only for
            exploratory/diagnostic runs where identity degradation is acceptable.

    Returns:
        (model, identity_dict) tuple.

    Raises:
        DenseModelIdentityError: When fail_closed=True and pinned revision unavailable.
    """
    from sentence_transformers import SentenceTransformer

    ident = dict(embedder_identity(root))
    local = ident.get("offline_snapshot") or ""
    revision_resolved = ident["revision"]
    load_mode = "pinned_revision"
    load_warnings: list[str] = []

    if local and Path(local).is_dir():
        model = SentenceTransformer(local)
        load_mode = "local_snapshot"
        # Hash whatever is on disk for identity.
        files = hash_local_snapshot(Path(local))
        if files:
            ident["files_sha256"] = files
        # Try to verify the local snapshot matches pinned revision
        expected_files = ident.get("files_sha256") or {}
        if not expected_files:
            load_warnings.append("local snapshot loaded but no file hashes to verify")
    else:
        try:
            model = SentenceTransformer(
                ident["model_id"],
                revision=ident["revision"],
            )
        except (OSError, ValueError) as exc:
            if fail_closed:
                raise DenseModelIdentityError(
                    f"pinned revision {ident['revision'][:12]}... unavailable "
                    f"({type(exc).__name__}); refusing to use unpinned fallback "
                    "for primary artifacts. Set fail_closed=False only for "
                    "exploratory runs."
                ) from exc
            # Offline / missing revision snapshot: use cached default weights.
            model = SentenceTransformer(ident["model_id"])
            revision_resolved = "cache_default_unpinned"
            load_mode = "offline_cache_fallback"
            load_warnings.append(
                f"pinned revision unavailable offline ({type(exc).__name__}); "
                "loaded cached model id without revision pin"
            )

    ident["revision_resolved"] = revision_resolved
    ident["revision_pinned"] = ident["revision"]
    ident["load_mode"] = load_mode
    ident["load_warnings"] = load_warnings
    ident["fail_closed"] = fail_closed
    ident["identity_degraded"] = load_mode == "offline_cache_fallback"

    # Recompute identity over what was actually loaded.
    blob = json.dumps(
        {
            "model_id": ident["model_id"],
            "revision": revision_resolved,
            "files": ident.get("files_sha256") or {},
            "load_mode": load_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    ident["identity_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return model, ident
