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


def load_sentence_transformer(root: Path | None = None):
    """Load the pinned SentenceTransformer; prefer local snapshot when present."""
    from sentence_transformers import SentenceTransformer

    ident = embedder_identity(root)
    local = ident.get("offline_snapshot") or ""
    if local and Path(local).is_dir():
        model = SentenceTransformer(local)
    else:
        model = SentenceTransformer(
            ident["model_id"],
            revision=ident["revision"],
        )
    return model, ident
