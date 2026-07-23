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


def _hf_hub_snapshot(model_id: str, revision: str) -> Path | None:
    """Return the local HF hub snapshot dir for ``model_id``@``revision`` if present."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    pinned = hub / slug / "snapshots" / revision
    if pinned.is_dir():
        return pinned
    return None


def _any_hf_hub_snapshot(model_id: str) -> Path | None:
    """Return any cached snapshot for ``model_id`` (prefer newest mtime)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    snaps = hub / slug / "snapshots"
    if not snaps.is_dir():
        return None
    dirs = [p for p in snaps.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


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
            When a different HF cache snapshot is used, identity is marked
            degraded but load still succeeds so fields stay truthful.

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
    identity_degraded = False

    if local and Path(local).is_dir():
        model = SentenceTransformer(local)
        load_mode = "local_snapshot"
        files = hash_local_snapshot(Path(local))
        if files:
            ident["files_sha256"] = files
        pin_files = dict(load_pin_manifest(root).get("files") or {})
        if pin_files and files == pin_files:
            revision_resolved = ident["revision"]
        else:
            # Do not claim the pin when local bytes are unverified / different.
            revision_resolved = f"local_snapshot:{hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]}"
            identity_degraded = True
            load_warnings.append(
                "local snapshot loaded without matching pin file hashes; "
                "revision_resolved is a content fingerprint, not the pin"
            )
    else:
        pinned_snap = _hf_hub_snapshot(ident["model_id"], ident["revision"])
        if pinned_snap is not None:
            model = SentenceTransformer(str(pinned_snap))
            load_mode = "hf_hub_pinned_snapshot"
            revision_resolved = ident["revision"]
            files = hash_local_snapshot(pinned_snap)
            if files:
                ident["files_sha256"] = files
        else:
            try:
                model = SentenceTransformer(
                    ident["model_id"],
                    revision=ident["revision"],
                )
                load_mode = "pinned_revision"
                revision_resolved = ident["revision"]
            except (OSError, ValueError, RuntimeError) as exc:
                alt = _any_hf_hub_snapshot(ident["model_id"])
                if alt is not None:
                    model = SentenceTransformer(str(alt))
                    load_mode = "hf_hub_unpinned_snapshot"
                    revision_resolved = alt.name
                    identity_degraded = True
                    load_warnings.append(
                        f"pinned revision {ident['revision'][:12]}... unavailable "
                        f"({type(exc).__name__}); loaded HF cache snapshot {alt.name}"
                    )
                    files = hash_local_snapshot(alt)
                    if files:
                        ident["files_sha256"] = files
                elif fail_closed:
                    raise DenseModelIdentityError(
                        f"pinned revision {ident['revision'][:12]}... unavailable "
                        f"({type(exc).__name__}); refusing to use unpinned fallback "
                        "for primary artifacts. Set fail_closed=False only for "
                        "exploratory runs."
                    ) from exc
                else:
                    # Offline / missing revision snapshot: use cached default weights.
                    model = SentenceTransformer(ident["model_id"])
                    revision_resolved = "cache_default_unpinned"
                    load_mode = "offline_cache_fallback"
                    identity_degraded = True
                    load_warnings.append(
                        f"pinned revision unavailable offline ({type(exc).__name__}); "
                        "loaded cached model id without revision pin"
                    )

    ident["revision_resolved"] = revision_resolved
    ident["revision_pinned"] = ident["revision"]
    ident["load_mode"] = load_mode
    ident["load_warnings"] = load_warnings
    ident["fail_closed"] = fail_closed
    ident["identity_degraded"] = identity_degraded or load_mode in {
        "offline_cache_fallback",
        "hf_hub_unpinned_snapshot",
    }

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
