"""Canonical internal RAG corpus freeze for Phase 4 controlled comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from nexus.baselines.retrieval import CorpusDocument


PROHIBITED_PATH_FRAGMENTS = (
    "/qa-dataset/",
    "\\qa-dataset\\",
    "oracle_v1.jsonl",
    "oracle_v1.manifest",
    "benchmarks/results/",
    "benchmarks\\results\\",
    "adjudication_packet",
    "adjudication_scores",
    "/hidden/",
    "\\hidden\\",
)

# Source docs only — exclude evaluation reports and gold datasets.
DEFAULT_SOURCE_GLOBS = (
    "sam-lm/experiments/*.md",
    "docs/production-profiles.md",
    "docs/stack-v1-freeze.md",
    "docs/CURRENT_STATE.md",
    "docs/external-evaluation-protocol.md",
    "STACK_RESULTS.md",
    "ANALYSIS_AND_ROADMAP.md",
    "README.md",
)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    start: int
    end: int
    heading: str = ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def assert_no_leakage(path: Path, text: str) -> None:
    lowered = path.as_posix().casefold()
    for frag in PROHIBITED_PATH_FRAGMENTS:
        if frag.casefold() in lowered:
            raise ValueError(f"prohibited corpus path: {path}")
    # Reject embedded gold-eval dumps inside non-doc paths
    rel = path.as_posix().casefold()
    if "qa-dataset" in rel and '"gold_answer"' in text:
        raise ValueError(f"possible gold leakage in {path}")


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[tuple[int, int, str]]:
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out


def build_canonical_corpus(
    root: Path,
    *,
    globs: Sequence[str] = DEFAULT_SOURCE_GLOBS,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    """Build and freeze the Phase-4 internal RAG corpus."""
    files: list[Path] = []
    for pattern in globs:
        files.extend(sorted(root.glob(pattern)))
    # Unique stable order
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in files:
        key = str(p.resolve())
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        uniq.append(p)

    file_entries = []
    chunks: list[ChunkRecord] = []
    for path in uniq:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        assert_no_leakage(path, text)
        rel = path.relative_to(root).as_posix()
        doc_id = _sha256_text(rel)[:16]
        file_entries.append(
            {
                "path": rel,
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
                "doc_id": doc_id,
            }
        )
        heading = path.stem
        for i, (start, end, piece) in enumerate(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunk_id = f"{doc_id}:{i:04d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=rel,
                    text=piece,
                    start=start,
                    end=end,
                    heading=heading,
                )
            )

    corpus_hasher = hashlib.sha256()
    for fe in file_entries:
        corpus_hasher.update(f"{fe['path']}:{fe['sha256']}\n".encode("utf-8"))
    chunk_hasher = hashlib.sha256()
    for ch in chunks:
        chunk_hasher.update(
            f"{ch.chunk_id}:{_sha256_text(ch.text)}\n".encode("utf-8")
        )

    return {
        "schema_version": "nexus-rag-corpus-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "globs": list(globs),
        "chunking": {
            "strategy": "char_window_v1",
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        "normalization": "utf-8 replace; CRLF->LF",
        "file_count": len(file_entries),
        "chunk_count": len(chunks),
        "files": file_entries,
        "corpus_sha256": corpus_hasher.hexdigest(),
        "chunks_sha256": chunk_hasher.hexdigest(),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "heading": c.heading,
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "text_sha256": _sha256_text(c.text),
            }
            for c in chunks
        ],
        "exclusions": list(PROHIBITED_PATH_FRAGMENTS),
        "note": (
            "Internal SAM/NEXUS development corpus for Phase 4 controlled RAG. "
            "Not a sealed external corpus."
        ),
    }


def documents_from_corpus(corpus: Mapping[str, Any]) -> list[CorpusDocument]:
    docs: list[CorpusDocument] = []
    for ch in corpus.get("chunks") or []:
        docs.append(
            CorpusDocument(
                doc_id=str(ch["chunk_id"]),
                text=str(ch["text"]),
                source=str(ch.get("source_path") or ""),
                metadata={
                    "doc_id": ch.get("doc_id"),
                    "heading": ch.get("heading"),
                    "start": ch.get("start"),
                    "end": ch.get("end"),
                },
            )
        )
    return docs


def format_evidence_blocks(
    hits: Sequence[dict[str, Any]], *, max_chars: int = 3500
) -> str:
    parts: list[str] = []
    used = 0
    for h in hits:
        block = (
            f"[{h.get('rank')}] id={h.get('doc_id')} score={h.get('score')}\n"
            f"{h.get('text', '')}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no evidence retrieved)"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k]
