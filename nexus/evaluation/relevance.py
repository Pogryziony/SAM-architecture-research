"""Gold entity/fact → document-chunk relevance mapping for retrieval metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


@dataclass(frozen=True)
class RelevanceMapping:
    question_id: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    method: str
    notes: str = ""
    confidence: str = "heuristic"  # explicit, heuristic, audited


@dataclass
class RelevanceAuditEntry:
    """Record for stratified auditing of relevance mappings."""
    question_id: str
    gold_answer: str
    gold_entities: list[str]
    chunk_id: str
    chunk_text_preview: str
    heuristic_relevant: bool
    auditor_relevant: bool | None = None
    auditor_notes: str = ""


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN_RE.findall(text or "")}


def build_relevance_for_question(
    question: Mapping[str, Any],
    corpus_chunks: Sequence[Mapping[str, Any]],
    *,
    min_token_hits: int = 2,  # Tightened from 1 to reduce over-matching
    min_answer_hits: int = 3,  # Tightened from 2 for stricter relevance
    max_relevant_chunks: int = 50,  # Cap to avoid near-all-corpus relevance
) -> RelevanceMapping:
    """Map gold entities / answer tokens onto corpus chunk IDs.

    Explicit ``relevant_chunk_ids`` / ``relevant_doc_ids`` on the question win.
    Otherwise match gold entity id tokens and distinctive gold-answer tokens
    against chunk text and source paths.

    Tightened heuristic (v2):
    - Requires at least 2 entity token hits (not 1)
    - Requires at least 3 distinctive answer token hits (not 2)
    - Caps relevant chunks to avoid near-all-corpus relevance
    - Records confidence level for downstream auditing
    """
    qid = str(question.get("id") or question.get("question_id") or "")
    if question.get("relevant_chunk_ids"):
        chunks = tuple(str(x) for x in question["relevant_chunk_ids"])
        docs = tuple(str(x) for x in (question.get("relevant_doc_ids") or []))
        return RelevanceMapping(qid, chunks, docs, "explicit_question_fields", confidence="explicit")

    gold_entities = [str(x) for x in (question.get("gold_entities") or [])]
    gold_answer = str(question.get("gold_answer") or "")
    entity_tokens: set[str] = set()
    for ent in gold_entities:
        entity_tokens |= _tokens(ent.replace("_", " "))
        entity_tokens.add(ent.casefold())

    # Distinctive answer tokens (length > 4 for tighter matching) excluding stop words
    stop = {
        "that", "this", "with", "from", "have", "been", "were", "their", "about",
        "which", "would", "could", "should", "there", "where", "what", "when",
        "does", "more", "most", "than", "then", "some", "into", "also", "other",
    }
    answer_tokens = {t for t in _tokens(gold_answer) if len(t) > 4 and t not in stop}

    # Score chunks and rank by relevance strength
    scored_chunks: list[tuple[str, str, int, int]] = []  # (cid, did, entity_score, answer_score)
    for ch in corpus_chunks:
        cid = str(ch.get("chunk_id") or "")
        did = str(ch.get("doc_id") or "")
        text = str(ch.get("text") or "")
        source = str(ch.get("source_path") or "")
        blob_tokens = _tokens(text) | _tokens(source) | {did.casefold(), cid.casefold()}
        entity_hits = len(entity_tokens & blob_tokens)
        answer_hits = len(answer_tokens & blob_tokens)

        # Tightened thresholds
        if entity_hits >= min_token_hits or answer_hits >= min_answer_hits:
            scored_chunks.append((cid, did, entity_hits, answer_hits))

    # Sort by relevance score (entity hits primary, answer hits secondary)
    scored_chunks.sort(key=lambda x: (x[2], x[3]), reverse=True)

    # Cap the number of relevant chunks
    hit_chunks: list[str] = []
    hit_docs: list[str] = []
    for cid, did, _, _ in scored_chunks[:max_relevant_chunks]:
        hit_chunks.append(cid)
        if did and did not in hit_docs:
            hit_docs.append(did)

    confidence = "heuristic"
    if len(scored_chunks) > max_relevant_chunks:
        # Warn if we had to cap
        confidence = "heuristic_capped"

    return RelevanceMapping(
        question_id=qid,
        relevant_chunk_ids=tuple(hit_chunks),
        relevant_doc_ids=tuple(hit_docs),
        method="gold_entity_answer_token_overlap_v2",
        notes=(
            f"entities={len(gold_entities)} entity_tokens={len(entity_tokens)} "
            f"answer_tokens={len(answer_tokens)} matched_chunks={len(hit_chunks)} "
            f"total_candidates={len(scored_chunks)}"
        ),
        confidence=confidence,
    )


def build_relevance_table(
    questions: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    chunks = list(corpus.get("chunks") or [])
    mappings = [build_relevance_for_question(q, chunks) for q in questions]
    rows = [
        {
            "question_id": m.question_id,
            "relevant_chunk_ids": list(m.relevant_chunk_ids),
            "relevant_doc_ids": list(m.relevant_doc_ids),
            "method": m.method,
            "notes": m.notes,
            "confidence": m.confidence,
        }
        for m in mappings
    ]

    # Compute statistics for audit reporting
    nonzero = sum(1 for r in rows if r["relevant_chunk_ids"])
    chunk_counts = [len(r["relevant_chunk_ids"]) for r in rows]
    median_chunks = sorted(chunk_counts)[len(chunk_counts) // 2] if chunk_counts else 0
    max_chunks = max(chunk_counts) if chunk_counts else 0
    capped_count = sum(1 for r in rows if r.get("confidence") == "heuristic_capped")

    return {
        "schema_version": "nexus-retrieval-relevance-v2",
        "questions_total": len(rows),
        "questions_with_relevant_chunks": nonzero,
        "corpus_sha256": corpus.get("corpus_sha256"),
        "relevance_statistics": {
            "median_relevant_chunks": median_chunks,
            "max_relevant_chunks": max_chunks,
            "capped_questions": capped_count,
            "method": "gold_entity_answer_token_overlap_v2",
            "note": "Tightened heuristic; capped at 50 chunks per question",
        },
        "rows": rows,
    }


def generate_audit_sample(
    relevance_table: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    *,
    sample_size: int = 30,
    stratify_by_confidence: bool = True,
) -> dict[str, Any]:
    """Generate a stratified sample for human auditing of relevance labels.

    Returns a packet with question-chunk pairs for manual verification.
    """
    import random

    chunks_by_id = {
        str(ch.get("chunk_id") or ""): ch
        for ch in (corpus.get("chunks") or [])
    }
    questions_by_id = {
        str(q.get("id") or q.get("question_id") or ""): q
        for q in questions
    }

    rows = relevance_table.get("rows") or []

    # Stratify sample
    if stratify_by_confidence:
        explicit = [r for r in rows if r.get("confidence") == "explicit"]
        heuristic = [r for r in rows if r.get("confidence") in ("heuristic", "heuristic_capped")]
        # Sample proportionally
        n_explicit = min(len(explicit), sample_size // 3)
        n_heuristic = min(len(heuristic), sample_size - n_explicit)
        sample_rows = (
            random.sample(explicit, n_explicit) +
            random.sample(heuristic, n_heuristic)
        )
    else:
        sample_rows = random.sample(rows, min(len(rows), sample_size))

    # Build audit entries
    entries = []
    for row in sample_rows:
        qid = row["question_id"]
        q = questions_by_id.get(qid, {})
        relevant_chunks = row.get("relevant_chunk_ids", [])[:5]  # Max 5 per question

        for cid in relevant_chunks:
            chunk = chunks_by_id.get(cid, {})
            entries.append({
                "question_id": qid,
                "question": q.get("question", ""),
                "gold_answer": q.get("gold_answer", ""),
                "gold_entities": q.get("gold_entities", []),
                "chunk_id": cid,
                "chunk_text_preview": str(chunk.get("text", ""))[:500],
                "heuristic_relevant": True,
                "auditor_relevant": None,
                "auditor_notes": "",
            })

    return {
        "schema_version": "nexus-relevance-audit-v1",
        "sample_size": len(entries),
        "stratified": stratify_by_confidence,
        "entries": entries,
        "instructions": (
            "For each entry, verify whether the chunk is truly relevant to answering "
            "the question. Set auditor_relevant to true/false and add notes if needed."
        ),
    }


def compute_relevance_precision(
    audit_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute precision metrics from completed audit."""
    entries = audit_results.get("entries") or []
    audited = [e for e in entries if e.get("auditor_relevant") is not None]

    if not audited:
        return {
            "precision": None,
            "n_audited": 0,
            "reason": "no audited entries",
        }

    true_positive = sum(1 for e in audited if e["heuristic_relevant"] and e["auditor_relevant"])
    false_positive = sum(1 for e in audited if e["heuristic_relevant"] and not e["auditor_relevant"])

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else None

    return {
        "precision": precision,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "n_audited": len(audited),
        "note": "Precision = TP / (TP + FP) for heuristic-labeled relevant chunks",
    }


def load_or_build_relevance(
    path: Path,
    questions: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    table = build_relevance_table(questions, corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return table


def relevant_chunks_for(
    table: Mapping[str, Any], question_id: str
) -> list[str]:
    for row in table.get("rows") or []:
        if str(row.get("question_id")) == question_id:
            return [str(x) for x in (row.get("relevant_chunk_ids") or [])]
    return []
