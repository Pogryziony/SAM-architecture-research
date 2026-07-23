"""Gold entity/fact → document-chunk relevance mapping for retrieval metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN_RE.findall(text or "")}


def build_relevance_for_question(
    question: Mapping[str, Any],
    corpus_chunks: Sequence[Mapping[str, Any]],
    *,
    min_token_hits: int = 1,
) -> RelevanceMapping:
    """Map gold entities / answer tokens onto corpus chunk IDs.

    Explicit ``relevant_chunk_ids`` / ``relevant_doc_ids`` on the question win.
    Otherwise match gold entity id tokens and distinctive gold-answer tokens
    against chunk text and source paths.
    """
    qid = str(question.get("id") or question.get("question_id") or "")
    if question.get("relevant_chunk_ids"):
        chunks = tuple(str(x) for x in question["relevant_chunk_ids"])
        docs = tuple(str(x) for x in (question.get("relevant_doc_ids") or []))
        return RelevanceMapping(qid, chunks, docs, "explicit_question_fields")

    gold_entities = [str(x) for x in (question.get("gold_entities") or [])]
    gold_answer = str(question.get("gold_answer") or "")
    entity_tokens: set[str] = set()
    for ent in gold_entities:
        entity_tokens |= _tokens(ent.replace("_", " "))
        entity_tokens.add(ent.casefold())
    # Distinctive answer tokens (length > 3) excluding stop-ish words
    stop = {"that", "this", "with", "from", "have", "been", "were", "their", "about"}
    answer_tokens = {t for t in _tokens(gold_answer) if len(t) > 3 and t not in stop}

    hit_chunks: list[str] = []
    hit_docs: list[str] = []
    for ch in corpus_chunks:
        cid = str(ch.get("chunk_id") or "")
        did = str(ch.get("doc_id") or "")
        text = str(ch.get("text") or "")
        source = str(ch.get("source_path") or "")
        blob_tokens = _tokens(text) | _tokens(source) | {did.casefold(), cid.casefold()}
        entity_hits = len(entity_tokens & blob_tokens)
        answer_hits = len(answer_tokens & blob_tokens)
        # Prefer entity id / alias hits; require stronger answer overlap alone
        if entity_hits >= min_token_hits or answer_hits >= max(2, min_token_hits + 1):
            hit_chunks.append(cid)
            if did and did not in hit_docs:
                hit_docs.append(did)

    return RelevanceMapping(
        question_id=qid,
        relevant_chunk_ids=tuple(hit_chunks),
        relevant_doc_ids=tuple(hit_docs),
        method="gold_entity_answer_token_overlap_v1",
        notes=(
            f"entities={len(gold_entities)} entity_tokens={len(entity_tokens)} "
            f"matched_chunks={len(hit_chunks)}"
        ),
    )


def build_relevance_table(
    questions: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    chunks = list(corpus.get("chunks") or [])
    rows = [
        {
            "question_id": m.question_id,
            "relevant_chunk_ids": list(m.relevant_chunk_ids),
            "relevant_doc_ids": list(m.relevant_doc_ids),
            "method": m.method,
            "notes": m.notes,
        }
        for m in (build_relevance_for_question(q, chunks) for q in questions)
    ]
    nonzero = sum(1 for r in rows if r["relevant_chunk_ids"])
    return {
        "schema_version": "nexus-retrieval-relevance-v1",
        "questions_total": len(rows),
        "questions_with_relevant_chunks": nonzero,
        "corpus_sha256": corpus.get("corpus_sha256"),
        "rows": rows,
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
