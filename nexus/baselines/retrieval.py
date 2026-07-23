"""Local retrieval baselines (BM25 first) for controlled RAG evaluation.

Retrieval-only evaluation is valid without an answer-generator LLM. Final-answer
RAG arms remain NOT_RUN until credentials and a pinned answer model exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class CorpusDocument:
    doc_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25Index:
    """Minimal Okapi BM25 index (no external dependency)."""

    documents: list[CorpusDocument]
    k1: float = 1.5
    b: float = 0.75
    corpus_id: str = ""
    chunking: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._docs_tokens = [tokenize(d.text) for d in self.documents]
        self._doc_len = [len(toks) or 1 for toks in self._docs_tokens]
        self._avgdl = (
            sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        )
        df: Counter[str] = Counter()
        for toks in self._docs_tokens:
            df.update(set(toks))
        n = len(self.documents) or 1
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        if not self.corpus_id:
            payload = json.dumps(
                [{"id": d.doc_id, "text": d.text} for d in self.documents],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self.corpus_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def index_id(self) -> str:
        identity = {
            "corpus_id": self.corpus_id,
            "k1": self.k1,
            "b": self.b,
            "n_docs": len(self.documents),
            "chunking": self.chunking,
            "algorithm": "okapi_bm25_local_v1",
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_terms = tokenize(query)
        scores: list[tuple[float, int]] = []
        for i, toks in enumerate(self._docs_tokens):
            tf = Counter(toks)
            score = 0.0
            dl = self._doc_len[i]
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1.0) / (denom or 1.0))
            scores.append((score, i))
        scores.sort(reverse=True)
        out: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(scores[: max(0, top_k)], start=1):
            doc = self.documents[idx]
            out.append(
                {
                    "rank": rank,
                    "doc_id": doc.doc_id,
                    "score": round(score, 6),
                    "source": doc.source,
                    "text": doc.text,
                }
            )
        return out


def corpus_from_graph_nodes(graph: Any, *, max_chars: int = 2000) -> list[CorpusDocument]:
    """Build a simple document corpus from graph node properties."""
    docs: list[CorpusDocument] = []
    nodes = getattr(graph, "_nodes", {}) or {}
    for node_id, node in nodes.items():
        props = getattr(node, "properties", {}) or {}
        aliases = getattr(node, "aliases", None) or []
        chunks = [
            str(node_id),
            " ".join(str(a) for a in aliases),
            str(props.get("summary") or props.get("description") or ""),
            str(props.get("finding") or props.get("key_finding") or ""),
        ]
        text = " | ".join(c for c in chunks if c.strip())[:max_chars]
        if not text.strip():
            continue
        docs.append(
            CorpusDocument(
                doc_id=str(node_id),
                text=text,
                source=str(getattr(node, "sources", [""])[0] if getattr(node, "sources", None) else ""),
                metadata={"type": getattr(node, "type", "")},
            )
        )
    return docs


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float | None:
    if not relevant_ids:
        return None
    top = set(retrieved_ids[:k])
    hits = sum(1 for r in relevant_ids if r in top)
    return hits / len(relevant_ids)


def mrr(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str]
) -> float | None:
    if not relevant_ids:
        return None
    rel = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def run_bm25_retrieval_eval(
    questions: Sequence[Mapping[str, Any]],
    documents: Sequence[CorpusDocument],
    *,
    dataset_id: str,
    top_k: int = 5,
    comparison_mode: str = "controlled",
    source_commit: str = "UNKNOWN",
) -> dict[str, Any]:
    """Execute BM25 retrieval-only evaluation (no answer generation)."""
    index = BM25Index(
        documents=list(documents),
        chunking={"strategy": "graph_node_property_concat_v1", "max_chars": 2000},
    )
    executed_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    for q in questions:
        qid = str(q.get("id") or "")
        question = str(q.get("question") or "")
        relevant = [
            str(x)
            for x in (
                q.get("gold_entities")
                or q.get("relevant_doc_ids")
                or q.get("gold_doc_ids")
                or []
            )
        ]
        hits = index.search(question, top_k=top_k)
        retrieved_ids = [h["doc_id"] for h in hits]
        r_at_k = recall_at_k(retrieved_ids, relevant, top_k)
        rr = mrr(retrieved_ids, relevant)
        if r_at_k is not None:
            recalls.append(r_at_k)
        if rr is not None:
            mrrs.append(rr)
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "retrieval_method": "bm25",
                "top_k": top_k,
                "retrieved": hits,
                "retrieved_doc_ids": retrieved_ids,
                "relevant_doc_ids": relevant,
                "metrics": {
                    "recall_at_k": {
                        "applicable": r_at_k is not None,
                        "value": r_at_k,
                        "k": top_k,
                        "numerator": None
                        if r_at_k is None
                        else r_at_k * max(len(relevant), 1),
                        "denominator": None if r_at_k is None else float(len(relevant)),
                        "reason": "no_gold_entities"
                        if r_at_k is None
                        else "gold_entities_as_relevant_docs",
                    },
                    "mrr": {
                        "applicable": rr is not None,
                        "value": rr,
                        "reason": "no_gold_entities"
                        if rr is None
                        else "gold_entities_as_relevant_docs",
                    },
                },
                "answer_generation": {
                    "status": "NOT_RUN",
                    "reason": (
                        "BM25 retrieval-only arm; answer generation requires "
                        "pinned NEXUS_LLM_MODEL + API credentials and budget"
                    ),
                },
                "terminal_outcome": "not_run",
                "comparison_mode": comparison_mode,
            }
        )

    return {
        "schema_version": "nexus-retrieval-eval-v1",
        "created_utc": executed_at,
        "system_id": "bm25_retrieval_only",
        "family": "rag_retrieval",
        "modern_rag": False,
        "comparison_mode": comparison_mode,
        "dataset_id": dataset_id,
        "dataset_sha256": hashlib.sha256(
            json.dumps(
                [{"id": q.get("id"), "question": q.get("question")} for q in questions],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "corpus_id": index.corpus_id,
        "index_id": index.index_id,
        "chunking": index.chunking,
        "embedding_model": None,
        "reranker": None,
        "fusion_method": None,
        "top_k": top_k,
        "source_commit": source_commit,
        "answer_generation_status": "NOT_RUN",
        "per_question": rows,
        "aggregates": {
            "questions_total": len(rows),
            "recall_at_k_mean": None
            if not recalls
            else round(sum(recalls) / len(recalls), 6),
            "recall_at_k_n": len(recalls),
            "mrr_mean": None if not mrrs else round(sum(mrrs) / len(mrrs), 6),
            "mrr_n": len(mrrs),
        },
        "status": "OK_RETRIEVAL_ONLY",
        "note": (
            "Do not compare retrieval-only scores against NEXUS answer "
            "correctness. Final-answer RAG remains NOT_RUN."
        ),
    }
