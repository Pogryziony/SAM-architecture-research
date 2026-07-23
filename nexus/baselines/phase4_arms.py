"""Phase 4 controlled/system-level arms using local Qwen 3.6."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from nexus.baselines.local_qwen import (
    FROZEN_CLOSED_BOOK_USER_TEMPLATE,
    FROZEN_EVIDENCE_USER_TEMPLATE,
    FROZEN_SYSTEM_PROMPT,
    LocalQwenAdapter,
    LocalQwenIdentity,
    discover_local_qwen,
)
from nexus.baselines.rag_corpus import (
    documents_from_corpus,
    format_evidence_blocks,
    reciprocal_rank_fusion,
)
from nexus.baselines.retrieval import BM25Index, mrr, recall_at_k
from nexus.evaluation.aggregate import aggregate_question_records
from nexus.evaluation.dataset_identity import hash_dataset
from nexus.evaluation.metrics import compute_proxy_key_fact_correct
from nexus.evaluation.relevance import relevant_chunks_for
from nexus.evaluation.schema import RESULT_SCHEMA_VERSION, TerminalOutcome, build_question_record
from nexus.evaluation.validate import assert_valid_result_artifact
from nexus.pipeline.config import CONFIG_IDENTITY_SCHEMA, ProductionNEXUSConfig


def _dataset_hash(questions: Sequence[Mapping[str, Any]]) -> str:
    """Full canonical dataset identity (gold/rubric fields included)."""
    return hash_dataset(questions)


def _prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    blob = (system_prompt + "\n\n" + user_prompt).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _terminal_from_answer(answer: str, *, error: str = "", timed_out: bool = False) -> TerminalOutcome:
    if timed_out:
        return TerminalOutcome.TIMED_OUT
    if error:
        return TerminalOutcome.FAILED
    if not answer.strip() or "insufficient" in answer.casefold():
        return TerminalOutcome.ABSTAINED
    return TerminalOutcome.ANSWERED


def _score_row(
    q: Mapping[str, Any],
    answer: str,
) -> dict[str, Any]:
    """Exploratory proxy scoring only — never primary grounded_correct."""
    should_abstain = bool(q.get("should_abstain", False))
    gold = str(q.get("gold_answer") or "")
    answer_correct = None
    try:
        from benchmarks.scoring import compute_fact_score

        fuzzy = float(compute_fact_score(answer, gold).get("fuzzy_accuracy") or 0.0)
        answer_correct = fuzzy >= 0.5
    except Exception:
        answer_correct = bool(gold) and gold.casefold() in answer.casefold()
    proxy = compute_proxy_key_fact_correct(
        answer=answer,
        gold_answer=gold,
        should_abstain=should_abstain,
        answer_correct=answer_correct,
    )
    return {
        "grounded_correct": {
            "name": "grounded_correct",
            "applicable": False,
            "value": None,
            "numerator": None,
            "denominator": 1.0,
            "reason": "pending_adjudication",
        },
        "proxy_key_fact_correct": proxy.to_dict(),
        "fact_fuzzy_accuracy": {
            "applicable": answer_correct is not None,
            "value": None if answer_correct is None else (1.0 if answer_correct else 0.0),
            "numerator": None if answer_correct is None else (1.0 if answer_correct else 0.0),
            "denominator": 1.0,
            "reason": "exploratory_key_fact_fuzzy_not_grounded_correct",
        },
    }


def _empty_metrics() -> dict[str, Any]:
    from nexus.evaluation.schema import empty_metric_applicability

    return empty_metric_applicability()


def _snapshot_resources() -> dict[str, Any]:
    """Capture current process-tree and LLM server resources."""
    try:
        from nexus.evaluation.process_resources import snapshot_llm_server_resources
        return snapshot_llm_server_resources()
    except Exception as exc:
        return {
            "schema_version": "nexus-llm-server-resources-v1",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_eval_artifact(
    *,
    system_id: str,
    profile: str,
    questions: Sequence[Mapping[str, Any]],
    rows: list[dict[str, Any]],
    comparison_mode: str,
    source_commit: str,
    config_hash: str,
    qwen_identity: Mapping[str, Any] | None,
    arm_metadata: Mapping[str, Any],
    status: str,
    arm_decoding_overrides: Mapping[str, Any] | None = None,
    dense_identity: Mapping[str, Any] | None = None,
    resource_snapshots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ds_hash = _dataset_hash(questions)

    # Collect resource snapshot if not provided
    resources = dict(resource_snapshots or {})
    if not resources:
        resources = {
            "end_of_run": _snapshot_resources(),
        }

    # Compute throughput metrics from rows
    latencies = [r.get("latency_ms") for r in rows if r.get("latency_ms")]
    total_latency_ms = sum(latencies) if latencies else 0
    throughput_qps = len(rows) / (total_latency_ms / 1000) if total_latency_ms > 0 else None

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "dataset_id": "oracle_v1",
        "dataset_sha256": ds_hash,
        "dataset_hash_schema": "nexus-canonical-dataset-v1",
        "system_id": system_id,
        "profile": profile,
        "config_hash": config_hash,
        "config_identity_schema": CONFIG_IDENTITY_SCHEMA,
        "comparison_mode": comparison_mode,
        "questions_total": len(rows),
        "per_question": rows,
        "aggregates": aggregate_question_records(rows),
        "status": status,
        "local_qwen_identity": dict(qwen_identity or {}),
        "dense_embedder_identity": dict(dense_identity or {}),
        "arm_decoding_overrides": dict(arm_decoding_overrides or {}),
        "arm_metadata": dict(arm_metadata),
        "adjudication_status": "PENDING_ADJUDICATION",
        "claim_eligibility": {
            "full_primary_superiority": False,
            "reason": (
                "human adjudication incomplete; proxy_key_fact_correct is "
                "exploratory only and must not be quoted as grounded_correct"
            ),
            "exploratory_auto_subset_ok": True,
        },
        "resource_usage": {
            **resources,
            "throughput_questions_per_second": throughput_qps,
            "total_latency_ms": total_latency_ms,
        },
    }
    assert_valid_result_artifact(artifact)
    return artifact


def run_closed_book_qwen(
    questions: Sequence[Mapping[str, Any]],
    adapter: LocalQwenAdapter,
    *,
    source_commit: str = "UNKNOWN",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    identity = adapter.identity.to_dict()
    rows: list[dict[str, Any]] = []
    executed_at = datetime.now(timezone.utc).isoformat()
    for i, q in enumerate(questions):
        qid = str(q["id"])
        if on_progress:
            on_progress(i + 1, len(questions), qid)
        user = FROZEN_CLOSED_BOOK_USER_TEMPLATE.format(question=str(q["question"]))
        gen = adapter.generate(user)
        outcome = _terminal_from_answer(
            gen.parsed_answer, error=gen.error, timed_out=gen.timed_out
        )
        metrics = _empty_metrics()
        if outcome in {TerminalOutcome.ANSWERED, TerminalOutcome.ABSTAINED}:
            metrics.update(_score_row(q, gen.parsed_answer))
        prompt_hash = _prompt_sha256(FROZEN_SYSTEM_PROMPT, user)
        rows.append(
            build_question_record(
                question_id=qid,
                domain=str(q.get("domain") or "sam"),
                question_type=str(q.get("category") or q.get("question_type") or "unknown"),
                dataset_id="oracle_v1",
                dataset_sha256=_dataset_hash(questions),
                system_id="qwen_3_6_closed_book_internal",
                profile="qwen_3_6_closed_book_internal",
                config_hash=identity["identity_hash"],
                config_identity_schema=CONFIG_IDENTITY_SCHEMA,
                model_id=identity["model_id"],
                checkpoint_id=identity["digest"],
                source_commit=source_commit,
                executed_at_utc=executed_at,
                terminal_outcome=outcome,
                question=str(q["question"]),
                final_answer=gen.parsed_answer,
                abstention=outcome is TerminalOutcome.ABSTAINED,
                metrics=metrics,
                latency_ms=gen.latency_ms,
                token_cost={
                    "prompt_eval_count": gen.prompt_eval_count,
                    "eval_count": gen.eval_count,
                    "tokens_per_second": gen.tokens_per_second,
                    "api_usd": 0.0,
                },
                execution_environment={
                    "raw_response": gen.raw_response,
                    "prompt": gen.prompt,
                    "system_prompt": gen.system_prompt,
                    "prompt_sha256": prompt_hash,
                    "prompt_eval_duration_ms": gen.time_to_first_token_ms,
                    "ttft_metric": "prompt_eval_duration_ms_nonstream_proxy",
                    "load_duration_ns": gen.load_duration_ns,
                    "error": gen.error,
                    "arm": "qwen_3_6_closed_book_internal",
                    "comparison_mode": "system_level",
                },
                comparison_mode="system_level",
                failure_category=(
                    "timed_out"
                    if gen.timed_out
                    else ("exception:LocalQwenError" if gen.error else "")
                ),
                diagnostic_message=gen.error,
            ).to_dict()
        )
    return build_eval_artifact(
        system_id="qwen_3_6_closed_book_internal",
        profile="qwen_3_6_closed_book_internal",
        questions=questions,
        rows=rows,
        comparison_mode="system_level",
        source_commit=source_commit,
        config_hash=identity["identity_hash"],
        qwen_identity=identity,
        arm_metadata={
            "family": "llm",
            "modern_rag": False,
            "is_placeholder": False,
            "label": "local Qwen 3.6 closed-book internal oracle only",
        },
        status="VALID",
    )


def run_rag_answer_arm(
    *,
    arm_id: str,
    questions: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    adapter: LocalQwenAdapter,
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    comparison_mode: str = "controlled",
    source_commit: str = "UNKNOWN",
    extra_meta: Mapping[str, Any] | None = None,
    relevance_table: Mapping[str, Any] | None = None,
    dense_identity: Mapping[str, Any] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    identity = adapter.identity.to_dict()
    rows: list[dict[str, Any]] = []
    executed_at = datetime.now(timezone.utc).isoformat()
    retrieval_rows: list[dict[str, Any]] = []
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "arm": arm_id,
                "qwen": identity["identity_hash"],
                "qwen_digest": identity.get("digest"),
                "corpus": corpus.get("corpus_sha256"),
                "chunks": corpus.get("chunks_sha256"),
                "extra": dict(extra_meta or {}),
                "system_prompt_sha256": hashlib.sha256(
                    FROZEN_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    for i, q in enumerate(questions):
        qid = str(q["id"])
        if on_progress:
            on_progress(i + 1, len(questions), qid)
        t_ret0 = time.perf_counter()
        hits = retrieve_fn(str(q["question"]))
        ret_ms = round((time.perf_counter() - t_ret0) * 1000, 3)
        evidence = format_evidence_blocks(hits)
        user = FROZEN_EVIDENCE_USER_TEMPLATE.format(
            question=str(q["question"]), evidence=evidence
        )
        gen = adapter.generate(user)
        outcome = _terminal_from_answer(
            gen.parsed_answer, error=gen.error, timed_out=gen.timed_out
        )
        metrics = _empty_metrics()
        if outcome in {TerminalOutcome.ANSWERED, TerminalOutcome.ABSTAINED}:
            metrics.update(_score_row(q, gen.parsed_answer))
        retrieved_ids = [str(h["doc_id"]) for h in hits]
        if relevance_table is not None:
            relevant = relevant_chunks_for(relevance_table, qid)
            relevance_reason = "gold_entity_fact_to_chunk_relevance_v1"
        else:
            relevant = [str(x) for x in (q.get("relevant_chunk_ids") or [])]
            relevance_reason = (
                "explicit_relevant_chunk_ids"
                if relevant
                else "missing_relevance_table_metrics_not_comparable"
            )
        r_at_k = recall_at_k(retrieved_ids, relevant, len(hits) or 5) if relevant else None
        rr = mrr(retrieved_ids, relevant) if relevant else None
        retrieval_rows.append(
            {
                "question_id": qid,
                "retrieved": hits,
                "relevant_chunk_ids": relevant,
                "recall_at_k": r_at_k,
                "mrr": rr,
                "retrieval_latency_ms": ret_ms,
            }
        )
        metrics["retrieval_recall_at_k"] = {
            "applicable": r_at_k is not None,
            "value": r_at_k,
            "numerator": None if r_at_k is None else r_at_k,
            "denominator": 1.0 if r_at_k is not None else None,
            "reason": relevance_reason,
        }
        metrics["retrieval_mrr"] = {
            "applicable": rr is not None,
            "value": rr,
            "numerator": None if rr is None else rr,
            "denominator": 1.0 if rr is not None else None,
            "reason": relevance_reason,
        }
        rows.append(
            build_question_record(
                question_id=qid,
                domain=str(q.get("domain") or "sam"),
                question_type=str(q.get("category") or "unknown"),
                dataset_id="oracle_v1",
                dataset_sha256=_dataset_hash(questions),
                system_id=arm_id,
                profile=arm_id,
                config_hash=config_hash,
                config_identity_schema=CONFIG_IDENTITY_SCHEMA,
                model_id=identity["model_id"],
                checkpoint_id=identity["digest"],
                source_commit=source_commit,
                executed_at_utc=executed_at,
                terminal_outcome=outcome,
                question=str(q["question"]),
                final_answer=gen.parsed_answer,
                citations=retrieved_ids,
                retrieved_documents=retrieved_ids,
                abstention=outcome is TerminalOutcome.ABSTAINED,
                metrics=metrics,
                latency_ms=round((gen.latency_ms or 0) + ret_ms, 3),
                token_cost={
                    "prompt_eval_count": gen.prompt_eval_count,
                    "eval_count": gen.eval_count,
                    "tokens_per_second": gen.tokens_per_second,
                    "api_usd": 0.0,
                },
                execution_environment={
                    "raw_response": gen.raw_response,
                    "prompt": gen.prompt,
                    "system_prompt": FROZEN_SYSTEM_PROMPT,
                    "prompt_sha256": _prompt_sha256(FROZEN_SYSTEM_PROMPT, user),
                    "retrieval_latency_ms": ret_ms,
                    "generation_latency_ms": gen.latency_ms,
                    "prompt_eval_duration_ms": gen.time_to_first_token_ms,
                    "ttft_metric": "prompt_eval_duration_ms_nonstream_proxy",
                    "error": gen.error,
                    "evidence_chars": len(evidence),
                    "arm": arm_id,
                    "comparison_mode": comparison_mode,
                    "corpus_sha256": corpus.get("corpus_sha256"),
                    **dict(extra_meta or {}),
                },
                comparison_mode=comparison_mode,
                failure_category=(
                    "timed_out"
                    if gen.timed_out
                    else ("exception:LocalQwenError" if gen.error else "")
                ),
                diagnostic_message=gen.error,
            ).to_dict()
        )

    artifact = build_eval_artifact(
        system_id=arm_id,
        profile=arm_id,
        questions=questions,
        rows=rows,
        comparison_mode=comparison_mode,
        source_commit=source_commit,
        config_hash=config_hash,
        qwen_identity=identity,
        dense_identity=dense_identity,
        arm_metadata={
            "family": "rag",
            "modern_rag": arm_id.startswith("hybrid"),
            "is_placeholder": False,
            "corpus_sha256": corpus.get("corpus_sha256"),
            "chunks_sha256": corpus.get("chunks_sha256"),
            "retrieval_sidecar": retrieval_rows,
            "relevance_table_sha256": (relevance_table or {}).get("corpus_sha256"),
            **dict(extra_meta or {}),
        },
        status="VALID",
    )
    return artifact


def make_bm25_retriever(corpus: Mapping[str, Any], *, top_k: int = 5):
    index = BM25Index(
        documents_from_corpus(corpus),
        chunking=dict(corpus.get("chunking") or {}),
        corpus_id=str(corpus.get("corpus_sha256") or "")[:16],
    )

    def _retrieve(question: str) -> list[dict[str, Any]]:
        return index.search(question, top_k=top_k)

    return _retrieve, {"retrieval_method": "bm25", "top_k": top_k, "index_id": index.index_id}


def make_dense_retriever(
    corpus: Mapping[str, Any],
    *,
    top_k: int = 5,
    model_name: str | None = None,
    fail_closed: bool = True,
):
    import numpy as np

    from nexus.baselines.dense_embedder import load_sentence_transformer

    docs = documents_from_corpus(corpus)
    model, embed_ident = load_sentence_transformer(fail_closed=fail_closed)
    if model_name and model_name != embed_ident["model_id"]:
        raise ValueError(
            f"dense model_name {model_name!r} disagrees with pin "
            f"{embed_ident['model_id']!r}"
        )
    texts = [d.text for d in docs]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    dim = int(emb.shape[1]) if emb.ndim == 2 else 0

    def _retrieve(question: str) -> list[dict[str, Any]]:
        q = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        qv = np.asarray(q, dtype=np.float32)[0]
        scores = emb @ qv
        order = np.argsort(-scores)[:top_k]
        hits = []
        for rank, idx in enumerate(order, start=1):
            d = docs[int(idx)]
            hits.append(
                {
                    "rank": rank,
                    "doc_id": d.doc_id,
                    "score": round(float(scores[int(idx)]), 6),
                    "source": d.source,
                    "text": d.text,
                }
            )
        return hits

    meta = {
        "retrieval_method": "dense",
        "top_k": top_k,
        "embedding_model": embed_ident["model_id"],
        "embedding_revision_pinned": embed_ident.get("revision_pinned") or embed_ident.get("revision"),
        "embedding_revision_resolved": embed_ident.get("revision_resolved") or embed_ident.get("revision"),
        "embedding_load_mode": embed_ident.get("load_mode", "unknown"),
        "embedding_identity_sha256": embed_ident["identity_sha256"],
        "embedding_files_sha256": embed_ident.get("files_sha256") or {},
        "embedding_identity_degraded": embed_ident.get("identity_degraded", False),
        "embedding_load_warnings": embed_ident.get("load_warnings", []),
        "vector_dim": dim,
        "similarity": "cosine_via_normalized_dot",
        "pooling": "sentence-transformers-default",
        "normalize": True,
    }
    # Return full identity for artifact inclusion
    return _retrieve, meta, embed_ident


def make_hybrid_retriever(
    corpus: Mapping[str, Any],
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    fail_closed: bool = True,
):
    bm25_fn, bm25_meta = make_bm25_retriever(corpus, top_k=candidate_k)
    dense_fn, dense_meta, dense_ident = make_dense_retriever(
        corpus, top_k=candidate_k, model_name=model_name, fail_closed=fail_closed
    )
    docs = {d.doc_id: d for d in documents_from_corpus(corpus)}

    def _retrieve(question: str) -> list[dict[str, Any]]:
        b_hits = bm25_fn(question)
        d_hits = dense_fn(question)
        fused = reciprocal_rank_fusion(
            [[h["doc_id"] for h in b_hits], [h["doc_id"] for h in d_hits]],
            k=rrf_k,
            top_k=top_k,
        )
        score_map = {
            **{h["doc_id"]: h for h in b_hits},
            **{h["doc_id"]: h for h in d_hits},
        }
        out = []
        for rank, (doc_id, score) in enumerate(fused, start=1):
            base = score_map.get(doc_id) or {
                "doc_id": doc_id,
                "text": docs[doc_id].text if doc_id in docs else "",
                "source": docs[doc_id].source if doc_id in docs else "",
            }
            out.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(float(score), 6),
                    "source": base.get("source", ""),
                    "text": base.get("text", ""),
                    "bm25_rank": next(
                        (h["rank"] for h in b_hits if h["doc_id"] == doc_id), None
                    ),
                    "dense_rank": next(
                        (h["rank"] for h in d_hits if h["doc_id"] == doc_id), None
                    ),
                }
            )
        return out

    meta = {
        "retrieval_method": "hybrid_rrf",
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "fusion_method": "reciprocal_rank_fusion",
        "bm25": bm25_meta,
        "dense": dense_meta,
    }
    return _retrieve, meta, dense_ident


def make_hybrid_qwen_rerank_retriever(
    corpus: Mapping[str, Any],
    adapter: LocalQwenAdapter,
    *,
    top_k: int = 5,
    candidate_k: int = 12,
    fail_closed: bool = True,
):
    """Hybrid candidates + Qwen listwise rerank (LLM reranker, not cross-encoder)."""
    hybrid_fn, hybrid_meta, dense_ident = make_hybrid_retriever(
        corpus, top_k=candidate_k, candidate_k=max(candidate_k, 20), fail_closed=fail_closed
    )

    def _retrieve(question: str) -> list[dict[str, Any]]:
        cands = hybrid_fn(question)
        if not cands:
            return []
        listing = "\n".join(
            f"{i+1}. id={h['doc_id']}\n{h['text'][:400]}" for i, h in enumerate(cands)
        )
        prompt = (
            f"Question: {question}\n\nCandidates:\n{listing}\n\n"
            f"Return the top {top_k} candidate ids as a comma-separated list only, "
            "best first."
        )
        gen = adapter.generate(
            prompt,
            decoding={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 0,
                "num_predict": 128,
                "think": False,
                "timeout_s": 180.0,
                "retry_max": 0,
            },
        )
        order: list[str] = []
        for token in gen.parsed_answer.replace("\n", ",").split(","):
            tok = token.strip()
            # accept bare ids present in candidates
            for h in cands:
                if h["doc_id"] in tok and h["doc_id"] not in order:
                    order.append(h["doc_id"])
        if not order:
            order = [h["doc_id"] for h in cands[:top_k]]
        by_id = {h["doc_id"]: h for h in cands}
        out = []
        for rank, doc_id in enumerate(order[:top_k], start=1):
            h = by_id[doc_id]
            out.append(
                {
                    **h,
                    "rank": rank,
                    "pre_rerank_rank": h.get("rank"),
                    "rerank_method": "qwen3.6_listwise",
                    "rerank_raw": gen.raw_response[:500],
                }
            )
        return out

    meta = {
        **hybrid_meta,
        "retrieval_method": "hybrid_rrf_qwen_listwise_rerank",
        "reranker": {
            "type": "llm_listwise",
            "model": adapter.model_id,
            "note": "Not a cross-encoder; explicitly an LLM reranker",
        },
        "top_k": top_k,
        "candidate_k": candidate_k,
    }
    return _retrieve, meta, dense_ident


def run_nexus_graph_evidence_qwen(
    questions: Sequence[Mapping[str, Any]],
    graph,
    adapter: LocalQwenAdapter,
    *,
    source_commit: str = "UNKNOWN",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluation-only: NEXUS evidence pack → Qwen generator (not production grounded)."""
    from nexus.pipeline.runner import NEXUSRunner
    from nexus.reasoning.model_interface import DummyModel

    config = ProductionNEXUSConfig.grounded()
    # Keep production fail-closed; DummyModel never used for final text.
    runner = NEXUSRunner(graph, config, model=DummyModel())
    identity = adapter.identity.to_dict()
    config_hash = hashlib.sha256(
        f"nexus_graph_evidence+{identity['identity_hash']}+{config.config_hash}".encode()
    ).hexdigest()[:16]
    rows: list[dict[str, Any]] = []
    executed_at = datetime.now(timezone.utc).isoformat()

    for i, q in enumerate(questions):
        qid = str(q["id"])
        if on_progress:
            on_progress(i + 1, len(questions), qid)
        qr = runner._run_single(qid, str(q["question"]), DummyModel())  # noqa: SLF001
        ep = qr.evidence_pack if isinstance(qr.evidence_pack, dict) else {}
        evidence_text = json.dumps(ep, ensure_ascii=False, indent=2)[:3500]
        if not evidence_text.strip() or evidence_text == "{}":
            evidence_text = (
                f"(no structured evidence; paths={qr.graph_paths_count}; "
                f"entities={qr.selected_entry_nodes or qr.predicted_entities})"
            )
        user = FROZEN_EVIDENCE_USER_TEMPLATE.format(
            question=str(q["question"]), evidence=evidence_text
        )
        gen = adapter.generate(user)
        outcome = _terminal_from_answer(
            gen.parsed_answer, error=gen.error, timed_out=gen.timed_out
        )
        metrics = _empty_metrics()
        if outcome in {TerminalOutcome.ANSWERED, TerminalOutcome.ABSTAINED}:
            metrics.update(_score_row(q, gen.parsed_answer))
        rows.append(
            build_question_record(
                question_id=qid,
                domain=str(q.get("domain") or "sam"),
                question_type=str(q.get("category") or "unknown"),
                dataset_id="oracle_v1",
                dataset_sha256=_dataset_hash(questions),
                system_id="nexus_graph_evidence_qwen_3_6_internal",
                profile="nexus_graph_evidence_qwen_3_6_internal",
                config_hash=config_hash,
                config_identity_schema=CONFIG_IDENTITY_SCHEMA,
                model_id=identity["model_id"],
                checkpoint_id=identity["digest"],
                source_commit=source_commit,
                executed_at_utc=executed_at,
                terminal_outcome=outcome,
                question=str(q["question"]),
                final_answer=gen.parsed_answer,
                structured_evidence=ep if isinstance(ep, dict) else {},
                entry_nodes=list(qr.selected_entry_nodes or []),
                abstention=outcome is TerminalOutcome.ABSTAINED,
                metrics=metrics,
                latency_ms=gen.latency_ms,
                token_cost={
                    "prompt_eval_count": gen.prompt_eval_count,
                    "eval_count": gen.eval_count,
                    "api_usd": 0.0,
                },
                execution_environment={
                    "raw_response": gen.raw_response,
                    "prompt": gen.prompt,
                    "nexus_config_hash": config.config_hash,
                    "nexus_allow_synth_fallback": config.allow_synth_fallback,
                    "graph_paths_count": qr.graph_paths_count,
                    "evaluation_only_adapter": True,
                    "production_grounded_unchanged": True,
                    "arm": "nexus_graph_evidence_qwen_3_6_internal",
                    "comparison_mode": "controlled",
                    "error": gen.error,
                },
                comparison_mode="controlled",
                failure_category=(
                    "timed_out"
                    if gen.timed_out
                    else ("exception:LocalQwenError" if gen.error else "")
                ),
                diagnostic_message=gen.error,
                reasoning_audit=dict(qr.reasoning_audit or {}),
            ).to_dict()
        )

    return build_eval_artifact(
        system_id="nexus_graph_evidence_qwen_3_6_internal",
        profile="nexus_graph_evidence_qwen_3_6_internal",
        questions=questions,
        rows=rows,
        comparison_mode="controlled",
        source_commit=source_commit,
        config_hash=config_hash,
        qwen_identity=identity,
        arm_metadata={
            "family": "nexus+llm",
            "modern_rag": False,
            "is_placeholder": False,
            "evaluation_only": True,
            "note": "Does not alter ProductionNEXUSConfig.grounded()",
        },
        status="VALID",
    )


# Fix closed_book: build_question_record may not accept adjudication_status
# Patch run_closed_book to not pass unknown kwargs — I already may have broken it.
# Check schema build_question_record signature.
