"""Phase 4 local Qwen 3.6 + RAG arm runner (manual/local benchmark job)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.baselines.local_qwen import LocalQwenAdapter, discover_local_qwen
from nexus.baselines.phase4_arms import (
    make_bm25_retriever,
    make_dense_retriever,
    make_hybrid_qwen_rerank_retriever,
    make_hybrid_retriever,
    run_closed_book_qwen,
    run_nexus_graph_evidence_qwen,
    run_rag_answer_arm,
)
from nexus.baselines.rag_corpus import build_canonical_corpus, format_evidence_blocks
from nexus.domain import load_domain_pack


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip()
    except Exception:
        return "UNKNOWN"


def _write(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _progress(i: int, n: int, qid: str) -> None:
    print(f"[{i}/{n}] {qid}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "health",
            "corpus",
            "closed_book",
            "long_context",
            "bm25_rag",
            "dense_rag",
            "hybrid_rag",
            "hybrid_rerank_rag",
            "nexus_graph_qwen",
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "phase4_rag_corpus_v1.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    # Prefer offline HF for dense
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.arm == "health":
        identity = discover_local_qwen()
        adapter = LocalQwenAdapter(identity)
        payload = adapter.health_check()
        _write(args.output, payload)
        print(json.dumps({"ok": payload["ok"], "model": identity.model_name}, sort_keys=True))
        return 0 if payload["ok"] else 2

    if args.arm == "corpus":
        corpus = build_canonical_corpus(ROOT)
        _write(args.output, corpus)
        print(
            json.dumps(
                {
                    "file_count": corpus["file_count"],
                    "chunk_count": corpus["chunk_count"],
                    "corpus_sha256": corpus["corpus_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    pack = load_domain_pack("sam")
    questions = pack.evaluation_tasks()
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    adapter = LocalQwenAdapter(discover_local_qwen())
    commit = _commit()

    if args.arm == "closed_book":
        art = run_closed_book_qwen(
            questions, adapter, source_commit=commit, on_progress=_progress
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if args.arm == "long_context":
        if not args.corpus.exists():
            raise SystemExit(f"corpus missing: {args.corpus}")
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        # Deterministic full-corpus assembly (chunk order as frozen)
        parts = []
        for ch in corpus["chunks"]:
            parts.append(f"## {ch['source_path']} [{ch['chunk_id']}]\n{ch['text']}")
        full = "\n\n".join(parts)
        # Rough token estimate; model context 262144
        est_tokens = len(full) // 4 + 512
        if est_tokens > int(adapter.identity.context_length) * 0.9:
            art = {
                "schema_version": "nexus-eval-result-v1",
                "status": "NOT_RUN",
                "system_id": "qwen_3_6_long_context_internal",
                "failure_reason": (
                    f"estimated tokens {est_tokens} exceed 90% of context "
                    f"{adapter.identity.context_length}; refusing silent truncation"
                ),
                "questions_total": 0,
                "per_question": [],
                "aggregates": {},
                "created_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "source_commit": commit,
                "dataset_id": "oracle_v1",
                "dataset_sha256": "0" * 64,
                "profile": "qwen_3_6_long_context_internal",
                "config_hash": "not_run",
                "config_identity_schema": "nexus-config-identity-v2",
                "comparison_mode": "system_level",
            }
            # Emit honest NOT_RUN via closed-book-like per-question if needed —
            # prefer schema-valid NOT_RUN rows:
            from nexus.baselines.adapters import run_baseline_eval

            # Fall through: build NOT_RUN rows manually using adapter path skipped
            from nexus.evaluation.schema import TerminalOutcome, build_question_record, empty_metric_applicability
            from nexus.evaluation.aggregate import aggregate_question_records
            from datetime import datetime, timezone

            rows = []
            executed = datetime.now(timezone.utc).isoformat()
            for q in questions:
                rows.append(
                    build_question_record(
                        question_id=str(q["id"]),
                        domain="sam",
                        question_type=str(q.get("category") or "unknown"),
                        dataset_id="oracle_v1",
                        dataset_sha256="0" * 64,
                        system_id="qwen_3_6_long_context_internal",
                        profile="qwen_3_6_long_context_internal",
                        config_hash="not_run",
                        config_identity_schema="nexus-config-identity-v2",
                        model_id=adapter.model_id,
                        checkpoint_id=adapter.identity.digest,
                        source_commit=commit,
                        executed_at_utc=executed,
                        terminal_outcome=TerminalOutcome.NOT_RUN,
                        question=str(q["question"]),
                        metrics=empty_metric_applicability(),
                        comparison_mode="system_level",
                        failure_category="not_run",
                        diagnostic_message=art["failure_reason"],
                    ).to_dict()
                )
            art = {
                "schema_version": "nexus-eval-result-v1",
                "created_utc": executed,
                "source_commit": commit,
                "dataset_id": "oracle_v1",
                "dataset_sha256": "0" * 64,
                "system_id": "qwen_3_6_long_context_internal",
                "profile": "qwen_3_6_long_context_internal",
                "config_hash": "not_run",
                "config_identity_schema": "nexus-config-identity-v2",
                "comparison_mode": "system_level",
                "questions_total": len(rows),
                "per_question": rows,
                "aggregates": aggregate_question_records(rows),
                "status": "NOT_RUN",
                "arm_metadata": {
                    "estimated_tokens": est_tokens,
                    "context_length": adapter.identity.context_length,
                    "truncation": "refused",
                },
            }
            _write(args.output, art)
            print(json.dumps({"status": "NOT_RUN", "est_tokens": est_tokens}, sort_keys=True))
            return 0

        # Fits: same corpus for every question
        import hashlib as _hashlib

        from nexus.baselines.local_qwen import (
            FROZEN_EVIDENCE_USER_TEMPLATE,
            FROZEN_SYSTEM_PROMPT,
        )
        from nexus.baselines.phase4_arms import (
            build_eval_artifact,
            _score_row,
            _terminal_from_answer,
            _empty_metrics,
            _dataset_hash,
            _prompt_sha256,
        )
        from nexus.evaluation.schema import build_question_record
        from datetime import datetime, timezone

        # Exact prefix bytes fed to the model (deterministic truncation).
        evidence_prefix = full[:120000]
        context_hash = _hashlib.sha256(full.encode("utf-8")).hexdigest()
        prefix_hash = _hashlib.sha256(evidence_prefix.encode("utf-8")).hexdigest()

        rows = []
        executed = datetime.now(timezone.utc).isoformat()
        identity = adapter.identity.to_dict()
        for i, q in enumerate(questions):
            _progress(i + 1, len(questions), str(q["id"]))
            evidence = evidence_prefix
            user = FROZEN_EVIDENCE_USER_TEMPLATE.format(
                question=str(q["question"]), evidence=evidence
            )
            gen = adapter.generate(
                user,
                decoding={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 0,
                    "num_predict": 256,
                    "think": False,
                    "timeout_s": 300.0,
                    "retry_max": 0,
                    "num_ctx": min(131072, adapter.identity.context_length),
                },
            )
            outcome = _terminal_from_answer(
                gen.parsed_answer, error=gen.error, timed_out=gen.timed_out
            )
            metrics = _empty_metrics()
            if outcome.value in {"answered", "abstained"}:
                metrics.update(_score_row(q, gen.parsed_answer))
            rows.append(
                build_question_record(
                    question_id=str(q["id"]),
                    domain="sam",
                    question_type=str(q.get("category") or "unknown"),
                    dataset_id="oracle_v1",
                    dataset_sha256=_dataset_hash(questions),
                    system_id="qwen_3_6_long_context_internal",
                    profile="qwen_3_6_long_context_internal",
                    config_hash=identity["identity_hash"],
                    config_identity_schema="nexus-config-identity-v2",
                    model_id=identity["model_id"],
                    checkpoint_id=identity["digest"],
                    source_commit=commit,
                    executed_at_utc=executed,
                    terminal_outcome=outcome,
                    question=str(q["question"]),
                    final_answer=gen.parsed_answer,
                    metrics=metrics,
                    latency_ms=gen.latency_ms,
                    comparison_mode="system_level",
                    execution_environment={
                        "context_sha256": context_hash,
                        "long_context_prefix_sha256": prefix_hash,
                        "prompt_sha256": _prompt_sha256(FROZEN_SYSTEM_PROMPT, user),
                        "context_chars": len(evidence),
                        "estimated_tokens": est_tokens,
                        "deterministic_truncation_chars": 120000,
                        "raw_response": gen.raw_response,
                        "error": gen.error,
                        "prompt_eval_duration_ms": gen.time_to_first_token_ms,
                        "ttft_metric": "prompt_eval_duration_ms_nonstream_proxy",
                    },
                    failure_category=(
                        "timed_out"
                        if gen.timed_out
                        else ("exception:LocalQwenError" if gen.error else "")
                    ),
                    diagnostic_message=gen.error,
                ).to_dict()
            )
        art = build_eval_artifact(
            system_id="qwen_3_6_long_context_internal",
            profile="qwen_3_6_long_context_internal",
            questions=questions,
            rows=rows,
            comparison_mode="system_level",
            source_commit=commit,
            config_hash=identity["identity_hash"],
            qwen_identity=identity,
            arm_metadata={
                "context_sha256": context_hash,
                "long_context_prefix_sha256": prefix_hash,
                "estimated_tokens": est_tokens,
                "deterministic_truncation_chars": 120000,
            },
            arm_decoding_overrides={
                "num_ctx": min(131072, adapter.identity.context_length),
                "timeout_s": 300.0,
                "num_predict": 256,
            },
            status="VALID",
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if not args.corpus.exists():
        raise SystemExit(f"corpus missing: {args.corpus}")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    from pathlib import Path as _Path

    from nexus.evaluation.relevance import load_or_build_relevance

    relevance_path = _Path("benchmarks/results/oracle_v1_retrieval_relevance_v1.json")
    relevance_table = load_or_build_relevance(relevance_path, questions, corpus)

    if args.arm == "bm25_rag":
        retrieve, meta = make_bm25_retriever(corpus, top_k=args.top_k)
        art = run_rag_answer_arm(
            arm_id="bm25_rag_qwen_3_6_internal",
            questions=questions,
            corpus=corpus,
            adapter=adapter,
            retrieve_fn=retrieve,
            comparison_mode="controlled",
            source_commit=commit,
            extra_meta=meta,
            relevance_table=relevance_table,
            on_progress=_progress,
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if args.arm == "dense_rag":
        retrieve, meta, dense_ident = make_dense_retriever(corpus, top_k=args.top_k)
        art = run_rag_answer_arm(
            arm_id="dense_rag_qwen_3_6_internal",
            questions=questions,
            corpus=corpus,
            adapter=adapter,
            retrieve_fn=retrieve,
            comparison_mode="controlled",
            source_commit=commit,
            extra_meta=meta,
            relevance_table=relevance_table,
            dense_identity=dense_ident,
            on_progress=_progress,
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if args.arm == "hybrid_rag":
        retrieve, meta, dense_ident = make_hybrid_retriever(corpus, top_k=args.top_k)
        art = run_rag_answer_arm(
            arm_id="hybrid_rag_qwen_3_6_internal",
            questions=questions,
            corpus=corpus,
            adapter=adapter,
            retrieve_fn=retrieve,
            comparison_mode="controlled",
            source_commit=commit,
            extra_meta=meta,
            relevance_table=relevance_table,
            dense_identity=dense_ident,
            on_progress=_progress,
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if args.arm == "hybrid_rerank_rag":
        retrieve, meta, dense_ident = make_hybrid_qwen_rerank_retriever(
            corpus, adapter, top_k=args.top_k
        )
        art = run_rag_answer_arm(
            arm_id="hybrid_rerank_rag_qwen_3_6_internal",
            questions=questions,
            corpus=corpus,
            adapter=adapter,
            retrieve_fn=retrieve,
            comparison_mode="controlled",
            source_commit=commit,
            extra_meta=meta,
            relevance_table=relevance_table,
            dense_identity=dense_ident,
            on_progress=_progress,
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    if args.arm == "nexus_graph_qwen":
        graph = pack.build_graph()
        art = run_nexus_graph_evidence_qwen(
            questions, graph, adapter, source_commit=commit, on_progress=_progress
        )
        _write(args.output, art)
        print(json.dumps({"status": art["status"], "n": art["questions_total"]}, sort_keys=True))
        return 0

    raise SystemExit(f"unknown arm {args.arm}")


if __name__ == "__main__":
    raise SystemExit(main())
