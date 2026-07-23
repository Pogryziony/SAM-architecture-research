"""BM25 retrieval-only evaluation on a domain pack corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from nexus.baselines.retrieval import corpus_from_graph_nodes, run_bm25_retrieval_eval
from nexus.domain import load_domain_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="mini", choices=("mini", "sam"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--comparison-mode", default="controlled")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    pack = load_domain_pack(args.domain)
    graph = pack.build_graph()
    questions = pack.evaluation_tasks()
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    docs = corpus_from_graph_nodes(graph)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "UNKNOWN"

    artifact = run_bm25_retrieval_eval(
        questions,
        docs,
        dataset_id="oracle_v1" if args.domain == "sam" else f"{args.domain}-tasks",
        top_k=args.top_k,
        comparison_mode=args.comparison_mode,
        source_commit=commit,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "answer_generation_status": artifact["answer_generation_status"],
                "recall_at_k_mean": artifact["aggregates"]["recall_at_k_mean"],
                "mrr_mean": artifact["aggregates"]["mrr_mean"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
