"""End-to-end performance campaign for ProductionNEXUSConfig profiles.

Supports mini-domain smoke and full SAM/`oracle_v1` measurement. Results from
the mini domain must never be presented as full-graph evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from nexus.domain import load_domain_pack
from nexus.evaluation.performance import measure_grounded_e2e
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.model_interface import DummyModel


def _build_config(profile: str, *, er3_dir: str = "") -> ProductionNEXUSConfig:
    if profile == "grounded":
        base = ProductionNEXUSConfig.grounded()
    elif profile == "lexical":
        base = ProductionNEXUSConfig.lexical_only()
    elif profile == "l1_acceptance":
        base = ProductionNEXUSConfig.l1_acceptance()
    else:
        raise ValueError(f"unsupported profile: {profile}")
    if er3_dir:
        # Re-bind grounded/lexical knobs onto the accepted ER3 identity.
        return ProductionNEXUSConfig.with_entity_ranker_v3(
            er3_dir,
            realizer_backend=base.realizer_backend,
            realizer_model_dir=base.realizer_model_dir,
            realizer_config_path=base.realizer_config_path,
            realizer_checkpoint_sha256=base.realizer_checkpoint_sha256,
            require_structured_provenance=base.require_structured_provenance,
            allow_synth_fallback=base.allow_synth_fallback,
        )
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="mini", choices=("mini", "sam"))
    parser.add_argument(
        "--profile",
        choices=("grounded", "lexical", "l1_acceptance"),
        default="grounded",
    )
    parser.add_argument(
        "--er3-dir",
        default="",
        help="Optional accepted ER3 checkpoint dir for lexical+ER3 measurement",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all evaluation tasks for the domain",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    setup: dict[str, float] = {}
    t0 = time.perf_counter()
    pack = load_domain_pack(args.domain)
    setup["domain_pack_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    t0 = time.perf_counter()
    graph = pack.build_graph()
    setup["graph_construction_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    questions = pack.evaluation_tasks()
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    t0 = time.perf_counter()
    config = _build_config(args.profile, er3_dir=args.er3_dir)
    setup["config_init_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    entity_resolver = None
    resolver_name = "lexical_in_runner"
    if args.er3_dir:
        t0 = time.perf_counter()
        from stack.pipeline.resolver import ER3Resolver

        entity_resolver = ER3Resolver.from_directory(args.er3_dir, graph)
        setup["checkpoint_loading_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        resolver_name = f"entity_ranker_v3:{args.er3_dir}"
    else:
        setup.setdefault("checkpoint_loading_ms", 0.0)

    t0 = time.perf_counter()
    runner = NEXUSRunner(
        graph, config, model=DummyModel(), entity_resolver=entity_resolver
    )
    setup["runner_init_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    setup.setdefault("index_construction_ms", 0.0)

    scope = "mini_domain" if args.domain == "mini" else "full_sam_oracle_v1"
    profile_name = (
        f"{args.profile}+er3" if args.er3_dir else args.profile
    )
    artifact = measure_grounded_e2e(
        runner,
        questions,
        warmup=args.warmup,
        repeats=args.repeats,
        profile_name=profile_name,
        setup_timings=setup,
        graph_meta={
            "domain_pack_id": pack.meta.domain_id,
            "domain_pack_version": pack.meta.version,
            "graph_node_count": graph.node_count,
            "graph_edge_count": graph.edge_count,
            "graph_snapshot_id": getattr(config.pipeline_id, "graph_snapshot_id", "")
            or f"{pack.meta.domain_id}:{pack.meta.version}",
            "er3_dir": args.er3_dir or None,
        },
        scope=scope,
    )
    try:
        artifact["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        artifact["source_commit"] = "UNKNOWN"
    artifact["dataset_id"] = (
        "oracle_v1" if args.domain == "sam" else f"{args.domain}-tasks"
    )
    artifact["domain_pack_version"] = pack.meta.version
    artifact["graph_node_count"] = graph.node_count
    artifact["graph_edge_count"] = graph.edge_count
    artifact["question_count"] = len(questions)
    artifact["resolver"] = resolver_name

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
                "scope": scope,
                "profile": profile_name,
                "question_count": len(questions),
                "graph_node_count": graph.node_count,
                "graph_edge_count": graph.edge_count,
                "warm_p50_ms": artifact["summary"]["warm_p50_ms"],
                "warm_p95_ms": artifact["summary"]["warm_p95_ms"],
                "warm_p99_ms": artifact["summary"]["warm_p99_ms"],
                "peak_rss_mb": artifact["summary"]["peak_rss_mb"],
                "latency_gate": artifact["budgets"]["latency_p50_gate"],
                "rss_gate": artifact["budgets"]["peak_rss_gate"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
