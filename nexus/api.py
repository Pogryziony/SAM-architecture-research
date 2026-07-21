"""Stable public API surface for NEXUS.

Prefer this module (and ``ProductionNEXUSConfig`` factories) over reaching into
internal packages. The library default Realizer remains ``synth``; production
QA should use ``ProductionNEXUSConfig.grounded()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner, PipelineResult, QuestionResult
from nexus.reasoning.answer import answer_question
from nexus.utils.config import NEXUSConfig

__all__ = [
    "InMemoryGraphStore",
    "NEXUSConfig",
    "NEXUSRunner",
    "PipelineResult",
    "ProductionNEXUSConfig",
    "QuestionResult",
    "answer_question",
    "ask",
    "build_default_graph",
    "main",
]


def build_default_graph(config: NEXUSConfig | None = None) -> InMemoryGraphStore:
    """Build the canonical project graph used by benchmarks."""
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, _provenance = build_benchmark_graph(config or NEXUSConfig())
    return graph


def ask(
    question: str,
    *,
    graph: InMemoryGraphStore | None = None,
    config: ProductionNEXUSConfig | NEXUSConfig | None = None,
    question_id: str = "cli-q0",
) -> QuestionResult:
    """Answer one question with the production runner.

    When *config* is omitted, uses ``ProductionNEXUSConfig.grounded()`` — the
    recommended production profile, not the library ``synth`` default.
    """
    if config is None:
        resolved_config = ProductionNEXUSConfig.grounded()
    elif isinstance(config, ProductionNEXUSConfig):
        resolved_config = config
    else:
        raise TypeError(
            "config must be ProductionNEXUSConfig or None; "
            "use ProductionNEXUSConfig.grounded()/pointer_copy()/lexical_only()"
        )

    store = graph if graph is not None else build_default_graph(resolved_config)
    runner = NEXUSRunner(graph=store, config=resolved_config)
    pipeline = runner.run(
        [{"id": question_id, "question": question}],
    )
    if pipeline.errors:
        raise RuntimeError("; ".join(pipeline.errors))
    if not pipeline.per_question:
        raise RuntimeError("NEXUSRunner returned no results")
    return pipeline.per_question[0]


def main(argv: list[str] | None = None) -> int:
    """CLI: ``nexus ask "..."`` or ``python -m nexus ask "..."``."""
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS graph-first QA (CPU-first, fail-closed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask_parser = sub.add_parser("ask", help="Answer a single question")
    ask_parser.add_argument("question", help="Natural-language question")
    ask_parser.add_argument(
        "--profile",
        choices=("grounded", "pointer_copy", "comparison_plan", "lexical", "synth"),
        default="grounded",
        help="Production profile (default: grounded). 'synth' is the library default.",
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON diagnostic record instead of plain text",
    )

    profiles_parser = sub.add_parser(
        "profiles",
        help="Print library default vs production profile guidance",
    )

    args = parser.parse_args(argv)

    if args.command == "profiles":
        from training.architecture_registry import describe_production_profiles

        print(json.dumps(describe_production_profiles(), indent=2, sort_keys=True))
        return 0

    factories = {
        "grounded": ProductionNEXUSConfig.grounded,
        "pointer_copy": ProductionNEXUSConfig.pointer_copy,
        "comparison_plan": ProductionNEXUSConfig.comparison_plan,
        "lexical": ProductionNEXUSConfig.lexical_only,
        "synth": lambda: ProductionNEXUSConfig.lexical_only(realizer_backend="synth"),
    }
    config = factories[args.profile]()
    try:
        result = ask(args.question, config=config)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2

    if args.json:
        payload: dict[str, Any] = {
            "question": result.question,
            "answer": result.answer,
            "intent": result.parsed_intent,
            "entities": result.predicted_entities,
            "verifier_passed": result.verifier_passed,
            "reasoning_action": result.reasoning_action,
            "config_profile": args.profile,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(result.answer or "")
    if not result.answer:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
