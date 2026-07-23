"""Run NEXUS through evaluation-result schema v1 and optional baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from nexus.baselines.adapters import run_baseline_eval
from nexus.domain import load_domain_pack
from nexus.evaluation import regenerate_aggregates, validate_result_artifact
from nexus.evaluation.dataset_identity import hash_dataset
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.model_interface import DummyModel, SynthesizingModel


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _load_questions(path: Path | None, domain: str, limit: int | None) -> list[dict]:
    if path is None:
        pack = load_domain_pack(domain)
        questions = pack.evaluation_tasks()
    else:
        if path.suffix == ".jsonl":
            questions = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            questions = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    return questions


def _nexus_config(profile: str) -> ProductionNEXUSConfig:
    factories = {
        "grounded": ProductionNEXUSConfig.grounded,
        "l1_acceptance": ProductionNEXUSConfig.l1_acceptance,
        "lexical": ProductionNEXUSConfig.lexical_only,
        "pointer_copy": ProductionNEXUSConfig.pointer_copy,
        "deterministic_render": ProductionNEXUSConfig.deterministic_render,
    }
    try:
        return factories[profile]()
    except KeyError as exc:
        raise SystemExit(f"unknown profile {profile}; known={sorted(factories)}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="nexus", help="nexus | baseline arm id")
    parser.add_argument("--profile", default="grounded")
    parser.add_argument("--domain", default="mini")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--comparison-mode",
        choices=("system_level", "controlled"),
        default="system_level",
    )
    parser.add_argument("--model", choices=("dummy", "synth"), default="dummy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    questions = _load_questions(args.dataset, args.domain, args.limit)
    dataset_id = args.dataset_id or (
        args.dataset.stem if args.dataset else f"{args.domain}-tasks"
    )
    ds_hash = hash_dataset(questions)
    source = _git_head()

    if args.arm == "nexus":
        pack = load_domain_pack(args.domain) if args.dataset is None else None
        graph = pack.build_graph() if pack is not None else None
        if graph is None:
            from benchmarks.run_benchmark import build_benchmark_graph

            graph, _ = build_benchmark_graph()
        config = _nexus_config(args.profile)
        model = DummyModel() if args.model == "dummy" else SynthesizingModel()
        runner = NEXUSRunner(graph, config, model=model)
        artifact = runner.run_eval(
            questions,
            dataset_id=dataset_id,
            dataset_sha256=ds_hash,
            source_sha=source,
            system_id=f"nexus_{args.profile}",
            profile=args.profile,
            domain_pack_id=pack.meta.domain_id if pack else "sam",
            domain_pack_version=pack.meta.version if pack else "sam-v1",
            model_id=model.name,
            comparison_mode=args.comparison_mode,
        )
    else:
        artifact = run_baseline_eval(
            args.arm,
            questions,
            dataset_id=dataset_id,
            dataset_sha256=ds_hash,
            comparison_mode=args.comparison_mode,
            source_commit=source,
        )

    # Prove aggregates are regenerable
    rebuilt = regenerate_aggregates(artifact)
    artifact["aggregates"] = rebuilt
    errors = validate_result_artifact(artifact)
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))

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
                "arm": args.arm,
                "profile": artifact.get("profile"),
                "questions_total": artifact["questions_total"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
