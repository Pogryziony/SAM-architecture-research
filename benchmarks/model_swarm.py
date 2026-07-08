"""
NEXUS Model Swarm — same graph, same 30 questions, 4 different models.

Compares 4 model backends side-by-side:
  a. qwen2.5-coder:3b   — current baseline (coding model)
  b. qwen2.5:latest     — instruct variant (general-purpose)
  c. llama3.2:3b        — alternative model family (Meta)
  d. SynthesizingModel  — template-based upper bound (no ML)

Collects: fuzzy_accuracy, exact_accuracy, hallucination_rate,
          verification_pass_rate, avg_latency, conciseness_ratio.

Usage:
    python benchmarks/model_swarm.py --limit 30 --output benchmarks/model_comparison.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import (
    ModelInterface,
    OllamaModel,
    SynthesizingModel,
)
from nexus.reasoning.verifier import Verifier, VerificationResult


# ---- Token counting (for conciseness) ----

def _count_tokens(text: str) -> int:
    return len(text.split())


# ---- Question loader ----

def load_questions(jsonl_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit and limit > 0:
        questions = questions[:limit]
    return questions


# ---- Accuracy scoring (from run_benchmark.py) ----

_FACT_PATTERNS = [
    (__import__("re").compile(r'\b(\d+\.?\d*\s*%)(?=\s|$|[,.);])'), "percentage"),
    (__import__("re").compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million\b', __import__("re").IGNORECASE), "number+million"),
    (__import__("re").compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:live\s+)?(slots?|examples?|tokens?|parameters?|params?|subkeys?|distractors?|vocabulary|hops?)\b', __import__("re").IGNORECASE), "number+unit"),
    (__import__("re").compile(r'\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b(?!\s*%)'), "large_number"),
    (__import__("re").compile(r'\b(\w+@\d+)\b'), "at_notation"),
    (__import__("re").compile(r'\b([Kk]=\d+)\b'), "k_notation"),
    (__import__("re").compile(r'\b(Exp_\d+_\d+[A-Z]?_\w+)\b'), "experiment_id"),
    (__import__("re").compile(r'\b(Concept_\w+)\b'), "concept_id"),
    (__import__("re").compile(r'\b(Decision_\w+)\b'), "decision_id"),
    (__import__("re").compile(r'\b(depends_on|validates|caused_by|contradicts|implements|mentioned_in|derived_from|related_to|replaces|blocked_by)\b', __import__("re").IGNORECASE), "relation"),
    (__import__("re").compile(r'\b(core_only|oracle_memory|retrieved_memory|random_memory|oracle_text_memory|oracle_filter|oracle_text_memory|retrieved_memory_external_text_query)\b', __import__("re").IGNORECASE), "sam_mode"),
    (__import__("re").compile(r'\b(Gate\s+\d+)\b', __import__("re").IGNORECASE), "gate_ref"),
]

import re as _re


def _extract_key_facts(text: str) -> set[str]:
    facts: set[str] = set()
    for pattern, fact_type in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact_str = match.group(0).strip().lower()
            fact_str = _re.sub(r'(\d),(\d)', r'\1\2', fact_str)
            facts.add(fact_str)
    return facts


def _extract_numbers(text: str) -> set[float]:
    numbers: set[float] = set()
    tokens: list[str] = _re.findall(r'[^\s]+', text)
    percent_re = _re.compile(r'^([\d,]+\.?\d*)\s*%$')

    for token in tokens:
        while token and token[-1] in ',.;:)!?' and token[-1] != '%':
            token = token[:-1]
        if not token:
            continue

        pm = percent_re.match(token)
        if pm:
            try:
                val = float(pm.group(1).replace(',', ''))
                numbers.add(round(val / 100.0, 10))
            except ValueError:
                pass
            continue

        stripped = token.replace(',', '')
        if stripped.replace('.', '', 1).isdigit():
            try:
                numbers.add(float(stripped))
            except ValueError:
                pass

    return numbers


def _fuzzy_number_match(pred_nums: set[float], gt_nums: set[float]) -> tuple[int, int]:
    if not gt_nums:
        return 0, 0

    pred_list: list[float | None] = list(pred_nums)
    gt_list = sorted(gt_nums, reverse=True)
    matches = 0

    for gt in gt_list:
        best_idx: int = -1
        best_err: float = float('inf')
        for i, pred in enumerate(pred_list):
            if pred is None:
                continue
            denom = max(abs(gt), 0.001)
            rel_err = abs(pred - gt) / denom
            abs_err = abs(pred - gt)
            if rel_err < 0.05 or abs_err < 0.001:
                if rel_err < best_err:
                    best_err = rel_err
                    best_idx = i
        if best_idx >= 0:
            matches += 1
            pred_list[best_idx] = None

    return matches, len(gt_nums)


def compute_key_fact_score(
    predicted_answer: str, ground_truth: str, use_fuzzy: bool = True
) -> dict[str, Any]:
    empty_detail: dict[str, Any] = {
        "gt_numbers": [],
        "pred_numbers": [],
        "fuzzy_matches": 0,
        "total_gt": 0,
        "fuzzy_score": 0.0,
        "exact_score": 0.0,
        "entity_overlap": [],
    }

    if "insufficient evidence" in predicted_answer.lower():
        return {
            "fuzzy_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "scoring_detail": empty_detail,
        }

    gt_facts = _extract_key_facts(ground_truth)
    pred_facts = _extract_key_facts(predicted_answer)

    if not gt_facts:
        empty_detail["fuzzy_score"] = None
        empty_detail["exact_score"] = None
        return {
            "fuzzy_accuracy": None,
            "exact_accuracy": None,
            "scoring_detail": empty_detail,
        }

    intersection = gt_facts & pred_facts
    exact_score: float = round(len(intersection) / len(gt_facts), 4)

    gt_nums = _extract_numbers(ground_truth)
    pred_nums = _extract_numbers(predicted_answer)
    entity_overlap = sorted(list(intersection))

    if gt_nums and use_fuzzy:
        fuzzy_matches, total_gt = _fuzzy_number_match(pred_nums, gt_nums)
        fuzzy_score: float | None = (
            round(fuzzy_matches / total_gt, 4) if total_gt > 0 else None
        )
        primary_accuracy: float = fuzzy_score if fuzzy_score is not None else exact_score
    else:
        fuzzy_matches = 0
        total_gt = 0
        fuzzy_score = None
        primary_accuracy = exact_score

    return {
        "fuzzy_accuracy": primary_accuracy,
        "exact_accuracy": exact_score,
        "scoring_detail": {
            "gt_numbers": sorted(list(gt_nums)),
            "pred_numbers": sorted(list(pred_nums)),
            "fuzzy_matches": fuzzy_matches,
            "total_gt": total_gt,
            "fuzzy_score": fuzzy_score,
            "exact_score": exact_score,
            "entity_overlap": entity_overlap,
        },
    }


# ---- Graph construction (same as run_benchmark.py) ----

def build_benchmark_graph() -> tuple[InMemoryGraphStore, dict[str, Any]]:
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    from nexus.ingestion.ingest_docs import ingest_directory

    graph = InMemoryGraphStore()

    if EXPERIMENTS_DIR.exists():
        graph = populate_graph(EXPERIMENTS_DIR, graph)

    docs_dir = _project_root / "docs"
    if docs_dir.exists():
        ingest_directory(docs_dir, graph)
    sam_docs_dir = _project_root / "sam-lm" / "docs"
    if sam_docs_dir.exists():
        ingest_directory(sam_docs_dir, graph)
    sam_exp_dir = _project_root / "sam-lm" / "experiments"
    if sam_exp_dir.exists():
        ingest_directory(sam_exp_dir, graph)

    provenance = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return graph, provenance


# ---- Pipeline runner ----

def run_nexus_pipeline(
    question_text: str,
    graph: InMemoryGraphStore,
    model: ModelInterface,
    verifier: Verifier,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = answer_question(
            question_text, graph, model=model, verifier=verifier,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "passed": False,
            "hallucination_rate": 1.0,
            "supported_count": 0,
            "unsupported_count": 0,
            "path_count": 0,
            "is_insufficient": False,
            "latency_s": round(elapsed, 4),
            "error": str(exc),
            "parsed_entity_ids": [],
        }
    elapsed = time.perf_counter() - t0

    verif: VerificationResult | None = result.get("verification")
    answer = result.get("answer", "")
    is_insufficient = "insufficient evidence" in answer.lower()

    if verif is not None:
        passed = verif.passed
        hall_rate = verif.hallucination_rate
        supported = verif.supported_count
        unsupported = len(verif.unsupported_claims)
    else:
        passed = True
        hall_rate = 0.0
        supported = 0
        unsupported = 0

    return {
        "answer": answer,
        "passed": passed,
        "hallucination_rate": round(hall_rate, 4),
        "supported_count": supported,
        "unsupported_count": unsupported,
        "path_count": result.get("path_count", 0),
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "error": None,
        "parsed_entity_ids": (
            result["parsed_query"].entity_ids if result.get("parsed_query") else []
        ),
    }


# ---- Check Ollama availability ----

def _check_ollama_available() -> tuple[bool, list[str]]:
    """Check if Ollama is running and return available models."""
    try:
        import urllib.request
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


# ---- Statistics helpers ----

def _avg(lst: list[float]) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


# ---- Display ----

def print_comparison_table(
    model_results: list[dict[str, Any]],
    total_questions: int,
) -> None:
    """Print a markdown comparison table of all models."""
    print()
    print("=" * 90)
    print("  NEXUS Model Swarm — Results Comparison")
    print("=" * 90)
    print(f"  Questions benchmarked: {total_questions}")
    print(f"  Graph built once, same 30 questions, 4 model backends.")
    print()

    # Header
    header = (
        f"  {'Model':<18} {'Fuzzy acc':>10} {'Exact acc':>10} "
        f"{'Halluc.':>9} {'Verify':>8} {'Latency':>8} {'Conciseness':>12}"
    )
    print(header)
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*9} {'-'*8} {'-'*8} {'-'*12}")

    best_fuzzy = ("", -1.0)
    for mr in model_results:
        s = mr["summary"]
        name = mr["name"]
        fuzzy = s["avg_accuracy"]
        exact = s["avg_exact_accuracy"]
        hall = s["avg_hallucination_rate"]
        verify = s["verification_pass_rate"]
        latency = s["avg_latency_s"]
        conciseness = s["avg_conciseness_ratio"]

        # Format all as strings
        fuzzy_str = f"{fuzzy:.1%}" if fuzzy is not None else "N/A"
        exact_str = f"{exact:.1%}" if exact is not None else "N/A"
        hall_str = f"{hall:.1%}"
        verify_str = f"{verify:.1%}"
        latency_str = f"{latency:.2f}s"
        conciseness_str = f"{conciseness:.1f}x"

        print(
            f"  {name:<18} {fuzzy_str:>10} {exact_str:>10} "
            f"{hall_str:>9} {verify_str:>8} {latency_str:>8} {conciseness_str:>12}"
        )

        if fuzzy is not None and fuzzy > best_fuzzy[1]:
            best_fuzzy = (name, fuzzy)

    print()
    print(f"  ** Best model by fuzzy accuracy: {best_fuzzy[0]} ({best_fuzzy[1]:.1%}) **")
    print("=" * 90)
    print()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Model Swarm — test same questions across 4 model backends"
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Number of questions to benchmark (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/model_comparison.json",
        help="Output file (default: benchmarks/model_comparison.json)"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    output_path = _project_root / args.output

    if not dataset_path.exists():
        print(f"Error: QA dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading questions from: {dataset_path}")
    questions = load_questions(str(dataset_path), args.limit)
    total = len(questions)
    print(f"Loaded {total} questions (limit={args.limit})")

    # Build graph once
    print("\nBuilding benchmark graph (deterministic)...")
    graph, graph_provenance = build_benchmark_graph()
    print(f"Graph ready: {graph_provenance['node_count']} nodes, "
          f"{graph_provenance['edge_count']} edges")

    verifier = Verifier(hallucination_threshold=0.2)

    # Define model backends to test
    ollama_ok, available_models = _check_ollama_available()

    model_backends: list[tuple[str, ModelInterface]] = []

    # a. qwen2.5-coder:3b — current baseline
    if "qwen2.5-coder:3b" in available_models:
        model_backends.append(
            ("qwen2.5-coder:3b", OllamaModel(model_name="qwen2.5-coder:3b"))
        )
    else:
        print("[skip] qwen2.5-coder:3b not available in Ollama")

    # b. qwen2.5:latest (instruct variant) — use qwen2.5:3b-instruct if available, else qwen2.5:latest
    instruct_model_name = None
    for candidate in ["qwen2.5:3b-instruct", "qwen2.5:3b", "qwen2.5:latest"]:
        if candidate in available_models:
            instruct_model_name = candidate
            break
    if instruct_model_name:
        model_backends.append(
            (instruct_model_name, OllamaModel(model_name=instruct_model_name))
        )
    else:
        print("[skip] No qwen2.5 instruct variant available")

    # c. llama3.2:3b — alternative model family
    llama_model_name = None
    for candidate in ["llama3.2:3b", "llama3.2:1b"]:
        if candidate in available_models:
            llama_model_name = candidate
            break
    if not llama_model_name:
        # Check if any llama model is available
        for m in available_models:
            if "llama" in m.lower():
                llama_model_name = m
                break
    if llama_model_name:
        model_backends.append(
            (llama_model_name, OllamaModel(model_name=llama_model_name))
        )
    else:
        print("[skip] No llama model available in Ollama")

    # d. SynthesizingModel — always available (template-based)
    model_backends.append(("SynthesizingModel", SynthesizingModel()))

    if not model_backends:
        print("Error: No model backends available!")
        sys.exit(1)

    print(f"\nTesting {len(model_backends)} model backends on {total} questions:\n")
    for name, _ in model_backends:
        print(f"  - {name}")

    # Run each model through all questions
    all_model_results: list[dict[str, Any]] = []

    for model_name, model in model_backends:
        print(f"\n{'='*70}")
        print(f"  Running: {model_name}")
        print(f"{'='*70}")

        results: list[dict[str, Any]] = []
        model_errors = 0

        for i, q in enumerate(questions, 1):
            qtext = q["question"]
            qid = q.get("id", f"q{str(i).zfill(3)}")
            ground_truth = q.get("answer", "")
            marker = f"[{i}/{total}]"

            # Run NEXUS pipeline
            pipeline_result = run_nexus_pipeline(qtext, graph, model, verifier)

            # Entity resolution accuracy
            gt_entity_ids: list[str] = q.get("entities", [])
            nexus_parsed_ids: list[str] = pipeline_result.get("parsed_entity_ids", [])
            entity_resolution_hit = bool(
                gt_entity_ids and any(
                    gid == pid or pid.startswith(gid + "_")
                    for gid in gt_entity_ids
                    for pid in nexus_parsed_ids
                )
            )
            pipeline_result["entity_resolution_hit"] = entity_resolution_hit
            pipeline_result["gt_entity_ids"] = gt_entity_ids

            # Compute accuracy
            scores = compute_key_fact_score(
                pipeline_result["answer"], ground_truth
            )
            pipeline_result["accuracy"] = scores["fuzzy_accuracy"]
            pipeline_result["exact_accuracy"] = scores["exact_accuracy"]
            pipeline_result["scoring_detail"] = scores["scoring_detail"]

            # Conciseness
            question_type = q.get("question_type", "factual")
            answer_tokens = _count_tokens(pipeline_result["answer"])
            gt_tokens = _count_tokens(ground_truth)
            ratio = round(answer_tokens / gt_tokens, 2) if gt_tokens > 0 else 999.0
            verbose_threshold = 5.0 if question_type in ("diagnostic", "multi-hop") else 3.0
            pipeline_result["conciseness"] = {
                "answer_tokens": answer_tokens,
                "ground_truth_tokens": gt_tokens,
                "ratio": ratio,
                "too_verbose": ratio > verbose_threshold,
            }

            if pipeline_result["error"]:
                model_errors += 1
                status = "ERR"
            elif pipeline_result["is_insufficient"]:
                status = "INS"
            elif pipeline_result["passed"]:
                status = "PASS"
            else:
                status = f"HALL({pipeline_result['hallucination_rate']:.0%})"

            fuzzy = scores["fuzzy_accuracy"]
            print(
                f"  {marker} {qid}: {status} | "
                f"fuzzy={fuzzy if fuzzy is not None else 'N/A'} | "
                f"paths={pipeline_result['path_count']} | "
                f"latency={pipeline_result['latency_s']:.3f}s"
            )

            results.append({
                "question_id": qid,
                "question": qtext,
                "ground_truth": ground_truth,
                "question_type": q.get("question_type", ""),
                "difficulty": q.get("difficulty", ""),
                "hops": q.get("hops", 1),
                "pipeline": pipeline_result,
            })

        # Compute summary for this model
        accuracies = [
            r["pipeline"]["accuracy"]
            for r in results
            if not r["pipeline"].get("error") and r["pipeline"]["accuracy"] is not None
        ]
        exact_accuracies = [
            r["pipeline"]["exact_accuracy"]
            for r in results
            if not r["pipeline"].get("error") and r["pipeline"]["exact_accuracy"] is not None
        ]
        hall_rates = [
            r["pipeline"]["hallucination_rate"]
            for r in results
            if not r["pipeline"].get("error")
        ]
        latencies = [
            r["pipeline"]["latency_s"]
            for r in results
            if not r["pipeline"].get("error")
        ]
        conciseness_ratios = [
            r["pipeline"]["conciseness"]["ratio"]
            for r in results
            if not r["pipeline"].get("error") and "conciseness" in r["pipeline"]
        ]
        verification_passed = sum(
            1 for r in results
            if r["pipeline"]["passed"] and not r["pipeline"].get("error")
        )

        summary = {
            "total_questions": total,
            "errors": model_errors,
            "avg_accuracy": _avg(accuracies),
            "avg_exact_accuracy": _avg(exact_accuracies),
            "avg_hallucination_rate": _avg(hall_rates),
            "verification_pass_rate": round(verification_passed / total, 4) if total > 0 else 0.0,
            "avg_latency_s": _avg(latencies),
            "avg_conciseness_ratio": _avg(conciseness_ratios),
            "scorable_count": len(accuracies),
        }

        all_model_results.append({
            "name": model_name,
            "results": results,
            "summary": summary,
        })

    # Print comparison table
    print_comparison_table(all_model_results, total)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "limit": args.limit,
            "models_tested": [m["name"] for m in all_model_results],
        },
        "graph_provenance": graph_provenance,
        "model_results": [
            {
                "name": mr["name"],
                "summary": mr["summary"],
                # Don't include full per-question results to keep file size manageable
                # — they're still available in the detailed per-model section
            }
            for mr in all_model_results
        ],
        "detailed_results": [
            {
                "name": mr["name"],
                "results": mr["results"],
            }
            for mr in all_model_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
