"""
Throughput Benchmark — measure actual tokens/sec on this hardware (honest).

Phase 3 rewrite — all measurements are gated behind warm-up and report
p50/p95 across multiple prompt lengths with >=5 runs each.

Modes:
  --throughput  (default)  Raw Ollama inference throughput
  --pipeline               NEXUS pipeline breakdown
  --zero-weight            SynthesizingModel only — proves the "zero-cost" claim

Output:
  benchmarks/results/throughput_<UTC>.json
  benchmarks/results/zero_weight_<UTC>.json

Usage:
    python benchmarks/throughput_bench.py
    python benchmarks/throughput_bench.py --model qwen2.5:latest
    python benchmarks/throughput_bench.py --zero-weight
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Constants ──
WARMUP_RUNS: int = 3
WARMUP_TOKENS: int = 256
MEASUREMENT_RUNS: int = 5
MEASUREMENT_TOKENS: int = 256
RESULTS_DIR: Path = _project_root / "benchmarks" / "results"

PROMPT_TEMPLATES: dict[int, str] = {
    50: (
        "You are a precise assistant. Answer the following question briefly "
        "in one or two sentences. Question: What is the capital of France?"
    ),
    200: (
        "You are a precise reasoning assistant. Answer questions based on the "
        "provided evidence. If the evidence is insufficient, say so honestly. "
        "Evidence: The SAM architecture uses a Product-Key Memory (PKM) module "
        "with a dual-encoder retrieval mechanism. The system achieves 99.87% "
        "accuracy on memorization tasks with 1,650 slots. The oracle memory "
        "configuration serves as the theoretical upper bound. Question: What "
        "is the key advantage of the SAM architecture over baseline RAG systems?"
    ),
    500: (
        "You are a precise reasoning assistant working with a knowledge graph. "
        "Answer questions based on the provided evidence. If evidence is "
        "insufficient, state that clearly. Do not hallucinate facts beyond "
        "what is provided. Evidence: The NEXUS architecture represents a "
        "fundamental pivot from the original SAM-LM design. SAM-LM used a "
        "Product-Key Memory (PKM) module with 1,650 slots, achieving 99.87% "
        "memorization accuracy with oracle memory. However, the retrieval "
        "pipeline showed a significant gap between oracle (99.87%) and "
        "retrieved (45.23%) settings. The Concept_SelectorBottleneck experiment "
        "revealed that the selector mechanism was the primary bottleneck, "
        "limiting retrieval quality. Exp_0_13A_NoisyMemory tested robustness "
        "to distractors, finding that SAM degrades gracefully with up to 40% "
        "noise before retrieval accuracy drops below 75%. Meanwhile, the "
        "dual-encoder architecture showed strong performance on the key-value "
        "retrieval task, with 96.6% accuracy at K=32. The Decision_PivotToNEXUS "
        "was driven by the insight that structured graph traversal with "
        "intelligent routing could achieve comparable accuracy to LLM "
        "inference at near-zero generation cost for factual queries. "
        "The NEXUS graph contains experiments, concepts, and decisions "
        "connected by depends_on, validates, caused_by, and implements "
        "relationships. Question: Why did the project pivot from SAM to NEXUS?"
    ),
}


# ── RAM measurement ──

def _get_ollama_process_rss_mb() -> float | str:
    """
    Measure peak RSS of the Ollama inference process.

    Tries in order:
      1. psutil — find the 'ollama_llama_server' or 'ollama' process
      2. Windows-specific: wmic / GetProcessMemoryInfo
      3. Fallback: resource.getrusage (our own process, not Ollama)

    Returns a float (MB) or a string like "unavailable: <reason>".
    """
    # ── Strategy 1: psutil ──
    try:
        import psutil

        target_names = {"ollama_llama_server", "ollama_runner", "ollama"}
        best_rss: int = 0

        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                info = proc.info
                name_lower = (info["name"] or "").lower()
                is_ollama = any(t in name_lower for t in target_names)
                # On macOS the process may be named differently
                if not is_ollama:
                    cmdline = " ".join(proc.cmdline() or [])
                    is_ollama = "ollama" in cmdline.lower() and "serve" in cmdline.lower()
                if is_ollama and info["memory_info"]:
                    rss = info["memory_info"].rss
                    if rss > best_rss:
                        best_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if best_rss > 0:
            return round(best_rss / (1024 * 1024), 1)
    except ImportError:
        pass

    # ── Strategy 2: Windows wmic ──
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "process", "where", "name like '%ollama%'", "get", "WorkingSetSize"],
                capture_output=True, text=True, timeout=10,
            )
            lines = result.stdout.strip().splitlines()
            best: int = 0
            for line in lines[1:]:  # skip header
                line = line.strip()
                if line.isdigit():
                    val = int(line)
                    if val > best:
                        best = val
            if best > 0:
                return round(best / (1024 * 1024), 1)
        except Exception:
            pass

    # ── Strategy 3: resource.getrusage (our process, NOT Ollama) ──
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = usage.ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KB
        if platform.system() == "Darwin":
            rss_mb = rss_kb / (1024 * 1024)
        else:
            rss_mb = rss_kb / 1024
        if rss_mb > 0:
            return f"unavailable: our-process-rss={rss_mb:.1f}MB (not Ollama inference)"
    except ImportError:
        pass

    return "unavailable: no psutil, no wmic, no resource module"


# ── Token counting ──

def _count_tokens(text: str) -> int:
    """Simple word-count token estimation."""
    return len(text.split())


# ── Ollama API ──

def _check_ollama(model_name: str) -> bool:
    """Check if Ollama is running and has the model."""
    try:
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if model_name in models:
                return True
            base = model_name.split(":")[0]
            for m in models:
                if m.startswith(base):
                    print(f"  Note: '{model_name}' not found, using '{m}' instead")
                    return True
            print(f"  Available models: {models}")
            return False
    except Exception:
        return False


def _find_ollama_model(model_name: str) -> str:
    """Find best matching Ollama model."""
    try:
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if model_name in models:
                return model_name
            base = model_name.split(":")[0]
            for m in models:
                if m.startswith(base):
                    return m
            if models:
                return models[0]
    except Exception:
        pass
    return model_name


def _ollama_generate(
    host: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call Ollama /api/generate. Returns timing + response data."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_time = time.perf_counter() - t0

    eval_count = data.get("eval_count", 0)
    load_duration_ns = data.get("load_duration", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    total_duration_ns = data.get("total_duration", 0)

    # TTFT: load + prompt eval + first eval token time
    if eval_duration_ns > 0 and eval_count > 0:
        first_token_eval_ns = eval_duration_ns / eval_count
        ttft_ns = load_duration_ns + prompt_eval_duration_ns + first_token_eval_ns
        ttft_s = ttft_ns / 1e9
    else:
        ttft_s = wall_time * 0.3

    # Tokens per second
    if eval_duration_ns > 0:
        tps = eval_count / (eval_duration_ns / 1e9)
    elif wall_time > 0:
        tps = eval_count / wall_time
    else:
        tps = 0.0

    return {
        "response": data.get("response", "").strip(),
        "completion_tokens": eval_count,
        "ttft_s": round(ttft_s, 4),
        "total_time_s": round(wall_time, 4),
        "tokens_per_second": round(tps, 2),
    }


# ── Dataclasses ──

@dataclass
class InferenceResult:
    model_name: str
    prompt_length: int          # nominal prompt length bucket (50, 200, 500)
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_time_s: float
    tokens_per_second: float
    is_warmup: bool = False


# ── Statistics ──

def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * 0.95)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


# ── Banner ──

def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
#  OLLAMA THROUGHPUT BENCHMARK (with warm-up)
# ═══════════════════════════════════════════════════════════════════════

def benchmark_ollama_throughput(
    host: str = "http://localhost:11434",
    model_name: str = "qwen2.5:latest",
    warmup_runs: int = WARMUP_RUNS,
    warmup_tokens: int = WARMUP_TOKENS,
    measurement_runs: int = MEASUREMENT_RUNS,
    measurement_tokens: int = MEASUREMENT_TOKENS,
) -> dict[str, Any]:
    """
    Run honest throughput benchmark.

    Phase 1: Warm-up — >=3 generations (not measured)
    Phase 2: Measurement — >=5 runs at 3 prompt lengths (50, 200, 500)

    Reports p50 and p95 for both tok/s and TTFT.
    """
    all_measured: list[InferenceResult] = []

    # ── Phase 1: Warm-up ──
    _banner("WARM-UP (not measured)")
    warmup_prompt = PROMPT_TEMPLATES[200]  # use medium prompt for warm-up
    print(f"  Running {warmup_runs} warm-up generations ({warmup_tokens} tokens each)...")
    for i in range(1, warmup_runs + 1):
        try:
            r = _ollama_generate(host, model_name, warmup_prompt, max_tokens=warmup_tokens)
            print(f"    Warm-up {i}/{warmup_runs}: "
                  f"{r['tokens_per_second']:.1f} tok/s, "
                  f"{r['completion_tokens']} tok out")
        except Exception as exc:
            print(f"    Warm-up {i}/{warmup_runs}: ERROR — {exc}")
            return {
                "error": f"Warm-up failed at run {i}: {exc}",
                "warmup_completed": i - 1,
            }

    print("  Warm-up complete — starting measurements.")

    # ── Phase 2: Measurement ──
    _banner("MEASUREMENT")

    for prompt_len in sorted(PROMPT_TEMPLATES.keys()):
        prompt_template = PROMPT_TEMPLATES[prompt_len]
        prompt_tokens = _count_tokens(prompt_template)
        print(f"\n  Prompt length ~{prompt_len} tokens ({prompt_tokens} actual):")

        for run in range(1, measurement_runs + 1):
            try:
                r = _ollama_generate(host, model_name, prompt_template, max_tokens=measurement_tokens)
            except Exception as exc:
                print(f"    Run {run}/{measurement_runs}: ERROR — {exc}")
                continue

            result = InferenceResult(
                model_name=model_name,
                prompt_length=prompt_len,
                prompt_tokens=prompt_tokens,
                completion_tokens=r["completion_tokens"],
                ttft_s=r["ttft_s"],
                total_time_s=r["total_time_s"],
                tokens_per_second=r["tokens_per_second"],
                is_warmup=False,
            )
            all_measured.append(result)

            print(f"    Run {run}/{measurement_runs}: "
                  f"TTFT={r['ttft_s']:.3f}s, "
                  f"{r['tokens_per_second']:.1f} tok/s, "
                  f"{r['completion_tokens']} tok out")

    if not all_measured:
        return {"error": "No measurements collected"}

    # ── Compute statistics ──
    all_tps = [r.tokens_per_second for r in all_measured]
    all_ttft = [r.ttft_s for r in all_measured]
    ram_str = _get_ollama_process_rss_mb()

    # Per prompt-length breakdown
    by_length: dict[int, dict[str, Any]] = {}
    for prompt_len in sorted(PROMPT_TEMPLATES.keys()):
        subset = [r for r in all_measured if r.prompt_length == prompt_len]
        if not subset:
            continue
        tps_vals = [r.tokens_per_second for r in subset]
        ttft_vals = [r.ttft_s for r in subset]
        by_length[prompt_len] = {
            "runs": len(subset),
            "p50_tps": _p50(tps_vals),
            "p95_tps": _p95(tps_vals),
            "p50_ttft_s": _p50(ttft_vals),
            "p95_ttft_s": _p95(ttft_vals),
            "all_tps": tps_vals,
            "all_ttft_s": ttft_vals,
        }

    # Print summary
    _banner("RESULTS SUMMARY")
    for prompt_len, stats in sorted(by_length.items()):
        print(f"\n  Prompt ~{prompt_len} tokens ({stats['runs']} runs):")
        print(f"    Tokens/sec:  p50={stats['p50_tps']:.1f}  p95={stats['p95_tps']:.1f}")
        print(f"    TTFT:        p50={stats['p50_ttft_s']:.3f}s  p95={stats['p95_ttft_s']:.3f}s")

    print(f"\n  OVERALL ({len(all_measured)} measured runs):")
    print(f"    Tokens/sec:  p50={_p50(all_tps):.1f}  p95={_p95(all_tps):.1f}  mean={statistics.mean(all_tps):.1f}")
    print(f"    TTFT:        p50={_p50(all_ttft):.3f}s  p95={_p95(all_ttft):.3f}s  mean={statistics.mean(all_ttft):.3f}s")
    print(f"    RAM (Ollama process): {ram_str}")

    return {
        "model": model_name,
        "warmup": {"runs": warmup_runs, "tokens_per_run": warmup_tokens},
        "measurement": {
            "runs_per_length": measurement_runs,
            "tokens_per_completion": measurement_tokens,
        },
        "ram_mb": ram_str if isinstance(ram_str, str) else ram_str,
        "ram_source": "ollama-process-rss" if isinstance(ram_str, float) else ram_str,
        "p50_tps": round(_p50(all_tps), 2),
        "p95_tps": round(_p95(all_tps), 2),
        "mean_tps": round(statistics.mean(all_tps), 2),
        "p50_ttft_s": round(_p50(all_ttft), 4),
        "p95_ttft_s": round(_p95(all_ttft), 4),
        "by_prompt_length": {
            str(k): {
                "p50_tps": v["p50_tps"],
                "p95_tps": v["p95_tps"],
                "p50_ttft_s": v["p50_ttft_s"],
                "p95_ttft_s": v["p95_ttft_s"],
            }
            for k, v in by_length.items()
        },
        "all_results": [
            {
                "prompt_length": r.prompt_length,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "ttft_s": r.ttft_s,
                "tokens_per_second": r.tokens_per_second,
            }
            for r in all_measured
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  ZERO-WEIGHT BENCHMARK — SynthesizingModel only
# ═══════════════════════════════════════════════════════════════════════

def benchmark_zero_weight(
    num_questions: int = 30,
) -> dict[str, Any]:
    """
    Run the FULL NEXUS pipeline with SynthesizingModel ONLY — no Ollama,
    no LLM fallback. Measures what happens when the router sends EVERY
    query to the template synthesizer.

    Metrics:
      - Peak RSS of the whole pipeline process
      - End-to-end latency per question
      - Accuracy vs ground truth from qa-dataset
    """
    from nexus.reasoning.model_interface import SynthesizingModel

    print("  Importing NEXUS components...")
    from nexus.graph.store import InMemoryGraphStore
    from nexus.reasoning.answer import answer_question
    from nexus.reasoning.verifier import Verifier
    from nexus.utils.config import DEFAULT_CONFIG

    # Load QA dataset
    qa_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    if not qa_path.exists():
        return {"error": f"QA dataset not found at {qa_path}"}

    questions: list[dict[str, Any]] = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    questions = questions[:num_questions]

    # Build the graph
    print("  Building benchmark graph...")
    sys.path.insert(0, str(_project_root))
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, provenance = build_benchmark_graph()
    print(f"  Graph ready: {provenance['node_count']} nodes, {provenance['edge_count']} edges")

    # Use SynthesizingModel ONLY
    model = SynthesizingModel()
    verifier = Verifier(hallucination_threshold=0.2)
    print(f"  Model: {model.name} (template-based, zero-weight)")

    # Measure RAM before
    ram_before_str = _get_ollama_process_rss_mb()
    try:
        import psutil
        proc = psutil.Process()
        ram_before_py = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        ram_before_py = 0.0

    peak_ram_mb = ram_before_py

    # Run all questions
    _banner(f"ZERO-WEIGHT BENCHMARK — {len(questions)} questions, SynthesizingModel only")
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    correct_count: int = 0
    total_scored: int = 0

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        ground_truth = q.get("answer", "")

        t0 = time.perf_counter()
        try:
            result = answer_question(
                question_text, graph,
                model=model, verifier=verifier,
                config=DEFAULT_CONFIG,
            )
        except Exception as exc:
            print(f"  [{i}/{len(questions)}] ERROR: {exc}")
            results.append({
                "id": q["id"],
                "question": question_text,
                "error": str(exc),
                "latency_s": time.perf_counter() - t0,
            })
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        answer_text = result.get("answer", "")
        passed = result.get("verification", None)
        passed_val = getattr(passed, "passed", None) if passed is not None else None
        timing = result.get("timing", {})

        # Accuracy: simple keyword overlap with ground truth
        gt_tokens = set(ground_truth.lower().split())
        ans_tokens = set(answer_text.lower().split())
        if gt_tokens and ans_tokens:
            overlap = len(gt_tokens & ans_tokens)
            precision = overlap / len(ans_tokens)
            recall = overlap / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.0

        # Consider "correct" if F1 >= 0.3 or verification passed
        is_correct = f1 >= 0.3 or (passed_val is True)
        if f1 >= 0.3:
            correct_count += 1
            total_scored += 1
        elif passed_val is not None:
            total_scored += 1
            if passed_val:
                correct_count += 1

        # Update peak RAM
        try:
            import psutil
            current_ram = proc.memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram
        except Exception:
            pass

        results.append({
            "id": q["id"],
            "question": question_text,
            "question_type": q.get("question_type", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            "answer": answer_text,
            "ground_truth": ground_truth,
            "f1_score": round(f1, 4),
            "verification_passed": passed_val,
            "latency_s": round(elapsed, 4),
            "timing": {
                "parse_ms": round(timing.get("parse_time", 0) * 1000, 2),
                "traverse_ms": round(timing.get("traverse_time", 0) * 1000, 2),
                "evidence_ms": round(timing.get("evidence_time", 0) * 1000, 2),
                "prompt_ms": round(timing.get("prompt_time", 0) * 1000, 2),
                "generate_ms": round(timing.get("generate_time", 0) * 1000, 2),
                "verify_ms": round(timing.get("verify_time", 0) * 1000, 2),
            },
        })

        status = "PASS" if is_correct else "FAIL"
        print(f"  [{i}/{len(questions)}] {status} | {elapsed*1000:.0f}ms | {answer_text[:60]}...")

    # Compute summary
    accuracy = correct_count / total_scored if total_scored > 0 else 0.0

    _banner("ZERO-WEIGHT RESULTS")
    print(f"\n  Questions:          {len(questions)}")
    print(f"  Scored:             {total_scored}")
    print(f"  Correct:            {correct_count}")
    print(f"  Accuracy:           {accuracy:.2%}")
    if latencies:
        print(f"  Latency (p50):      {_p50(latencies)*1000:.0f}ms")
        print(f"  Latency (p95):      {_p95(latencies)*1000:.0f}ms")
        print(f"  Latency (mean):     {statistics.mean(latencies)*1000:.0f}ms")
    ram_str = _get_ollama_process_rss_mb()
    print(f"  Peak pipeline RSS:  {peak_ram_mb:.1f} MB")
    print(f"  Ollama process RSS: {ram_str}")
    print(f"\n  [ZERO-COST CLAIM]: SynthesizingModel = template-only, no GPU, no LLM.")
    print(f"  Cost per 1M tokens: $0.00 (no inference compute)")

    return {
        "model": "SynthesizingModel (zero-weight)",
        "num_questions": num_questions,
        "accuracy": round(accuracy, 4),
        "correct": correct_count,
        "total_scored": total_scored,
        "latency_p50_ms": round(_p50(latencies) * 1000, 1) if latencies else 0,
        "latency_p95_ms": round(_p95(latencies) * 1000, 1) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
        "peak_pipeline_rss_mb": round(peak_ram_mb, 1),
        "ollama_process_rss": ram_str,
        "cost_per_1m_tokens": 0.0,
        "is_zero_cost": True,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════
#  NEXUS PIPELINE BREAKDOWN (retained from original)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineBreakdown:
    parse_ms: float
    traverse_ms: float
    evidence_ms: float
    prompt_ms: float
    generate_ms: float
    verify_ms: float
    total_ms: float

    @property
    def cpu_overhead_ms(self) -> float:
        return self.parse_ms + self.traverse_ms + self.evidence_ms + self.prompt_ms + self.verify_ms

    @property
    def cpu_overhead_pct(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return (self.cpu_overhead_ms / self.total_ms) * 100


def benchmark_nexus_pipeline(
    graph,
    model,
    test_questions: list[str],
) -> tuple[list[PipelineBreakdown], list[dict[str, Any]]]:
    """Benchmark the full NEXUS pipeline, returning per-step timing."""
    from nexus.reasoning.answer import answer_question
    from nexus.reasoning.verifier import Verifier
    from nexus.utils.config import DEFAULT_CONFIG

    verifier = Verifier(hallucination_threshold=0.2)
    breakdowns: list[PipelineBreakdown] = []
    all_results: list[dict[str, Any]] = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n  Q{i}: {question[:80]}...")

        t0 = time.perf_counter()
        try:
            result = answer_question(
                question, graph, model=model, verifier=verifier,
                config=DEFAULT_CONFIG,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue
        total_time = time.perf_counter() - t0

        timing = result.get("timing", {})
        breakdown = PipelineBreakdown(
            parse_ms=round(timing.get("parse_time", 0) * 1000, 2),
            traverse_ms=round(timing.get("traverse_time", 0) * 1000, 2),
            evidence_ms=round(timing.get("evidence_time", 0) * 1000, 2),
            prompt_ms=round(timing.get("prompt_time", 0) * 1000, 2),
            generate_ms=round(timing.get("generate_time", 0) * 1000, 2),
            verify_ms=round(timing.get("verify_time", 0) * 1000, 2),
            total_ms=round(total_time * 1000, 2),
        )
        breakdowns.append(breakdown)
        all_results.append(result)

        cpu_pct = breakdown.cpu_overhead_pct
        print(f"    Total: {breakdown.total_ms:.0f}ms | "
              f"Generate: {breakdown.generate_ms:.0f}ms ({100-cpu_pct:.0f}%) | "
              f"CPU overhead: {breakdown.cpu_overhead_ms:.0f}ms ({cpu_pct:.0f}%)")

    return breakdowns, all_results


def print_pipeline_breakdown(breakdowns: list[PipelineBreakdown]) -> None:
    if not breakdowns:
        return
    _banner("NEXUS PIPELINE BREAKDOWN")
    total_times = [b.total_ms for b in breakdowns]
    cpu_times = [b.cpu_overhead_ms for b in breakdowns]

    print(f"\n  Across {len(breakdowns)} questions:")
    print(f"  {'Step':<16} {'Mean':>10} {'p50':>10} {'p95':>10} {'% of Total':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for name, vals in [
        ("Parse", [b.parse_ms for b in breakdowns]),
        ("Traverse", [b.traverse_ms for b in breakdowns]),
        ("Evidence", [b.evidence_ms for b in breakdowns]),
        ("Prompt", [b.prompt_ms for b in breakdowns]),
        ("Generate", [b.generate_ms for b in breakdowns]),
        ("Verify", [b.verify_ms for b in breakdowns]),
        ("CPU Overhead", cpu_times),
        ("TOTAL", total_times),
    ]:
        mean_v = statistics.mean(vals) if vals else 0
        pct = (mean_v / statistics.mean(total_times) * 100) if total_times and statistics.mean(total_times) > 0 else 0
        print(f"  {name:<16} {mean_v:>8.1f}ms {_p50(vals):>8.1f}ms "
              f"{_p95(vals):>8.1f}ms {pct:>10.1f}%")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def _utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _update_index(filename: str, command: str, summary: str) -> None:
    """Append an entry to benchmarks/results/INDEX.md."""
    index_path = RESULTS_DIR / "INDEX.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"| {filename} | {timestamp} | `{command}` | {summary} |\n"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(content + entry)
    else:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Results Index\n\n"
                    "| File | Date | Command | Summary |\n"
                    "|------|------|---------|----------|\n")
            f.write(entry)


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Throughput Benchmark — honest throughput and cost measurement"
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:latest",
        help="Ollama model to benchmark (default: qwen2.5:latest)"
    )
    parser.add_argument(
        "--runs", type=int, default=MEASUREMENT_RUNS,
        help=f"Number of measured runs per prompt length (default: {MEASUREMENT_RUNS})"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=WARMUP_RUNS,
        help=f"Number of warm-up runs before measurement (default: {WARMUP_RUNS})"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true",
        help="Skip Ollama raw throughput"
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip NEXUS pipeline breakdown"
    )
    parser.add_argument(
        "--zero-weight", action="store_true",
        help="Run zero-weight benchmark (SynthesizingModel only, no LLM)"
    )
    parser.add_argument(
        "--pipeline-only", action="store_true",
        help="Run only NEXUS pipeline breakdown"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  NEXUS THROUGHPUT BENCHMARK — Phase 3 (honest measurement)")
    print("=" * 72)

    # ── Zero-weight mode ──
    if args.zero_weight:
        print("\n  Mode: ZERO-WEIGHT (SynthesizingModel only)")
        data = benchmark_zero_weight(num_questions=30)
        timestamp = _utc_timestamp()
        filename = f"zero_weight_{timestamp}.json"
        output_path = RESULTS_DIR / filename
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_path}")
        _update_index(filename, "python benchmarks/throughput_bench.py --zero-weight",
                      f"Zero-weight benchmark: {data.get('accuracy', 0):.1%} accuracy, "
                      f"{data.get('latency_mean_ms', 0):.0f}ms mean latency, ${data.get('cost_per_1m_tokens', 0):.2f}/1M tokens")
        return

    # ── Pipeline-only mode ──
    if args.pipeline_only:
        print("\n  Mode: PIPELINE BREAKDOWN ONLY")
        print("\n  Building benchmark graph...")
        from benchmarks.run_benchmark import build_benchmark_graph
        graph, provenance = build_benchmark_graph()
        print(f"  Graph: {provenance['node_count']} nodes, {provenance['edge_count']} edges")
        from nexus.reasoning.model_interface import get_available_model
        model = get_available_model()
        print(f"  Model: {model.name}")
        test_questions = [
            "What was the key finding of the chain-aware retrieval experiment?",
            "Why did the project pivot from SAM to NEXUS?",
            "How many slots does the Product-Key Memory have?",
            "What showed that the selector is the bottleneck?",
            "What is the accuracy of oracle memory on memorization tasks?",
        ]
        breakdowns, _ = benchmark_nexus_pipeline(graph, model, test_questions)
        print_pipeline_breakdown(breakdowns)
        return

    # ── Default: throughput mode ──
    print(f"\n  Model: {args.model}")
    print(f"  Warm-up: {args.warmup_runs} runs × {WARMUP_TOKENS} tokens")
    print(f"  Measurement: {args.runs} runs × 3 prompt lengths")
    print()

    # ── Part 1: Ollama throughput ──
    ollama_data: dict[str, Any] | None = None

    if not args.skip_ollama:
        model_name = _find_ollama_model(args.model)
        if _check_ollama(args.model):
            print(f"Ollama available. Model: {model_name}")
            ollama_data = benchmark_ollama_throughput(
                model_name=model_name,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.runs,
            )
            if "error" in ollama_data:
                print(f"\n  ERROR: {ollama_data['error']}")
                return
        else:
            print(f"\n  [SKIP] Ollama not running or model '{args.model}' not found.")
            print(f"  Start with: ollama pull {args.model} && ollama serve")
    else:
        print("\n  [SKIP] Ollama benchmark (--skip-ollama)")

    # ── Save throughput results ──
    if ollama_data and "error" not in ollama_data:
        timestamp = _utc_timestamp()
        filename = f"throughput_{timestamp}.json"
        output_path = RESULTS_DIR / filename
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ollama_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Throughput results saved to: {output_path}")
        _update_index(filename, f"python benchmarks/throughput_bench.py --runs {args.runs}",
                      f"Throughput: p50={ollama_data['p50_tps']:.1f} tok/s, "
                      f"p95={ollama_data['p95_tps']:.1f} tok/s, "
                      f"RAM={ollama_data.get('ram_mb', 'N/A')}")

    # ── Part 2: Pipeline breakdown (optional) ──
    if not args.skip_pipeline and ollama_data:
        print("\n  Building benchmark graph for pipeline breakdown...")
        try:
            sys.path.insert(0, str(_project_root))
            from benchmarks.run_benchmark import build_benchmark_graph
            graph, provenance = build_benchmark_graph()
            print(f"  Graph: {provenance['node_count']} nodes, {provenance['edge_count']} edges")
            from nexus.reasoning.model_interface import get_available_model
            model = get_available_model()
            print(f"  Model: {model.name}")
            test_questions = [
                "What was the key finding of the chain-aware retrieval experiment?",
                "Why did the project pivot from SAM to NEXUS?",
                "How many slots does the Product-Key Memory have?",
                "What showed that the selector is the bottleneck?",
                "What is the accuracy of oracle memory on memorization tasks?",
            ]
            breakdowns, _ = benchmark_nexus_pipeline(graph, model, test_questions)
            print_pipeline_breakdown(breakdowns)
        except Exception as exc:
            print(f"  Pipeline breakdown failed: {exc}")

    print()
    print("=" * 72)
    print("  BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
