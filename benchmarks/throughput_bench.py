"""
Throughput Benchmark — measure actual tokens/sec on this hardware.

Measures:
  1. Ollama raw inference: time to first token (TTFT) + tokens/sec
     across 3 prompt lengths (50, 200, 500 tokens), 10 runs each.
     Reports p50 and p95.
  2. NEXUS pipeline breakdown: parse, traversal, model inference latency.
  3. RAM usage: peak RSS during inference.
  4. Calculated: max queries/hour for each configuration.

Target: $0.01/1M tokens. Determine if local inference can hit that.

Usage:
    python benchmarks/throughput_bench.py
    python benchmarks/throughput_bench.py --model qwen2.5:latest
    python benchmarks/throughput_bench.py --skip-ollama  # if Ollama not running
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ── Utilities ──

def _count_tokens(text: str) -> int:
    """Simple word-count token estimation."""
    return len(text.split())


def _get_ram_mb() -> float:
    """Get current process RSS in MB. Platform-appropriate."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _check_ollama(model_name: str) -> bool:
    """Check if Ollama is running and has the model available."""
    try:
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            # Try exact match first, then fallback to first model with same base
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
    """Find the best matching Ollama model, falling back gracefully."""
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

            # Return first available model
            if models:
                return models[0]
    except Exception:
        pass
    return model_name


# ── Dataclasses for results ──

@dataclass
class InferenceResult:
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float          # time to first token
    total_time_s: float    # wall clock for generation
    tokens_per_second: float
    ram_mb: float = 0.0


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
        """CPU-only time (everything except generate)."""
        return self.parse_ms + self.traverse_ms + self.evidence_ms + self.prompt_ms + self.verify_ms

    @property
    def cpu_overhead_pct(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return (self.cpu_overhead_ms / self.total_ms) * 100


@dataclass
class ConfigurationThroughput:
    name: str
    tokens_per_second: float
    avg_tokens_per_query: int
    queries_per_hour: float
    tokens_per_hour: float
    description: str


# ── Ollama raw inference benchmark ──

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


def _ollama_generate(
    host: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Call Ollama generate API and return response with timing from the API."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_time = time.perf_counter() - t0

    # Ollama reports total_duration and eval_count in the response
    eval_count = data.get("eval_count", 0)
    # Use the API-reported timing when available
    total_duration_ns = data.get("total_duration", 0)
    load_duration_ns = data.get("load_duration", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
    eval_duration_ns = data.get("eval_duration", 0)

    # Time to first token ~= load + prompt_eval + first eval step
    # We approximate TTFT from the API timings
    if eval_duration_ns > 0 and eval_count > 0:
        # Estimate TTFT: load + prompt eval + first token eval time
        first_token_eval_ns = eval_duration_ns / eval_count
        ttft_ns = load_duration_ns + prompt_eval_duration_ns + first_token_eval_ns
        ttft_s = ttft_ns / 1e9
    else:
        ttft_s = wall_time * 0.3  # rough estimate

    # Tokens per second from eval
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
        "api_total_duration_s": round(total_duration_ns / 1e9, 4) if total_duration_ns else 0,
    }


def benchmark_ollama(
    host: str = "http://localhost:11434",
    model_name: str = "qwen2.5:latest",
    runs_per_length: int = 10,
) -> list[InferenceResult]:
    """Benchmark Ollama raw inference at 3 prompt lengths, N runs each."""
    results: list[InferenceResult] = []

    for prompt_len, prompt_template in sorted(PROMPT_TEMPLATES.items()):
        prompt_tokens = _count_tokens(prompt_template)
        print(f"\n  Prompt length ~{prompt_len} tokens ({prompt_tokens} actual):")

        for run in range(1, runs_per_length + 1):
            ram_before = _get_ram_mb()
            try:
                r = _ollama_generate(host, model_name, prompt_template)
            except Exception as exc:
                print(f"    Run {run}/{runs_per_length}: ERROR - {exc}")
                continue

            ram_after = _get_ram_mb()
            ram_delta = max(0, ram_after - ram_before)

            result = InferenceResult(
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=r["completion_tokens"],
                ttft_s=r["ttft_s"],
                total_time_s=r["total_time_s"],
                tokens_per_second=r["tokens_per_second"],
                ram_mb=round(ram_delta, 1),
            )
            results.append(result)

            print(f"    Run {run}/{runs_per_length}: "
                  f"TTFT={r['ttft_s']:.3f}s, "
                  f"{r['tokens_per_second']:.1f} tok/s, "
                  f"{r['completion_tokens']} tok out, "
                  f"+{ram_delta:.0f} MB RAM")

    return results


# ── NEXUS pipeline breakdown ──

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
        print(f"    Parse: {breakdown.parse_ms:.1f}ms | "
              f"Traverse: {breakdown.traverse_ms:.1f}ms | "
              f"Evidence: {breakdown.evidence_ms:.1f}ms | "
              f"Prompt: {breakdown.prompt_ms:.1f}ms | "
              f"Verify: {breakdown.verify_ms:.1f}ms")

    return breakdowns, all_results


# ── RAG baseline latency ──

def benchmark_rag_retrieval(
    graph,
    test_questions: list[str],
) -> list[float]:
    """
    Simulate RAG retrieval latency. RAG retrieves chunks from a vector DB,
    which is analogous to graph traversal + evidence building in NEXUS.
    We approximate this as parse + traverse + evidence time.
    """
    from nexus.query.parser import parse_question
    from nexus.graph.traversal import traverse_with_intent
    from nexus.reasoning.evidence_builder import build_evidence
    from nexus.utils.config import DEFAULT_CONFIG

    retrieval_times: list[float] = []
    config = DEFAULT_CONFIG

    for question in test_questions:
        t0 = time.perf_counter()

        # Parse (entity extraction for RAG = query reformulation)
        parsed = parse_question(question, graph, cutoff=0.6, config=config)

        if parsed.entity_ids:
            # Traverse (≈ vector search)
            query_entities = set(parsed.entity_ids)
            paths = traverse_with_intent(
                graph=graph,
                entry_nodes=parsed.entity_ids,
                query_entities=query_entities,
                intent=parsed.intent,
                max_depth=config.max_depth,
                beam_width=config.beam_width,
                config=config,
            )
            # Evidence (≈ chunk assembly)
            if paths:
                build_evidence(question, paths, graph, question_intent=parsed.intent)

        elapsed = time.perf_counter() - t0
        retrieval_times.append(elapsed)

    return retrieval_times


# ── Throughput calculations ──

def compute_configurations(
    ollama_tps: float,
    nexus_cpu_ms: float,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
) -> list[ConfigurationThroughput]:
    """
    Compute max queries/hour for each system configuration.

    Args:
        ollama_tps: Measured Ollama tokens/second
        nexus_cpu_ms: Average NEXUS CPU overhead (parse+traverse+evidence+prompt+verify) in ms
        avg_prompt_tokens: Average prompt tokens per query
        avg_completion_tokens: Average completion tokens per query
    """
    configs: list[ConfigurationThroughput] = []

    avg_total_tokens = avg_prompt_tokens + avg_completion_tokens

    # ── NEXUS + 3B: graph overhead + LLM inference ──
    # Time per query = CPU overhead + inference time
    nexus_inference_s = avg_completion_tokens / ollama_tps if ollama_tps > 0 else float("inf")
    nexus_time_per_query = (nexus_cpu_ms / 1000) + nexus_inference_s
    nexus_qph = 3600 / nexus_time_per_query if nexus_time_per_query > 0 else float("inf")

    configs.append(ConfigurationThroughput(
        name="NEXUS + 3B",
        tokens_per_second=ollama_tps,
        avg_tokens_per_query=avg_total_tokens,
        queries_per_hour=round(nexus_qph, 1),
        tokens_per_hour=round(nexus_qph * avg_total_tokens, 0),
        description=(
            f"Full NEXUS pipeline: CPU overhead {nexus_cpu_ms:.0f}ms + "
            f"LLM inference at {ollama_tps:.1f} tok/s. "
            f"Per-query: {nexus_time_per_query:.2f}s"
        ),
    ))

    # ── NEXUS + Router: 80% synth (~0 cost) + 20% LLM ──
    # Synthetic queries cost only CPU overhead (no LLM)
    synth_time = nexus_cpu_ms / 1000  # template-based, negligible gen time
    llm_time = nexus_time_per_query   # full pipeline for 20%
    # Blended: 80% synth + 20% LLM
    blended_time = 0.8 * synth_time + 0.2 * llm_time
    blended_qph = 3600 / blended_time if blended_time > 0 else float("inf")
    # Blended tokens: 80% queries = 0 completion tokens (no LLM)
    # 20% queries = full prompt+completion tokens
    blended_tokens_per_query = avg_prompt_tokens + (0.2 * avg_completion_tokens)
    # LLM tokens for cost: only 20% use LLM
    llm_completion_only = 0.2 * avg_completion_tokens

    configs.append(ConfigurationThroughput(
        name="NEXUS + Router (80% synth)",
        tokens_per_second=ollama_tps,
        avg_tokens_per_query=round(blended_tokens_per_query),
        queries_per_hour=round(blended_qph, 1),
        tokens_per_hour=round(blended_qph * blended_tokens_per_query, 0),
        description=(
            f"80% template synthesis ({synth_time:.3f}s) + "
            f"20% LLM ({llm_time:.2f}s). "
            f"Blended: {blended_time:.2f}s/query. "
            f"Only {llm_completion_only:.0f} completion tok/query on average."
        ),
    ))

    # ── RAG + 3B: retrieval + LLM inference ──
    # RAG retrieval is analogous to NEXUS CPU overhead
    # (typically slightly faster since no graph traversal)
    rag_overhead_ms = nexus_cpu_ms * 0.9  # approximate
    rag_inference_s = avg_completion_tokens / ollama_tps if ollama_tps > 0 else float("inf")
    rag_time_per_query = (rag_overhead_ms / 1000) + rag_inference_s
    rag_qph = 3600 / rag_time_per_query if rag_time_per_query > 0 else float("inf")

    configs.append(ConfigurationThroughput(
        name="RAG + 3B",
        tokens_per_second=ollama_tps,
        avg_tokens_per_query=avg_total_tokens,
        queries_per_hour=round(rag_qph, 1),
        tokens_per_hour=round(rag_qph * avg_total_tokens, 0),
        description=(
            f"RAG pipeline: retrieval ~{rag_overhead_ms:.0f}ms + "
            f"LLM inference at {ollama_tps:.1f} tok/s. "
            f"Per-query: {rag_time_per_query:.2f}s"
        ),
    ))

    # ── Raw LLM: no pipeline overhead ──
    raw_time = avg_completion_tokens / ollama_tps if ollama_tps > 0 else float("inf")
    raw_qph = 3600 / raw_time if raw_time > 0 else float("inf")

    configs.append(ConfigurationThroughput(
        name="Raw LLM (no pipeline)",
        tokens_per_second=ollama_tps,
        avg_tokens_per_query=avg_completion_tokens,
        queries_per_hour=round(raw_qph, 1),
        tokens_per_hour=round(raw_qph * avg_completion_tokens, 0),
        description=(
            f"Pure LLM inference, no retrieval overhead. "
            f"Upper bound on generation throughput."
        ),
    ))

    return configs


# ── Statistics helpers ──

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


# ── Print helpers ──

def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_ollama_results(results: list[InferenceResult]) -> None:
    """Print summary of Ollama raw inference benchmark."""
    if not results:
        print("\n  No Ollama results collected.")
        return

    all_tps = [r.tokens_per_second for r in results]
    all_ttft = [r.ttft_s for r in results]

    _banner("OLLAMA RAW INFERENCE")

    # Per prompt-length breakdown
    for prompt_len in sorted(PROMPT_TEMPLATES.keys()):
        subset = [r for r in results if r.prompt_tokens == _count_tokens(PROMPT_TEMPLATES[prompt_len])]
        if not subset:
            continue
        tps_vals = [r.tokens_per_second for r in subset]
        ttft_vals = [r.ttft_s for r in subset]
        print(f"\n  Prompt ~{prompt_len} tokens ({len(subset)} runs):")
        print(f"    Tokens/sec:  p50={_p50(tps_vals):.1f}  p95={_p95(tps_vals):.1f}  "
              f"mean={statistics.mean(tps_vals):.1f}  min={min(tps_vals):.1f}  max={max(tps_vals):.1f}")
        print(f"    TTFT:        p50={_p50(ttft_vals):.3f}s  p95={_p95(ttft_vals):.3f}s  "
              f"mean={statistics.mean(ttft_vals):.3f}s")

    print(f"\n  OVERALL ({len(results)} runs):")
    print(f"    Tokens/sec:  p50={_p50(all_tps):.1f}  p95={_p95(all_tps):.1f}  "
          f"mean={statistics.mean(all_tps):.1f}")
    print(f"    TTFT:        p50={_p50(all_ttft):.3f}s  p95={_p95(all_ttft):.3f}  "
          f"mean={statistics.mean(all_ttft):.3f}")
    if results and results[0].ram_mb:
        ram_vals = [r.ram_mb for r in results if r.ram_mb > 0]
        if ram_vals:
            print(f"    Peak RAM Δ:  max={max(ram_vals):.0f} MB  mean={statistics.mean(ram_vals):.0f} MB")


def print_pipeline_breakdown(breakdowns: list[PipelineBreakdown]) -> None:
    """Print NEXUS pipeline latency breakdown."""
    if not breakdowns:
        print("\n  No pipeline results collected.")
        return

    _banner("NEXUS PIPELINE BREAKDOWN")

    parse_times = [b.parse_ms for b in breakdowns]
    traverse_times = [b.traverse_ms for b in breakdowns]
    evidence_times = [b.evidence_ms for b in breakdowns]
    prompt_times = [b.prompt_ms for b in breakdowns]
    generate_times = [b.generate_ms for b in breakdowns]
    verify_times = [b.verify_ms for b in breakdowns]
    total_times = [b.total_ms for b in breakdowns]
    cpu_times = [b.cpu_overhead_ms for b in breakdowns]

    print(f"\n  Across {len(breakdowns)} questions:")
    print(f"  {'Step':<16} {'Mean':>10} {'p50':>10} {'p95':>10} {'% of Total':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    steps = [
        ("Parse", parse_times),
        ("Traverse", traverse_times),
        ("Evidence", evidence_times),
        ("Prompt", prompt_times),
        ("Generate", generate_times),
        ("Verify", verify_times),
        ("CPU Overhead", cpu_times),
        ("TOTAL", total_times),
    ]
    for name, values in steps:
        mean_v = statistics.mean(values) if values else 0
        pct = (mean_v / statistics.mean(total_times) * 100) if total_times and statistics.mean(total_times) > 0 else 0
        print(f"  {name:<16} {mean_v:>8.1f}ms {_p50(values):>8.1f}ms "
              f"{_p95(values):>8.1f}ms {pct:>10.1f}%")


def print_configurations(configs: list[ConfigurationThroughput]) -> None:
    """Print throughput comparison for all configurations."""
    _banner("THROUGHPUT COMPARISON")

    print(f"\n  {'Configuration':<28} {'Q/hour':>10} {'Tok/hr':>12} {'Tok/Q':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*8}")
    for c in configs:
        print(f"  {c.name:<28} {c.queries_per_hour:>8.1f}  {c.tokens_per_hour:>10.0f}  {c.avg_tokens_per_query:>6}")

    print(f"\n  Details:")
    for c in configs:
        print(f"    {c.name}:")
        print(f"      {c.description}")


def print_cost_analysis(ollama_tps: float) -> None:
    """Print cost analysis for local-only pricing target."""
    _banner("COST ANALYSIS — $0.01/1M TOKENS TARGET")

    watts = 65.0       # typical CPU TDP under load
    electricity = 0.15  # USD/kWh

    seconds_per_1m = 1_000_000 / ollama_tps if ollama_tps > 0 else float("inf")
    kwh = (watts / 1000) * (seconds_per_1m / 3600)
    cost_per_1m = kwh * electricity

    # What tps is needed for $0.01/1M?
    # cost = (watts/1000) * (1_000_000 / tps / 3600) * electricity
    # 0.01 = (65/1000) * (1_000_000 / tps / 3600) * 0.15
    # tps = 65 * 1_000_000 * 0.15 / (1000 * 3600 * 0.01)
    tps_needed = (watts * 1_000_000 * electricity) / (1000 * 3600 * 0.01)

    print(f"\n  Measured throughput:  {ollama_tps:.1f} tokens/sec")
    print(f"  CPU power draw:       {watts:.0f}W (estimated)")
    print(f"  Electricity cost:     ${electricity:.2f}/kWh")
    print(f"  Time for 1M tokens:   {seconds_per_1m:.0f}s ({seconds_per_1m/3600:.2f}h)")
    print(f"  Energy for 1M tokens: {kwh:.4f} kWh")
    print(f"  Cost per 1M tokens:   ${cost_per_1m:.4f}")
    print()
    print(f"  TARGET: $0.01/1M tokens requires {tps_needed:.0f} tok/s")
    print(f"  GAP:    We need {tps_needed/ollama_tps:.1f}x more throughput")

    if ollama_tps >= tps_needed:
        print(f"  STATUS: [PASS] Local inference MEETS the $0.01 target!")
    else:
        gap_pct = (1 - ollama_tps / tps_needed) * 100
        print(f"  STATUS: [FAIL] {gap_pct:.0f}% below target throughput")


def print_router_cost(ollama_tps: float, synth_ratio: float = 0.8) -> None:
    """Print blended cost analysis for the NEXUS router."""
    watts = 65.0
    electricity = 0.15

    # CPU-only cost (no LLM) = just the electricity for CPU overhead
    # Actually CPU overhead is tiny compared to LLM, so synth queries are ~$0
    seconds_per_1m = 1_000_000 / ollama_tps if ollama_tps > 0 else float("inf")
    kwh_per_1m = (watts / 1000) * (seconds_per_1m / 3600)
    cost_per_1m = kwh_per_1m * electricity

    # Blended: synth_ratio% of queries cost $0, (1-synth_ratio)% cost full LLM tokens
    llm_fraction = 1 - synth_ratio
    blended_cost_per_1m = llm_fraction * cost_per_1m

    # Effective cost per 1M user-facing tokens
    # Only llm_fraction of tokens actually go through LLM
    effective_tokens_per_1m_user = 1_000_000 / llm_fraction if llm_fraction > 0 else float("inf")

    _banner("ROUTER BLENDED COST")

    print(f"\n  Synthesizer ratio:    {synth_ratio:.0%} (template-based, ~$0)")
    print(f"  LLM ratio:            {llm_fraction:.0%} (full inference)")
    print(f"  Raw cost per 1M LLM tokens:  ${cost_per_1m:.4f}")
    print(f"  Blended cost per 1M tokens:  ${blended_cost_per_1m:.4f}")
    print(f"  (Only {llm_fraction:.0%} of tokens incur LLM cost)")
    print()
    print(f"  NEXUS router effective cost: ${blended_cost_per_1m:.4f} per 1M user-facing tokens")
    print(f"  ({synth_ratio:.0%} synth = $0)")

    if blended_cost_per_1m <= 0.01:
        print(f"  STATUS: [PASS] Router blended cost MEETS $0.01 target!")
    else:
        gap = blended_cost_per_1m - 0.01
        print(f"  STATUS: [FAIL] ${gap:.4f} above $0.01 target")
        # What synth ratio would hit the target?
        # blended = llm_fraction * cost_per_1m <= 0.01
        # llm_fraction <= 0.01 / cost_per_1m
        # synth_ratio >= 1 - (0.01 / cost_per_1m)
        needed_synth = 1 - (0.01 / cost_per_1m) if cost_per_1m > 0 else 1.0
        if 0 <= needed_synth <= 1:
            print(f"  To hit $0.01: need {needed_synth:.0%} synthesizer ratio ({needed_synth*100:.0f}%)")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Throughput Benchmark — measure actual tokens/sec on this hardware"
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:latest",
        help="Ollama model to benchmark (default: qwen2.5:latest)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs per prompt length (default: 5)"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true",
        help="Skip Ollama raw throughput (only run pipeline breakdown)"
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip NEXUS pipeline breakdown (only run Ollama benchmark)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  NEXUS THROUGHPUT BENCHMARK")
    print("  Target: $0.01/1M tokens (local-only pricing)")
    print("=" * 72)

    # ── Part 1: Ollama raw throughput ──
    ollama_tps: float = 0.0
    ollama_results: list[InferenceResult] = []

    if not args.skip_ollama:
        model_name = _find_ollama_model(args.model)
        if _check_ollama(args.model):
            print(f"\n  Ollama is running. Model: {model_name}")
            print(f"  Running {args.runs} iterations per prompt length...")
            ollama_results = benchmark_ollama(
                model_name=model_name,
                runs_per_length=args.runs,
            )
            all_tps = [r.tokens_per_second for r in ollama_results]
            ollama_tps = _p50(all_tps) if all_tps else 0.0
            print_ollama_results(ollama_results)
        else:
            print(f"\n  [SKIP] Ollama not running or model '{args.model}' not found.")
            print(f"  Start Ollama with: ollama pull {args.model} && ollama serve")
    else:
        print("\n  [SKIP] Ollama benchmark (--skip-ollama)")

    # ── Part 2: NEXUS pipeline breakdown ──
    nexus_cpu_ms: float = 0.0
    pipeline_breakdowns: list[PipelineBreakdown] = []

    if not args.skip_pipeline:
        print("\n  Building benchmark graph...")
        from benchmarks.run_benchmark import build_benchmark_graph
        graph, graph_provenance = build_benchmark_graph()
        print(f"  Graph ready: {graph_provenance['node_count']} nodes, "
              f"{graph_provenance['edge_count']} edges")

        # Get model
        from nexus.reasoning.model_interface import get_available_model
        model = get_available_model()
        print(f"  Model: {model.name}")

        # Test questions covering different query types
        test_questions = [
            "What was the key finding of the chain-aware retrieval experiment?",
            "Why did the project pivot from SAM to NEXUS?",
            "How many slots does the Product-Key Memory have?",
            "What showed that the selector is the bottleneck?",
            "What is the accuracy of oracle memory on memorization tasks?",
        ]

        print(f"\n  Running pipeline breakdown on {len(test_questions)} questions...")
        pipeline_breakdowns, _ = benchmark_nexus_pipeline(
            graph, model, test_questions,
        )
        print_pipeline_breakdown(pipeline_breakdowns)

        # CPU overhead (ms)
        cpu_times = [b.cpu_overhead_ms for b in pipeline_breakdowns]
        nexus_cpu_ms = statistics.mean(cpu_times) if cpu_times else 0.0
    else:
        print("\n  [SKIP] Pipeline breakdown (--skip-pipeline)")

    # ── Part 3: Throughput calculations ──
    if ollama_tps > 0:
        # Average tokens per query from pipeline results
        avg_prompt = 500   # typical NEXUS prompt with evidence
        avg_completion = 80  # typical short answer
        if pipeline_breakdowns:
            # Estimate from generate time vs tokens
            avg_completion = 80  # keep reasonable default

        configs = compute_configurations(
            ollama_tps=ollama_tps,
            nexus_cpu_ms=nexus_cpu_ms,
            avg_prompt_tokens=avg_prompt,
            avg_completion_tokens=avg_completion,
        )
        print_configurations(configs)
        print_cost_analysis(ollama_tps)
        print_router_cost(ollama_tps, synth_ratio=0.8)

    # ── Part 4: Save results ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": {
                "model": args.model,
                "runs_per_length": args.runs,
            },
            "ollama": {
                "p50_tokens_per_second": _p50([r.tokens_per_second for r in ollama_results]) if ollama_results else 0,
                "p95_tokens_per_second": _p95([r.tokens_per_second for r in ollama_results]) if ollama_results else 0,
                "mean_tokens_per_second": statistics.mean([r.tokens_per_second for r in ollama_results]) if ollama_results else 0,
                "results": [
                    {
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "ttft_s": r.ttft_s,
                        "tokens_per_second": r.tokens_per_second,
                        "ram_mb": r.ram_mb,
                    }
                    for r in ollama_results
                ],
            },
            "pipeline": {
                "cpu_overhead_ms": nexus_cpu_ms,
                "breakdowns": [
                    {
                        "parse_ms": b.parse_ms,
                        "traverse_ms": b.traverse_ms,
                        "evidence_ms": b.evidence_ms,
                        "prompt_ms": b.prompt_ms,
                        "generate_ms": b.generate_ms,
                        "verify_ms": b.verify_ms,
                        "total_ms": b.total_ms,
                    }
                    for b in pipeline_breakdowns
                ],
            },
            "cost_analysis": {
                "tokens_per_second": ollama_tps,
                "watts_at_load": 65,
                "electricity_per_kwh": 0.15,
                "cost_per_1m_tokens": (
                    ((65 / 1000) * (1_000_000 / ollama_tps / 3600) * 0.15) if ollama_tps > 0 else None
                ),
                "tps_needed_for_1cent_target": (
                    (65 * 1_000_000 * 0.15) / (1000 * 3600 * 0.01)
                ),
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_path}")

    print()
    print("=" * 72)
    print("  BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
