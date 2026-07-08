"""
RAM & Throughput Measurement — peak RSS per architecture arm + warmed throughput.

Measures on real hardware (NOT CI-capable):
  1. Peak RSS for zero-weight (SynthesizingModel only, no Ollama)
  2. Peak RSS for NEXUS + 3B (Ollama + SynthesizingModel via FallbackModel)
  3. Peak RSS for RAG + 3B (chunk retrieval + Ollama, embeddings in memory)
  4. Warmed Ollama throughput (5 warmup, 10 measurement at 3 prompt lengths)

Uses psutil.Process().memory_info().rss for all measurements.
Output: benchmarks/results/ram_throughput_<timestamp>.json

Usage:
    python benchmarks/ram_throughput.py
    python benchmarks/ram_throughput.py --skip-ollama  # zero-weight + RAG only
    python benchmarks/ram_throughput.py --ram-only      # skip throughput recalibration
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Ensure project root on sys.path ──
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

RESULTS_DIR = _project_root / "benchmarks" / "results"

# ── Prompt templates for throughput ──
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

# ── Warm-up and measurement constants ──
WARMUP_RUNS: int = 5
WARMUP_TOKENS: int = 256
MEASUREMENT_RUNS: int = 10
MEASUREMENT_TOKENS: int = 256
ZERO_WEIGHT_QUESTIONS: int = 30


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    return len(text.split())


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


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_machine_info() -> dict[str, str]:
    """Get hardware info for the results file."""
    info: dict[str, str] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    # Try to get CPU model name
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=10,
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) > 1:
                info["cpu_model"] = lines[1].strip()
        except Exception:
            info["cpu_model"] = platform.processor() or "unknown"
    else:
        try:
            result = subprocess.run(
                ["cat", "/proc/cpuinfo"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            info["cpu_model"] = platform.processor() or "unknown"

    # Get total RAM
    try:
        import psutil
        info["ram_total_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        info["ram_total_gb"] = "unknown"

    return info


# ── RSS measurement ──

def _get_process_rss_mb() -> float:
    """Get current process RSS in MB using psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _get_ollama_rss_mb() -> float | str:
    """Measure peak RSS of the Ollama inference process using psutil."""
    try:
        import psutil

        target_names = {"ollama_llama_server", "ollama_runner", "ollama"}
        best_rss: int = 0

        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                info = proc.info
                name_lower = (info["name"] or "").lower()
                is_ollama = any(t in name_lower for t in target_names)
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

    return "unavailable: psutil not installed"


# ── Ollama API ──

def _check_ollama(model_name: str) -> tuple[bool, str]:
    """Check if Ollama is running and has the model. Returns (available, resolved_name)."""
    try:
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            if model_name in models:
                return True, model_name
            base = model_name.split(":")[0]
            for m in models:
                if m.startswith(base):
                    print(f"  Note: '{model_name}' not found, using '{m}' instead")
                    return True, m
            print(f"  Available models: {models}")
            return False, model_name
    except Exception:
        return False, model_name


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


# ═══════════════════════════════════════════════════════════════════════
#  ARM 1: ZERO-WEIGHT — SynthesizingModel only, no Ollama
# ═══════════════════════════════════════════════════════════════════════

def measure_zero_weight_ram() -> dict[str, Any]:
    """
    Measure peak RSS of the SynthesizingModel pipeline.
    Load graph, run 30 questions, measure peak RSS during end-to-end pipeline.
    """
    from nexus.reasoning.model_interface import SynthesizingModel

    print("  Importing NEXUS components...")
    from nexus.graph.store import InMemoryGraphStore
    from nexus.reasoning.answer import answer_question
    from nexus.reasoning.verifier import Verifier
    from nexus.utils.config import DEFAULT_CONFIG

    # Measure RSS before
    rss_before = _get_process_rss_mb()
    print(f"  RSS before loading: {rss_before:.1f} MB")

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
    questions = questions[:ZERO_WEIGHT_QUESTIONS]

    # Build the graph
    print("  Building benchmark graph...")
    sys.path.insert(0, str(_project_root))
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, provenance = build_benchmark_graph()
    print(f"  Graph ready: {provenance['node_count']} nodes, {provenance['edge_count']} edges")

    rss_after_graph = _get_process_rss_mb()
    print(f"  RSS after graph load: {rss_after_graph:.1f} MB (delta: {rss_after_graph - rss_before:.1f} MB)")

    # Use SynthesizingModel ONLY
    model = SynthesizingModel()
    verifier = Verifier(hallucination_threshold=0.2)
    print(f"  Model: {model.name} (template-based, zero-weight)")

    rss_after_model = _get_process_rss_mb()
    print(f"  RSS after model init: {rss_after_model:.1f} MB (delta: {rss_after_model - rss_before:.1f} MB)")

    # Track peak
    peak_ram_mb = rss_after_model

    # Run all questions
    _banner(f"ZERO-WEIGHT BENCHMARK — {len(questions)} questions, SynthesizingModel only")
    latencies: list[float] = []
    peak_times: list[tuple[float, str]] = []  # (rss, label) for peak tracking

    for i, q in enumerate(questions, 1):
        question_text = q["question"]

        t0 = time.perf_counter()
        try:
            result = answer_question(
                question_text, graph,
                model=model, verifier=verifier,
                config=DEFAULT_CONFIG,
            )
        except Exception as exc:
            print(f"  [{i}/{len(questions)}] ERROR: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Update peak RAM
        try:
            import psutil
            current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram
                peak_times.append((current_ram, f"Q{i}"))
        except Exception:
            pass

        answer_text = result.get("answer", "")
        status = "OK" if result.get("verification") else "FAIL"
        print(f"  [{i}/{len(questions)}] {status} | {elapsed*1000:.0f}ms | {answer_text[:60]}...")

    rss_after = _get_process_rss_mb()

    # Cleanup: delete references to allow GC
    import gc
    del graph
    del model
    del verifier
    del questions
    gc.collect()

    rss_after_cleanup = _get_process_rss_mb()
    print(f"\n  RSS after cleanup: {rss_after_cleanup:.1f} MB")

    _banner("ZERO-WEIGHT RAM RESULTS")
    print(f"  RSS before:      {rss_before:.1f} MB")
    print(f"  RSS peak:        {peak_ram_mb:.1f} MB")
    print(f"  RSS delta:       {peak_ram_mb - rss_before:.1f} MB")
    print(f"  RSS after:       {rss_after_cleanup:.1f} MB")
    print(f"  Latency (p50):   {_p50(latencies)*1000:.0f}ms")
    print(f"  Latency (p95):   {_p95(latencies)*1000:.0f}ms")

    return {
        "arm": "zero-weight",
        "rss_before_mb": round(rss_before, 1),
        "rss_peak_mb": round(peak_ram_mb, 1),
        "rss_delta_mb": round(peak_ram_mb - rss_before, 1),
        "rss_after_mb": round(rss_after_cleanup, 1),
        "rss_after_graph_mb": round(rss_after_graph, 1),
        "rss_after_model_mb": round(rss_after_model, 1),
        "num_questions": ZERO_WEIGHT_QUESTIONS,
        "num_completed": len(latencies),
        "latency_p50_ms": round(_p50(latencies) * 1000, 1) if latencies else 0,
        "latency_p95_ms": round(_p95(latencies) * 1000, 1) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  ARM 2: NEXUS + 3B — Ollama + SynthesizingModel (FallbackModel)
# ═══════════════════════════════════════════════════════════════════════

def measure_nexus_3b_ram(model_name: str) -> dict[str, Any]:
    """
    Measure peak RSS of NEXUS with FallbackModel (Ollama + SynthesizingModel).
    Load graph, run 30 questions through full pipeline, measure peak RSS.
    """
    print("  Importing NEXUS components...")
    from nexus.graph.store import InMemoryGraphStore
    from nexus.reasoning.answer import answer_question
    from nexus.reasoning.model_interface import FallbackModel, OllamaModel, SynthesizingModel
    from nexus.reasoning.verifier import Verifier
    from nexus.utils.config import DEFAULT_CONFIG

    # Measure RSS before
    rss_before = _get_process_rss_mb()
    ollama_rss_before = _get_ollama_rss_mb()
    print(f"  RSS before loading: {rss_before:.1f} MB (Ollama: {ollama_rss_before})")

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
    questions = questions[:ZERO_WEIGHT_QUESTIONS]

    # Build the graph
    print("  Building benchmark graph...")
    sys.path.insert(0, str(_project_root))
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, provenance = build_benchmark_graph()
    print(f"  Graph ready: {provenance['node_count']} nodes, {provenance['edge_count']} edges")

    rss_after_graph = _get_process_rss_mb()
    print(f"  RSS after graph load: {rss_after_graph:.1f} MB")

    # Create FallbackModel (Ollama + SynthesizingModel)
    synth = SynthesizingModel()
    ollama_model = OllamaModel(model_name=model_name)
    model = FallbackModel(primary=ollama_model, fallback=synth)
    verifier = Verifier(hallucination_threshold=0.2)
    print(f"  Model: {model.name}")

    rss_after_model = _get_process_rss_mb()
    print(f"  RSS after model init: {rss_after_model:.1f} MB")

    # Track peak
    peak_ram_mb = rss_after_model

    # Run all questions
    _banner(f"NEXUS+3B BENCHMARK — {len(questions)} questions, FallbackModel")
    latencies: list[float] = []

    for i, q in enumerate(questions, 1):
        question_text = q["question"]

        t0 = time.perf_counter()
        try:
            result = answer_question(
                question_text, graph,
                model=model, verifier=verifier,
                config=DEFAULT_CONFIG,
            )
        except Exception as exc:
            print(f"  [{i}/{len(questions)}] ERROR: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Update peak RAM
        try:
            import psutil
            current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram
        except Exception:
            pass

        answer_text = result.get("answer", "")
        print(f"  [{i}/{len(questions)}] | {elapsed*1000:.0f}ms | {answer_text[:60]}...")

    rss_after = _get_process_rss_mb()
    ollama_rss_after = _get_ollama_rss_mb()

    # Cleanup
    import gc
    del graph
    del model
    del synth
    del ollama_model
    del verifier
    del questions
    gc.collect()

    rss_after_cleanup = _get_process_rss_mb()
    print(f"\n  RSS after cleanup: {rss_after_cleanup:.1f} MB")

    _banner("NEXUS+3B RAM RESULTS")
    print(f"  RSS before:      {rss_before:.1f} MB")
    print(f"  RSS peak:        {peak_ram_mb:.1f} MB")
    print(f"  RSS delta:       {peak_ram_mb - rss_before:.1f} MB")
    print(f"  RSS after:       {rss_after_cleanup:.1f} MB")
    print(f"  Ollama RSS:      {ollama_rss_after}")
    print(f"  Latency (p50):   {_p50(latencies)*1000:.0f}ms")
    print(f"  Latency (p95):   {_p95(latencies)*1000:.0f}ms")

    return {
        "arm": "nexus-3b",
        "rss_before_mb": round(rss_before, 1),
        "rss_peak_mb": round(peak_ram_mb, 1),
        "rss_delta_mb": round(peak_ram_mb - rss_before, 1),
        "rss_after_mb": round(rss_after_cleanup, 1),
        "rss_after_graph_mb": round(rss_after_graph, 1),
        "rss_after_model_mb": round(rss_after_model, 1),
        "ollama_rss_before": ollama_rss_before,
        "ollama_rss_after": ollama_rss_after,
        "num_questions": ZERO_WEIGHT_QUESTIONS,
        "num_completed": len(latencies),
        "latency_p50_ms": round(_p50(latencies) * 1000, 1) if latencies else 0,
        "latency_p95_ms": round(_p95(latencies) * 1000, 1) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  ARM 3: RAG + 3B — chunk retrieval + Ollama, embeddings in memory
# ═══════════════════════════════════════════════════════════════════════

def measure_rag_3b_ram(model_name: str) -> dict[str, Any]:
    """
    Measure peak RSS of RAG pipeline.
    Load embedding index (all-MiniLM-L6 for all chunks), run 30 questions,
    measure peak RSS during end-to-end pipeline.
    """
    # Measure RSS before
    rss_before = _get_process_rss_mb()
    ollama_rss_before = _get_ollama_rss_mb()
    print(f"  RSS before loading: {rss_before:.1f} MB (Ollama: {ollama_rss_before})")

    # Import local modules
    from nexus.reasoning.model_interface import OllamaModel

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
    questions = questions[:ZERO_WEIGHT_QUESTIONS]

    # Import RAG components
    print("  Loading RAG pipeline components...")
    from benchmarks.rag_baseline import (
        chunk_documents, RAGEmbedder, retrieve_top_k, build_rag_prompt,
        CORPUS_DIRS,
    )

    # Find all .md files
    all_docs: list[Path] = []
    for corpus_dir in CORPUS_DIRS:
        if corpus_dir.exists():
            all_docs.extend(sorted(corpus_dir.glob("**/*.md")))

    if not all_docs:
        return {"error": "No .md documents found in corpus dirs"}

    print(f"  Found {len(all_docs)} .md documents")

    # Chunk documents
    print("  Chunking documents...")
    chunks = chunk_documents(all_docs)
    print(f"  Produced {len(chunks)} chunks")

    rss_after_chunking = _get_process_rss_mb()
    print(f"  RSS after chunking: {rss_after_chunking:.1f} MB")

    # Load embeddings (this loads all-MiniLM-L6 model + computes/caches embeddings)
    print("  Loading embedding model and embeddings...")
    embedder = RAGEmbedder()
    chunk_embeddings = embedder.embed_chunks(chunks)

    rss_after_embeddings = _get_process_rss_mb()
    print(f"  RSS after embeddings loaded: {rss_after_embeddings:.1f} MB "
          f"(delta: {rss_after_embeddings - rss_before:.1f} MB)")

    # Create Ollama model
    ollama_model = OllamaModel(model_name=model_name)
    print(f"  Model: {ollama_model.name}")

    rss_after_model = _get_process_rss_mb()
    print(f"  RSS after model init: {rss_after_model:.1f} MB")

    # Track peak
    peak_ram_mb = rss_after_model

    # Run all questions
    _banner(f"RAG+3B BENCHMARK — {len(questions)} questions, chunk retrieval + Ollama")
    latencies: list[float] = []

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        ground_truth = q.get("answer", "")

        t0 = time.perf_counter()

        # Embed query
        query_emb = embedder.embed_query(question_text)

        # Retrieve top-k chunks
        retrieved = retrieve_top_k(query_emb, chunk_embeddings, chunks, k=5)

        # Build RAG prompt
        rag_prompt = build_rag_prompt(question_text, retrieved)

        # Generate with Ollama
        try:
            response = ollama_model.generate(rag_prompt)
        except Exception as exc:
            print(f"  [{i}/{len(questions)}] ERROR: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Update peak RAM
        try:
            import psutil
            current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram
        except Exception:
            pass

        answer_text = response[:80] if isinstance(response, str) else str(response)[:80]
        print(f"  [{i}/{len(questions)}] | {elapsed*1000:.0f}ms | {answer_text}...")

    rss_after = _get_process_rss_mb()
    ollama_rss_after = _get_ollama_rss_mb()

    # Cleanup
    import gc
    del chunks
    del chunk_embeddings
    if embedder._model is not None:
        del embedder._model
        embedder._model = None
    del embedder
    del ollama_model
    del questions
    gc.collect()

    rss_after_cleanup = _get_process_rss_mb()
    print(f"\n  RSS after cleanup: {rss_after_cleanup:.1f} MB")

    _banner("RAG+3B RAM RESULTS")
    print(f"  RSS before:         {rss_before:.1f} MB")
    print(f"  RSS peak:           {peak_ram_mb:.1f} MB")
    print(f"  RSS delta:          {peak_ram_mb - rss_before:.1f} MB")
    print(f"  RSS after:          {rss_after_cleanup:.1f} MB")
    print(f"  RSS embeddings:     {rss_after_embeddings - rss_before:.1f} MB added")
    print(f"  Ollama RSS:         {ollama_rss_after}")
    print(f"  Latency (p50):      {_p50(latencies)*1000:.0f}ms")
    print(f"  Latency (p95):      {_p95(latencies)*1000:.0f}ms")

    return {
        "arm": "rag-3b",
        "rss_before_mb": round(rss_before, 1),
        "rss_peak_mb": round(peak_ram_mb, 1),
        "rss_delta_mb": round(peak_ram_mb - rss_before, 1),
        "rss_after_mb": round(rss_after_cleanup, 1),
        "rss_after_chunking_mb": round(rss_after_chunking, 1),
        "rss_after_embeddings_mb": round(rss_after_embeddings, 1),
        "rss_after_model_mb": round(rss_after_model, 1),
        "ollama_rss_before": ollama_rss_before,
        "ollama_rss_after": ollama_rss_after,
        "num_chunks": len(chunks) if 'chunks' in dir() else 0,
        "num_questions": ZERO_WEIGHT_QUESTIONS,
        "num_completed": len(latencies),
        "latency_p50_ms": round(_p50(latencies) * 1000, 1) if latencies else 0,
        "latency_p95_ms": round(_p95(latencies) * 1000, 1) if latencies else 0,
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  WARMED OLLAMA THROUGHPUT BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def benchmark_warmed_throughput(
    host: str = "http://localhost:11434",
    model_name: str = "qwen2.5:latest",
    warmup_runs: int = WARMUP_RUNS,
    warmup_tokens: int = WARMUP_TOKENS,
    measurement_runs: int = MEASUREMENT_RUNS,
    measurement_tokens: int = MEASUREMENT_TOKENS,
) -> dict[str, Any]:
    """
    Run honest throughput benchmark with proper warm-up.
    Phase 1: 5 warm-up generations (256 tokens each, not measured)
    Phase 2: 10 runs at 3 prompt lengths (50, 200, 500 tokens), 256 completion tokens
    """
    all_measured: list[dict[str, Any]] = []

    # ── Phase 1: Warm-up ──
    _banner("WARM-UP (not measured)")
    warmup_prompt = PROMPT_TEMPLATES[200]
    print(f"  Running {warmup_runs} warm-up generations ({warmup_tokens} tokens each)...")
    for i in range(1, warmup_runs + 1):
        try:
            r = _ollama_generate(host, model_name, warmup_prompt, max_tokens=warmup_tokens)
            print(f"    Warm-up {i}/{warmup_runs}: "
                  f"{r['tokens_per_second']:.1f} tok/s, "
                  f"{r['completion_tokens']} tok out")
        except Exception as exc:
            print(f"    Warm-up {i}/{warmup_runs}: ERROR — {exc}")
            return {"error": f"Warm-up failed at run {i}: {exc}"}

    print("  Warm-up complete — model is now hot. Starting measurements.")
    time.sleep(2)  # brief pause

    # ── Phase 2: Measurement ──
    _banner("MEASUREMENT (warmed model)")

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

            all_measured.append({
                "prompt_length": prompt_len,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": r["completion_tokens"],
                "ttft_s": r["ttft_s"],
                "total_time_s": r["total_time_s"],
                "tokens_per_second": r["tokens_per_second"],
            })

            print(f"    Run {run}/{measurement_runs}: "
                  f"TTFT={r['ttft_s']:.3f}s, "
                  f"{r['tokens_per_second']:.1f} tok/s, "
                  f"{r['completion_tokens']} tok out")

    if not all_measured:
        return {"error": "No measurements collected"}

    # ── Compute statistics ──
    all_tps = [r["tokens_per_second"] for r in all_measured]
    all_ttft = [r["ttft_s"] for r in all_measured]

    # Per prompt-length breakdown
    by_length: dict[int, dict[str, Any]] = {}
    for prompt_len in sorted(PROMPT_TEMPLATES.keys()):
        subset = [r for r in all_measured if r["prompt_length"] == prompt_len]
        if not subset:
            continue
        tps_vals = [r["tokens_per_second"] for r in subset]
        ttft_vals = [r["ttft_s"] for r in subset]
        by_length[prompt_len] = {
            "runs": len(subset),
            "p50_tps": _p50(tps_vals),
            "p95_tps": _p95(tps_vals),
            "mean_tps": round(statistics.mean(tps_vals), 2) if tps_vals else 0.0,
            "p50_ttft_s": _p50(ttft_vals),
            "p95_ttft_s": _p95(ttft_vals),
            "mean_ttft_s": round(statistics.mean(ttft_vals), 4) if ttft_vals else 0.0,
        }

    # ── Ollama process RAM ──
    ollama_rss = _get_ollama_rss_mb()

    _banner("WARMED THROUGHPUT RESULTS")
    for prompt_len, stats in sorted(by_length.items()):
        print(f"\n  Prompt ~{prompt_len} tokens ({stats['runs']} runs):")
        print(f"    Tokens/sec:  p50={stats['p50_tps']:.1f}  p95={stats['p95_tps']:.1f}  mean={stats['mean_tps']:.1f}")
        print(f"    TTFT:        p50={stats['p50_ttft_s']:.3f}s  p95={stats['p95_ttft_s']:.3f}s  mean={stats['mean_ttft_s']:.3f}s")

    print(f"\n  OVERALL ({len(all_measured)} measured runs):")
    print(f"    Tokens/sec:  p50={_p50(all_tps):.1f}  p95={_p95(all_tps):.1f}  mean={statistics.mean(all_tps):.1f}")
    print(f"    TTFT:        p50={_p50(all_ttft):.3f}s  p95={_p95(all_ttft):.3f}s")
    print(f"    Ollama RSS:  {ollama_rss}")

    return {
        "model": model_name,
        "warmup": {"runs": warmup_runs, "tokens_per_run": warmup_tokens},
        "measurement": {
            "runs_per_length": measurement_runs,
            "tokens_per_completion": measurement_tokens,
        },
        "ollama_rss": ollama_rss,
        "p50_tps": round(_p50(all_tps), 2),
        "p95_tps": round(_p95(all_tps), 2),
        "mean_tps": round(statistics.mean(all_tps), 2),
        "p50_ttft_s": round(_p50(all_ttft), 4),
        "p95_ttft_s": round(_p95(all_ttft), 4),
        "by_prompt_length": {
            str(k): {
                "p50_tps": round(v["p50_tps"], 2),
                "p95_tps": round(v["p95_tps"], 2),
                "mean_tps": v["mean_tps"],
                "p50_ttft_s": round(v["p50_ttft_s"], 4),
                "p95_ttft_s": round(v["p95_ttft_s"], 4),
            }
            for k, v in by_length.items()
        },
        "all_results": all_measured,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAM & Throughput Measurement — peak RSS per arm + warmed throughput"
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:latest",
        help="Ollama model to use (default: qwen2.5:latest)"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true",
        help="Skip Ollama-dependent arms (NEXUS+3B, RAG+3B, throughput)"
    )
    parser.add_argument(
        "--ram-only", action="store_true",
        help="Measure RAM only, skip throughput recalibration"
    )
    parser.add_argument(
        "--throughput-only", action="store_true",
        help="Run throughput only, skip RAM measurements"
    )
    parser.add_argument(
        "--zero-weight-only", action="store_true",
        help="Run zero-weight arm only"
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  RAM & THROUGHPUT MEASUREMENT")
    print("  Real hardware measurement — NOT CI-capable")
    print("=" * 72)

    # ── Get machine info ──
    machine_info = _get_machine_info()
    print(f"\n  Machine: {machine_info.get('cpu_model', 'unknown')}")
    print(f"  OS: {machine_info['os']} {machine_info['os_version']}")
    print(f"  RAM total: {machine_info.get('ram_total_gb', 'unknown')} GB")

    # ── Check Ollama availability ──
    ollama_available = False
    resolved_model = args.model
    if not args.skip_ollama:
        ollama_available, resolved_model = _check_ollama(args.model)
        if ollama_available:
            print(f"\n  Ollama: available (model: {resolved_model})")
        else:
            print(f"\n  Ollama: NOT available (model: {args.model})")
            print("  Will skip Ollama-dependent measurements.")
    else:
        print("\n  Ollama: skipped (--skip-ollama)")

    # ── Get Ollama model details ──
    model_info: dict[str, str] = {"name": resolved_model}
    if ollama_available:
        try:
            url = "http://localhost:11434/api/show"
            payload = json.dumps({"name": resolved_model}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                show_data = json.loads(resp.read().decode("utf-8"))
                details = show_data.get("details", {})
                model_info["family"] = details.get("family", "unknown")
                model_info["parameter_size"] = details.get("parameter_size", "unknown")
                model_info["quantization_level"] = details.get("quantization_level", "unknown")
        except Exception:
            pass

    print(f"  Model info: {model_info}")

    # ── Collect results ──
    results: dict[str, Any] = {
        "timestamp": _utc_timestamp(),
        "machine": machine_info,
        "model": model_info,
    }

    if not args.throughput_only:
        # ── Arm 1: Zero-weight ──
        _banner("ARM 1: ZERO-WEIGHT RAM MEASUREMENT")
        print("\n  Measuring SynthesizingModel pipeline peak RSS...")
        try:
            results["zero_weight"] = measure_zero_weight_ram()
        except Exception as exc:
            print(f"\n  ERROR measuring zero-weight: {exc}")
            import traceback
            traceback.print_exc()
            results["zero_weight"] = {"error": str(exc)}

        if not args.zero_weight_only and ollama_available:
            # ── Arm 2: NEXUS + 3B ──
            _banner("ARM 2: NEXUS+3B RAM MEASUREMENT")
            print("\n  Measuring NEXUS + FallbackModel pipeline peak RSS...")
            try:
                results["nexus_3b"] = measure_nexus_3b_ram(resolved_model)
            except Exception as exc:
                print(f"\n  ERROR measuring NEXUS+3B: {exc}")
                import traceback
                traceback.print_exc()
                results["nexus_3b"] = {"error": str(exc)}

            # ── Arm 3: RAG + 3B ──
            _banner("ARM 3: RAG+3B RAM MEASUREMENT")
            print("\n  Measuring RAG + chunk retrieval peak RSS...")
            try:
                results["rag_3b"] = measure_rag_3b_ram(resolved_model)
            except Exception as exc:
                print(f"\n  ERROR measuring RAG+3B: {exc}")
                import traceback
                traceback.print_exc()
                results["rag_3b"] = {"error": str(exc)}

    if not args.ram_only and ollama_available and not args.zero_weight_only:
        # ── Warmed throughput ──
        _banner("WARMED OLLAMA THROUGHPUT")
        print(f"\n  Model: {resolved_model}")
        throughput_data = benchmark_warmed_throughput(
            model_name=resolved_model,
            warmup_runs=WARMUP_RUNS,
            measurement_runs=MEASUREMENT_RUNS,
        )
        if "error" in throughput_data:
            print(f"\n  ERROR: {throughput_data['error']}")
        else:
            results["warmed_throughput"] = throughput_data

    # ── Compute cost estimates ──
    if "warmed_throughput" in results and "error" not in results["warmed_throughput"]:
        tp = results["warmed_throughput"]
        tps = tp["p50_tps"]
        if tps > 0:
            watts = 65.0
            elec = 0.15
            seconds_1m = 1_000_000 / tps
            kwh = (watts / 1000) * (seconds_1m / 3600)
            cost_1m = kwh * elec
            tps_needed = (watts * 1_000_000 * elec) / (1000 * 3600 * 0.01)

            results["cost_estimates"] = {
                "watts_at_load": watts,
                "electricity_per_kwh": elec,
                "tokens_per_second": tps,
                "cost_per_1m_tokens": round(cost_1m, 6),
                "tps_needed_for_1cent_target": round(tps_needed, 1),
                "cost_per_1m_tokens_llm_only": f"${cost_1m:.4f}",
                "cost_per_1m_tokens_router_80pct": f"${cost_1m * 0.2:.6f} (80% synth = $0)",
            }
            print(f"\n  Cost model (warmed, p50={tps:.1f} tok/s):")
            print(f"    ${cost_1m:.4f}/1M tokens (raw LLM)")
            print(f"    ${cost_1m * 0.2:.6f}/1M tokens (router, 80% synth)")
            print(f"    Need {tps_needed:.0f} tok/s for $0.01/1M target")

    # ── Save results ──
    timestamp = _utc_timestamp()
    filename = f"ram_throughput_{timestamp}.json"
    output_path = RESULTS_DIR / filename
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    print(f"  RESULTS SAVED TO: {output_path}")
    print(f"{'=' * 72}")

    # ── Summary table ──
    print("\n  PEAK RSS SUMMARY:")
    print(f"  {'Arm':<20} {'RSS Before':>12} {'RSS Peak':>12} {'RSS Delta':>12} {'RSS After':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for arm_key, label in [("zero_weight", "Zero-weight"), ("nexus_3b", "NEXUS + 3B"), ("rag_3b", "RAG + 3B")]:
        arm_data = results.get(arm_key, {})
        if "error" in arm_data:
            print(f"  {label:<20} {'ERROR':>12} {arm_data['error'][:40]}")
        else:
            print(f"  {label:<20} {arm_data.get('rss_before_mb', '?'):>11} MB "
                  f"{arm_data.get('rss_peak_mb', '?'):>11} MB "
                  f"{arm_data.get('rss_delta_mb', '?'):>11} MB "
                  f"{arm_data.get('rss_after_mb', '?'):>11} MB")

    if "warmed_throughput" in results and "error" not in results["warmed_throughput"]:
        tp = results["warmed_throughput"]
        print(f"\n  WARMED THROUGHPUT: p50={tp['p50_tps']:.1f} tok/s, p95={tp['p95_tps']:.1f} tok/s")


if __name__ == "__main__":
    main()
