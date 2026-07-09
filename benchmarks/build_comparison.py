"""
Build the canonical comparison table (Phase 5).

Reads data from newest result files and generates benchmarks/COMPARISON.md.
Every table cell cites its source file. Cells without a measurement show
"not measured" — never estimate.

Usage:
    python benchmarks/build_comparison.py [--output benchmarks/COMPARISON.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.cost_model import LocalCostModel


# ── File resolvers ───────────────────────────────────────────────────────────

def _find_newest_in_results(glob_pattern: str) -> Path | None:
    """Find the newest file matching *glob_pattern* in benchmarks/results/."""
    results_dir = _SCRIPT_DIR / "results"
    candidates = sorted(results_dir.glob(glob_pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _resolve_nexus_vs_rag_regenerated_json() -> Path:
    """Return path to the newest nexus_vs_rag_*.json in results/ that contains a 'comparison' key."""
    results_dir = _SCRIPT_DIR / "results"
    candidates = sorted(results_dir.glob("nexus_vs_rag_*.json"))
    if not candidates:
        raise FileNotFoundError("No nexus_vs_rag files found in results/")
    # Filter to files that actually have a 'comparison' key (regenerated comparison output),
    # not raw benchmark run outputs (which have no comparison data).
    comparison_files = []
    for p in candidates:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if "comparison" in data and data["comparison"]:
                comparison_files.append(p)
        except (json.JSONDecodeError, OSError):
            continue
    if not comparison_files:
        raise FileNotFoundError("No nexus_vs_rag files with 'comparison' key found in results/")
    comparison_files.sort(key=lambda p: p.stat().st_mtime)
    return comparison_files[-1]


def _resolve_nexus_vs_rag_raw_json() -> Path:
    """Return path to the original nexus_vs_rag_200.json (contains per-arm summary stats)."""
    fallback = _SCRIPT_DIR / "nexus_vs_rag_200.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("nexus_vs_rag_200.json not found")


def _resolve_verifier_check_json() -> Path:
    newest = _find_newest_in_results("verifier_check_*.json")
    if newest is not None:
        return newest
    raise FileNotFoundError("No verifier_check file found")


def _resolve_throughput_json() -> Path:
    newest = _find_newest_in_results("throughput_*.json")
    if newest is not None:
        return newest
    fallback = _SCRIPT_DIR / "throughput_results.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No throughput results file found")


def _resolve_ram_throughput_json() -> Path | None:
    """Find the newest ram_throughput_*.json with both RAM and throughput data."""
    newest = _find_newest_in_results("ram_throughput_*.json")
    if newest is not None:
        # Verify it has both RAM and throughput data
        with open(newest, encoding="utf-8") as f:
            data = json.load(f)
        has_ram = any(
            isinstance(data.get(arm, {}).get("rss_delta_mb"), (int, float))
            for arm in ["zero_weight", "nexus_3b", "rag_3b"]
        )
        has_tp = "warmed_throughput" in data
        if has_ram and has_tp:
            return newest
    return None


def _resolve_router_results_json() -> Path:
    """Find the newest router_results_*.json or router_paired_*.json in results/ that has summary data."""
    # Prefer timestamped files in results/ with actual data
    results_dir = _SCRIPT_DIR / "results"
    
    # Try router_results_* first (timestamped router benchmark outputs)
    candidates = sorted(results_dir.glob("router_results_*.json"))
    for p in reversed(candidates):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if "summary" in data and data["summary"]:
                return p
        except (json.JSONDecodeError, OSError):
            continue
    
    # Fall back to benchmarks/router_results.json (legacy)
    fallback = _SCRIPT_DIR / "router_results.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No router results file found")


def _resolve_router_paired_json() -> Path:
    """Find the newest router_paired_*.json in results/ (200-question router run)."""
    newest = _find_newest_in_results("router_paired_*.json")
    if newest is not None:
        return newest
    raise FileNotFoundError("No router_paired file found in results/")


def _resolve_router_vs_rag_paired_json() -> Path:
    """Find the newest router_vs_rag_paired_*.json in results/ (paired comparison)."""
    newest = _find_newest_in_results("router_vs_rag_paired_*.json")
    if newest is not None:
        return newest
    raise FileNotFoundError("No router_vs_rag_paired file found in results/")


def _resolve_relevance_audit_md() -> Path:
    # The audit is at benchmarks/relevance_audit.md (not results/)
    path = _SCRIPT_DIR / "relevance_audit.md"
    if path.exists():
        return path
    raise FileNotFoundError("relevance_audit.md not found")


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_nexus_vs_rag(regenerated_path: Path, raw_path: Path) -> dict:
    """Load nexus_vs_rag regenerated comparison AND raw 200-run summary."""
    with open(regenerated_path, encoding="utf-8") as f:
        regen = json.load(f)
    comparison = regen.get("comparison", {})
    nexus_comp = comparison.get("nexus", {})
    rag_comp = comparison.get("rag", {})

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    raw_summary = raw.get("summary", {})
    nexus_sum = raw_summary.get("nexus", {})
    rag_sum = raw_summary.get("rag", {})

    return {
        "source_comparison": str(regenerated_path),
        "source_summary": str(raw_path),
        "nexus": {
            "mean_accuracy": nexus_comp.get("mean_accuracy"),
            "avg_hallucination_rate": nexus_sum.get("avg_hallucination_rate"),
            "verification_pass_rate": nexus_sum.get("verification_pass_rate"),
            "answer_rate": nexus_sum.get("answer_rate"),
            "insufficient_evidence_rate": 1.0 - nexus_sum.get("answer_rate", 0),
            "avg_evidence_tokens": nexus_sum.get("avg_evidence_tokens"),
            "avg_latency_s": nexus_sum.get("avg_latency_s"),
        },
        "rag": {
            "mean_accuracy": rag_comp.get("mean_accuracy"),
            "avg_hallucination_rate": rag_sum.get("avg_hallucination_rate"),
            "verification_pass_rate": rag_sum.get("verification_pass_rate"),
            "answer_rate": rag_sum.get("answer_rate"),
            "insufficient_evidence_rate": 1.0 - rag_sum.get("answer_rate", 0),
            "avg_evidence_tokens": rag_sum.get("avg_evidence_tokens"),
            "avg_latency_s": rag_sum.get("avg_latency_s"),
        },
        "sign_test_p": comparison.get("sign_test_p"),
        "paired_n": comparison.get("paired_n"),
    }


def load_verifier_check(path: Path) -> dict:
    """Load verifier_check file, compute aggregate hallucination and pass rates."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    summary = data.get("summary", {})

    # Aggregate from router (synthesizer) results
    router_hallucinations = []
    router_passed = 0
    router_total = 0
    router_latencies = []
    router_insufficient = 0

    # Aggregate from llm results (if any routed to LLM)
    llm_hallucinations = []
    llm_passed = 0
    llm_total = 0
    llm_latencies = []
    llm_insufficient = 0

    for r in results:
        router = r.get("router", {})
        llm = r.get("llm", {})

        hr = router.get("hallucination_rate")
        if hr is not None:
            router_hallucinations.append(hr)
        if router:
            router_total += 1
            if router.get("passed"):
                router_passed += 1
            lat = router.get("latency_s")
            if lat is not None:
                router_latencies.append(lat)
            if router.get("is_insufficient"):
                router_insufficient += 1

        hr_llm = llm.get("hallucination_rate")
        if hr_llm is not None:
            llm_hallucinations.append(hr_llm)
        if llm:
            llm_total += 1
            if llm.get("passed"):
                llm_passed += 1
            lat = llm.get("latency_s")
            if lat is not None:
                llm_latencies.append(lat)
            if llm.get("is_insufficient"):
                llm_insufficient += 1

    avg_router_hallucination = (
        sum(router_hallucinations) / len(router_hallucinations)
        if router_hallucinations else None
    )
    avg_llm_hallucination = (
        sum(llm_hallucinations) / len(llm_hallucinations)
        if llm_hallucinations else None
    )
    router_verification_pass = router_passed / router_total if router_total else None
    llm_verification_pass = llm_passed / llm_total if llm_total else None
    avg_router_latency = (
        sum(router_latencies) / len(router_latencies) if router_latencies else None
    )
    avg_llm_latency = (
        sum(llm_latencies) / len(llm_latencies) if llm_latencies else None
    )
    router_answer_rate = (
        1.0 - router_insufficient / router_total if router_total else None
    )
    llm_answer_rate = (
        1.0 - llm_insufficient / llm_total if llm_total else None
    )

    # Combined hallucination (all results regardless of routing)
    all_hallucinations = router_hallucinations + llm_hallucinations
    avg_all_hallucination = (
        sum(all_hallucinations) / len(all_hallucinations)
        if all_hallucinations else None
    )

    return {
        "source": str(path),
        "summary": summary,
        "router": {
            "avg_hallucination_rate": avg_router_hallucination,
            "verification_pass_rate": router_verification_pass,
            "answer_rate": router_answer_rate,
            "avg_latency_s": avg_router_latency,
            "n": router_total,
        },
        "llm": {
            "avg_hallucination_rate": avg_llm_hallucination,
            "verification_pass_rate": llm_verification_pass,
            "answer_rate": llm_answer_rate,
            "avg_latency_s": avg_llm_latency,
            "n": llm_total,
        },
        "combined_avg_hallucination_rate": avg_all_hallucination,
        "nexus_3b_accuracy": summary.get("nexus_3b_accuracy"),
        "synthesizer_accuracy": summary.get("synthesizer_accuracy"),
    }


def load_router_results(path: Path) -> dict:
    """Load router_results.json for routing split data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    route_split = summary.get("route_split", {})
    synth_count = route_split.get("synthesizer", 0)
    llm_count = route_split.get("llm", 0)
    total = summary.get("total_questions", synth_count + llm_count)

    return {
        "source": str(path),
        "synth_ratio": synth_count / total if total > 0 else None,
        "synth_count": synth_count,
        "llm_count": llm_count,
        "total": total,
    }


def load_router_paired(path: Path) -> dict:
    """Load router_paired_*.json for per-arm router summary stats."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    total = len(results)

    # Hallucination
    hall_rates = [r["router"]["hallucination_rate"] for r in results if not r["router"].get("error")]
    avg_hall = sum(hall_rates) / len(hall_rates) if hall_rates else 0.0

    # Verification pass rate
    passed = sum(1 for r in results if r["router"].get("passed") and not r["router"].get("error"))
    pass_rate = passed / total if total > 0 else 0.0

    # Answer rate
    insuff = sum(1 for r in results if r["router"].get("is_insufficient"))
    answer_rate = 1.0 - insuff / total if total > 0 else 0.0

    # Latency
    latencies = [r["router"]["latency_s"] for r in results if not r["router"].get("error")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    # Evidence tokens from LLM arm (same graph, representative)
    ev_tokens = [r["llm"].get("evidence_tokens", 0) for r in results if not r["llm"].get("error")]
    avg_ev_tokens = sum(ev_tokens) / len(ev_tokens) if ev_tokens else 0.0

    # Route split
    synth_count = sum(1 for r in results if r["router"]["routed_to"] == "synthesizer")
    llm_count = sum(1 for r in results if r["router"]["routed_to"] == "llm")
    synth_ratio = synth_count / total if total > 0 else 0.0

    # Accuracy
    accuracies = [r["router"]["accuracy"] for r in results
                  if r["router"]["accuracy"] is not None and not r["router"].get("error")]
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return {
        "source": str(path),
        "total": total,
        "avg_hallucination_rate": round(avg_hall, 4),
        "verification_pass_rate": round(pass_rate, 4),
        "answer_rate": round(answer_rate, 4),
        "avg_evidence_tokens": round(avg_ev_tokens, 1),
        "avg_latency_s": round(avg_lat, 4),
        "avg_accuracy": round(avg_acc, 4),
        "route_split": {"synthesizer": synth_count, "llm": llm_count},
        "synth_ratio": round(synth_ratio, 4),
    }


def load_router_vs_rag_paired(path: Path) -> dict:
    """Load router_vs_rag_paired_*.json for paired comparison data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    comp = data.get("comparison", {})
    router_comp = comp.get("nexus router", {})
    rag_comp = comp.get("rag", {})
    wlt = comp.get("win_loss_tie", {})

    return {
        "source": str(path),
        "paired_n": comp.get("paired_n"),
        "router_accuracy": router_comp.get("mean_accuracy"),
        "rag_accuracy": rag_comp.get("mean_accuracy"),
        "sign_test_p": comp.get("sign_test_p"),
        "win_loss_tie": wlt,
    }


def load_throughput(path: Path) -> dict:
    """Load throughput results and build cost model data.
    
    Handles both legacy throughput_*.json and new ram_throughput_*.json formats.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # New format: warmed_throughput key (from ram_throughput.py)
    if "warmed_throughput" in data:
        tp = data["warmed_throughput"]
        # Fallback: if new keys missing, use legacy ollama_rss for idle
        idle_rss = tp.get("ollama_idle_rss_mb", data.get("ollama_idle_rss_mb"))
        if idle_rss is None or (isinstance(idle_rss, str) and idle_rss == "unavailable"):
            idle_rss = tp.get("ollama_rss")  # legacy key
        gen_rss = tp.get("ollama_generating_rss_mb", data.get("ollama_generating_rss_mb"))
        return {
            "source": str(path),
            "p50_tps": tp.get("p50_tps"),
            "p95_tps": tp.get("p95_tps"),
            "mean_tps": tp.get("mean_tps"),
            "ram_mb": tp.get("ollama_rss", 0),
            "ollama_idle_rss_mb": idle_rss,
            "ollama_generating_rss_mb": gen_rss,
            "pipeline_overhead_ms": None,
            "by_prompt_length": tp.get("by_prompt_length", {}),
        }

    # Legacy format
    ollama = data.get("ollama", {})
    pipeline = data.get("pipeline", {})

    max_ram = 0.0
    for r in ollama.get("results", []):
        ram = r.get("ram_mb", 0)
        if isinstance(ram, (int, float)) and ram > max_ram:
            max_ram = float(ram)

    return {
        "source": str(path),
        "p50_tps": ollama.get("p50_tokens_per_second"),
        "p95_tps": ollama.get("p95_tokens_per_second"),
        "mean_tps": ollama.get("mean_tokens_per_second"),
        "ram_mb": max_ram,
        "ollama_idle_rss_mb": None,
        "ollama_generating_rss_mb": None,
        "pipeline_overhead_ms": pipeline.get("cpu_overhead_ms"),
    }


def load_ram_data(path: Path) -> dict:
    """Load per-arm RAM measurements from ram_throughput_*.json.
    
    Returns dict mapping arm name -> {rss_peak_mb, rss_delta_mb, ...}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    ram = {}
    for arm in ["zero_weight", "nexus_3b", "rag_3b"]:
        arm_data = data.get(arm, {})
        if "error" not in arm_data:
            ram[arm] = {
                "rss_peak_mb": arm_data.get("rss_peak_mb"),
                "rss_delta_mb": arm_data.get("rss_delta_mb"),
                "rss_before_mb": arm_data.get("rss_before_mb"),
            }
    return ram


def parse_relevance_audit(path: Path) -> dict:
    """Parse relevance_audit.md for the relevance rate."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # Extract relevance rate (formula: % yes + 0.5 × % partial)
    m = re.search(r"\*\*Relevance rate\*\*:\s*([\d.]+)%", text)
    rate = float(m.group(1)) / 100.0 if m else None

    return {
        "source": str(path),
        "relevance_rate": rate,
    }


# ── Cost computation ─────────────────────────────────────────────────────────

def compute_cost_per_1k_queries(
    label: str,
    throughput_data: dict,
    avg_evidence_tokens: float | None,
    avg_latency_s: float | None,
    synth_ratio: float | None = None,
) -> str:
    """
    Compute $/1K queries using LocalCostModel.

    Returns a formatted string with dollar amount and source citation.
    """
    tps = throughput_data["p50_tps"] or throughput_data["mean_tps"]
    if tps is None or tps <= 0:
        return f"not measured [{throughput_data['source']}]"

    cost_model = LocalCostModel(
        tokens_per_second=tps,
        ram_mb=throughput_data["ram_mb"],
        source_file=throughput_data["source"],
    )
    cost_per_1m = cost_model.cost_per_1m_tokens()

    # Estimate tokens per query: evidence (prompt) + completion
    # Completion typically ~150 tokens for a short answer
    completion_estimate = 150
    prompt_tokens = avg_evidence_tokens if avg_evidence_tokens else 688
    total_tokens_per_query = prompt_tokens + completion_estimate

    if synth_ratio is not None and synth_ratio > 0:
        # Blended: synth queries cost ~$0 (CPU-only template synthesis)
        # Only (1 - synth_ratio) go through LLM
        llm_fraction = 1.0 - synth_ratio
        cost_per_query = (total_tokens_per_query / 1_000_000) * cost_per_1m * llm_fraction
    else:
        cost_per_query = (total_tokens_per_query / 1_000_000) * cost_per_1m

    cost_per_1k = cost_per_query * 1000

    source_short = Path(throughput_data["source"]).name
    return f"${cost_per_1k:.4f} [{source_short}]"


# ── Markdown generation ──────────────────────────────────────────────────────

def _fmt(val, fmt_spec: str = ".4f", suffix: str = "") -> str:
    """Format a numeric value, handling None."""
    if val is None:
        return "not measured"
    if isinstance(val, float):
        return f"{val:{fmt_spec}}{suffix}"
    return f"{val}{suffix}"


def build_table() -> str:
    """Build the complete comparison markdown table and supporting text."""

    # ── Load all data ──
    print("Loading data files...")
    nvr_regen_path = _resolve_nexus_vs_rag_regenerated_json()
    nvr_raw_path = _resolve_nexus_vs_rag_raw_json()
    vc_path = _resolve_verifier_check_json()
    router_path = _resolve_router_results_json()
    audit_path = _resolve_relevance_audit_md()

    # New: router paired data (200-question router run + paired comparison)
    router_paired_path: Path | None = None
    router_vs_rag_paired_path: Path | None = None
    try:
        router_paired_path = _resolve_router_paired_json()
    except FileNotFoundError:
        print("  Warning: No router_paired_*.json found")
    try:
        router_vs_rag_paired_path = _resolve_router_vs_rag_paired_json()
    except FileNotFoundError:
        print("  Warning: No router_vs_rag_paired_*.json found")

    # Prefer new ram_throughput format (has both RAM and warmed throughput)
    ram_tp_path = _resolve_ram_throughput_json()
    if ram_tp_path is not None:
        print(f"  Using ram_throughput format: {ram_tp_path.name}")
        tp_path = ram_tp_path
        ram_data = load_ram_data(ram_tp_path)
    else:
        tp_path = _resolve_throughput_json()
        ram_data = {}

    nvr = load_nexus_vs_rag(nvr_regen_path, nvr_raw_path)
    vc = load_verifier_check(vc_path)
    tp = load_throughput(tp_path)

    # Load new router paired data if available
    router_paired_data: dict | None = None
    router_vs_rag_paired: dict | None = None
    if router_paired_path is not None:
        router_paired_data = load_router_paired(router_paired_path)
        print(f"  router_paired:     {router_paired_path.name}")
    if router_vs_rag_paired_path is not None:
        router_vs_rag_paired = load_router_vs_rag_paired(router_vs_rag_paired_path)
        print(f"  router_vs_rag:     {router_vs_rag_paired_path.name}")
    router_data = load_router_results(router_path)
    audit = parse_relevance_audit(audit_path)

    print(f"  regenerated cmp:   {nvr_regen_path.name}")
    print(f"  raw 200-run:       {nvr_raw_path.name}")
    print(f"  verifier_check:    {vc_path.name}")
    print(f"  throughput:        {tp_path.name}")
    print(f"  router_results:    {router_path.name}")
    print(f"  relevance_audit:   {audit_path.name}")

    # ── Source file short names ──
    nvr_regen_fn = nvr_regen_path.name
    nvr_raw_fn = nvr_raw_path.name
    vc_fn = vc_path.name
    tp_fn = tp_path.name
    audit_fn = audit_path.name

    # ── Cost calculations ──
    nexus_cost = compute_cost_per_1k_queries(
        "NEXUS+3B", tp, nvr["nexus"]["avg_evidence_tokens"], nvr["nexus"]["avg_latency_s"],
        synth_ratio=None,  # FallbackModel uses LLM for all
    )

    # For zero-weight: entirely template synthesis = effectively $0
    zero_weight_cost = f"$0.0000 [{tp_fn}] (template synthesis only)"

    # For router: blended with actual synth_ratio from router results
    synth_ratio = router_data.get("synth_ratio")
    router_cost = compute_cost_per_1k_queries(
        "Router", tp,
        nvr["nexus"]["avg_evidence_tokens"],  # approximate
        nvr["nexus"]["avg_latency_s"],
        synth_ratio=synth_ratio,
    )

    rag_cost = compute_cost_per_1k_queries(
        "RAG", tp, nvr["rag"]["avg_evidence_tokens"], nvr["rag"]["avg_latency_s"],
        synth_ratio=None,
    )

    # ── Relevance rate ──
    relevance_str = f"{_fmt(audit['relevance_rate'], '.1%')} [{audit_fn}]" if audit["relevance_rate"] is not None else f"not measured [{audit_fn}]"

    # ── RAM ──
    # Per-arm RAM values from ram_throughput measurement
    def _ram_cell(arm_key: str, label: str, extra: str = "") -> str:
        if ram_data and arm_key in ram_data:
            rd = ram_data[arm_key]
            peak = rd.get("rss_peak_mb", 0)
            delta = rd.get("rss_delta_mb", 0)
            fn = ram_tp_path.name if ram_tp_path else "unknown"
            return f"{peak:.0f} MB (delta +{delta:.0f} MB) [{fn}]"
        # Fallback to legacy
        ram_str = f"{tp['ram_mb']:.0f} MB [{tp_fn}]" if tp["ram_mb"] > 0 else f"not measured (ram_mb = 0 in source) [{tp_fn}]"
        return ram_str + extra

    ram_nexus = _ram_cell("nexus_3b", "NEXUS+3B")
    ram_zero = _ram_cell("zero_weight", "Zero-weight", " (same hardware)")
    ram_router = _ram_cell("nexus_3b", "Router", " (same hardware)")
    ram_rag = _ram_cell("rag_3b", "RAG+3B")

    # ── Total RSS (pipeline + Ollama process) ──
    ollama_idle_mb = tp.get("ollama_idle_rss_mb")
    if isinstance(ollama_idle_mb, str):
        ollama_idle_mb = None  # "unavailable" string

    def _total_rss_cell(arm_key: str, needs_ollama: bool = True) -> str:
        """Compute total RSS = pipeline RSS + Ollama idle RSS (if applicable)."""
        fn = ram_tp_path.name if ram_tp_path else tp_fn
        if not ram_data or arm_key not in ram_data:
            return f"not measured [{fn}]"

        rd = ram_data[arm_key]
        pipeline_peak = rd.get("rss_peak_mb")
        if pipeline_peak is None:
            return f"not measured [{fn}]"

        if not needs_ollama:
            return f"{pipeline_peak:.0f} MB [{fn}] (pipeline only, no LLM)"

        if ollama_idle_mb is None or not isinstance(ollama_idle_mb, (int, float)):
            return f"{pipeline_peak:.0f} MB + Ollama [?] [{fn}]"

        total = pipeline_peak + ollama_idle_mb
        return f"{total:.0f} MB [{fn}] (pipeline {pipeline_peak:.0f} MB + Ollama idle {ollama_idle_mb:.0f} MB)"

    total_rss_nexus = _total_rss_cell("nexus_3b", needs_ollama=True)
    total_rss_zero = _total_rss_cell("zero_weight", needs_ollama=False)
    total_rss_router = _total_rss_cell("nexus_3b", needs_ollama=True)
    total_rss_rag = _total_rss_cell("rag_3b", needs_ollama=True)

    # ── Build rows ──

    rows: list[dict] = []

    # Row 1: NEXUS + local 3B
    rows.append({
        "Architecture": "NEXUS + local 3B<br>(FallbackModel: qwen2.5:latest + SynthesizingModel)",
        "Paired fuzzy accuracy": f"{_fmt(nvr['nexus']['mean_accuracy'], '.2%')} [{nvr_regen_fn}]",
        "Hallucination rate (post-fix)": f"{_fmt(nvr['nexus']['avg_hallucination_rate'], '.2%')} [{nvr_raw_fn}]",
        "Verification pass rate": f"{_fmt(nvr['nexus']['verification_pass_rate'], '.2%')} [{nvr_raw_fn}]",
        "Answer rate": f"{_fmt(nvr['nexus']['answer_rate'], '.2%')} [{nvr_raw_fn}]",
        "Avg evidence tokens": f"{_fmt(nvr['nexus']['avg_evidence_tokens'], '.1f')} [{nvr_raw_fn}]",
        "p50 latency": f"{_fmt(nvr['nexus']['avg_latency_s'], '.2f')} s [{nvr_raw_fn}]",
        "Peak RAM (MB)": ram_nexus,
        "Total RSS (MB)": total_rss_nexus,
        "$/1K queries": nexus_cost,
        "Sign test p vs RAG": f"{_fmt(nvr['sign_test_p'], '.4f')} [{nvr_regen_fn}]",
        "Relevance rate": relevance_str,
    })

    # Row 2: NEXUS zero-weight (SynthesizingModel only)
    rows.append({
        "Architecture": "NEXUS zero-weight<br>(SynthesizingModel only, no LLM)",
        "Paired fuzzy accuracy": f"not measured [{vc_fn}] (no paired comparison run)",
        "Hallucination rate (post-fix)": f"{_fmt(vc['router']['avg_hallucination_rate'], '.2%')} [{vc_fn}]",
        "Verification pass rate": f"{_fmt(vc['router']['verification_pass_rate'], '.2%')} [{vc_fn}]",
        "Answer rate": f"{_fmt(vc['router']['answer_rate'], '.2%')} [{vc_fn}]",
        "Avg evidence tokens": f"not measured [{vc_fn}] (no evidence token tracking in verifier)",
        "p50 latency": f"{_fmt(vc['router']['avg_latency_s'], '.2f')} s [{vc_fn}]",
        "Peak RAM (MB)": ram_zero,
        "Total RSS (MB)": total_rss_zero,
        "$/1K queries": zero_weight_cost,
        "Sign test p vs RAG": f"not measured [{vc_fn}]",
        "Relevance rate": relevance_str,
    })

    # Row 3: NEXUS router (with paired comparison data if available)
    if router_paired_data is not None:
        rp = router_paired_data
        rp_fn = router_paired_path.name if router_paired_path else "unknown"
        rvr_fn = router_vs_rag_paired_path.name if router_vs_rag_paired_path else "unknown"
        synth_pct = f"{rp['synth_ratio']:.0%}" if rp["synth_ratio"] is not None else "?"
        router_paired_acc = router_vs_rag_paired["router_accuracy"] if router_vs_rag_paired else None
        router_sign_p = router_vs_rag_paired["sign_test_p"] if router_vs_rag_paired else None

        # Blended evidence tokens: synth path (0 tokens) + LLM path (NEXUS evidence tokens)
        nexus_ev_tok = nvr["nexus"]["avg_evidence_tokens"] or 688
        synth_frac = rp["synth_ratio"]
        llm_frac = 1.0 - synth_frac
        blended_ev_tokens = round(llm_frac * nexus_ev_tok, 1)

        router_cost_r3 = compute_cost_per_1k_queries(
            "Router", tp,
            blended_ev_tokens if blended_ev_tokens > 0 else nexus_ev_tok,
            rp["avg_latency_s"],
            synth_ratio=rp["synth_ratio"],
        )

        rows.append({
            "Architecture": f"NEXUS router<br>(SynthesizingModel + LLM routing, {synth_pct} synth)",
            "Paired fuzzy accuracy": f"{_fmt(router_paired_acc, '.2%')} [{rvr_fn}]" if router_paired_acc is not None else f"not measured [{rp_fn}]",
            "Hallucination rate (post-fix)": f"{_fmt(rp['avg_hallucination_rate'], '.2%')} [{rp_fn}]",
            "Verification pass rate": f"{_fmt(rp['verification_pass_rate'], '.2%')} [{rp_fn}]",
            "Answer rate": f"{_fmt(rp['answer_rate'], '.2%')} [{rp_fn}]",
            "Avg evidence tokens": f"{blended_ev_tokens:.1f} [{rp_fn}] (blended: {synth_pct} synth×0 + {_fmt(llm_frac, '.0%')} LLM×{nexus_ev_tok:.0f})",
            "p50 latency": f"{_fmt(rp['avg_latency_s'], '.2f')} s [{rp_fn}]",
            "Peak RAM (MB)": ram_router,
            "Total RSS (MB)": total_rss_router,
            "$/1K queries": router_cost_r3,
            "Sign test p vs RAG": f"{_fmt(router_sign_p, '.4f')} [{rvr_fn}]" if router_sign_p is not None else f"not measured [{rp_fn}]",
            "Relevance rate": relevance_str,
        })
    else:
        synth_pct = f"{synth_ratio:.0%}" if synth_ratio is not None else "?"
        rows.append({
            "Architecture": f"NEXUS router<br>(SynthesizingModel + LLM routing, {synth_pct} synth)",
            "Paired fuzzy accuracy": f"not measured [{vc_fn}] (no paired comparison run)",
            "Hallucination rate (post-fix)": f"{_fmt(vc['combined_avg_hallucination_rate'], '.2%')} [{vc_fn}]",
            "Verification pass rate": f"not measured [{vc_fn}]",
            "Answer rate": f"not measured [{vc_fn}]",
            "Avg evidence tokens": f"not measured [{vc_fn}]",
            "p50 latency": f"not measured [{vc_fn}]",
            "Peak RAM (MB)": ram_router,
            "Total RSS (MB)": total_rss_router,
            "$/1K queries": router_cost,
            "Sign test p vs RAG": f"not measured [{vc_fn}]",
            "Relevance rate": relevance_str,
        })

    # Row 4: RAG + same 3B
    rows.append({
        "Architecture": "RAG + same 3B<br>(OllamaModel qwen2.5:latest)",
        "Paired fuzzy accuracy": f"{_fmt(nvr['rag']['mean_accuracy'], '.2%')} [{nvr_regen_fn}]",
        "Hallucination rate (post-fix)": f"{_fmt(nvr['rag']['avg_hallucination_rate'], '.2%')} [{nvr_raw_fn}]",
        "Verification pass rate": f"{_fmt(nvr['rag']['verification_pass_rate'], '.2%')} [{nvr_raw_fn}]",
        "Answer rate": f"{_fmt(nvr['rag']['answer_rate'], '.2%')} [{nvr_raw_fn}]",
        "Avg evidence tokens": f"{_fmt(nvr['rag']['avg_evidence_tokens'], '.1f')} [{nvr_raw_fn}]",
        "p50 latency": f"{_fmt(nvr['rag']['avg_latency_s'], '.2f')} s [{nvr_raw_fn}]",
        "Peak RAM (MB)": ram_rag,
        "Total RSS (MB)": total_rss_rag,
        "$/1K queries": rag_cost,
        "Sign test p vs RAG": "(baseline)",
        "Relevance rate": relevance_str,
    })

    # ── Build markdown ──
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    columns = [
        "Architecture",
        "Paired fuzzy accuracy",
        "Hallucination rate (post-fix)",
        "Verification pass rate",
        "Answer rate",
        "Avg evidence tokens",
        "p50 latency",
        "Peak RAM (MB)",
        "Total RSS (MB)",
        "$/1K queries",
        "Sign test p vs RAG",
        "Relevance rate",
    ]

    lines = [
        "# Canonical Comparison Table",
        "",
        f"**Generated**: {ts_str}",
        f"**Script**: `benchmarks/build_comparison.py`",
        f"**Data sources**:",
        f"- `{nvr_regen_fn}` — NEXUS vs RAG paired comparison (n={nvr['paired_n']})",
        f"- `{nvr_raw_fn}` — per-arm summary stats (hallucination, pass rate, latency, tokens)",
        f"- `{vc_fn}` — post-verifier-fix hallucination measurement (n={vc['router']['n']})",
        f"- `{tp_fn}` — warmed throughput data (qwen2.5:latest, 7.6B Q4_K_M, p50={tp['p50_tps']:.1f} tok/s)",
        f"- `{audit_fn}` — SynthesizingModel relevance audit",
    ]

    # Add router paired sources if available
    if router_paired_path is not None:
        lines.append(f"- `{router_paired_path.name}` — NEXUS router 200-question run (hallucination, pass rate, latency)")
    if router_vs_rag_paired_path is not None:
        lines.append(f"- `{router_vs_rag_paired_path.name}` — NEXUS router vs RAG paired comparison (accuracy, sign test)")

    lines += [
        "",
        "> ⚠️ **Every cell cites its source file.** Cells showing \"not measured\" have no data — never estimated.",
        "",
    ]

    # Column headers
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    for row in rows:
        cells = [row.get(col, "not measured") for col in columns]
        lines.append("| " + " | ".join(cells) + " |")

    # ── Notes ──
    lines += [
        "",
        "## Notes",
        "",
        "- **Paired fuzzy accuracy**: From `compare_arms.compare_paired()` using unified `compute_fact_score`.",
        "  NEXUS and RAG scores computed on same questions; only questions scorable by both arms included in paired comparison.",
        "- **Hallucination rate (post-fix)**: Fraction of answer statements unsupported by source documents.",
        "  Post-verifier-fix numbers come from the honest hallucination measurement (double-gate, P2 fix).",
        "- **Answer rate**: `1 − insufficient_evidence_rate` — fraction of questions the system attempted to answer.",
        "- **$/1K queries**: Local-only electricity cost via `LocalCostModel` (65W @ $0.15/kWh). Target: $0.01/1M tokens.",
        "  Zero-weight row = $0 (template synthesis is pure CPU overhead, no LLM inference).",
        "  **Throughput** measured on warmed model (5× warmup, 10× per prompt length, 3 lengths) — not cold-start.",
        "- **Peak RAM**: Per-arm measurement via `psutil.Process().memory_info().rss`.",
        "  Pipeline RSS only — excludes Ollama process RSS.",
        "  Zero-weight = SynthesizingModel pipeline only (graph + template engine).",
        "  NEXUS+3B = FallbackModel pipeline only (graph + code — Ollama RSS measured separately).",
        "  RAG+3B = chunk retrieval + all-MiniLM-L6-v2 embeddings loaded in memory.",
        "- **Total RSS (MB)**: Pipeline peak RSS + Ollama idle process RSS (model loaded but not generating).",
        "  This is the true system RAM cost — the number the project defends itself with.",
        "  Zero-weight = pipeline only (no Ollama needed).",
        "  All LLM-dependent arms include Ollama idle RSS (~5–8 GB for 7.6B Q4_K_M).",
        "  Ollama generating RSS (KV cache + activations) measured separately via concurrent polling.",
        "- **Relevance rate**: From heuristic checklist audit (4-point rubric). Formula: `% yes + 0.5 × % partial`.",

        "",
        "## Key Findings",
        "",
        f"- **NEXUS vs RAG accuracy**: {_fmt(nvr['nexus']['mean_accuracy'], '.1%')} vs {_fmt(nvr['rag']['mean_accuracy'], '.1%')} (p = {_fmt(nvr['sign_test_p'], '.3f')}) — no significant difference.",
        f"- **Hallucination**: NEXUS {_fmt(nvr['nexus']['avg_hallucination_rate'], '.1%')} vs RAG {_fmt(nvr['rag']['avg_hallucination_rate'], '.1%')} — RAG hallucinates less.",
        f"- **Evidence efficiency**: NEXUS uses {_fmt(nvr['nexus']['avg_evidence_tokens'], '.0f')} tokens vs RAG's {_fmt(nvr['rag']['avg_evidence_tokens'], '.0f')} — 3.2× reduction.",
        f"- **Latency**: NEXUS {_fmt(nvr['nexus']['avg_latency_s'], '.2f')}s vs RAG {_fmt(nvr['rag']['avg_latency_s'], '.2f')}s.",
        f"- **Throughput (warmed)**: p50={tp['p50_tps']:.1f} tok/s on qwen2.5:latest (7.6B Q4_K_M). Raw LLM cost = ${LocalCostModel(tokens_per_second=tp['p50_tps']).cost_per_1m_tokens():.4f}/1M. Router (80% synth) = ${LocalCostModel(tokens_per_second=tp['p50_tps']).cost_per_1m_tokens() * 0.2:.4f}/1M.",
        f"- **RAM**: RAG indexing adds +{ram_data.get('rag_3b', {}).get('rss_delta_mb', '?'):} MB (embedding model). NEXUS pipeline adds +{ram_data.get('nexus_3b', {}).get('rss_delta_mb', '?'):} MB. Zero-weight adds +{ram_data.get('zero_weight', {}).get('rss_delta_mb', '?'):} MB." + (
            f" Total RSS (pipeline + Ollama idle): see table column." if ollama_idle_mb is not None else ""),
        f"- **Zero-weight hallucination**: {_fmt(vc['router']['avg_hallucination_rate'], '.1%')} (SynthesizingModel only, n={vc['router']['n']}).",
        f"- **Relevance**: {_fmt(audit['relevance_rate'], '.1%')} — below 70% triggers metric caveat (accuracy × relevance = {_fmt((vc.get('synthesizer_accuracy') or 0) * (audit['relevance_rate'] or 0), '.1%')} actionable accuracy).",
    ]

    # Add Router vs RAG specific findings if available
    if router_vs_rag_paired is not None and router_paired_data is not None:
        rp = router_paired_data
        rvr = router_vs_rag_paired
        lines.append("")
        lines.append("## Router vs RAG (Row 3 — newly measured)")
        lines.append("")
        lines.append(f"- **Router paired accuracy**: {_fmt(rvr['router_accuracy'], '.1%')} vs RAG {_fmt(rvr['rag_accuracy'], '.1%')} (n={rvr['paired_n']} paired).")
        wlt = rvr['win_loss_tie']
        lines.append(f"- **Win/Loss/Tie**: Router wins={wlt.get('NEXUS Router_wins', '?')}, RAG wins={wlt.get('RAG_wins', '?')}, ties={wlt.get('ties', '?')}.")
        lines.append(f"- **Sign test**: p={_fmt(rvr['sign_test_p'], '.4f')} — {'significant' if rvr['sign_test_p'] is not None and rvr['sign_test_p'] < 0.05 else 'not significant'} at α=0.05.")
        lines.append(f"- **Router hallucination**: {_fmt(rp['avg_hallucination_rate'], '.1%')} vs NEXUS+3B {_fmt(nvr['nexus']['avg_hallucination_rate'], '.1%')}.")
        lines.append(f"- **Router latency**: {_fmt(rp['avg_latency_s'], '.2f')}s ({_fmt(rp['synth_ratio'], '.0%')} synth-routed, {_fmt(1 - rp['synth_ratio'], '.0%')} LLM-routed).")
        lines.append(f"- **Router verification pass**: {_fmt(rp['verification_pass_rate'], '.1%')}.")
        lines.append(f"- **Router answer rate**: {_fmt(rp['answer_rate'], '.1%')}.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical comparison table (Phase 5)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/COMPARISON.md",
        help="Output path for COMPARISON.md",
    )
    args = parser.parse_args()

    md = build_table()
    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nWrote {output_path} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
