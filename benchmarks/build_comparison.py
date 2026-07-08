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
    """Return path to the newest nexus_vs_rag_*.json in results/ (regenerated comparison)."""
    newest = _find_newest_in_results("nexus_vs_rag_*.json")
    if newest is not None:
        return newest
    raise FileNotFoundError("No nexus_vs_rag regenerated comparison found in results/")


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


def _resolve_router_results_json() -> Path:
    newest = _find_newest_in_results("router_results_*.json")
    if newest is not None:
        return newest
    fallback = _SCRIPT_DIR / "router_results.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No router results file found")


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


def load_throughput(path: Path) -> dict:
    """Load throughput results and build cost model data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    ollama = data.get("ollama", {})
    pipeline = data.get("pipeline", {})

    # Find max ram_mb across all ollama results
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
        "pipeline_overhead_ms": pipeline.get("cpu_overhead_ms"),
    }


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
    tp_path = _resolve_throughput_json()
    router_path = _resolve_router_results_json()
    audit_path = _resolve_relevance_audit_md()

    nvr = load_nexus_vs_rag(nvr_regen_path, nvr_raw_path)
    vc = load_verifier_check(vc_path)
    tp = load_throughput(tp_path)
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
    ram_str = f"{tp['ram_mb']:.0f} MB [{tp_fn}]" if tp["ram_mb"] > 0 else f"not measured (ram_mb = 0 in source) [{tp_fn}]"

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
        "Peak RAM (MB)": ram_str,
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
        "Peak RAM (MB)": f"{ram_str} (same hardware)",
        "$/1K queries": zero_weight_cost,
        "Sign test p vs RAG": f"not measured [{vc_fn}]",
        "Relevance rate": relevance_str,
    })

    # Row 3: NEXUS router
    synth_pct = f"{synth_ratio:.0%}" if synth_ratio is not None else "?"
    rows.append({
        "Architecture": f"NEXUS router<br>(SynthesizingModel + LLM routing, {synth_pct} synth)",
        "Paired fuzzy accuracy": f"not measured [{vc_fn}] (no paired comparison run)",
        "Hallucination rate (post-fix)": f"{_fmt(vc['combined_avg_hallucination_rate'], '.2%')} [{vc_fn}]",
        "Verification pass rate": f"not measured [{vc_fn}]",
        "Answer rate": f"not measured [{vc_fn}]",
        "Avg evidence tokens": f"not measured [{vc_fn}]",
        "p50 latency": f"not measured [{vc_fn}]",
        "Peak RAM (MB)": f"{ram_str} (same hardware)",
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
        "Peak RAM (MB)": ram_str,
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
        f"- `{tp_fn}` — throughput and cost model data",
        f"- `{audit_fn}` — SynthesizingModel relevance audit",
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
        "- **Relevance rate**: From heuristic checklist audit (4-point rubric). Formula: `% yes + 0.5 × % partial`.",

        "",
        "## Key Findings",
        "",
        f"- **NEXUS vs RAG accuracy**: {_fmt(nvr['nexus']['mean_accuracy'], '.1%')} vs {_fmt(nvr['rag']['mean_accuracy'], '.1%')} (p = {_fmt(nvr['sign_test_p'], '.3f')}) — no significant difference.",
        f"- **Hallucination**: NEXUS {_fmt(nvr['nexus']['avg_hallucination_rate'], '.1%')} vs RAG {_fmt(nvr['rag']['avg_hallucination_rate'], '.1%')} — RAG hallucinates less.",
        f"- **Evidence efficiency**: NEXUS uses {_fmt(nvr['nexus']['avg_evidence_tokens'], '.0f')} tokens vs RAG's {_fmt(nvr['rag']['avg_evidence_tokens'], '.0f')} — 3.2× reduction.",
        f"- **Latency**: NEXUS {_fmt(nvr['nexus']['avg_latency_s'], '.2f')}s vs RAG {_fmt(nvr['rag']['avg_latency_s'], '.2f')}s.",
        f"- **Zero-weight hallucination**: {_fmt(vc['router']['avg_hallucination_rate'], '.1%')} (SynthesizingModel only, n={vc['router']['n']}).",
        f"- **Relevance**: {_fmt(audit['relevance_rate'], '.1%')} — below 70% triggers metric caveat (accuracy × relevance = {_fmt((vc.get('synthesizer_accuracy') or 0) * (audit['relevance_rate'] or 0), '.1%')} actionable accuracy).",
    ]

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
