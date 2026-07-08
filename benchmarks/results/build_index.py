"""
Auto-generate benchmarks/results/INDEX.md from all JSON files in the directory.

Scans the results directory, extracts metadata and key metrics from each JSON,
and writes a Markdown table summarizing every result file.

Usage:
    python benchmarks/results/build_index.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent


def _format_size(bytes_val: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_val < 1024:
            return f"{bytes_val:.0f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.0f} TB"


def _format_date(timestamp: float) -> str:
    """Human-readable UTC date from mtime."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _extract_metrics(data: object) -> str:
    """Extract key summary metrics from a JSON file.

    Handles several known schemas:
      - nexus_vs_rag_* (regenerated): comparison + total_questions
      - nexus_vs_rag_* (full run): summary with top-level config
      - verifier_check_*: graph_provenance + verification results
      - relevance_sample.json: array of Q&A items
      - synth_*: config + results/summary
    """
    if isinstance(data, list):
        return f"questions={len(data)}"

    if not isinstance(data, dict):
        return "-"

    parts: list[str] = []

    # ── comparison block (nexus_vs_rag regenerated) ──────────────
    comparison = data.get("comparison", {}) or {}
    if comparison:
        paired_n = comparison.get("paired_n", "?")
        wlt = comparison.get("win_loss_tie", {}) or {}
        nexus_mean = comparison.get("nexus", {}).get("mean_accuracy", None)
        rag_mean = comparison.get("rag", {}).get("mean_accuracy", None)
        parts.append(f"N={paired_n}")
        if nexus_mean is not None:
            parts.append(f"NEXUS={nexus_mean:.1%}")
        if rag_mean is not None:
            parts.append(f"RAG={rag_mean:.1%}")
        w = wlt.get("NEXUS_wins", "?")
        l = wlt.get("RAG_wins", "?")
        t = wlt.get("ties", "?")
        parts.append(f"W/L/T={w}/{l}/{t}")
        if "sign_test_p" in comparison:
            parts.append(f"p={comparison['sign_test_p']:.4g}")

    # ── summary block (full run) ─────────────────────────────────
    summary = data.get("summary", {}) or {}
    if summary:
        if "total_questions" in summary:
            parts.append(f"questions={summary['total_questions']}")
        if "nexus_coverage" in summary:
            parts.append(f"nexus_cov={summary['nexus_coverage']:.1%}")
        if "rag_coverage" in summary:
            parts.append(f"rag_cov={summary['rag_coverage']:.1%}")
        if "hallucination_rate" in summary:
            parts.append(f"halluc={summary['hallucination_rate']:.1%}")

    # ── total_questions at top level ──────────────────────────────
    if "total_questions" in data and not any("questions=" in p for p in parts):
        parts.append(f"questions={data['total_questions']}")

    # ── graph_provenance ──────────────────────────────────────────
    gp = data.get("graph_provenance", {}) or {}
    if gp:
        node_count = gp.get("node_count")
        edge_count = gp.get("edge_count")
        if node_count is not None:
            parts.append(f"nodes={node_count}")
        if edge_count is not None:
            parts.append(f"edges={edge_count}")

    # ── config block ──────────────────────────────────────────────
    config = data.get("config", {}) or {}
    if config:
        model = config.get("model") or config.get("llm_backend") or config.get("underlying_model")
        if model:
            parts.append(f"model={model}")
        if "limit" in config:
            parts.append(f"limit={config['limit']}")

    # ── results array length ──────────────────────────────────────
    results = data.get("results")
    if isinstance(results, list) and not any("questions=" in p for p in parts):
        parts.append(f"results={len(results)}")

    return ", ".join(parts) if parts else "-"


def _extract_reproduce(data: object, filename: str) -> str:
    """Try to extract a reproduction command or provide reasonable default."""
    if isinstance(data, list):
        return "see source"

    if not isinstance(data, dict):
        return "see source"

    # Check for build command in graph_provenance
    gp = data.get("graph_provenance", {}) or {}
    cmd = gp.get("build_command")
    if cmd:
        return f"`{cmd}`"

    # Check for source_file in regenerated output
    source = data.get("source_file")
    if source:
        return f"`python benchmarks/regenerate_comparison.py {source}`"

    # nexus_vs_rag → likely from regenerate or run_benchmark
    if filename.startswith("nexus_vs_rag"):
        if "comparison" in data:
            return "`python benchmarks/regenerate_comparison.py`"
        return "`python benchmarks/run_benchmark.py`"

    # verifier check
    if filename.startswith("verifier_"):
        return "`python benchmarks/verifier_check.py`"

    # synth
    if filename.startswith("synth"):
        return "`python benchmarks/synth_eval.py`"

    # relevance sample
    if filename.startswith("relevance"):
        return "see source"

    return "see source"


def build_index() -> str:
    """Scan results directory and return INDEX.md content."""
    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if not json_files:
        return "# Benchmarks Results Index\n\n*No result files found.*\n"

    rows: list[str] = []
    for filepath in json_files:
        stat = filepath.stat()
        filename = filepath.name
        date_str = _format_date(stat.st_mtime)
        size_str = _format_size(stat.st_size)

        try:
            with open(filepath, encoding="utf-8") as fh:
                data = json.load(fh)
            metrics = _extract_metrics(data)
            reproduce = _extract_reproduce(data, filename)
        except Exception:
            metrics = "(unparseable)"
            reproduce = "see source"

        rows.append(
            f"| `{filename}` | {date_str} | {size_str} | {metrics} | {reproduce} |"
        )

    header = (
        "# Benchmarks Results Index\n\n"
        f"*Auto-generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from {len(json_files)} result files.*\n\n"
        "| File | Date (UTC) | Size | Key Metrics | Command to Reproduce |\n"
        "|---|---|---|---|---|\n"
    )

    return header + "\n".join(rows) + "\n"


def main() -> None:
    content = build_index()
    index_path = RESULTS_DIR / "INDEX.md"
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"Wrote index: {index_path}")
    print(f"  {len(content.splitlines()) - 5} data rows")


if __name__ == "__main__":
    main()
