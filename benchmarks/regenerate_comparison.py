"""
Regenerate NEXUS-vs-RAG comparison using the unified scorer.

Loads existing results (answers + ground truths) from an existing JSON,
re-scores EVERYTHING with ``benchmarks.scoring.compute_fact_score``,
computes a PAIRED comparison via ``benchmarks.compare_arms.compare_paired``,
and saves the result to ``benchmarks/results/``.

Usage:
    python benchmarks/regenerate_comparison.py [input_json] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure benchmarks/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from benchmarks.scoring import compute_fact_score
from benchmarks.compare_arms import compare_paired, pretty_print_comparison


def regenerate(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Re-score and regenerate the comparison.

    Returns the full result dict (saved to JSON and also printed).
    """
    # ── Load existing results ───────────────────────────────────────
    print(f"Loading: {input_path}")
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    results = data["results"]
    total = len(results)
    print(f"  {total} questions loaded")

    # ── Re-score everything ─────────────────────────────────────────
    nexus_scores: list[float | None] = []
    rag_scores: list[float | None] = []

    for r in results:
        ground_truth = r["ground_truth"]
        nexus_answer = r["nexus"]["answer"]
        rag_answer = r["rag"]["answer"]

        # NEXUS
        ns = compute_fact_score(nexus_answer or "", ground_truth)
        r["nexus"]["accuracy"] = ns["fuzzy_accuracy"]
        r["nexus"]["exact_accuracy"] = ns["exact_accuracy"]
        r["nexus"]["scoring_detail"] = ns["scoring_detail"]

        # RAG
        rs = compute_fact_score(rag_answer or "", ground_truth)
        r["rag"]["accuracy"] = rs["fuzzy_accuracy"]
        r["rag"]["exact_accuracy"] = rs["exact_accuracy"]
        r["rag"]["scoring_detail"] = rs["scoring_detail"]

        nexus_scores.append(ns["fuzzy_accuracy"])
        rag_scores.append(rs["fuzzy_accuracy"])

    # ── Compute paired comparison ───────────────────────────────────
    comparison = compare_paired(nexus_scores, rag_scores, "NEXUS", "RAG")

    pretty_print_comparison(comparison)

    # ── Build output ────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%SZ")

    output: dict[str, object] = {
        "generated_at": timestamp.isoformat(),
        "source_file": str(input_path),
        "scorer": "benchmarks.scoring.compute_fact_score (unified)",
        "total_questions": total,
        "comparison": comparison,
        "results": results,  # re-scored results with updated accuracy fields
    }

    # ── Save ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"nexus_vs_rag_{ts_str}.json"
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved regenerated comparison to: {output_path}")

    # ── Update INDEX.md ─────────────────────────────────────────────
    _update_index(output_dir, filename, timestamp, comparison)

    return output


def _update_index(
    output_dir: Path,
    filename: str,
    timestamp: datetime,
    comparison: dict[str, object],
) -> None:
    """Create or append to benchmarks/results/INDEX.md."""
    index_path = output_dir / "INDEX.md"

    paired_n = comparison.get("paired_n", 0)
    wlt = comparison.get("win_loss_tie", {})
    p_val = comparison.get("sign_test_p", "N/A")

    nexus_acc = comparison.get("nexus", {}).get("mean_accuracy", "N/A")
    rag_acc = comparison.get("rag", {}).get("mean_accuracy", "N/A")

    if isinstance(nexus_acc, float):
        nexus_str = f"{nexus_acc:.2%}"
    else:
        nexus_str = str(nexus_acc)
    if isinstance(rag_acc, float):
        rag_str = f"{rag_acc:.2%}"
    else:
        rag_str = str(rag_acc)

    date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    entry = (
        f"| {date_str} | `{filename}` | "
        f"`python benchmarks/regenerate_comparison.py` | "
        f"N={paired_n}, NEXUS={nexus_str}, RAG={rag_str}, "
        f"W/L/T={wlt.get('NEXUS_wins','?')}/{wlt.get('RAG_wins','?')}/{wlt.get('ties','?')}, "
        f"p={p_val} |"
    )

    if index_path.exists():
        with open(index_path, encoding="utf-8") as fh:
            existing = fh.read()
        # Append after the header section
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(existing.rstrip() + "\n" + entry + "\n")
    else:
        header = (
            "# Benchmarks Results Index\n\n"
            "| Date (UTC) | File | Command | Summary |\n"
            "|---|---|---|---|\n"
        )
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(header + entry + "\n")

    print(f"Updated index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate NEXUS-vs-RAG comparison with unified scorer",
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="benchmarks/nexus_vs_rag_200.json",
        help="Path to existing results JSON (default: benchmarks/nexus_vs_rag_200.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory for regenerated output (default: benchmarks/results)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    regenerate(input_path, output_dir)


if __name__ == "__main__":
    main()
