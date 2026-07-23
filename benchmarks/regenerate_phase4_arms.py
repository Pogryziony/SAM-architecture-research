"""Regenerate all Phase-4 Qwen arms with repaired identity (requires Ollama)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
RUNNER = ROOT / "benchmarks" / "run_phase4_arms.py"

ARMS = (
    ("closed_book", "phase4_qwen_closed_book_oracle_v1_repair.json"),
    ("long_context", "phase4_qwen_long_context_oracle_v1_repair.json"),
    ("bm25_rag", "phase4_bm25_rag_qwen_oracle_v1_repair.json"),
    ("dense_rag", "phase4_dense_rag_qwen_oracle_v1_repair.json"),
    ("hybrid_rag", "phase4_hybrid_rag_qwen_oracle_v1_repair.json"),
    ("hybrid_rerank_rag", "phase4_hybrid_rerank_rag_qwen_oracle_v1_repair.json"),
    ("nexus_graph_qwen", "phase4_nexus_graph_evidence_qwen_oracle_v1_repair.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--promote", action="store_true", help="Replace canonical filenames")
    args = parser.parse_args()

    # Health first (skip if a fresh health artifact already exists unless forcing)
    health = RESULTS / "phase4_qwen_health_repair.json"
    if health.exists() and not args.only:
        print("skip existing", health)
    else:
        if health.exists():
            health.unlink()
        subprocess.check_call(
            [sys.executable, str(RUNNER), "--arm", "health", "--output", str(health)],
            cwd=ROOT,
        )

    selected = ARMS
    if args.only:
        selected = [a for a in ARMS if a[0] in args.only]
    for arm, out_name in selected:
        out = RESULTS / out_name
        if out.exists():
            print("skip existing", out)
            continue
        cmd = [
            sys.executable,
            str(RUNNER),
            "--arm",
            arm,
            "--output",
            str(out),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        print("RUNNING", arm, flush=True)
        subprocess.check_call(cmd, cwd=ROOT)
        print("DONE", arm, flush=True)

    if args.promote:
        mapping = {
            "phase4_qwen_closed_book_oracle_v1_repair.json": "phase4_qwen_closed_book_oracle_v1.json",
            "phase4_qwen_long_context_oracle_v1_repair.json": "phase4_qwen_long_context_oracle_v1.json",
            "phase4_bm25_rag_qwen_oracle_v1_repair.json": "phase4_bm25_rag_qwen_oracle_v1.json",
            "phase4_dense_rag_qwen_oracle_v1_repair.json": "phase4_dense_rag_qwen_oracle_v1.json",
            "phase4_hybrid_rag_qwen_oracle_v1_repair.json": "phase4_hybrid_rag_qwen_oracle_v1.json",
            "phase4_hybrid_rerank_rag_qwen_oracle_v1_repair.json": "phase4_hybrid_rerank_rag_qwen_oracle_v1.json",
            "phase4_nexus_graph_evidence_qwen_oracle_v1_repair.json": "phase4_nexus_graph_evidence_qwen_oracle_v1.json",
            "phase4_qwen_health_repair.json": "phase4_qwen_health.json",
        }
        for src_name, dst_name in mapping.items():
            src = RESULTS / src_name
            dst = RESULTS / dst_name
            if not src.exists():
                continue
            if dst.exists():
                bak = RESULTS / (dst_name + ".pre_identity_repair.json")
                if not bak.exists():
                    dst.replace(bak)
                else:
                    dst.unlink()
            src.replace(dst)
            print("promoted", dst_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
