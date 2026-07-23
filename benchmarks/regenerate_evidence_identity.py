"""Regenerate evidence-foundation identity artifacts from this checkout.

1. Rebuild canonical graph snapshot provenance
2. Build retrieval relevance table (entity/fact → chunk)
3. Re-run NEXUS grounded on oracle_v1 via run_eval_v1 (answers regenerated)
4. Write end-to-end evidence manifest
5. Tombstone invalid metadata-only rebind artifact

Qwen Phase-4 arms still require Ollama:
  python benchmarks/run_phase4_arms.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nexus.baselines.rag_corpus import build_canonical_corpus
from nexus.evaluation.dataset_identity import assert_primary_dataset_hash, hash_dataset
from nexus.evaluation.evidence_manifest import (
    build_evidence_manifest,
    write_evidence_manifest,
)
from nexus.evaluation.relevance import build_relevance_table
from nexus.ingestion.canonical_graph import build_canonical_sam_graph


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ORACLE = ROOT / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def load_oracle() -> list[dict]:
    rows = []
    for line in ORACLE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_nexus_grounded(out: Path) -> dict:
    if out.exists():
        out.unlink()
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "run_eval_v1.py"),
        "--arm",
        "nexus",
        "--profile",
        "grounded",
        "--domain",
        "sam",
        "--dataset",
        str(ORACLE),
        "--dataset-id",
        "oracle_v1",
        "--model",
        "dummy",
        "--comparison-mode",
        "system_level",
        "--output",
        str(out),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    return json.loads(out.read_text(encoding="utf-8"))


def tombstone_rebind() -> None:
    old = RESULTS / "eval_oracle_v1_grounded_phase3_dataset_rebind.json"
    if not old.exists():
        return
    tomb = json.loads(old.read_text(encoding="utf-8"))
    tomb["status"] = "INVALID_METADATA_REBIND"
    tomb["superseded_by"] = "eval_oracle_v1_grounded_evidence_repair.json"
    tomb["note"] = (
        "Metadata-only dataset rebinding is forbidden. Use the regenerated "
        "evidence_repair artifact from regenerate_evidence_identity.py."
    )
    old.write_text(
        json.dumps(tomb, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nexus-rerun", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    commit = _git(["git", "rev-parse", "HEAD"])
    tree = _git(["git", "rev-parse", "HEAD^{tree}"])
    questions = load_oracle()
    ds_hash = hash_dataset(questions)
    print("dataset_sha256", ds_hash)
    print("source_commit", commit)

    graph_id = "skipped"
    if not args.skip_graph:
        _graph, prov = build_canonical_sam_graph()
        graph_id = prov["graph_snapshot_id"]
        (RESULTS / "canonical_graph_snapshot.json").write_text(
            json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("graph_snapshot_id", graph_id, "nodes", prov["node_count"])

    corpus = build_canonical_corpus(
        ROOT,
        globs=(
            "README.md",
            "docs/**/*.md",
            "sam-lm/docs/**/*.md",
            "sam-lm/experiments/**/*.md",
        ),
        chunk_size=800,
        overlap=100,
    )
    rel = build_relevance_table(questions, corpus)
    rel_path = RESULTS / "oracle_v1_retrieval_relevance_v1.json"
    rel_path.write_text(
        json.dumps(rel, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "relevance_nonzero",
        rel["questions_with_relevant_chunks"],
        "/",
        rel["questions_total"],
    )
    if rel["questions_with_relevant_chunks"] <= 0:
        raise SystemExit("relevance table has zero hits; refusing empty retrieval gold")

    nexus_path = RESULTS / "eval_oracle_v1_grounded_evidence_repair.json"
    if args.skip_nexus_rerun:
        if not nexus_path.exists():
            raise SystemExit("missing nexus artifact and --skip-nexus-rerun set")
        art = json.loads(nexus_path.read_text(encoding="utf-8"))
    else:
        art = run_nexus_grounded(nexus_path)
        art["source_tree"] = tree
        art["dataset_hash_schema"] = "nexus-canonical-dataset-v1"
        art["graph_snapshot_id"] = graph_id
        art["claim_eligibility"] = {
            "full_primary_superiority": False,
            "reason": "human adjudication incomplete; exploratory proxies only",
        }
        nexus_path.write_text(
            json.dumps(art, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    assert_primary_dataset_hash(art, questions)
    if art.get("source_commit") != commit:
        raise SystemExit(
            f"source_commit mismatch: artifact={art.get('source_commit')} head={commit}"
        )

    tombstone_rebind()

    manifest = build_evidence_manifest(
        root=ROOT,
        dataset_id="oracle_v1",
        dataset_sha256=ds_hash,
        graph_snapshot_id=graph_id,
        model_identities={"nexus_grounded": "DummyModel+ProductionNEXUSConfig.grounded"},
        config_hashes={"nexus_grounded": str(art.get("config_hash") or "")},
        prompt_sha256={},
        artifact_paths=[nexus_path, rel_path],
        adjudication_status="PENDING_ADJUDICATION",
    )
    write_evidence_manifest(RESULTS / "evidence_manifest_v1.json", manifest)
    print("wrote", nexus_path)
    print("manifest", manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
