"""Derive docs/CURRENT_STATE.md tables from validated evidence manifests."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "CURRENT_STATE.md"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    commit = _git_head()
    manifest = _load("evidence_manifest_v1.json") or {}
    nexus = _load("eval_oracle_v1_grounded_evidence_repair.json") or _load(
        "eval_oracle_v1_grounded_phase3.json"
    )
    rel = _load("oracle_v1_retrieval_relevance_v1.json") or {}
    arms = [
        ("Qwen closed-book", "phase4_qwen_closed_book_oracle_v1.json"),
        ("Qwen long-context", "phase4_qwen_long_context_oracle_v1.json"),
        ("BM25 RAG+Qwen", "phase4_bm25_rag_qwen_oracle_v1.json"),
        ("Dense RAG+Qwen", "phase4_dense_rag_qwen_oracle_v1.json"),
        ("Hybrid RAG+Qwen", "phase4_hybrid_rag_qwen_oracle_v1.json"),
        ("Hybrid+rerank RAG+Qwen", "phase4_hybrid_rerank_rag_qwen_oracle_v1.json"),
        ("NEXUS graph-evidence+Qwen", "phase4_nexus_graph_evidence_qwen_oracle_v1.json"),
        ("NEXUS grounded (evidence repair)", "eval_oracle_v1_grounded_evidence_repair.json"),
    ]
    rows = []
    for label, fname in arms:
        art = _load(fname)
        if art is None:
            rows.append(f"| {label} | MISSING | `{fname}` |")
            continue
        status = art.get("status") or "UNKNOWN"
        src = str(art.get("source_commit") or "")[:12]
        ds = str(art.get("dataset_sha256") or "")[:12]
        rows.append(
            f"| {label} | {status} @ `{src}` ds=`{ds}` | `{fname}` |"
        )

    nonzero = rel.get("questions_with_relevant_chunks", "?")
    total = rel.get("questions_total", "?")
    body = f"""# NEXUS current state (canonical source of truth)

**Document role:** canonical current-state attestation for this repository.  
**Supersedes for status claims:** informal “validated architecture” wording in older docs.  
**Analyzed HEAD:** `{commit}`  
**Generated (UTC):** {datetime.now(timezone.utc).strftime("%Y-%m-%d")} via `benchmarks/sync_current_state.py`  
**Evidence manifest:** `{manifest.get("manifest_sha256", "not-built")}`

> If this file and another document disagree on *current* status, **this file wins**.
> Regenerate with: `python benchmarks/sync_current_state.py`

---

## Active architecture

| Item | Value |
|------|-------|
| Active architecture | **NEXUS** |
| Recommended production profile | `ProductionNEXUSConfig.grounded()` with `allow_synth_fallback=false` |
| Local LLM for Phase 4 comparisons | Ollama `qwen3.6:latest` (full digest enforced) |
| Kuzu | Experimental; deferred (ADR-0001) |
| Dataset identity | full canonical record hash (`nexus-canonical-dataset-v1`) |
| Retrieval relevance | entity/fact→chunk map (`{nonzero}/{total}` nonzero) |

---

## Supported claims

- NEXUS active; SAM Classic archived; safe profiles fail closed.
- Internal L1 beat deterministic placeholders (`VALIDATED_INTERNAL`).
- Schema-v1 terminals; regenerable aggregates; denominators exposed.
- Paired stats refuse placeholders, `NOT_RUN`, pending adjudication, and mixed comparison families.
- Exploratory `proxy_key_fact_correct` must not be quoted as primary `grounded_correct`.
- Evidence-bearing dual adjudication packets are required; empty evidence export is refused.
- Sealed-evaluator handoff package exists; sealed run **BLOCKED** until independent evaluator completes it.
- Metadata-only dataset rebinding is **invalid**.

`SynthesizingModel` / `EvidenceBlindModel` are **not** LLMs.

---

## Unsupported claims

- General LLM superiority  
- General modern-RAG superiority  
- Sealed external generalization  
- Completed dual human adjudication (responses still required)  
- Authoritative Kuzu  

**NO FULL SUPERIORITY VERDICT — human adjudication incomplete.**

---

## Artifact status (auto)

| Arm | Status | Artifact |
|-----|--------|----------|
{chr(10).join(rows)}
| Human adjudication | PENDING | `phase4_adjudication_export/` |
| Sealed external | BLOCKED | `evaluator_handoff/` |

NEXUS grounded source_commit in latest repair artifact: `{str((nexus or {}).get("source_commit") or "n/a")}`

---

## Next acceptance gates

1. Import dual human annotator responses; compute κ; resolve disagreements.  
2. Bind complete primary metrics; publish family-wide Holm-corrected paired stats.  
3. Regenerate all Phase-4 Qwen arms from this checkout (Ollama required).  
4. Independent evaluator + sealed external corpus.  
5. Revisit Kuzu only if product scope requires persistence.

---

## Where to look

| Need | Location |
|------|----------|
| This file | `docs/CURRENT_STATE.md` |
| Evidence manifest | `benchmarks/results/evidence_manifest_v1.json` |
| Artifact governance | `docs/ARTIFACT_GOVERNANCE.md` |
| Phase reports | `docs/EVIDENCE_REPORT_PHASE*.md` |
"""
    OUT.write_text(body, encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
