# Evidence Report — Phase 4 identity repair (evidence foundation)

**Branch:** `fix/evidence-foundation-and-roadmap`  
**Supersedes for identity claims:** stale `source_commit` / question-only hashes in Phase 1–4 artifacts authored at `d4ebf4e` / `5181031`.

## What was repaired

| Gap | Fix |
|-----|-----|
| Question-only `dataset_sha256` | `nexus.evaluation.dataset_identity.hash_dataset` over gold/rubric fields |
| Metadata-only rebind | `_rebind_nexus_dataset.py` refuses; tombstoned rebind artifact |
| NEXUS arm provenance | Regenerated `eval_oracle_v1_grounded_evidence_repair.json` via `run_eval_v1` |
| Invalid RAG recall | `oracle_v1_retrieval_relevance_v1.json` (191/191 nonzero); metrics use chunk IDs |
| Proxy vs primary | `proxy_key_fact_correct` only; `grounded_correct` not applicable without adjudication |
| Empty human packets | Evidence-bearing export under `phase4_adjudication_export_evidence_v1/` |
| Full Qwen digest | Exact digest match required |
| Dense pin | revision + optional file hashes in `benchmarks/pins/` |
| Prompt / long-context | `prompt_sha256` + `long_context_prefix_sha256` |
| TTFT honesty | Labeled `prompt_eval_duration_ms_nonstream_proxy` |
| Holm family | `multiple_comparison.holm_adjust` / `apply_holm_to_comparisons` |
| Graph ownership | `nexus.ingestion.canonical_graph` |
| Evidence manifest | `benchmarks/results/evidence_manifest_v1.json` |
| Docs sync | `benchmarks/sync_current_state.py` |
| Repro | `Dockerfile` + `requirements.lock.txt` + `docs/ARTIFACT_GOVERNANCE.md` |

## Still incomplete (honest gates)

- Dual human responses: **PENDING** (packets ready; κ tooling ready).
- Phase-4 Qwen arm JSON files still carry pre-repair `source_commit` / question-only hashes until regenerated with Ollama via `run_phase4_arms.py`.
- Sealed external evaluation: **BLOCKED** (`evaluator_handoff/`).
- Full superiority: **not claimed**.

## Regeneration commands

```bash
python benchmarks/regenerate_evidence_identity.py
python benchmarks/export_dual_adjudication.py
python benchmarks/sync_current_state.py
# Requires local Ollama qwen3.6:latest with full digest:
python benchmarks/run_phase4_arms.py --arm closed_book --output benchmarks/results/phase4_qwen_closed_book_oracle_v1_repair.json
```
