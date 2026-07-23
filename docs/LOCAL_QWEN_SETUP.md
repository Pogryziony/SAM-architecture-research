# Local Qwen 3.6 setup (Phase 4)

**Pinned model:** Ollama tag `qwen3.6:latest`  
**Digest:** `07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522`  
**Do not substitute** `qwen2.5:*` or other tags without a new preregistration.

## Prerequisites

1. Ollama installed and running (`ollama serve`).
2. Model present: `ollama list` must show `qwen3.6:latest` with the digest above.
3. For dense RAG: local HF cache of `sentence-transformers/all-MiniLM-L6-v2`; prefer `HF_HUB_OFFLINE=1`.

## Health check

```bash
python benchmarks/run_phase4_arms.py --arm health --output benchmarks/results/phase4_qwen_health.json
```

## Full arms (manual / local job — not PR CI)

```bash
powershell -File benchmarks/run_phase4_all_arms.ps1
```

Or per arm via `benchmarks/run_phase4_arms.py --arm ...`.

## Notes

- Generation uses `think: false`.
- Safe production `grounded()` does **not** call Qwen as fallback.
- NEXUS-graph+Qwen is evaluation-only (`nexus_graph_evidence_qwen_3_6_internal`).
