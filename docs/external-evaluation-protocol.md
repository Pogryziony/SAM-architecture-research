# Sealed external evaluation protocol

**Status:** Specified tooling + controls (no sealed hidden-test result published).  
**Canonical status:** see [`CURRENT_STATE.md`](CURRENT_STATE.md).  
**Evaluator handoff package:** [`../evaluator_handoff/`](../evaluator_handoff/) (run itself remains `BLOCKED`).  
**Schema for future runs:** `nexus-eval-result-v1`.

## Purpose

Preserve the SAM `oracle_v1` suite as an **internal regression contract**. Do not
promote it as an untouched external test. External evaluation requires
independent domains, frozen corpora, and a hidden question set held by a
separate evaluator.

## Required controls

1. **Independent corpora** — source documents selected without access to hidden questions.
2. **Frozen source versions** — URL/commit + SHA-256 recorded before graph build.
3. **Frozen ingestion** — graph snapshot identity sealed before test disclosure.
4. **Hidden questions** — created or held by a separate evaluator role.
5. **No test-driven repair** — graph/alias/template edits after disclosure are protocol violations.
6. **Predefined configurations** — profile + `config_hash` + identity schema pinned.
7. **Preregistered metrics** — primary = grounded correct answer rate (all questions).
8. **Power analysis** — minimum detectable effect documented before the sealed run.
9. **One sealed final run** — additional runs are exploratory and unlabeled as final.

## Hidden-set composition (minimum)

Single-hop facts; genuine 2-/3-hop; comparisons; causal/dependency; temporal;
contradictions; ambiguous entities; distractors; unanswerable; changed facts;
paraphrases; typos; supported multilingual cases.

## Tracks

### System-level comparison

Each architecture receives a strong reasonable configuration under declared
resource constraints (RSS, latency, API budget).

### Controlled comparison

Hold answer model, corpus, and evaluation procedure constant; vary only
retrieval/evidence architecture.

## Commands (when credentials exist)

```bash
# List fair baseline arms (placeholders flagged)
python -c "from nexus.baselines import list_arms; print([a.arm_id for a in list_arms()])"

# Emit NOT_RUN for real LLM arm without credentials
python -c "from nexus.baselines import get_arm, BaselineRequest, run_baseline_or_not_run; \
print(run_baseline_or_not_run(get_arm('closed_book_llm'), BaselineRequest('closed_book_llm','q0','Who won?')).status)"

# Domain pack smoke (mini domain — no SAM core changes)
python -c "from nexus.domain import load_domain_pack; p=load_domain_pack('mini'); \
print(p.meta.version, p.build_graph().node_count, len(p.evaluation_tasks()))"
```

Environment prerequisites for real LLM arms:

- `NEXUS_LLM_API_KEY`
- `NEXUS_LLM_MODEL` (version-pinned string)
- Optional: `NEXUS_LLM_BASE_URL`, `NEXUS_LLM_API_VERSION`

## Explicit non-goals

- Do not generate a hidden-test success artifact unless separation controls are real.
- Do not describe placeholder arms as modern RAG or actual LLMs.
- Do not unseal by editing the graph after seeing failures.
