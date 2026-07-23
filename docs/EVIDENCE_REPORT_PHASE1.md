# Evidence report — Phase 1 (evaluation foundation)

**Analyzed commit (start):** `518103188e797d1aded310ff56134045264f6968`  
**Phase goal:** Make NEXUS claims, identity, evaluation records, and baseline honesty match repository evidence — without fabricating external LLM/RAG wins.

## Implemented changes (this phase)

1. Canonical source of truth: `docs/CURRENT_STATE.md`
2. Claim hygiene for architecture validation (placeholders ≠ LLMs)
3. Config identity v2 (complete behavior hash, deep-freeze maps, portable paths)
4. Evaluation schema `nexus-eval-result-v1` + validator + primary metric helpers
5. Paired stats helpers (bootstrap CI, McNemar, Cohen's dz)
6. Fair baseline registry with honest `NOT_RUN` / placeholder flags
7. Domain-pack interface + `mini` second domain + `sam` pack wrapper
8. Graph mutation fixes (stale index purge; edge evidence/temporal updates)
9. Explicit safe CLI profiles; JSON emits profile + config hash
10. Kuzu labeled experimental
11. Licensing inventory + backlog + sealed external protocol + human rubric

## Behavior changed

| Area | Before | After |
|------|--------|-------|
| Architecture verdict label | `VALIDATED` | `VALIDATED_INTERNAL` (+ claim_scope) |
| Config hash payload | Incomplete (boosts/type_priority/ER3 SHAs omitted) | `nexus-config-identity-v2` full payload |
| Nested `type_priority` | Mutable dict after freeze | `MappingProxyType` |
| Duplicate `add_edge` | Silently discarded updates | Updates confidence/evidence/temporal |
| Node update indexes | Stale type/alias/property possible | Purge + reindex |
| CLI profiles | Missing l1/deterministic | Full named set; identity in JSON |

## Tests executed / results

| Suite | Result | Notes |
|-------|--------|-------|
| Focused Phase-1 tests (44) | **44 passed** | config identity, eval schema, graph mutation, domain packs, baselines, docs, architecture validation helpers, public API, pipeline config |
| Full `tests/` excluding heavy torch jobs | **707 passed**, 2 failed, 1 skipped | Failures: `tests/test_entity_candidate.py` semantic stage — HuggingFace SSL `CERTIFICATE_VERIFY_FAILED` fetching `all-MiniLM-L6-v2` (environment; unrelated to this phase) |
| `tests/test_architecture.py` | run separately below | nexus↛stack boundary |
| Optional Kuzu suite | **NOT_RUN** | `kuzu` package not installed |
| Optional ER3 neural torch job | **NOT_RUN** this phase | torch available locally (`2.12.0+cpu`) but heavy job not re-executed after base suite |
| Fair baseline smoke | **NOT_RUN** arm status | `python benchmarks/run_fair_baselines.py --arm closed_book_llm` → `{"status":"NOT_RUN"}` |

Frozen JSON under `benchmarks/results/*.json` and `models/` were **not** modified. Only the maintained index `benchmarks/results/INDEX.md` was updated for claim wording.

## Benchmarks actually performed

- No new full `oracle_v1` architecture campaign re-run in this phase (historical artifact preserved).
- Fair-baseline harness smoke emitted honest `NOT_RUN` without credentials.
- No real LLM/RAG API evaluations.

## Experiments not performed

- Sealed external multi-domain hidden test
- Full AnswerPlan / realizer training
- End-to-end `grounded()` latency/RSS campaign
- Kuzu authoritative migration
- Re-training of any encoder/realizer weights

## Artifacts created

- Docs listed above
- New modules under `nexus/evaluation/`, `nexus/baselines/`, `nexus/domain/`
- `benchmarks/run_fair_baselines.py`
- New regression tests (config identity, eval schema, graph mutation, domain packs, baselines, docs consistency)

## Remaining blockers

See `docs/REMAINING_WORK_BACKLOG.md` (B2–B4 especially).

## Claims now supported vs still unsupported

| Claim | Status |
|-------|--------|
| NEXUS is active; SAM Classic archived | Supported |
| Internal L1 contract beat deterministic placeholders on `oracle_v1` | Supported (historical artifact) |
| Recommended production profile is `grounded()` | Supported |
| Config identity covers behavior-changing fields (v2) | Supported by tests |
| NEXUS outperforms real LLMs / modern RAG | **Unsupported** |
| External sealed generalization | **Unsupported** |
| Single e2e ≤500 MB/≤500 ms on `grounded()` | **Unsupported** (component evidence only) |

## Compatibility effects

- New `ProductionNEXUSConfig.config_hash` values differ from historical v1 hashes.
- Historical artifacts remain readable; validators accept legacy schemas only with `legacy_schema: true` (new runs set this).
- Architecture validation decision string changed for **new** runs; committed historical JSON unchanged.
