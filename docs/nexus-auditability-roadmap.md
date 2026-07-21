# NEXUS auditability and reasoning roadmap

Status: active implementation plan.  This document separates implemented
capabilities from proposed work; it is not evidence that later stages passed.

## Goal

Make every NEXUS answer replayable and measurable while preserving the stack
constraints: CPU-only execution, no required LLM in the reasoning path, peak
RSS at or below 500 MB for the complete stack, and end-to-end p50 at or below
500 ms.

## Repository gap analysis

| Capability | State before this work | Required change |
|---|---|---|
| Typed graph | Present (`Node`, `Edge`, relation vocabulary) | Preserve and evolve compatibly |
| Source references | Free-form node/edge strings | Measure coverage now; introduce structured source records later |
| Bounded traversal | Depth and beam width are bounded | Add explicit expansion/time budgets and report truncation |
| Evidence | Structured evidence pack exists | Attach stable proof steps and source references |
| Verification | Hallucination verifier exists | Include verifier output in a composite, transparent audit |
| Counter-case | `contradicts` edge exists | Surface contradictions for every selected proof |
| Answer readiness | No canonical diagnostic | Add decomposable readiness score and recommendation |
| Oracle NEXUS evaluation | Entry-node override exists only at function level | Add fail-closed canonical runner mode |
| Rules | Not implemented as a formal engine | Add versioned, bounded rules after oracle baseline |
| Temporal knowledge | Timestamps are operational only | Add valid-time and observed-time semantics later |

## Foundation implemented in this change

1. `nexus.reasoning.audit` creates deterministic proof steps from selected
   graph paths.
2. Each proof step receives a stable ID and collects edge evidence plus node
   source references.
3. Adjacent `contradicts` edges are reported as counter-evidence.
4. A decomposable readiness diagnostic records evidence quality, provenance
   coverage, verifier support, path relevance, and opposition clarity.
5. The recommendation is one of `answer`, `conditional_answer`, or `abstain`.
   It is diagnostic and does not mutate the answer.
6. The canonical runner serializes proof validity, proof/counter counts,
   provenance coverage, readiness score, action, and real path scores.
7. `NEXUSRunner.run_oracle()` requires non-empty `gold_entities` for every
   question and cannot silently fall back to SAM, ER3, or lexical resolution.
8. Readiness thresholds are part of `NEXUSConfig`, validation, serialization,
   and the production config hash.

## Measurement contract

Run two independent arms on the same questions:

| Arm | Input entities | Purpose |
|---|---|---|
| `oracle` | Frozen `gold_entities` | Measures graph traversal, evidence, realization, and verification |
| `predicted` | SAM/ER3/lexical output | Measures the complete stack |

Every oracle record must eventually contain:

```json
{
  "id": "q-0001",
  "question": "...",
  "gold_entities": ["Entity_A"],
  "gold_answer": ["Entity_B"],
  "gold_path": [
    ["Entity_A", "depends_on", "Entity_B"]
  ],
  "should_abstain": false
}
```

Minimum report fields:

- answer exact match and token/entity F1;
- gold-path recall and proof validity;
- provenance coverage;
- contradiction precision/recall;
- answer/conditional/abstain distribution;
- selective accuracy versus coverage;
- p50, p95, and p99 per stage;
- peak RSS and graph size;
- source SHA, config hash, dataset hash, and evaluation mode.

Readiness thresholds must remain diagnostic until calibrated on validation data
and evaluated once on a frozen test split.  They must not be tuned on the test
split or retroactively applied to historical artifacts.

## Next implementation stages

### Stage 1: frozen oracle benchmark

- Create versioned train/validation/test records covering direct lookup,
  two-hop, three/four-hop, contradictions, no-answer, and temporal questions.
- Add gold answer and gold proof-path scoring.
- Add paired `oracle` versus `predicted` reporting.
- Add publication guards rejecting missing records, hashes, or per-question
  diagnostics.

**Partial implementation (2026-07-21):** Frozen contract `oracle_v1` lives in
`benchmarks/qa-dataset/oracle_v1.jsonl` (+ manifest; **188 records** including
curated `oracle_families_v1` coverage for direct/two-hop/three-hop/
contradiction/no-answer/temporal). Build with
`python benchmarks/build_frozen_oracle_dataset.py`. Paired publication schema
`nexus-oracle-vs-predicted-v2` defaults the predicted arm to
`UnionResolver` (lexical∪frozen ER3 with question-gated handoff; no new
training) and reports ER decomposition (`pool_recall`, `entry_recall`, proof
`gold_entity_coverage`, path recall). Current full union+focus artifact:
`benchmarks/results/oracle_vs_predicted_union_focus_full_*.json`
(`predicted.entry_recall_mean≈0.95`, `pool_recall_mean=1.0`,
`gold_path_recall_mean≈0.96`, `gold_entity_coverage_mean≈0.61`). Path ranking
now scores against `path_score_focus` leading entries (default 2) with
`max_paths=12`, so hub fillers no longer bury gold edges outside the proof
window. Raw ER3-only / pre-focus union paired artifacts remain historical
baselines. Remaining gap is proof entity coverage (≈0.61 vs ≈0.70 target);
further family expansion and bi-temporal gold remain open.

Gate: 100% valid records, deterministic rerun, and complete provenance for the
benchmark artifact.

### Stage 2: explicit traversal budgets

- Add maximum expanded edges/nodes and an optional monotonic time budget.
- Return traversal statistics and a `truncated` reason.
- Treat budget exhaustion as a visible conditional answer or abstention, never
  as a complete search.

**Partial implementation (2026-07-21):** `NEXUSConfig.max_expanded_edges` /
`max_expanded_nodes` / `max_traversal_ms` and `TraversalStats` report
truncation. Answer pipeline attaches `traversal_stats` to results; truncated
search forces `conditional_answer`. Path ranking also exposes
`path_score_focus` / `max_paths` so large ER handoffs stay expandable while
proof selection stays focused. Synthetic campaign
`python benchmarks/run_traversal_budget_campaign.py --output ...` measures
small/medium/large graphs against NEXUS hard p50/p95/RSS limits and requires
tight-budget truncation. Reference-CPU preregistration remains open.

Gate: no unbounded traversal; p95 and RSS remain inside the assigned NEXUS
budget on small, medium, and large synthetic graphs.

### Stage 3: structured provenance

- Replace free-form source strings with stable source IDs and records.
- Keep compatibility adapters for existing ingestion.
- Record locator, content hash, observed time, extraction method, and source
  reliability.
- Prevent unconditional answers when required provenance is incomplete.

**Partial implementation (2026-07-21):** `SourceRecord` adapters + ingestion
`properties["provenance"]` attach. Audit computes
`structured_provenance_coverage`; `require_structured_provenance=True`
(default on `ProductionNEXUSConfig.grounded()`) blocks unconditional answers
when only free-form sources are present. Free-form coverage still gates the
library default path.

Gate: 100% provenance coverage for accepted benchmark answers and reproducible
source resolution.

### Stage 4: formal rule engine

- Introduce a restricted Datalog-like rule representation.
- Bound rule depth, activations, and recursion.
- Version every rule and record premises plus rule ID in each inference step.
- Keep asserted and inferred facts distinct.

Gate: 100% valid rule proofs and target F1 fixed in preregistration before the
frozen run.

### Stage 5: contradiction policy and calibration

- Distinguish contradiction, supersession, different validity periods, and
  source disagreement.
- Add conflict-resolution policies based on explicit source metadata.
- Calibrate readiness and abstention using risk-coverage curves, Brier score,
  and expected calibration error.

Gate: contradiction F1 and high-confidence answer accuracy meet preregistered
thresholds; no unresolved conflict yields an unconditional answer.

### Stage 6: bi-temporal knowledge

- Add `valid_from`/`valid_to` and `observed_at`/`retracted_at`.
- Support `as_valid_at` and `as_known_at` queries.
- Add point-in-time replay tests that prohibit using knowledge learned later.

Gate: deterministic historical replay and no look-ahead leakage.

### Stage 7: deterministic realization

- Render L1 and L2 answers exclusively from the structured answer and audit.
- Require every generated statement to map to a proof step or explicit
  uncertainty statement.
- Keep any learned realizer optional and outside the zero-LLM acceptance path.

Gate: zero unsupported statements and identical output for identical structured
input.

## Proposed resource allocation

The complete stack limit is 500 MB RSS and 500 ms p50.  NEXUS should target:

| Metric | Target | Hard limit |
|---|---:|---:|
| Peak RSS | 200 MB | 250 MB |
| p50 | 150 ms | 250 ms |
| p95 | 300 ms | 450 ms |
| Proof validity | 100% | 100% |
| Provenance for unconditional answers | 100% | 100% |
| Unsupported statement rate | 0% | 1% |

These are proposed engineering budgets.  They become experiment gates only
after preregistration against a defined graph size and reference CPU.
