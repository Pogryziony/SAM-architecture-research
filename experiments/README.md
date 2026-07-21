# NEXUS Experiments

These experiments test the graph-first reasoning pipeline. They are separate from
the archived SAM experiments in `../sam-lm/experiments/`.

## Experiment tracks

| Track | Question | Directory | Status |
|-------|----------|-----------|--------|
| Entity extraction | Can we reliably extract typed entities from project artifacts? | `entity-extraction/` | **Implemented** (`evaluate_extraction.py`) |
| Relation extraction | Can we extract typed, directed relationships between entities? | `relation-extraction/` | **Implemented** (`evaluate_relations.py`); structural `sub_experiment` edges excluded from F1; current F1=100% on gold; CI interim gate ≥50% |
| Graph traversal | Does traversal return relevant paths for domain questions? | `graph-traversal/` | **Scaffold** — unit coverage lives in `tests/test_traversal_budgets.py` and `tests/test_scoring.py`; dedicated experiment scripts not yet added |
| Path ranking | Can we score and rank paths so the correct reasoning chain is top-K? | `path-ranking/` | **Scaffold** — ranking logic is in `nexus/graph/scoring.py` with tests; dedicated experiment scripts not yet added |

Empty directories marked **Scaffold** are intentional placeholders. Do not cite them as active evaluation tracks until scripts and gold sets exist.

## Running experiments

Implemented tracks contain their own eval scripts. Example:

```bash
python experiments/relation-extraction/evaluate_relations.py
python experiments/entity-extraction/evaluate_extraction.py
```

Frozen oracle dataset + paired oracle/predicted reporting:

```bash
python benchmarks/build_frozen_oracle_dataset.py --force
# Default predicted arm = lexical∪ER3 union handoff (no training).
# Preserves ER3 top-k membership; diversifies pack order; mention-aware focus.
python benchmarks/run_oracle_vs_predicted.py --predicted-resolver union --dummy-model --output benchmarks/results/oracle_vs_predicted_union_recall_full_TIMESTAMP.json
```

Synthetic traversal budget campaign (small/medium/large; prereg `EXPERIMENT_TRAVERSAL_BUDGETS_V1.md`):

```bash
python benchmarks/run_traversal_budget_campaign.py --output benchmarks/results/traversal_budget_campaign_TIMESTAMP.json
# CI small gate:
python benchmarks/ci_stage_gates.py --gate traversal
```

AnswerPlan / Realizer training status (no full training by default):

```bash
python benchmarks/record_answer_plan_status.py --force
```

L1 acceptance paired publish (default realizer is `deterministic_render`):

```bash
python benchmarks/run_oracle_vs_predicted.py --predicted-resolver union --realizer-backend deterministic_render --model synth --output benchmarks/results/oracle_vs_predicted_union_l1_det_full_TIMESTAMP.json
```

Grounded hybrid publish (pointer/comparison + path-render fallback):

```bash
python benchmarks/run_oracle_vs_predicted.py --predicted-resolver union --realizer-backend grounded_v1 --model synth --output benchmarks/results/oracle_vs_predicted_union_grounded_full_TIMESTAMP.json
```

Stage 5 contradiction frozen eval (`EXPERIMENT_CONTRADICTION_POLICY_V2.md`):

```bash
python benchmarks/eval_contradiction_policy.py --mode frozen --output benchmarks/results/contradiction_policy_frozen_TIMESTAMP.json
```

Stage 4 rule corpus (development + frozen under `rule-engine-v2`):

```bash
python benchmarks/eval_rule_engine.py --mode development --output benchmarks/results/rule_corpus_v1_dev_eval_TIMESTAMP.json
python benchmarks/eval_rule_engine.py --mode frozen --output benchmarks/results/rule_corpus_v1_frozen_eval_TIMESTAMP.json
```

Stage 5 contradiction F1 / calibration campaign (`EXPERIMENT_CONTRADICTION_POLICY_V1.md`):

```bash
python benchmarks/eval_contradiction_policy.py --output benchmarks/results/contradiction_policy_campaign_TIMESTAMP.json
```

Stage 6 bi-temporal oracle replay:

```bash
python benchmarks/run_bitemporal_replay.py --output benchmarks/results/bitemporal_replay_TIMESTAMP.json
```

Canonical graph rebuild (deterministic content hash; schema v2 bi-temporal fields):

```bash
python benchmarks/build_canonical_graph.py --print-hash-only
```

## Relationship to SAM experiments

The SAM experiments (in `../sam-lm/experiments/`) tested the associative memory
approach (embedding-based retrieval + gated memory injection). Those experiments
are **archived** — their findings informed the NEXUS design but are not part of
the active NEXUS research.
