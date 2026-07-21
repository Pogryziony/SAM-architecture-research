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
# Path ranking uses mention-aware path_score_focus, max_paths=12, and
# diversified ungrounded handoff. Use er3|lexical for historical baselines.
python benchmarks/run_oracle_vs_predicted.py --predicted-resolver union --dummy-model --output benchmarks/results/oracle_vs_predicted_union_cov_full_TIMESTAMP.json
```

Synthetic traversal budget campaign (small/medium/large):

```bash
python benchmarks/run_traversal_budget_campaign.py --output benchmarks/results/traversal_budget_campaign_TIMESTAMP.json
```

Canonical graph rebuild (deterministic content hash):

```bash
python benchmarks/build_canonical_graph.py --print-hash-only
```

## Relationship to SAM experiments

The SAM experiments (in `../sam-lm/experiments/`) tested the associative memory
approach (embedding-based retrieval + gated memory injection). Those experiments
are **archived** — their findings informed the NEXUS design but are not part of
the active NEXUS research.
