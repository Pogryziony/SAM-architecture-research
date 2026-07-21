# NEXUS Experiments

These experiments test the graph-first reasoning pipeline. They are separate from
the archived SAM experiments in `../sam-lm/experiments/`.

## Experiment tracks

| Track | Question | Directory | Status |
|-------|----------|-----------|--------|
| Entity extraction | Can we reliably extract typed entities from project artifacts? | `entity-extraction/` | **Implemented** (`evaluate_extraction.py`) |
| Relation extraction | Can we extract typed, directed relationships between entities? | `relation-extraction/` | **Implemented** (`evaluate_relations.py`); structural `sub_experiment` edges excluded from F1; production keeps `enable_cooccurrence_edges=false`; interim F1 gate ≥50% in CI |
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
python benchmarks/build_frozen_oracle_dataset.py
python benchmarks/run_oracle_vs_predicted.py --output benchmarks/results/oracle_vs_predicted_TIMESTAMP.json
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
