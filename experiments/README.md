# NEXUS Experiments

These experiments test the graph-first reasoning pipeline. They are separate from
the archived SAM experiments in `../sam-lm/experiments/`.

## Experiment tracks

| Track | Question | Directory |
|-------|----------|-----------|
| Entity extraction | Can we reliably extract typed entities from project artifacts? | `entity-extraction/` |
| Relation extraction | Can we extract typed, directed relationships between entities? | `relation-extraction/` |
| Graph traversal | Does traversal return relevant paths for domain questions? | `graph-traversal/` |
| Path ranking | Can we score and rank paths so the correct reasoning chain is top-K? | `path-ranking/` |

## Running experiments

Each experiment directory contains its own config and run scripts.
See the individual directories for details.

## Relationship to SAM experiments

The SAM experiments (in `../sam-lm/experiments/`) tested the associative memory
approach (embedding-based retrieval + gated memory injection). Those experiments
are **archived** — their findings informed the NEXUS design but are not part of
the active NEXUS research.
