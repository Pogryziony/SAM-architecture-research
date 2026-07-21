# EXPERIMENT: Traversal Budgets V1 — Reference-CPU Preregistration

**Pre-registered**: 2026-07-21  
**Status**: ACTIVE — Stage 2 reference-CPU preregistration  
**Repository**: SAM-architecture-research  
**Campaign schema**: `nexus-traversal-budget-campaign-v1`  
**Runner**: `python benchmarks/run_traversal_budget_campaign.py --output ...`

---

## Purpose

Pin the Stage 2 synthetic traversal-budget campaign against a named reference
CPU profile so PASS/FAIL cannot drift silently across machines. Budgets and
hard limits are fixed here; they must not be retuned against a failing CI host.

## Reference CPU profile

| Field | Value |
|---|---|
| `reference_cpu_label` | `github-actions-ubuntu-latest-x86_64` |
| `os_family` | `Linux` |
| `python_min` | `3.11` |
| Allowed CI runners | GitHub Actions `ubuntu-latest` (x86_64) |
| Local publication | Allowed when artifact records `platform` + `cpu_model` + `python_version` |

The campaign artifact MUST emit:

- `preregistration_id`: `traversal-budgets-v1`
- `reference_cpu_label` (above)
- `platform` (`sys.platform`)
- `python_version`
- `cpu_model` (best-effort string; empty string if unavailable)
- `source_sha`
- hard limits matching this document

## Graph sizes (fixed)

| Size | Nodes | Branching |
|---|---:|---:|
| small | 50 | 2 |
| medium | 500 | 3 |
| large | 5_000 | 3 |

## Hard limits (fixed; from auditability roadmap)

| Metric | Hard limit |
|---|---:|
| Peak RSS | 250 MB |
| Latency p50 (default budgets) | 250 ms |
| Latency p95 (default budgets) | 450 ms |

## Behavioral gates

1. Default budgets (`max_expanded_edges=10000`, `max_expanded_nodes=5000`) must
   **not** truncate on any size.
2. Tight budgets (`max_expanded_edges=8`, `max_expanded_nodes=8`) must truncate
   on every size.
3. Artifact validation is fail-closed via
   `benchmarks.run_traversal_budget_campaign.validate_campaign_artifact`.

## CI policy

- PR CI runs the **small** size campaign plus unit tests
  (`tests/test_traversal_budgets.py`, `tests/test_traversal_budget_campaign.py`).
- Full small/medium/large publication remains a local or nightly command; the
  committed PASS artifact must cite this preregistration id.

## Non-goals

- Retuning beam width / depth to chase a flaky host
- Claiming a laptop-specific SLA as the production gate
- Replacing InMemoryGraphStore with Kuzu inside this experiment
