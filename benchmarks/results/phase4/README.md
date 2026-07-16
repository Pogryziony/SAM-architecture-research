# NEXUS Phase 0–4 evidence

This directory is the immutable pre-training evidence bundle generated from
source commit `7b1920947a155404c58bf98701fdd4f8d54c696e`.

The final decision is recorded in `phase4_readiness.json`:

```text
GO_FOR_REALIZER_TRAINING
blocking_checks: []
```

## Contents

- `oracle.json`: 181-case graph proof/provenance publication guard;
- `stage0.json`: registered 30-case NEXUS versus offline lexical RAG baseline;
- `stage2_seed*/`: registered Stage 2 under `PYTHONHASHSEED=0,1,42`;
- `stage3/`: complete 110-turn dialogue protocol using the verified ER3 model;
- `readiness.json`: exact model/config/dataset readiness decision;
- `preflight.json`: CPU forward/backward validation without writing weights;
- `overfit_smoke.json`: 50-step loss-decrease check without writing weights;
- `phase4_readiness.json`: all-or-nothing aggregate decision.

Every JSON artifact has an adjacent `.sha256` file. The three Stage 2 runs
share one canonical content hash. Dataset records and their manifest live in
`data/distillation/realizer_v1/`.

No full Realizer training was run while producing this bundle. The authorized
next action is the generation-aware 1→3→5 epoch pilot described in
`docs/nexus-realizer-pretraining-status.md`.
