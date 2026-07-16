# NEXUS pilot integrity

This document describes the contracts that must hold before another Realizer
or Entity Ranker V3 (ER3) training run is considered valid.

## Why the correction was necessary

The first preset implementation resolved and printed values such as epochs,
batch size, patience and learning rate, but most of those values never reached
the real training loops. ER3 still used hard-coded training values, while
Realizer v2 only applied the epoch override. A `smoke` command could therefore
look short without actually controlling all work performed.

The ER3 checkpoint path had a similar integrity gap: the resolver verified one
file and could then load another. Benchmark evidence also mixed smoke runs
with registered gates and stored hashes that could not match their final files.

## Correct contracts

### Training parameters

The effective configuration is resolved in this order:

```text
explicit CLI value > preset > model default > committed config
```

The resolved values are passed to the trainer and written to the output
manifest. Realizer manifests contain `effective_training_config` and
`effective_config_sha256`. ER3 selection artifacts contain the complete
`ER3TrainingConfig`.

### Checkpoint identity

The verified ER3 checkpoint is currently committed with its `config.json`,
`vocab.json` and manifest so Phase 3 can be reproduced without private storage.
Before PyTorch deserialization, the resolver checks both byte size and SHA-256
from the manifest. An explicit `--weights-path` or `ER3_WEIGHTS_PATH` may still
be used, but the exact verified path is the one passed to `torch.load`.

### Dependency direction

The enforced direction is:

```text
stack/ -> nexus/
```

NEXUS defines resolver and dialogue behaviours but does not import their stack
implementations. The composition layer creates ER3, lexical and dialogue-aware
resolvers and injects them into `NEXUSRunner`.

### Resolver diagnostics

Every resolver reports a `ResolutionResult` containing:

- selected entity IDs;
- all candidates and optional scores;
- raw candidate-pool size before top-K;
- resolver name and version;
- threshold and rejection reason;
- fallback flag;
- resolver latency.

Dialogue-state overhead is measured separately from ER3 inference and total
pipeline latency. The Stage 3 5ms threshold applies to dialogue-state work,
not to the whole neural resolution path.

This replaces the previous use of private `_canonical_ids` fields.

### Benchmark identity

- `registered_stage2_v1` means exactly 30 ordered cases.
- Any other size is `smoke_stage2_N` and cannot produce a registered PASS.
- Canonical hashes exclude runtime latency and output filenames.
- Exact file hashes live in adjacent `.sha256` sidecars.
- Invalid historical evidence is preserved and listed in
  `benchmarks/results/artifact_status.json`.

## Verified readiness result

The complete Phase 0–4 aggregator is implemented in
`benchmarks/check_phase4_readiness.py`. A successful result requires all of
the following at the same time:

1. 7,127 verifier-passed unique train-only pairs and zero known leakage;
2. a valid 181-case oracle;
3. a valid 30-case Stage 0 paired baseline;
4. registered Stage 2 PASS under seeds 0, 1 and 42 with one canonical hash;
5. a passing 110-turn Stage 3 run with the verified ER3 bundle;
6. `READY_FOR_TRAINING`, CPU preflight and a decreasing 50-step overfit smoke;
7. a default Realizer policy of no more than 5 epochs and patience 3.

The current verified run satisfies those conditions. This authorizes a short
Realizer pilot; it does not authorize deployment or claim that trained answer
quality has already passed.
