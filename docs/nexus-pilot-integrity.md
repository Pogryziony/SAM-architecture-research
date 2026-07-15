# NEXUS pilot integrity

This document describes the contracts that must hold before another Realizer
or Entity Ranker V3 (ER3) training run is considered valid.

## Why the correction was necessary

The first preset implementation resolved and printed values such as epochs,
batch size, patience and learning rate, but most of those values never reached
the real training loops. ER3 still used hard-coded training values, while
Realizer v2 only applied the epoch override. A `smoke` command could therefore
look short without actually controlling all work performed.

The external ER3 checkpoint path had a similar integrity gap: the resolver
verified the requested external file and then loaded `model_dir/weights.pt`
instead. Benchmark evidence also mixed smoke runs with registered gates and
stored hashes that could not match their own final files.

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

### External weights

Only `config.json`, `vocab.json` and manifests belong in Git. A checkpoint is
supplied through `--weights-path` or `ER3_WEIGHTS_PATH`. Before PyTorch
deserialization, the resolver checks both byte size and SHA-256 from the model
manifest. The verified path is passed directly to `torch.load`.

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

This replaces the previous use of private `_canonical_ids` fields.

### Benchmark identity

- `registered_stage2_v1` means exactly 30 ordered cases.
- Any other size is `smoke_stage2_N` and cannot produce a registered PASS.
- Canonical hashes exclude runtime latency and output filenames.
- Exact file hashes live in adjacent `.sha256` sidecars.
- Invalid historical evidence is preserved and listed in
  `benchmarks/results/artifact_status.json`.

## Current blockers

1. The committed Stage 0 v2 artifact is invalid because RAG answered 0/30.
2. Registered Stage 2 must be regenerated for three hash seeds.
3. Stage 3 must be rerun with the verified external ER3 checkpoint.
4. The existing 7,127-pair dataset must be reproduced and hash-checked.
5. Readiness must be regenerated from those corrected inputs.

No full training is authorized while any blocker remains.
