# NEXUS Realizer v1 — training status

**Decision: historical v1 checkpoints remain rejected. Grounded v2 passes;
stable neural v2 is ready only for a bounded 1–3 epoch pilot.**

The original failure report below is retained as historical evidence. The
verified recovery and corrected diagnosis are documented in
`docs/realizer-v2-quality-recovery.md`.

## Pilot outcome (2026-07-16)

The first controlled pilot training (`run_20260716T100428Z`) completed 4 epochs
(stopped early by gen quality regression). Despite loss decreasing from 191.8 to
3.5, both checkpoint epochs (1 and 3) exhibit catastrophic mode collapse:

- **Epoch 1**: 1 unique output across 100 validation samples
- **Epoch 3**: 4 unique outputs across 100 validation samples

All registered answer-quality gates FAIL:
- Relevance: 0% (threshold: >= 77%)
- Accuracy: 0% (threshold: >= baseline - 2pp)
- Naturalness: 0 points (threshold: >= 5pt improvement)
- Hallucination: 100% (threshold: <= baseline)

**Initial hypothesis (superseded)**: byte tokenization and model capacity were
suspected. Direct diagnostics later identified pathological tied-head
initialization (initial loss about 191.8 versus expected 5.56) and a fully
extractive target contract as the primary causes. Tokenization and capacity
remain future experiments, not verified root causes.

Full report: `benchmarks/results/realizer/run_20260716T100428Z/pilot_report.json`
Training artifacts: `models/realizer/run_20260716T100428Z/`

## Completed recovery and next step

Implemented: stable initialization, evidence-first serialization, strict
grounding fallback, text-level checkpoint metrics, initial-loss gate and
mode-collapse stop. Grounded validation is 100% exact with 0% hallucination.
The remaining step is one short neural pilot, stopped after epoch 1 unless raw
neural F1, grounding and uniqueness justify continuing to at most epoch 3.

## Confirmed results

| Area | Result | Status |
|---|---:|---|
| Acquired train-only inventory | 8,282 records | PASS |
| Verifier/audit-passed unique pairs | 7,127 | PASS |
| Train/validation records | 5,693 / 1,434 | PASS |
| Validation share | 20.12% | PASS |
| Known split leakage | 0 | PASS |
| Oracle | 181 cases, publication guard valid | PASS |
| Stage 0 | 30 cases, both arms answer, paired N=25 | VALID |
| Stage 2 | relevance 78.33%, seeds 0/1/42 identical | PASS |
| Stage 3 | 110 turns, reference resolution 87.5% | PASS |
| Stage 3 single-turn regression | 0.00pp | PASS |
| Stage 3 dialogue-state p50 | 0.048ms | PASS, below 5ms |
| ER3 checkpoint | 3,487,600 bytes, manifest SHA-256 verified | PASS |
| Realizer parameters | 2,770,752 | PASS, below 50M |
| CPU preflight | forward/backward, no weights written | PASS |
| 50-step overfit smoke | loss 190.477 → 142.881, no weights written | PASS |
| Default training policy | 5 epochs, patience 3 | PASS |

The first historical 50-epoch CPU run remains rejected. Its loss converged,
but answer relevance, accuracy, naturalness and hallucination regressed. The
failure was traced to the original greedy byte decoder entering repetition
loops. Repetition penalty `1.2` and no-repeat trigram blocking fixed coherence,
but only registered post-training metrics can select a checkpoint.

## What was fixed

- Presets now control the actual Realizer and ER3 training loops.
- Effective training values and their canonical hash are recorded in manifests.
- ER3 loads the exact file whose size and SHA-256 passed verification.
- The checkpoint can be reproduced directly from the repository; an explicit
  external checkpoint remains supported.
- NEXUS receives structured entity-resolution results by injection.
- ER3 static entity projections are precomputed instead of rebuilt per query.
- Stage 0 has a deterministic, dependency-free lexical RAG backend.
- Stage 2 distinguishes the exact registered 30-case protocol from smoke runs
  and is deterministic across `PYTHONHASHSEED=0,1,42`.
- Stage 3 measures dialogue-state, resolver and full-pipeline latency separately.
- Dataset reproduction can use the immutable archived acquisition snapshot;
  every archived record and its provenance are still hash-checked.
- Evaluation, readiness, preflight and smoke artifacts use exact `.sha256`
  sidecars and refuse accidental overwrite.
- `check_phase4_readiness.py` fails closed unless every Phase 0–4 input agrees.

## Final launch contract

Training may start only when `benchmarks/check_phase4_readiness.py` reports
`GO_FOR_REALIZER_TRAINING`. It checks:

1. at least 5,000 unique verifier-passed train-only pairs;
2. a valid oracle and Stage 0 baseline;
3. registered Stage 2 PASS for all three required hash seeds with one canonical hash;
4. a complete passing Stage 3 run using the verified ER3 bundle;
5. exact dataset/config identity in readiness, preflight and overfit smoke;
6. immutable sidecars for every evidence artifact;
7. a default training limit of at most 5 epochs with patience at most 3.

## Recommended next run

Use the generation-aware trainer and promote checkpoints progressively:

```bash
# 1 epoch: plumbing and generation smoke
python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset smoke \
  --manifest data/distillation/realizer_v1/manifest.json

# Continue only if generation and registered quality do not regress.
python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset quick \
  --manifest data/distillation/realizer_v1/manifest.json

# Final pilot ceiling: 5 epochs, patience 3.
python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset pilot \
  --manifest data/distillation/realizer_v1/manifest.json
```

Stop after any failed generation-aware or registered answer-quality gate. The
8- and 12-epoch presets require a separate decision backed by improvement in
relevance, naturalness, accuracy and hallucination. A 50-epoch preset has been
removed because training loss alone did not predict usable answers.

## First pilot: REALIZER_PILOT_FAIL

Run `run_20260716T100428Z` completed 4 epochs. Mode collapse confirmed at both
checkpoint epochs (1 and 3). No checkpoint passes any answer-quality gate.
Training artifacts preserved for diagnostics at `models/realizer/run_20260716T100428Z/`.
Full evaluation at `benchmarks/results/realizer/run_20260716T100428Z/`.
