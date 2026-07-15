# NEXUS Realizer v1 — training status

**Decision: BLOCKED pending regenerated Stage 0, registered Stage 2 and Stage 3 evidence.**

The repository has enough unique train-only data and a working CPU Realizer,
but it is no longer correct to describe the model as “not trained”. The first
50-epoch run completed externally. Its loss converged, yet post-training answer
quality regressed because the original greedy byte decoder entered repetition
loops. Repetition penalty `1.2` plus no-repeat trigram blocking removed those
loops in decoder diagnostics, but the corrected model path has not yet passed
the complete registered evaluation suite.

## Confirmed results

| Area | Result | Status |
|---|---:|---|
| Unique acquired train-only targets | 8,282 | PASS |
| Verifier/audit-passed pairs | 7,127 | PASS |
| Train/validation records | 5,693 / 1,434 | PASS |
| Validation share | 20.12% | PASS |
| Known split leakage | 0 | PASS |
| Realizer parameters | 2,770,752 | PASS, below 50M |
| First CPU training | 50 epochs, best validation loss 1.778 | COMPLETE, historical |
| Original post-training relevance, 30 cases | 58.33% | REGRESSION |
| Original post-training hallucination, 30 cases | 90.32% | REGRESSION |
| Decoder diagnostic | repetition penalty + trigram block removed loops | IMPLEMENTED |
| Current committed Stage 0 | RAG 0/30, paired N=0 | INVALID |
| Current committed Stage 2 | two 5-case smoke runs | NOT A REGISTERED GATE |
| Latest Stage 3 | 15.62% resolution, 12.166ms p50 | FAIL |

Model weights and generated datasets remain outside Git. Their manifests and
SHA-256 values identify the required external artifacts.

## Integrity fixes in the current implementation

- Presets now flow into the actual Realizer and ER3 trainers instead of being
  printed without changing the training loop.
- Effective training parameters and their canonical hash are written to run
  manifests.
- `--list-presets` works without PyTorch for Realizer v2.
- ER3 loads the exact external file that passed size and SHA-256 verification.
- NEXUS receives entity resolution through an injected structured result; it
  no longer imports `stack.*` or inspects resolver private fields.
- Stage 2 uses exactly `registered_stage2_v1` for 30 cases. Other sizes are
  smoke runs and cannot report a registered PASS.
- Stage 2 serialized artifacts use a `.sha256` sidecar, avoiding a
  self-referential hash.
- Stage 3 runs through the injected resolver and records candidates, scores,
  selected entry nodes, state updates and separate resolver/pipeline latency.

## Required order before the next pilot

1. Materialize the external ER3 checkpoint identified by its manifest.
2. Reproduce the 7,127-pair dataset from exact Git blobs and confirm all hashes.
3. Run a valid 30-case Stage 0 with both NEXUS and RAG producing answers.
4. Run registered Stage 2 on exactly 30 cases for `PYTHONHASHSEED=0,1,42`.
5. Run the complete 110-turn Stage 3 with the injected ER3 resolver.
6. Regenerate readiness, preflight and overfit-smoke evidence.
7. Only if every blocking check passes, run Realizer pilots for 1, 3 and at
   most 5 epochs.

## Safe command outline

```bash
# Metadata-only; does not need torch.
python benchmarks/train_nexus_realizer_v2.py --list-presets

# Registered Stage 2 uses exactly 30 cases.
PYTHONHASHSEED=0 python benchmarks/run_stage2_stage3.py \
  --stage 2 --limit 30 --output-dir /tmp/nexus-stage2-seed0

# ER3 evaluation requires the external checkpoint.
ER3_WEIGHTS_PATH=/external/er3/weights.pt \
python benchmarks/run_stage2_stage3.py \
  --stage 3 --er3 \
  --er3-dir models/encoder/entity_ranker_v3_20260715T191041Z \
  --output-dir /tmp/nexus-stage3

# After readiness passes: deliberately short generation-aware pilots.
python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset smoke --manifest /external/realizer/manifest.json

python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset quick --epochs 3 \
  --manifest /external/realizer/manifest.json

python benchmarks/train_nexus_realizer_v2.py \
  --mode pilot --preset quick --epochs 5 \
  --manifest /external/realizer/manifest.json
```

Do not run the 12-, 25- or 50-epoch presets until the 1→3→5 sequence improves
registered relevance, accuracy, naturalness and hallucination simultaneously.
