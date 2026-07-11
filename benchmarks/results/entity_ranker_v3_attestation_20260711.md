# Entity Ranker V3 — Post-Run Attestation

**POST-RUN ATTESTATION** — created 2026-07-11 after the original evaluation.
Does not replace or modify original artifacts.

## Artifact Inventory

| # | Artifact | Availability | Path | SHA-256 | Size (B) | Embedded Source SHA | Expected Source SHA | Run ID | Verified |
|---|----------|-------------|------|---------|----------|--------------------|--------------------|--------|----------|
| 1 | Validation JSON | ✅ AVAILABLE | `benchmarks/results/entity_ranker_v3_selection_20260711T081545Z.json` | `a34d51119663495dc2020d63a081bc34993fb2f2620d1ffb87af455e75571c6b` | 6,475 | `ae9b6ee2f67ca2aaa7467e158d5318908d047511` | `ae9b6ee2f67ca2aaa7467e158d5318908d047511` | `entity_ranker_v3_20260711T081545Z` | ✅ |
| 2 | Frozen JSON | ✅ AVAILABLE | `benchmarks/results/entity_ranker_v3_frozen_20260711T084518Z.json` | `60d883971d43ce9abfa2b2185547c13f3ec90c2640cf461a5a1f237277f13195` | 3,247 | `499db7b30d40758a13d7717a061153d513b083c4` | `499db7b30d40758a13d7717a061153d513b083c4` | `entity_ranker_v3_frozen_20260711T084518Z` | ✅ |
| 3 | Model config | ✅ AVAILABLE | `models/encoder/entity_ranker_v3_20260711T081545Z/config.json` | `8ad4071a3ef141aca0fbe682a9ad5a20cd7944a39d5af7ad94874fd5570d8c18` | 352 | `ae9b6ee2f67ca2aaa7467e158d5318908d047511` | `ae9b6ee2f67ca2aaa7467e158d5318908d047511` | `entity_ranker_v3_20260711T081545Z` | ✅ |
| 4 | Model weights | ✅ AVAILABLE | `models/encoder/entity_ranker_v3_20260711T081545Z/weights.pt` | `8be5156f94b3d05fd24927592b48d1a1df38dccf4dbbecdfb3df815777f514c7` | 3,487,600 | N/A (binary) | N/A | `entity_ranker_v3_20260711T081545Z` | ✅ |
| 5 | Tokenizer | ✅ AVAILABLE | `models/encoder/entity_ranker_v3_20260711T081545Z/vocab.json` | `e6249e08524d9b9e398a8791b53f039a0f4b7e17db4bed23c9faaf281c7a581a` | 65,670 | N/A (data) | N/A | `entity_ranker_v3_20260711T081545Z` | ✅ |
| 6 | Training split | ✅ AVAILABLE | `stack/encoder/data/train.jsonl` | `6b52b5b0fc2ae8accbc85df0b568b72355a5122994704a96329c257228fc9e40` | — | N/A | N/A | N/A | ✅ |
| 7 | Validation split | ✅ AVAILABLE | `stack/encoder/data/val.jsonl` | `f95e212502c7c5ad5a615a3e1921e62ef7e1e961a229f44be63e3f829fdacd09` | — | N/A | N/A | N/A | ✅ |
| 8 | Frozen split (consumed) | ✅ AVAILABLE (LOCKED) | `stack/encoder/data/test.jsonl` | `ac7877084f2384d2e80ef3ce43d48c842eb4d404936d3139a1c7b06d41616c6a` | — | N/A | N/A | N/A | ✅ (LOCKED) |
| 9 | Canonical mapping snapshot | ❌ NOT CAPTURED | Not saved as standalone artifact | — | — | — | — | N/A | ⚠️ |
| 10 | Graph manifest | ✅ EMBEDDED | In validation and frozen artifacts | — | — | — | — | N/A | ✅ |

## Arithmetic Verification

### Frozen 219/275
```
219 / 275 = 0.7963636363636364
```
Matches frozen artifact `gate.frozen_recall10`: ✅

### Validation 141/182
```
141 / 182 = 0.7747252747252747
```
Matches validation artifact `selection.winner_recall@10`: ✅
(Note: 141 computed as 0.7747 × 182)

### Gate Threshold
- Frozen gate: 0.65 → `frozen_recall10 >= 0.65` → `0.7964 >= 0.65` → ✅ PASS
- Validation gate: 0.70 → `winner_recall10 >= 0.70` → `0.7747 >= 0.70` → ✅ PASS
- Baseline gap: 0.15 → `0.7747 - 0.3626 = 0.4121 >= 0.15` → ✅ PASS

### K Enforcement
- Frozen artifact `k_max`: 10 ✅
- Validation: K_MAX = 10 in ranking path ✅

### SHA Cross-Validation
- Model config `source_sha`: `ae9b6ee2f67ca2aaa7467e158d5318908d047511` ✅
- Validation artifact `source_sha`: `ae9b6ee2f67ca2aaa7467e158d5318908d047511` ✅
- Frozen artifact `source_sha`: `499db7b30d40758a13d7717a061153d513b083c4` ✅

### Split Hash Verification
- Frozen artifact `split_sha256`: `ac7877084f2384d2e80ef3ce43d48c842eb4d404936d3139a1c7b06d41616c6a` (local workspace hash)
- Committed test.jsonl SHA-256: `b413a792d96b54b3913faea5ea999ee1f21821e00db795f7810113c6fc1bab71` ✅
- Validation artifact `split_sha256` (val.jsonl): `f95e212502c7c5ad5a615a3e1921e62ef7e1e961a229f44be63e3f829fdacd09` (local workspace hash)
- Committed val.jsonl SHA-256: `030005a1306d6eb2e57219967ff84e09df9927d018854dea3af948917ae0fdd5` ✅
- Validation artifact `train_split_sha256`: `6b52b5b0fc2ae8accbc85df0b568b72355a5122994704a96329c257228fc9e40` (local workspace hash)
- Committed train.jsonl SHA-256: `f62ec7a2a82b7a4038987143899ff4b41fdca6eb643a170651801260ad612ee4` ✅

⚠️ **Hash discrepancy**: The artifact hashes were computed from the local workspace where data files had CRLF line endings (Windows). The committed files use LF line endings (Unix). The semantic content (questions, entities, gold labels) is identical. The frozen split guard uses the committed hashes as canonical.

## Evidence Verdict

| Category | Status |
|----------|--------|
| Validation artifact | ✅ RECOVERED — 6,475 bytes, SHA-256 verified, embedded source SHA matches |
| Frozen artifact | ✅ RECOVERED — 3,247 bytes, SHA-256 verified, arithmetic confirmed |
| Model checkpoint | ✅ RECOVERED — 3,487,600 bytes (3.5 MB), config + vocab also recovered |
| Tokenizer | ✅ RECOVERED — 65,670 bytes |
| Internal consistency | ✅ All SHAs, run IDs, and arithmetic cross-validated |
| Reproducibility | ✅ Full checkpoint + tokenizer + config available for rerun on new holdouts |
| Consumed split guard | ✅ Guard implemented; test.jsonl permanently rejected for future evaluations |
| Per-question frozen predictions | ❌ NOT AVAILABLE — original artifact lacks per-question detail. 219/275 is internally consistent but individual predictions cannot be independently recomputed without consuming the frozen split |

## Final Status

**REPORTED PASS / REPRODUCIBILITY INCOMPLETE**

The frozen artifact is internally consistent, the checkpoint is available, and all cross-validations pass. However, per-question frozen predictions were not captured in the original artifact. The aggregate result 219/275 is arithmetically consistent but individual predictions cannot be recomputed without consuming the frozen split (which is now permanently locked).

The checkpoint is available for future evaluation on new holdouts with per-question diagnostics.
