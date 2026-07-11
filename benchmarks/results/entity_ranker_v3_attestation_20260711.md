# Entity Ranker V3 — Post-Run Attestation

**POST-RUN ATTESTATION** — created 2026-07-11 after the original evaluation.
Does not replace or modify original artifacts.

**Weights policy**: Model weights (`.pt` files) are tracked in git at commit
`b6d832d`. The `.gitignore` pattern `models/**/*.pt` prevents accidental
future commits. SHA-256 hashes are recorded here for audit.

## Artifact Inventory

| # | Artifact | Committed LF SHA-256 | Committed Size | Original Local CRLF SHA-256 | Original Local Size | Semantic SHA-256 | Source SHA |
|---|----------|---------------------|---------------|---------------------------|--------------------|-----------------|------------|
| 1 | Validation JSON | `8f0542b73aaa4f0c528c02f3ee190005713ac707880045b497eb43987c34846d` | 6,295 B | `a34d5111...` | 6,475 B | N/A | `ae9b6ee2...` |
| 2 | Frozen JSON | `df2c51c66ca5e26b641c8dc3c3da355f8292d32525b2d22f791d5356a16e1538` | 3,151 B | `60d88397...` | 3,247 B | N/A | `499db7b...` |
| 3 | Model config | `c14051c80af3c72ee8a2d6c7915da69bd6de978c25f9a5cf9cd5866dd2ce6d95` | 343 B | `8ad4071a...` | 352 B | N/A | `ae9b6ee2...` |
| 4 | Weights | `8be5156f94b3d05fd24927592b48d1a1df38dccf4dbbecdfb3df815777f514c7` | 3,487,600 B | N/A | N/A | N/A | N/A |
| 5 | Vocabulary | `e69d55991720318689487832d1b38e16b60164b90e3b792d9adef7aa2ace8364` | 62,575 B | `e6249e08...` | 65,670 B | N/A | N/A |
| 6 | train.jsonl | `f62ec7a2a82b7a4038987143899ff4b41fdca6eb643a170651801260ad612ee4` | 157,068 B | `6b52b5b0...` | — | `87038d74...` | N/A |
| 7 | val.jsonl | `030005a1306d6eb2e57219967ff84e09df9927d018854dea3af948917ae0fdd5` | 64,774 B | `f95e2125...` | — | `82f859e5...` | N/A |
| 8 | test.jsonl (CONSUMED) | `b413a792d96b54b3913faea5ea999ee1f21821e00db795f7810113c6fc1bab71` | 79,591 B | `ac787708...` | — | `37c6fe6e...` | N/A |
| 9 | Canonical mapping | ❌ NOT CAPTURED | — | — | — | — | — |
| 10 | Model manifest | ✅ COMMITTED in `manifest.json` | — | — | — | — | — |

All "Committed LF" hashes computed from `git show HEAD:<path> | sha256sum`.
All "Original Local CRLF" hashes from the Windows workspace where the experiment was run.
Semantic hashes are line-ending insensitive canonical JSONL digests.

## Arithmetic Verification

- `219 / 275 = 0.7963636363636364` — matches frozen artifact `gate.frozen_recall10` ✅
- `141 / 182 ≈ 0.7747` — matches validation artifact `selection.winner_recall@10` ✅

## Gate Verification

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Frozen recall@10 | 79.64% | ≥65% | ✅ PASS |
| Validation recall@10 | 77.47% | ≥70% | ✅ PASS |
| Baseline gap | 41.21 pp | ≥15 pp | ✅ PASS |
| K enforcement | 10 | ≤10 | ✅ |
| Denominator (val) | 150q / 182 gold | exact | ✅ |
| Denominator (frozen) | 225q / 275 gold | exact | ✅ |

## Cross-Validation

| Check | Result |
|-------|--------|
| Model config source_sha = validation source_sha | ✅ `ae9b6ee2...` |
| Frozen source_sha = `499db7b...` | ✅ |
| Consumed split guard rejects LF hash | ✅ |
| Consumed split guard rejects CRLF hash | ✅ |
| Consumed split guard rejects semantic hash | ✅ |
| Candidate-pool invariant (min ≥ 59) | ✅ |

## Evidence Verdict

| Category | Status |
|----------|--------|
| Validation artifact | ✅ RECOVERED — 6,295 B committed |
| Frozen artifact | ✅ RECOVERED — 3,151 B committed |
| Model checkpoint | ✅ RECOVERED — 3.5 MB, SHA-256 verified |
| Tokenizer | ✅ RECOVERED — 62,575 B committed |
| Per-question frozen predictions | ❌ NOT AVAILABLE |
| Canonical mapping snapshot | ❌ NOT CAPTURED |

## Final Status

**REPORTED PASS / REPRODUCIBILITY INCOMPLETE**

All recoverable evidence is committed and SHA-256 verified. The aggregate
frozen result (219/275 = 79.64%) is arithmetically consistent. Per-question
frozen predictions and canonical mapping snapshot were not captured in the
original artifact. The consumed frozen split is permanently locked.
