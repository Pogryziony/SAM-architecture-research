# STAGE1B_NEGATIVE.md — Associative Encoder v2 Gate Failure

**Date**: 2026-07-10  
**Status**: **STAGE 1B GATES FAILED** — program STOPPED per immutable protocol.

The metrics in this document are the historical pre-R1 artifact. The current frozen-split reference is `benchmarks/results/stage1b_honest_20260710_102235Z.json`; the post-fix diagnostic rerun is `stage1b_honest_20260710_105503Z.json`. Historical result files are preserved.  
**Pre-registered in**: EXPERIMENT_SAM_NEXUS_STACK.md §Stage 1B

---

## Gate Results (Test Split: 225 questions)

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_accuracy | 18.8% | >= 65% | **FAIL** |
| resolution_rate | 100.0% | >= 100% (no regression) | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 82.2% | >= 85% | **FAIL** |
| RSS delta | 6.8 MB | <= 150 MB | PASS |
| inference p50 | 0.6 ms | <= 50 ms | PASS |

**2 of 6 gates FAIL**. The program stops here.

---

## Per-Head Metrics

| Metric | Value |
|--------|-------|
| Combined entity_accuracy | 18.8% (lexical baseline: 18.8%, no improvement) |
| Entity re-ranker precision | 11.5% (when enabled; disabled in final eval) |
| Entity re-ranker recall | ~98% (near-perfect — predicts almost all candidates) |
| Resolution rate | 100.0% (unchanged) |
| Intent accuracy (rule + model) | 82.2% (up from 65.3% in Stage 1) |
| Rule coverage | ~43% of test questions |
| Rule accuracy | ~96% on matched cases |
| Model intent accuracy | ~70% on unmatched cases |
| RSS delta | 6.8 MB (well within budget) |
| Inference p50 | 0.6 ms (well within budget) |
| Parameters | 555,017 |
| Training epochs | 30 (early stopped at intent plateau) |

---

## What Worked

### 1. Rule-First Intent Classification

The rule-based intent classifier is the **clear success** of Stage 1b. Intent accuracy improved from 65.3% (Stage 1) to 82.2% (+16.9 pp). The rules cover 43% of the test set at 96% accuracy. The remaining 57% is handled by the encoder model at ~70% accuracy. This validates the hypothesis that symbolic rules can handle the templated patterns in the QA dataset, leaving the model to handle genuinely ambiguous cases.

### 2. Char N-Gram Tokenizer and Sequential Component

The char tri/penta-gram hashing and pseudo-sequential GRU work correctly at the implementation level: feature dimension is reasonable (3,517), training converges, inference is fast (0.6 ms p50). The architecture composes with the rule classifier without conflicts.

### 3. Resource Budget

RSS delta (6.8 MB) and inference latency (0.6 ms) are well within the 150 MB / 50 ms gates. The CPU-only training constraint is maintained throughout.

---

## What Failed and Why

### Failure 1: Entity Accuracy (18.8% vs 65%)

**Root cause**: The entity re-ranker cannot bridge the gap from lexical entity spotting (18.8% accuracy) to the gate threshold (65%). There are three compounding factors:

1. **Lexical baseline is too weak**. The fuzzy substring match in the NEXUS parser resolves entities at only 18.8% precision on the test split. The entities "Decision_PivotToNEXUS" and "Concept_ArchitectureWorks" appear frequently in GT but are never found by substring matching because their names don't appear in question text.

2. **Re-ranker training class imbalance**. With 1–3 positive entities out of 20 candidates per question (5–15% positive ratio), weighted BCE (pos_weight=15) pushes all entity scores upward. This gives ~98% recall but only ~11.5% precision. The re-ranker cannot learn to discriminate candidates because the signal from 1–3 positives is overwhelmed by 17–19 negatives.

3. **Entity pool mismatch**. The model was trained on only 21 entity types (the entities appearing in the 375-question training split). The graph has 366 nodes. The re-ranker cannot generalize to unseen entity types — it overfits to the 21 training entities.

**Why the hypothesis fails**: The entity re-ranker was designed to score top-K candidates from the lexical index. But (a) the lexical index provides poor recall on semantic entities (e.g., concepts, decisions), and (b) with 21 entity types and 1:19 positive ratio, no amount of loss weighting can produce a discriminative re-ranker. The re-ranker hypothesis is valid for the chain-set retriever design (where candidates are pre-filtered to relevant ones), but the NEXUS lexical index doesn't provide a meaningful candidate set to re-rank.

### Failure 2: Intent Accuracy (82.2% vs 85%)

**Root cause**: While rules improved intent dramatically, the remaining 2.8 pp gap is at the limit of what the model can achieve on 375 training examples. The 57% of questions not matched by rules are the genuinely ambiguous cases (e.g., "What is the difference between controlled distractors and realistic distractors?" — labeled comparative; "What is the role of the verifier?" — labeled factual; "How does the SAM gate work?" — labeled factual). These cases require semantic understanding that 375 training examples cannot provide.

**Why the hypothesis partially holds**: Rule-first intent was the right architectural decision (+16.9 pp improvement). But the model's capacity ceiling on the residual cases is ~70%, limited by training data size. Closing the gap would require either (a) more training data, (b) a larger model, or (c) more comprehensive rules. None of these are available within the Stage 1b constraints.

---

## Failure Hypothesis (Final)

The core hypothesis of Stage 1b was that four architectural changes (rule-first intent, char n-grams, re-ranker, focal loss) would solve the Stage 1 failures. The evidence:

1. **Rule-first intent**: **Confirmed** — improved intent from 65.3% to 82.2%. But insufficient alone to clear 85%.
2. **Char n-grams + sequential component**: **Unclear** — the model's intent accuracy on unmatched cases (~70%) is only marginally better than Stage 1 (~65%). The architecture works but the training data is the binding constraint.
3. **Entity re-ranker**: **Falsified** — the re-ranker cannot produce discriminative scores with 1:19 class imbalance on 21 entity types. The design assumes a pre-filtered candidate set from the lexical index, but the lexical index doesn't provide meaningful candidates for semantic/concept entities.
4. **Focal loss / class weights**: **Insufficient** — focal loss with class weights helps but cannot overcome the fundamental 1:19 imbalance for entity scoring.

**The negative is about the hypothesis**: The entity re-ranker design is structurally incompatible with the NEXUS lexical entity spotter. The re-ranker requires a candidate set with high recall (most GT entities present) and moderate precision (not all 366 graph nodes). The NEXUS lexical index provides candidates at 18.8% recall — most GT entities are missing from the candidate set, so re-ranking cannot help.

---

## Implications for the SAM+NEXUS Stack

The experiment protocol states: "If Stage 1b also fails: the negative is about the hypothesis, not the implementation, and STOP is final."

The SAM-as-encoder layer (Stage 1) cannot meet the entity accuracy gate with the current entity spotting infrastructure. This does not invalidate the NEXUS graph traversal approach — the lexical entity spotter works at 18.8% precision but 100% resolution rate (every question gets at least one entity). The remaining entity resolution gap would need:

1. **Improved entity spotting** — embedding-based semantic matching (already present via NodeEmbeddingIndex but not scored in these metrics) or LLM-based entity extraction.
2. **Entity expansion** — graph traversal from spotted entities to related entities could compensate for missed GT entities.
3. **Ablation of the entity accuracy gate** — the 65% metric may be inappropriately high for a system where entity spotting is supplementary to graph traversal.

The intent classification improvement (82.2%) demonstrates that the rule-first symbolic approach is viable and reduces model dependency for structured QA datasets.

---

## Appendix: 20 Worst Entity Cases

All cases share the same pattern: GT entities are "Decision_PivotToNEXUS" or concept-type entities that never appear verbatim in question text. The lexical index resolves experiment-type entities (e.g., "Exp_0_6_Validation") from text fragments but misses abstract entities entirely.

| Case | Question | GT Entities | Resolved |
|------|----------|-------------|----------|
| q538 | What is the significance of the oracle memory experiment achieving 99.87% accuracy? | Decision_PivotToNEXUS | Exp_0_6_Validation, Exp_0_12_Selection, ... |
| q539 | What is the significance of the chain-set BCE retriever achieving 100% all_required@32? | Decision_PivotToNEXUS | Concept_ChainRetrieval, Exp_0_11_ChainRetrieval, ... |
| q541 | What is the significance of the noise tolerance experiment showing 91.6% at +8 distractors? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, ... |
| q543 | What is the significance of the dense dataset improving retrieval from 6.9% to 99.0% Rec@8? | Decision_PivotToNEXUS | Exp_0_5_DenseDataset, Exp_0_12_Selection, ... |
| q544 | What is the significance of the pivot from SAM to NEXUS? | Decision_PivotToNEXUS | Exp_0_12_Selection, Exp_0_11_ChainRetrieval, ... |
| q545 | What is the significance of the NEXUS CPU-first design principle? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, ... |
| q546 | What is the significance of the NEXUS verifier being rule-based? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, ... |
| q548 | What is the significance of the multi-positive BCE loss vs InfoNCE? | Decision_PivotToNEXUS | Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, ... |
| q549 | What is the significance of the 3-hop collapse at +16 distractors? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Concept_NoiseTolerance, ... |
| q579–583 | What was the goal/key challenge/breakthrough/surprise/lesson of the 'Selection & Noise' phase? | Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory | Exp_0_13A_NoisyMemory, ... (misses 2/3 GT) |
| q584–588 | What if SAM had been tested on real-world data / chain-set discovered earlier / entity extraction is bottleneck? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Exp_0_7_ExternalText, ... |
| q591–597 | How many Experiment nodes / Which experiment node has most edges / What is the longest dependency chain? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, ... |
| q607 | Where would you find the SAM experiment reports? | Decision_PivotToNEXUS | Exp_0_13A_NoisyMemory, Exp_0_11_ChainRetrieval, ... |

**Pattern**: In 15/20 worst cases, "Decision_PivotToNEXUS" is the only GT entity and never appears in resolved entities. The lexical index cannot match decision-type nodes because they have no text-surface overlap with questions. This is a structural limitation of substring-based entity spotting, not a problem the encoder can fix.

---

## Program Status: **STOPPED**

Per EXPERIMENT_SAM_NEXUS_STACK.md §Stage 1B Verdict Rule: "If Stage 1b also fails: the negative is about the hypothesis, not the implementation, and STOP is final."

The experiment is stopped. All artifacts are preserved in this commit. The program can be restarted only with a new pre-registered hypothesis that addresses the entity spotting limitation.

### Case 1: q552
**Question**: Compare the all_required@K at K=8, 16, 32 for chain-set BCE across different SAM configurations: K=8: 81.03%, K=16: 96.53%, K=32: 100.00%. What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_11_ChainRetrieval, Concept_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_12_Selection, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: Exp_0_12_Selection, Concept_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_11_ChainRetrieval
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: []

### Case 2: q538
**Question**: What is the significance of the oracle memory experiment achieving 99.87% accuracy?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_6_Validation, Exp_0_12_Selection, Exp_0_2_CompactPKM, Exp_0_13A_NoisyMemory, Exp_0_6_Validation_oracle_text_memory
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_2_CompactPKM, Exp_0_12_Selection, Exp_0_6_Validation_oracle_text_memory, Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 3: q539
**Question**: What is the significance of the chain-set BCE retriever achieving 100% all_required@32?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Concept_ChainRetrieval, Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_12_Selection, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Concept_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 4: q541
**Question**: What is the significance of the noise tolerance experiment showing 91.6% at +8 distractors?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+1, Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+0
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_13A_NoisyMemory_noise_+1, Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+0, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 5: q543
**Question**: What is the significance of the dense dataset improving retrieval from 6.9% to 99.0% Rec@8?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_5_DenseDataset, Exp_0_12_Selection, Exp_0_Diagnosis, Exp_0_2_CompactPKM, Exp_0_6_Validation_retrieval_dual_encoder
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_2_CompactPKM, Exp_0_12_Selection, Exp_0_Diagnosis, Exp_0_6_Validation_retrieval_dual_encoder, Exp_0_5_DenseDataset
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 6: q548
**Question**: What is the significance of the multi-positive BCE loss vs InfoNCE for chain retrieval?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_12_Selection, Concept_ChainRetrieval
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Concept_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 7: q549
**Question**: What is the significance of the 3-hop collapse at +16 distractors?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_11_ChainRetrieval, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory_noise_+16, Concept_NoiseTolerance
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_13A_NoisyMemory, Concept_NoiseTolerance, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 8: q555
**Question**: How would you add a new experiment result to the NEXUS graph?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, Exp_0_3_PKM_Candidates, Exp_0_7_ExternalText, Exp_0_11_ChainRetrieval
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_7_ExternalText, Exp_0_3_PKM_Candidates, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 9: q579
**Question**: What was the goal of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+2
**Missed**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection
**Extra**: Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 10: q580
**Question**: What was the key challenge of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+2
**Missed**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection
**Extra**: Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 11: q581
**Question**: What was the breakthrough moment of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+2
**Missed**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection
**Extra**: Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 12: q582
**Question**: What was the biggest surprise of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+2
**Missed**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection
**Extra**: Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 13: q583
**Question**: What was the lesson for NEXUS of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+2
**Missed**: Exp_0_13B_RealisticDistractors, Exp_0_12_Selection
**Extra**: Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 14: q584
**Question**: What if SAM had been tested on real-world data from the start?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_7_ExternalText, Exp_0_11_ChainRetrieval, Exp_0_12_Selection, Exp_0_5_DenseDataset
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Exp_0_13A_NoisyMemory, Exp_0_7_ExternalText, Exp_0_5_DenseDataset, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 15: q585
**Question**: What if the chain-set BCE retriever had been discovered at experiment 0.6 instead of 0.11?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Concept_ChainRetrieval, Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Decision_PivotToNEXUS
**Extra**: Concept_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_6_Validation, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 16: q588
**Question**: What if entity extraction turns out to be the bottleneck for NEXUS?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, Exp_0_3_PKM_Candidates, Exp_0_12_Selection, Exp_0_11_ChainRetrieval
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_3_PKM_Candidates, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 17: q591
**Question**: How many Experiment nodes are in the NEXUS graph?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_10_RequiredSet, Exp_0_Diagnosis, Exp_0_13B_RealisticDistractors, Exp_0_3_PKM_Candidates
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_13B_RealisticDistractors, Exp_0_10_RequiredSet, Exp_0_3_PKM_Candidates
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: []

### Case 18: q596
**Question**: What is the longest experiment dependency chain in the NEXUS graph?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_12_Selection, Exp_0_13A_NoisyMemory, Exp_0_Diagnosis
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: []

### Case 19: q597
**Question**: Which experiment node has the most incoming edges?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_Diagnosis, Exp_0_3_PKM_Candidates, Exp_0_7_ExternalText, Exp_0_8_Aggregation
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_3_PKM_Candidates
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: []

### Case 20: q607
**Question**: Where would you find the SAM experiment reports?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_13A_NoisyMemory, Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_Diagnosis, Exp_0_3_PKM_Candidates
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_Diagnosis, Exp_0_13A_NoisyMemory, Exp_0_10_RequiredSet, Exp_0_3_PKM_Candidates, Exp_0_11_ChainRetrieval
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: []

---

## Post-Mortem Addendum — 2026-07-10

### What happened after the STOP

1. **Stage 1b gate evaluation FAILED** on 2026-07-10 at commit 55af1ce: intent_accuracy 82.2% (threshold 85%), entity_accuracy 18.8% (threshold 65%). This file was erroneously written as STAGE1_NEGATIVE.md.

2. **Eval set was switched** from the frozen 225-question test split to a curated 60 subset. This produced passing numbers (entity 100%, intent 100%) at commit dcfe780.

3. **This file was deleted** when STAGE1_NEGATIVE.md was overwritten by the pass-declaring eval_gates.py output.

4. **PASS was declared** (commit dcfe780) based on the curated-60 evaluation. Stages 2, 3, and 5 were built on this declaration.

### Rectification

- This file restored from commit 768b132 (original Stage 1b negative content) on 2026-07-10.
- The PASS declared at dcfe780 is **retracted** pending honest re-evaluation on the frozen test split (Phase R1 of protocol repair).
- Protocol violations documented in PROTOCOL_VIOLATIONS.md.

---

## Phase R1 — Honest Re-evaluation (2026-07-10)

**Commit**: 45a774f (frozen 225-question split, canonical label normalization)  
**Result**: **HONEST FAIL**

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| entity_accuracy | 4.4% | ≥65% | **FAIL** |
| resolution_rate | 100% | no regression | PASS |
| paraphrase_drop | 0.0pp | <10pp | PASS |
| intent_accuracy | 85.3% | ≥85% | PASS |
| RSS delta | 6.7 MB | ≤150 MB | PASS |
| inference p50 | 18.0 ms | ≤50 ms | PASS |

5/6 gates passed. Pipeline entity accuracy dominated by lexical fallback (1.3%). Encoder-only precision: 1.1%.

Per decision tree: R1 failed → R2 skipped → R3 (baseline fix) executed → program STOPPED after R3.

---

### Honest Re-evaluation (2026-07-10, metric fix)

Evaluated on frozen 225-question test split with corrected metric definitions.
The ambiguous `entity_accuracy` metric has been replaced with consistently named
`entity_precision`, `entity_recall`, `entity_f1`, and `exact_entity_accuracy`.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| entity_recall | 4.4% | ≥65% | **FAIL** |
| entity_precision | 1.1% | measured | — |
| entity_f1 | 1.7% | measured | — |
| exact_entity_accuracy | 4.4% | measured | — |
| intent_accuracy | 85.3% | ≥85% | PASS |
| resolution_rate | 100% | no regression | PASS |
| paraphrase_drop | 0.0 pp | <10 pp | PASS |
| RSS delta | 7.0 MB | ≤150 MB | PASS |
| inference p50 | 18.2 ms | ≤50 ms | PASS |

Stage 1B: **FAILED** — entity_recall at 4.4% vs 65% threshold is far below requirement.
The pre-registered 65% threshold was historically measured as recall (correct GT matches / total GT entities),
which is now consistently named `entity_recall`. The metric fix preserves the original gate semantics.

Baseline lexical path achieves entity_recall = 5.5% (entity_precision = 1.3%, entity_f1 = 2.1%).
The encoder does not improve entity resolution over the lexical baseline — it marginally worsens
recall (4.4% vs 5.5%) due to the encoder scores being near-random for unseen entity types.