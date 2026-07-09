# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — see details below.

## Gate Results

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_accuracy | 0.0% | >= 65% | FAIL |
| resolution_rate | 100.0% | >= 100% (no regression) | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 82.2% | >= 85% | FAIL |
| RSS delta | 6.8 MB | <= 150 MB | PASS |
| inference p50 | 0.6 ms | <= 50 ms | PASS |

## Per-Head Metrics

- **Entity precision** (encoder-only): 0.0%
- **Entity resolution rate** (encoder-only): 0.0%
- **Combined entity_accuracy**: 18.8%
- **Combined resolution_rate**: 100.0%
- **Intent accuracy**: 82.2%
- **Paraphrase drop**: 0.0 pp
- **Inference p50**: 0.6 ms
- **RSS delta**: 6.8 MB
- **Parameters**: 555,017

## Failure Hypothesis

The encoder was trained on only 375 questions (with augmentation to 1181) covering just 21 unique entity types. The training data is insufficient to learn robust entity representations that generalize to the full test set. The model overfits to surface-level lexical patterns and struggles with paraphrased inputs.

Key issues:
1. Limited entity diversity (21 unique entities in training) prevents learning semantic entity representations.
2. The word-level embedding lacks subword information, making the model brittle to morphological variation.
3. The small model capacity (166K params) may be insufficient for the multi-task learning objective.

## 20 Worst Cases

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
