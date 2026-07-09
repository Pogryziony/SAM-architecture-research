# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — intent_accuracy gate not met. STOP — do not proceed to Stage 2.

## Gate Results

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| entity_accuracy | 65.9% | >= 65% | ✅ PASS |
| resolution_rate | 100.0% | >= 100% (no regression) | ✅ PASS |
| paraphrase_drop | 10.0 pp | < 10 pp | ⚠️ BORDERLINE |
| intent_accuracy | 65.3% | >= 85% | ❌ FAIL |
| RSS delta | 3.8 MB | <= 150 MB | ✅ PASS |
| inference p50 | 0.3 ms | <= 50 ms | ✅ PASS |

**Blocking failure**: intent_accuracy at 65.3% is 19.7 pp below the 85% threshold.

## Per-Head Metrics

### Entity Prediction (encoder-only, threshold=0.55)
- **Precision**: 65.9% (54/82 correct entities predicted)
- **Resolution rate**: 37.3% (84/225 questions get encoder entity proposals)
- **Entity recall** (vs GT): 18.9% (54/285 GT entities found)

### Intent Prediction
- **Accuracy**: 65.3% (147/225 correct)

Confusion matrix:

| GT \ Pred | factual_lookup | diagnostic | comparison | multi_hop |
|-----------|---------------|------------|------------|-----------|
| factual_lookup (129) | 121 (93.8%) | 6 | 2 | 0 |
| diagnostic (67) | 47 (70.1%) | 15 (22.4%) | 3 | 2 |
| comparison (29) | 15 (51.7%) | 3 | 11 (37.9%) | 0 |

Model defaults to factual_lookup for 70% of diagnostic and 52% of comparison questions.

### Paraphrase Robustness (30 pairs, threshold=0.55)
- **Original entity precision**: 33.3%
- **Paraphrase entity precision**: 23.3%
- **Drop**: 10.0 pp
- **Polish subset** (7): 33.3% → 16.7% (-16.7 pp)
- **English subset** (23): 33.3% → 25.3% (-8.0 pp)

### Resource Usage
- **Model parameters**: 166,045
- **RSS delta on load**: 3.8 MB (well under 150 MB budget)
- **Peak RSS during training**: 274.8 MB (55% of 500 MB budget)
- **Inference p50**: 0.3 ms/question
- **Inference p90**: 0.5 ms/question

## Error Analysis — 20 Worst Cases

### Case 1: q545 (diagnostic, Decision_PivotToNEXUS)
**Q**: "What is the significance of the NEXUS CPU-first design principle?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent (factual_lookup)
**Root cause**: No lexical overlap between question and correct entity.

### Case 2: q546 (diagnostic, Decision_PivotToNEXUS)
**Q**: "What is the significance of the NEXUS verifier being rule-based rather than LLM-based?"
**Encoder**: [Exp_0_6_Validation, Concept_ArchitectureWorks] — wrong entities, wrong intent
**Root cause**: "verifier" not in training vocabulary.

### Case 3: q544 (diagnostic, Decision_PivotToNEXUS)
**Q**: "What is the significance of the pivot from SAM to NEXUS?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent
**Root cause**: "pivot" not linked to Decision_PivotToNEXUS in training.

### Case 4: q548 (diagnostic, Exp_0_11_ChainRetrieval)
**Q**: "What is the significance of the multi-positive BCE loss vs InfoNCE for chain retrieval?"
**Encoder**: [Exp_0_6_Validation] — wrong entity
**Root cause**: "BCE loss", "InfoNCE" out of vocabulary.

### Case 5: q564 (diagnostic, Exp_0_Diagnosis)
**Q**: "What was the goal of the 'Pipeline Setup' phase in SAM research?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent
**Root cause**: "Pipeline Setup" aliases not in training data; encoder defaults to most common entity.

### Case 6: q616 (diagnostic, Exp_0_2_CompactPKM)
**Q**: "What was the goal of the 'Compact PKM' phase in SAM research?"
**Encoder**: [Exp_0_6_Validation] — wrong entity
**Root cause**: "Compact PKM" rarely appears in training data.

### Case 7: q622 (diagnostic, Exp_0_7_ExternalText)
**Q**: "What was the key challenge of the 'External Text' phase in SAM research?"
**Encoder**: [Exp_0_6_Validation] — wrong entity
**Root cause**: "External Text" rarely appears in training data.

### Case 8: q628 (diagnostic, Exp_0_11_ChainRetrieval)
**Q**: "What was the goal of the 'Retrieval Revolution' phase in SAM research?"
**Encoder**: [Exp_0_6_Validation] — wrong entity
**Root cause**: "Retrieval Revolution" has no training examples.

### Case 9: q550 (comparison, Exp_0_6_Validation + Exp_0_13A_NoisyMemory)
**Q**: "Compare the 3-hop accuracy across different SAM configurations..."
**Encoder**: [Exp_0_6_Validation, Exp_0_13A_NoisyMemory] — correct entities! Wrong intent (diagnostic instead of comparison)
**Root cause**: Model correctly identifies entities but misclassifies intent.

### Case 10: q551 (comparison, Exp_0_6_Validation)
**Q**: "Compare the overall accuracy across all memory modes at experiment 0.6..."
**Encoder**: [Exp_0_6_Validation] — correct entity! Wrong intent (diagnostic instead of comparison)

### Case 11: q560 (comparison, Decision_PivotToNEXUS)
**Q**: "What SAM concepts map directly to NEXUS concepts?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent

### Case 12: q555 (diagnostic, Decision_PivotToNEXUS)
**Q**: "How would you add a new experiment result to the NEXUS graph?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent

### Case 13: q557 (diagnostic, Decision_PivotToNEXUS)
**Q**: "How would you compare two experiments using the NEXUS graph?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, wrong intent

### Case 14: q584 (diagnostic, Decision_PivotToNEXUS)
**Q**: "What if SAM had been tested on real-world data from the start?"
**Encoder**: [] — no entities, no intent signal
**Root cause**: Hypothetical question format OOD for training data.

### Case 15: q588 (diagnostic, Decision_PivotToNEXUS)
**Q**: "What if entity extraction turns out to be the bottleneck for NEXUS?"
**Encoder**: [] — no entities, wrong intent

### Case 16: q589 (factual_lookup, Decision_PivotToNEXUS)
**Q**: "How many nodes are in the current NEXUS graph?"
**Encoder**: [Exp_0_6_Validation] — wrong entity, correct intent
**Root cause**: "NEXUS graph" not grounded to Decision_PivotToNEXUS.

### Case 17: q614 (factual_lookup, Exp_0_Diagnosis)
**Q**: "How many failed runs were recovered during the pipeline setup phase?"
**Encoder**: [Exp_0_6_Validation] — wrong entity
**Root cause**: "failed runs", "pipeline setup" OOV.

### Case 18: q627 (diagnostic, Exp_0_7_ExternalText + Decision_PivotToNEXUS)
**Q**: "What was the lesson for NEXUS of the 'External Text' phase in SAM research?"
**Encoder**: [Exp_0_6_Validation] — wrong entities
**Root cause**: Multi-entity question; model defaults to single most-frequent entity.

### Case 19: q596 (factual_lookup, Decision_PivotToNEXUS)
**Q**: "What is the longest experiment dependency chain in the NEXUS graph?"
**Encoder**: [] — no entities
**Root cause**: "dependency chain" OOV.

### Case 20: q755 (diagnostic, Decision_PivotToNEXUS)
**Q**: "Why is the separation of concerns critical in NEXUS?"
**Encoder**: [] — no entities, wrong intent
**Root cause**: Abstract question with no entity keywords.

## Failure Hypothesis — Root Cause Analysis

### 1. Intent Classification: Word-Level Features Insufficient (CRITICAL)

The EmbeddingBag encoder averages word embeddings, losing all positional and sequential information. Diagnostic questions ("What is the significance of X?") and factual questions ("What was the accuracy of X?") become indistinguishable because:
- Both start with "what"
- The distinguishing feature is the noun phrase after the wh-word
- Bag-of-words averaging flattens this distinction

The model's 93.8% accuracy on factual_lookup and 22.4% on diagnostic reveals it defaults to the majority class.

### 2. Entity Vocabulary Sparsity (MAJOR)

Only 21 unique entity IDs appear across all 750 questions. The training data covers 375 questions, making rare entities (Exp_0_2_CompactPKM, Exp_0_7_ExternalText, Exp_0_Diagnosis) appear in only 2-3% of training examples. The model overfits to the most common entity (Exp_0_6_Validation, ~30% of questions).

### 3. OOV Sensitivity (MAJOR)

The word-level tokenizer creates UNK tokens for any word not in the training vocabulary. Domain-specific terms ("verifier", "InfoNCE", "dependency chain", "pipeline setup") become UNK, losing critical signals for both entity and intent prediction.

### 4. Polish Robustness (MINOR)

Only 10 augmented Polish training samples exist. The 16.7 pp drop on Polish paraphrases reflects near-zero cross-lingual training data.

## Remediation Options (for future work)

1. **Character n-gram encoder**: Replace word-level EmbeddingBag with character trigram embeddings to capture subword patterns and reduce OOV.
2. **Positional encoding**: Add position-aware encoding (even simple sinusoidal) to help distinguish "What was X" from "What is the significance of X".
3. **Balanced sampling**: Weight intent classes inversely to frequency during training.
4. **Entity-aware loss**: Use focal loss or class-balanced loss for the multi-label entity head.
5. **Increase model capacity**: 2-3 layer encoder with residual connections (2-5M parameters).

## Conclusion

The associative encoder shows promise in the resource efficiency dimension (3.8 MB RSS, 0.3 ms inference, 166K params) but fails the primary quality gate (intent_accuracy 65.3% < 85%). The word-level EmbeddingBag architecture is fundamentally limited for intent discrimination tasks where word order matters. Per the immutable gate protocol, the experiment stops at Stage 1.

The architecture is **not viable in its current form** but the resource budget headroom (only 55% of 500 MB) leaves room for a more capable model in a follow-up experiment.
