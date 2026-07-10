# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — see details below.

## Gate Results

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_recall (pipeline) | 50.5% | >= 65% | FAIL |
| entity_precision (pipeline) | 12.4% | measured | — |
| entity_f1 (pipeline) | 19.9% | measured | — |
| resolution_rate | 100.0% | >= baseline | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 85.3% | >= 85% | PASS |
| RSS delta | 6.6 MB | <= 150 MB | PASS |
| inference p50 | 26.1 ms | <= 50 ms | PASS |

## Per-Head Metrics

- **Entity precision** (pipeline): 12.4%
- **Entity recall** (pipeline): 50.5%
- **Entity F1** (pipeline): 19.9%
- **Exact entity accuracy** (all GT matched): 48.0%
- **Encoder-only precision**: 12.4%
- **Encoder-only recall**: 50.5%
- **Encoder-only F1**: 19.9%
- **Resolution rate**: 100.0%
- **Intent accuracy** (canonical labels): 85.3%
- **Paraphrase drop**: 0.0 pp
- **Inference p50**: 26.1 ms
- **RSS delta**: 6.6 MB
- **Parameters**: 555,017

## Per-Intent-Class Breakdown

| Intent | Count | Intent Acc | Enc Precision |
|--------|-------|------------|---------------|
| factual_lookup | 129 | 87.6% | 11.9% |
| diagnostic | 67 | 80.6% | 12.2% |
| comparison | 29 | 86.2% | 14.5% |

## Failure Hypothesis

The encoder was trained on only 375 questions (with augmentation to 1181) covering just 21 unique entity types. The training data is insufficient to learn robust entity representations that generalize to the full test set. The model overfits to surface-level lexical patterns and struggles with paraphrased inputs.

Key issues:
1. Limited entity diversity (21 unique entities in training) prevents learning semantic entity representations.
2. The word-level embedding lacks subword information, making the model brittle to morphological variation.
3. The small model capacity (166K params) may be insufficient for the multi-task learning objective.

## 20 Worst Cases

### Case 1: q564
**Question**: What was the goal of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_5_DenseDataset, Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_Diagnosis, Exp_0_11_ChainRetrieval, Exp_0_6_Validation, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset
**Extra**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_chain_set_bce', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 2: q566
**Question**: What was the breakthrough moment of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_5_DenseDataset, Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_Diagnosis, Exp_0_11_ChainRetrieval, Exp_0_6_Validation, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset
**Extra**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_chain_set_bce', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 3: q567
**Question**: What was the biggest surprise of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_5_DenseDataset, Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_Diagnosis, Exp_0_11_ChainRetrieval, Exp_0_6_Validation, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset
**Extra**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_chain_set_bce', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 4: q568
**Question**: What was the lesson for NEXUS of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_5_DenseDataset, Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_Diagnosis, Exp_0_11_ChainRetrieval, Exp_0_6_Validation, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset
**Extra**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_chain_set_bce', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 5: q569
**Question**: What was the goal of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Resolved**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Extra**: Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Exp_0_11_ChainRetrieval, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_6_Validation', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop']

### Case 6: q570
**Question**: What was the key challenge of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Resolved**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_6_Validation_core_only, Exp_0_2_CompactPKM, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Extra**: Exp_0_6_Validation_core_only, Exp_0_11_ChainRetrieval, Exp_0_2_CompactPKM, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_2_CompactPKM', 'Exp_0_6_Validation', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall']

### Case 7: q571
**Question**: What was the breakthrough moment of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Resolved**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Extra**: Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Exp_0_11_ChainRetrieval, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_6_Validation', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop']

### Case 8: q572
**Question**: What was the biggest surprise of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Resolved**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Extra**: Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Exp_0_11_ChainRetrieval, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_6_Validation', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop']

### Case 9: q573
**Question**: What was the lesson for NEXUS of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Resolved**: Exp_0_6_Validation, Exp_0_11_ChainRetrieval, Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Exp_0_9_OracleFilter, Exp_0_8_Aggregation, Exp_0_7_ExternalText
**Extra**: Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Exp_0_11_ChainRetrieval, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_6_Validation', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop']

### Case 10: q532
**Question**: What is relation extraction?
**GT entities**: Exp_0_6_Validation
**Resolved**: Step_3:_Relation_Extraction, relation_extraction, Decision_PivotToNEXUS, Step_2:_Entity_Extraction, entity_extraction
**Missed**: Exp_0_6_Validation
**Extra**: Decision_PivotToNEXUS, Step_3:_Relation_Extraction, Step_2:_Entity_Extraction, entity_extraction, relation_extraction
**GT intent**: factual_lookup, **Pred intent**: multi_hop ✗
**Encoder entities**: ['Decision_PivotToNEXUS', 'Step_2:_Entity_Extraction', 'Step_3:_Relation_Extraction', 'entity_extraction', 'relation_extraction']

### Case 11: q533
**Question**: What is mmap?
**GT entities**: Exp_0_6_Validation
**Resolved**: Decision_PivotToNEXUS, Mmap, current_status_what, entity_location_map, repository_map
**Missed**: Exp_0_6_Validation
**Extra**: Decision_PivotToNEXUS, entity_location_map, Mmap, current_status_what, repository_map
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: ['Decision_PivotToNEXUS', 'Mmap', 'current_status_what', 'entity_location_map', 'repository_map']

### Case 12: q534
**Question**: What is KuzuDB?
**GT entities**: Exp_0_6_Validation
**Resolved**: Decision_PivotToNEXUS, Kuzudb, current_status_what, roadmap_what, datahub
**Missed**: Exp_0_6_Validation
**Extra**: Decision_PivotToNEXUS, datahub, Kuzudb, current_status_what, roadmap_what
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: ['Decision_PivotToNEXUS', 'Kuzudb', 'current_status_what', 'datahub', 'roadmap_what']

### Case 13: q536
**Question**: What is a hard negative?
**GT entities**: Exp_0_6_Validation
**Resolved**: Hard_Negative, Hard_Negative_Training, Decision_PivotToNEXUS, the_problem_is_distractor_quality,_train_on_hard_negatives, current_status_what
**Missed**: Exp_0_6_Validation
**Extra**: Decision_PivotToNEXUS, the_problem_is_distractor_quality,_train_on_hard_negatives, Hard_Negative_Training, Hard_Negative, current_status_what
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: ['Decision_PivotToNEXUS', 'Hard_Negative', 'Hard_Negative_Training', 'current_status_what', 'the_problem_is_distractor_quality,_train_on_hard_negatives']

### Case 14: q537
**Question**: What is a controlled distractor?
**GT entities**: Exp_0_6_Validation
**Resolved**: Exp_0_13A_NoisyMemory, Decision_PivotToNEXUS, Controlled_Noisy_Memory_Path_(0.13a/0.13b), Concept_NoiseTolerance, controlled_noisy_memory_path
**Missed**: Exp_0_6_Validation
**Extra**: Decision_PivotToNEXUS, controlled_noisy_memory_path, Exp_0_13A_NoisyMemory, Concept_NoiseTolerance, Controlled_Noisy_Memory_Path_(0.13a/0.13b)
**GT intent**: factual_lookup, **Pred intent**: factual_lookup ✓
**Encoder entities**: ['Concept_NoiseTolerance', 'Controlled_Noisy_Memory_Path_(0.13a/0.13b)', 'Decision_PivotToNEXUS', 'Exp_0_13A_NoisyMemory', 'controlled_noisy_memory_path']

### Case 15: q538
**Question**: What is the significance of the oracle memory experiment achieving 99.87% accuracy?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_6_Validation, Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_6_Validation, Exp_0_5_DenseDataset
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_2_CompactPKM', 'Exp_0_3_PKM_Candidates', 'Exp_0_5_DenseDataset', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 16: q539
**Question**: What is the significance of the chain-set BCE retriever achieving 100% all_required@32?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_11_ChainRetrieval, Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval_chain_set_bce, Metric_Exp_0_11_ChainRetrieval_chain_set_bce_total_wall_s, Exp_0_11_ChainRetrieval_sam_chain_aware
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval, Exp_0_11_ChainRetrieval_chain_set_bce, Exp_0_11_ChainRetrieval_sam_chain_aware, Metric_Exp_0_11_ChainRetrieval_chain_set_bce_total_wall_s
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_chain_set_bce', 'Exp_0_11_ChainRetrieval_sam_chain_aware', 'Metric_Exp_0_11_ChainRetrieval_chain_set_bce_total_wall_s']

### Case 17: q540
**Question**: What is the significance of the selector achieving only 50% precision?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_12_Selection, Exp_0_2_CompactPKM, Exp_0_6_Validation_core_only, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_12_Selection, Metric_Exp_0_6_Validation_core_only_val_accuracy_overall, Exp_0_6_Validation_core_only, Exp_0_2_CompactPKM, Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_12_Selection', 'Exp_0_2_CompactPKM', 'Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop']

### Case 18: q541
**Question**: What is the significance of the noise tolerance experiment showing 91.6% at +8 distractors?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_Diagnosis, Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_2_CompactPKM, Exp_0_6_Validation
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_6_Validation, Exp_0_5_DenseDataset
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_2_CompactPKM', 'Exp_0_3_PKM_Candidates', 'Exp_0_5_DenseDataset', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 19: q543
**Question**: What is the significance of the dense dataset improving retrieval from 6.9% to 99.0% Rec@8?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM, Exp_0_10_RequiredSet, Exp_0_6_Validation
**Missed**: Decision_PivotToNEXUS
**Extra**: Exp_0_10_RequiredSet, Exp_0_2_CompactPKM, Exp_0_Diagnosis, Exp_0_6_Validation, Exp_0_5_DenseDataset
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_10_RequiredSet', 'Exp_0_2_CompactPKM', 'Exp_0_5_DenseDataset', 'Exp_0_6_Validation', 'Exp_0_Diagnosis']

### Case 20: q544
**Question**: What is the significance of the pivot from SAM to NEXUS?
**GT entities**: Decision_PivotToNEXUS
**Resolved**: Exp_0_11_ChainRetrieval, Exp_0_6_Validation, Exp_0_11_ChainRetrieval_sam_chain_aware, Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_single_hop, Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_overall
**Missed**: Decision_PivotToNEXUS
**Extra**: Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_overall, Exp_0_11_ChainRetrieval, Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_single_hop, Exp_0_6_Validation, Exp_0_11_ChainRetrieval_sam_chain_aware
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_11_ChainRetrieval', 'Exp_0_11_ChainRetrieval_sam_chain_aware', 'Exp_0_6_Validation', 'Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_overall', 'Metric_Exp_0_11_ChainRetrieval_sam_chain_aware_val_accuracy_single_hop']
