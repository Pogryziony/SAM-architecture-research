# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — see details below.

## Gate Results

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_recall (pipeline) | 5.8% | >= 65% | FAIL |
| entity_precision (pipeline) | 1.4% | measured | — |
| entity_f1 (pipeline) | 2.3% | measured | — |
| resolution_rate | 100.0% | >= baseline | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 85.3% | >= 85% | PASS |
| RSS delta | 6.6 MB | <= 150 MB | PASS |
| inference p50 | 25.1 ms | <= 50 ms | PASS |

## Per-Head Metrics

- **Entity precision** (pipeline): 1.4%
- **Entity recall** (pipeline): 5.8%
- **Entity F1** (pipeline): 2.3%
- **Exact entity accuracy** (all GT matched): 7.1%
- **Encoder-only precision**: 1.4%
- **Encoder-only recall**: 4.7%
- **Encoder-only F1**: 2.2%
- **Resolution rate**: 100.0%
- **Intent accuracy** (canonical labels): 85.3%
- **Paraphrase drop**: 0.0 pp
- **Inference p50**: 25.1 ms
- **RSS delta**: 6.6 MB
- **Parameters**: 555,017

## Per-Intent-Class Breakdown

| Intent | Count | Intent Acc | Enc Precision |
|--------|-------|------------|---------------|
| factual_lookup | 129 | 87.6% | 2.1% |
| diagnostic | 67 | 80.6% | 1.1% |
| comparison | 29 | 86.2% | 0.0% |

## Failure Hypothesis

The encoder was trained on only 375 questions (with augmentation to 1181) covering just 21 unique entity types. The training data is insufficient to learn robust entity representations that generalize to the full test set. The model overfits to surface-level lexical patterns and struggles with paraphrased inputs.

Key issues:
1. Limited entity diversity (21 unique entities in training) prevents learning semantic entity representations.
2. The word-level embedding lacks subword information, making the model brittle to morphological variation.
3. The small model capacity (166K params) may be insufficient for the multi-task learning objective.

## 20 Worst Cases

### Case 1: q564
**Question**: What was the goal of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Resolved**: [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction, Retriever_To_Sam_Integration
**Missed**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Extra**: [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, Retriever_To_Sam_Integration, Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only']

### Case 2: q565
**Question**: What was the key challenge of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Resolved**: —_Sam_Validation, [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, [troubleshooting](troubleshooting.md), Experiment_0_—_Pipeline_Diagnosis
**Missed**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Extra**: [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', '—_Sam_Validation']

### Case 3: q566
**Question**: What was the breakthrough moment of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Resolved**: —_Sam_Validation, [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction
**Missed**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Extra**: [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_oracle_memory', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 4: q567
**Question**: What was the biggest surprise of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Resolved**: —_Sam_Validation, [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction
**Missed**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Extra**: [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 5: q568
**Question**: What was the lesson for NEXUS of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Resolved**: —_Sam_Validation, Step_3:_Relation_Extraction, Nodes_(entities)_+_Edges_(relationships), [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md)
**Missed**: Exp_0_3_PKM_Candidates, Exp_0_2_CompactPKM, Exp_0_5_DenseDataset, Exp_0_Diagnosis
**Extra**: Step_3:_Relation_Extraction, [repository_Map](repository_Map.md), Nodes_(entities)_+_Edges_(relationships), [troubleshooting](troubleshooting.md), —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 6: q569
**Question**: What was the goal of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter, Exp_0_8_Aggregation
**Resolved**: Experiment_0.6_—_Full_Validation, End_To_End_Pipeline_Validation, Gate_1_(rec@8_≥_80%):_Passed, Revalidate_Learned_Selector, Hard_Negative_Training
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Hard_Negative_Training, End_To_End_Pipeline_Validation, Revalidate_Learned_Selector
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 7: q570
**Question**: What was the key challenge of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter, Exp_0_8_Aggregation
**Resolved**: Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_Results_(poc_Validation), Experiment_0.6_—_Full_Validation, [repository_Map](repository_Map.md), Gate_1_(rec@8_≥_80%):_Passed
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0_Results_(poc_Validation), [repository_Map](repository_Map.md), Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 8: q571
**Question**: What was the breakthrough moment of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter, Exp_0_8_Aggregation
**Resolved**: —_Sam_Validation, Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', 'Metric_Exp_0_6_Validation_core_only_num_live_slots', 'Validation']

### Case 9: q572
**Question**: What was the biggest surprise of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter, Exp_0_8_Aggregation
**Resolved**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Iterative_Querying, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Iterative_Querying
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 10: q573
**Question**: What was the lesson for NEXUS of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter, Exp_0_8_Aggregation
**Resolved**: Validation, Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Step_3:_Relation_Extraction, Experiment_0.5_—_Dense_Dataset_Fix
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Validation, Experiment_0.5_—_Dense_Dataset_Fix, Step_3:_Relation_Extraction
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 11: q579
**Question**: What was the goal of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**Missed**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Extra**: "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', '—_Sam_Validation']

### Case 12: q580
**Question**: What was the key challenge of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_—_Pipeline_Diagnosis, [repository_Map](repository_Map.md), "the_Latent_Memory_Path_Is_Fragile"
**Missed**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Extra**: [repository_Map](repository_Map.md), "the_Latent_Memory_Path_Is_Fragile", Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 13: q581
**Question**: What was the breakthrough moment of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**Missed**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Extra**: "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_num_live_slots']

### Case 14: q582
**Question**: What was the biggest surprise of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**Missed**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Extra**: "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+2', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_num_live_slots']

### Case 15: q583
**Question**: What was the lesson for NEXUS of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Step_3:_Relation_Extraction, Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_—_Pipeline_Diagnosis
**Missed**: Exp_0_12_Selection, Exp_0_13B_RealisticDistractors, Exp_0_13A_NoisyMemory
**Extra**: Step_3:_Relation_Extraction, Integration_Step, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, —_Sam_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+2', 'Exp_0_13A_NoisyMemory_noise_+4', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots']

### Case 16: q550
**Question**: Compare the 3-hop accuracy across different SAM configurations: core_only (22.00%) vs oracle_memory (100%) vs controlled noise +8 (79.33%) vs +16 (39.00%). What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+4, Exp_0_13A_NoisyMemory_noise_+1
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+1, Exp_0_13A_NoisyMemory_noise_+4
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_oracle_memory', 'Metric_Exp_0_6_Validation_core_only_best_val_loss', 'Metric_Exp_0_6_Validation_core_only_num_live_slots', 'Metric_Exp_0_6_Validation_core_only_total_wall_s']

### Case 17: q551
**Question**: Compare the overall accuracy across all memory modes at experiment 0.6 across different SAM configurations: core_only=68.74%, random_memory=68.74%, retrieved_memory=68.74%, oracle_memory=99.87%, oracle_text=100%. What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: Experiment_0.6_—_Full_Validation, Exp_0_6_Validation_oracle_text_memory, Exp_0_6_Validation_oracle_memory, Exp_0_6_Validation_random_memory, Updated_Decision_Gates_(post_Experiment_0.11)
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: Experiment_0.6_—_Full_Validation, Exp_0_6_Validation_random_memory, Updated_Decision_Gates_(post_Experiment_0.11), Exp_0_6_Validation_oracle_text_memory, Exp_0_6_Validation_oracle_memory
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Metric_Exp_0_6_Validation_core_only_best_val_loss', 'Metric_Exp_0_6_Validation_core_only_num_live_slots', 'Metric_Exp_0_6_Validation_core_only_param_count', 'Metric_Exp_0_6_Validation_core_only_total_wall_s']

### Case 18: q552
**Question**: Compare the all_required@K at K=8, 16, 32 for chain-set BCE across different SAM configurations: K=8: 81.03%, K=16: 96.53%, K=32: 100.00%. What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: —_Sam_Validation, Chain_Set_Bce, Integration_Step, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, "the_Latent_Memory_Path_Is_Fragile"
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: Chain_Set_Bce, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, —_Sam_Validation, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'no_validation_ever_ran_|_per_epoch_eval_in_all_training_scripts_|', '—_Sam_Validation']

### Case 19: q553
**Question**: Compare the all_required@K at K=8, 32, 64 for dual encoder across different SAM configurations: K=8: 26.34%, K=32: 26.84%, K=64: 27.29%. What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: sam_tiny_dense.yaml, Train_Dual_Encoder(), Train_Sam_Full_Compact, Exp_0_6_Validation_retrieval_dual_encoder, Sam_Lm_Experiment_0.6_—_Final_Validation_Report
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: sam_tiny_dense.yaml, Train_Sam_Full_Compact, Train_Dual_Encoder(), Sam_Lm_Experiment_0.6_—_Final_Validation_Report, Exp_0_6_Validation_retrieval_dual_encoder
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Exp_0_6_Validation_retrieval_dual_encoder', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'sam_tiny_dense.yaml', '—_Sam_Validation']

### Case 20: q554
**Question**: Compare the 3-hop accuracy under noise at +1, +2, +4, +8, +16 distractors across different SAM configurations: +1: 99.50%, +2: 98.17%, +4: 95.00%, +8: 79.33%, +16: 39.00%. What does this tell us?
**GT entities**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Resolved**: Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_single_hop, Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_two_hop, Exp_0_13A_NoisyMemory_noise_+16, Exp_0_13A_NoisyMemory_noise_+2, Exp_0_13A_NoisyMemory_noise_+8
**Missed**: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
**Extra**: Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_two_hop, Exp_0_13A_NoisyMemory_noise_+8, Exp_0_13A_NoisyMemory_noise_+16, Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_single_hop, Exp_0_13A_NoisyMemory_noise_+2
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_two_hop']
