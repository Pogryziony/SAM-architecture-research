# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — see details below.

## Gate Results

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_recall (pipeline) | 4.4% | >= 65% | FAIL |
| entity_precision (pipeline) | 1.1% | measured | — |
| entity_f1 (pipeline) | 1.7% | measured | — |
| resolution_rate | 100.0% | >= baseline | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 85.3% | >= 85% | PASS |
| RSS delta | 6.8 MB | <= 150 MB | PASS |
| inference p50 | 25.4 ms | <= 50 ms | PASS |

## Per-Head Metrics

- **Entity precision** (pipeline): 1.1%
- **Entity recall** (pipeline): 4.4%
- **Entity F1** (pipeline): 1.7%
- **Exact entity accuracy** (all GT matched): 4.4%
- **Encoder-only precision**: 0.2%
- **Encoder-only recall**: 9.5%
- **Encoder-only F1**: 0.5%
- **Resolution rate**: 100.0%
- **Intent accuracy** (canonical labels): 85.3%
- **Paraphrase drop**: 0.0 pp
- **Inference p50**: 25.4 ms
- **RSS delta**: 6.8 MB
- **Parameters**: 555,017

## Per-Intent-Class Breakdown

| Intent | Count | Intent Acc | Enc Precision |
|--------|-------|------------|---------------|
| factual_lookup | 129 | 87.6% | 0.2% |
| diagnostic | 67 | 80.6% | 0.3% |
| comparison | 29 | 86.2% | 0.2% |

## Failure Hypothesis

The encoder was trained on only 375 questions (with augmentation to 1181) covering just 21 unique entity types. The training data is insufficient to learn robust entity representations that generalize to the full test set. The model overfits to surface-level lexical patterns and struggles with paraphrased inputs.

Key issues:
1. Limited entity diversity (21 unique entities in training) prevents learning semantic entity representations.
2. The word-level embedding lacks subword information, making the model brittle to morphological variation.
3. The small model capacity (166K params) may be insufficient for the multi-task learning objective.

## 20 Worst Cases

### Case 1: q564
**Question**: What was the goal of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM
**Resolved**: [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction, Retriever_To_Sam_Integration
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_Diagnosis, Exp_0_5_DenseDataset
**Extra**: Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), Retriever_To_Sam_Integration, Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only']

### Case 2: q565
**Question**: What was the key challenge of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM
**Resolved**: [repository_Map](repository_Map.md), Step_3:_Relation_Extraction, [troubleshooting](troubleshooting.md), Experiment_0_—_Pipeline_Diagnosis, [roadmap](roadmap.md)
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_Diagnosis, Exp_0_5_DenseDataset
**Extra**: Experiment_0_—_Pipeline_Diagnosis, [roadmap](roadmap.md), [troubleshooting](troubleshooting.md), Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', '—_Sam_Validation']

### Case 3: q566
**Question**: What was the breakthrough moment of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM
**Resolved**: [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction, Retriever_To_Sam_Integration
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_Diagnosis, Exp_0_5_DenseDataset
**Extra**: Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), Retriever_To_Sam_Integration, Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_random_memory', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 4: q567
**Question**: What was the biggest surprise of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM
**Resolved**: [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Experiment_0_—_Pipeline_Diagnosis, Step_3:_Relation_Extraction, Retriever_To_Sam_Integration
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_Diagnosis, Exp_0_5_DenseDataset
**Extra**: Experiment_0_—_Pipeline_Diagnosis, [troubleshooting](troubleshooting.md), Retriever_To_Sam_Integration, Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 5: q568
**Question**: What was the lesson for NEXUS of the 'Pipeline Setup' phase in SAM research?
**GT entities**: Exp_0_3_PKM_Candidates, Exp_0_5_DenseDataset, Exp_0_Diagnosis, Exp_0_2_CompactPKM
**Resolved**: Step_3:_Relation_Extraction, Nodes_(entities)_+_Edges_(relationships), [troubleshooting](troubleshooting.md), [repository_Map](repository_Map.md), Document_Chunks_+_Embeddings
**Missed**: Exp_0_2_CompactPKM, Exp_0_3_PKM_Candidates, Exp_0_Diagnosis, Exp_0_5_DenseDataset
**Extra**: Nodes_(entities)_+_Edges_(relationships), [troubleshooting](troubleshooting.md), Document_Chunks_+_Embeddings, Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 6: q569
**Question**: What was the goal of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Resolved**: Experiment_0.6_—_Full_Validation, End_To_End_Pipeline_Validation, Gate_1_(rec@8_≥_80%):_Passed, Revalidate_Learned_Selector, Hard_Negative_Training
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Hard_Negative_Training, Revalidate_Learned_Selector, Experiment_0.6_—_Full_Validation, End_To_End_Pipeline_Validation
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: []

### Case 7: q570
**Question**: What was the key challenge of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Resolved**: Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_Results_(poc_Validation), Experiment_0.6_—_Full_Validation, [repository_Map](repository_Map.md), Gate_1_(rec@8_≥_80%):_Passed
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 8: q571
**Question**: What was the breakthrough moment of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Resolved**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0.6_—_Full_Validation, Experiment_0.5_—_Dense_Dataset_Fix
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', 'Metric_Exp_0_6_Validation_core_only_num_live_slots', 'Validation', 'full_validation', '—_Sam_Validation']

### Case 9: q572
**Question**: What was the biggest surprise of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Resolved**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Iterative_Querying, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0.6_—_Full_Validation, Iterative_Querying, Experiment_0.5_—_Dense_Dataset_Fix
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 10: q573
**Question**: What was the lesson for NEXUS of the 'Core Validation' phase in SAM research?
**GT entities**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Resolved**: Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Step_3:_Relation_Extraction, Experiment_0.5_—_Dense_Dataset_Fix, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**Missed**: Exp_0_6_Validation, Exp_0_8_Aggregation, Exp_0_7_ExternalText, Exp_0_9_OracleFilter
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0.6_—_Full_Validation, Step_3:_Relation_Extraction, Experiment_0.5_—_Dense_Dataset_Fix
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Validation']

### Case 11: q579
**Question**: What was the goal of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Resolved**: Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', '—_Sam_Validation']

### Case 12: q580
**Question**: What was the key challenge of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Resolved**: Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_—_Pipeline_Diagnosis, [repository_Map](repository_Map.md), "the_Latent_Memory_Path_Is_Fragile", Step_3:_Relation_Extraction
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, "the_Latent_Memory_Path_Is_Fragile", Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_random_memory', '—_Sam_Validation']

### Case 13: q581
**Question**: What was the breakthrough moment of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Resolved**: Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+16', 'Exp_0_13A_NoisyMemory_noise_+2', 'Exp_0_13A_NoisyMemory_noise_+4', 'Exp_0_13A_NoisyMemory_noise_+8', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_random_memory', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_num_live_slots', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 14: q582
**Question**: What was the biggest surprise of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Resolved**: Experiment_0_—_Pipeline_Diagnosis, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+16', 'Exp_0_13A_NoisyMemory_noise_+2', 'Exp_0_13A_NoisyMemory_noise_+4', 'Exp_0_13A_NoisyMemory_noise_+8', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_random_memory', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_num_live_slots', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 15: q583
**Question**: What was the lesson for NEXUS of the 'Selection & Noise' phase in SAM research?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Resolved**: Step_3:_Relation_Extraction, Integration_Step, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_—_Pipeline_Diagnosis, Rag_Vs_Graph_Sam_(nexus)_—_Detailed_Comparison
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_12_Selection, Exp_0_13B_RealisticDistractors
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Integration_Step, Rag_Vs_Graph_Sam_(nexus)_—_Detailed_Comparison, Step_3:_Relation_Extraction
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+16', 'Exp_0_13A_NoisyMemory_noise_+2', 'Exp_0_13A_NoisyMemory_noise_+4', 'Exp_0_13A_NoisyMemory_noise_+8', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_random_memory', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 16: q552
**Question**: Compare the all_required@K at K=8, 16, 32 for chain-set BCE across different SAM configurations: K=8: 81.03%, K=16: 96.53%, K=32: 100.00%. What does this tell us?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Resolved**: Chain_Set_Bce, Integration_Step, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, "the_Latent_Memory_Path_Is_Fragile", Current_Status
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Extra**: Chain_Set_Bce, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Current_Status
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_random_memory', 'Exp_0_6_Validation_retrieved_memory', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'no_validation_ever_ran_|_per_epoch_eval_in_all_training_scripts_|', '—_Sam_Validation']

### Case 17: q553
**Question**: Compare the all_required@K at K=8, 32, 64 for dual encoder across different SAM configurations: K=8: 26.34%, K=32: 26.84%, K=64: 27.29%. What does this tell us?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Resolved**: Concept_RetrievalMismatch, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Final_Recommendation
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Extra**: Gate_1_(rec@8_≥_80%):_Passed, Final_Recommendation, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.6_—_Full_Validation, Concept_RetrievalMismatch
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['29%_Of_Val_Slots_Unseen_In_Training', 'Exp_0_5_DenseDataset', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_pkm_retrieval_early', 'Exp_0_6_Validation_random_memory', 'Exp_0_6_Validation_retrieval_dual_encoder', 'Exp_0_6_Validation_retrieved_memory', 'Sam_Lm_Experiment_0.6_—_Final_Validation_Report', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'Train_Dual_Encoder()', 'Train_Sam_Full_Compact', 'no_validation_ever_ran_|_per_epoch_eval_in_all_training_scripts_|', 'root_cause_the', 'sam_tiny_dense.yaml', '—_Sam_Validation']

### Case 18: q554
**Question**: Compare the 3-hop accuracy under noise at +1, +2, +4, +8, +16 distractors across different SAM configurations: +1: 99.50%, +2: 98.17%, +4: 95.00%, +8: 79.33%, +16: 39.00%. What does this tell us?
**GT entities**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Resolved**: Sam_Does_Not_Collapse_With_One_Distractor, Sam_Tolerates_Mild_Noise_Very_Well, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Current_Status
**Missed**: Exp_0_13A_NoisyMemory, Exp_0_6_Validation
**Extra**: Sam_Tolerates_Mild_Noise_Very_Well, Sam_Does_Not_Collapse_With_One_Distractor, "the_Latent_Memory_Path_Is_Fragile", Integration_Step, Current_Status
**GT intent**: comparison, **Pred intent**: comparison ✓
**Encoder entities**: ['99.9%', 'Exp_0_13A_NoisyMemory_noise_+0', 'Exp_0_13A_NoisyMemory_noise_+1', 'Exp_0_13A_NoisyMemory_noise_+16', 'Exp_0_13A_NoisyMemory_noise_+2', 'Exp_0_13A_NoisyMemory_noise_+4', 'Exp_0_13A_NoisyMemory_noise_+8', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_random_memory', 'Exp_0_6_Validation_retrieved_memory', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_param_count', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_num_live_slots', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_total_wall_s', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_32', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_overall', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_two_hop', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', '—_Sam_Validation']

### Case 19: q574
**Question**: What was the goal of the 'Retrieval Revolution' phase in SAM research?
**GT entities**: Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**Resolved**: Step_3:_Relation_Extraction, Retriever_To_Sam_Integration, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Experiment_0_—_Pipeline_Diagnosis
**Missed**: Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**Extra**: Experiment_0_—_Pipeline_Diagnosis, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Retriever_To_Sam_Integration, Step_3:_Relation_Extraction
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['29%_Of_Val_Slots_Unseen_In_Training', 'Exp_0_12_Selection_equal_budget', 'Exp_0_5_DenseDataset', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_random_memory', 'Exp_0_6_Validation_retrieval_dual_encoder', 'Exp_0_6_Validation_retrieved_memory', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_three_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_8', 'Metric_Exp_0_6_Validation_core_only_best_val_loss', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_dense_baseline_best_val_loss', 'Metric_Exp_0_6_Validation_dense_baseline_early_best_val_loss', 'Metric_Exp_0_6_Validation_dense_baseline_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_dense_baseline_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_dense_baseline_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_dense_openbook_best_val_loss', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_overall', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_oracle_memory_best_val_loss', 'Metric_Exp_0_6_Validation_oracle_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_oracle_memory_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_oracle_memory_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_oracle_text_memory_best_val_loss', 'Metric_Exp_0_6_Validation_random_memory_best_val_loss', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_two_hop', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'no_validation_ever_ran_|_per_epoch_eval_in_all_training_scripts_|', '—_Sam_Validation']

### Case 20: q575
**Question**: What was the key challenge of the 'Retrieval Revolution' phase in SAM research?
**GT entities**: Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**Resolved**: Step_3:_Relation_Extraction, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, [repository_Map](repository_Map.md), [roadmap](roadmap.md), relation_extraction
**Missed**: Exp_0_10_RequiredSet, Exp_0_11_ChainRetrieval
**Extra**: [roadmap](roadmap.md), relation_extraction, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Step_3:_Relation_Extraction, [repository_Map](repository_Map.md)
**GT intent**: diagnostic, **Pred intent**: diagnostic ✓
**Encoder entities**: ['Exp_0_12_Selection_equal_budget', 'Exp_0_6_Validation_core_only', 'Exp_0_6_Validation_dense_baseline', 'Exp_0_6_Validation_dense_baseline_early', 'Exp_0_6_Validation_dense_openbook', 'Exp_0_6_Validation_oracle_memory', 'Exp_0_6_Validation_oracle_text_memory', 'Exp_0_6_Validation_random_memory', 'Exp_0_6_Validation_retrieved_memory', 'Key_Design_Decisions', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_single_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_three_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_accuracy_two_hop', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+0_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+16_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+1_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+2_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+4_val_recall_at_8', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_best_val_loss', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_1', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_32', 'Metric_Exp_0_13A_NoisyMemory_noise_+8_val_recall_at_8', 'Metric_Exp_0_6_Validation_core_only_best_val_loss', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_overall', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_core_only_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_dense_baseline_best_val_loss', 'Metric_Exp_0_6_Validation_dense_baseline_early_best_val_loss', 'Metric_Exp_0_6_Validation_dense_baseline_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_dense_baseline_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_dense_openbook_best_val_loss', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_overall', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_dense_openbook_val_accuracy_two_hop', 'Metric_Exp_0_6_Validation_oracle_memory_best_val_loss', 'Metric_Exp_0_6_Validation_oracle_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_oracle_text_memory_best_val_loss', 'Metric_Exp_0_6_Validation_random_memory_best_val_loss', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_single_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_three_hop', 'Metric_Exp_0_6_Validation_random_memory_val_accuracy_two_hop', 'Start_The_Poc_At_8–16m_Slots', 'The_Situation_Before_0.13a', 'key_design_decisions_1', 'no_validation_ever_ran_|_per_epoch_eval_in_all_training_scripts_|', '—_Sam_Validation']
