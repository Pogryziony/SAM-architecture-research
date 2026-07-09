# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure

Status: **GATES FAILED** — see details below.

## Gate Results

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| entity_accuracy | 0.0% | >= 65% | FAIL |
| resolution_rate | 100.0% | >= 100% (no regression) | PASS |
| paraphrase_drop | 0.0 pp | < 10 pp | PASS |
| intent_accuracy | 38.3% | >= 85% | FAIL |
| RSS delta | 6.6 MB | <= 150 MB | PASS |
| inference p50 | 32.7 ms | <= 50 ms | PASS |

## Per-Head Metrics

- **Entity precision** (encoder-only): 0.0%
- **Entity resolution rate** (encoder-only): 96.7%
- **Combined entity_accuracy**: 100.0%
- **Combined resolution_rate**: 100.0%
- **Intent accuracy**: 38.3%
- **Paraphrase drop**: 0.0 pp
- **Inference p50**: 32.7 ms
- **RSS delta**: 6.6 MB
- **Parameters**: 555,017

## Failure Hypothesis

The encoder was trained on only 375 questions (with augmentation to 1181) covering just 21 unique entity types. The training data is insufficient to learn robust entity representations that generalize to the full test set. The model overfits to surface-level lexical patterns and struggles with paraphrased inputs.

Key issues:
1. Limited entity diversity (21 unique entities in training) prevents learning semantic entity representations.
2. The word-level embedding lacks subword information, making the model brittle to morphological variation.
3. The small model capacity (166K params) may be insufficient for the multi-task learning objective.

## 20 Worst Cases

### Case 1: q141
**Question**: What research question did the compact PKM retrieval experiment investigate?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Experiment_0.6_—_Full_Validation, Step_3:_Relation_Extraction, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**Missed**: (none)
**Extra**: Step_3:_Relation_Extraction, Experiment_0_Results_(poc_Validation), Experiment_0.6_—_Full_Validation, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 2: q142
**Question**: Which research phase does the compact PKM retrieval experiment belong to?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Changes_Made_For_Experiment_0.3, Experiment_0.6_—_Full_Validation, Retrieval_Diagnostics
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), Changes_Made_For_Experiment_0.3, Retrieval_Diagnostics
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 3: q143
**Question**: What problem or limitation from previous experiments did the compact PKM retrieval experiment address?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Step_3:_Relation_Extraction, Integration_Step
**Missed**: (none)
**Extra**: Step_3:_Relation_Extraction, Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Integration_Step
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 4: q144
**Question**: What was the significance of the compact PKM retrieval experiment for the overall SAM research arc?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.6_—_Full_Validation, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 5: q145
**Question**: If the compact PKM retrieval experiment had failed, what would have been the consequence?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Experiment_0_Results_(poc_Validation), Experiment_0.6_—_Full_Validation, Controlled_Random_Distractors
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Controlled_Random_Distractors, Experiment_0_Results_(poc_Validation)
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 6: q146
**Question**: What experiment directly builds on the findings of the compact PKM retrieval experiment?
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.6_—_Full_Validation
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**GT intent**: , **Pred intent**: multi_hop ✗
**Encoder entities**: []

### Case 7: q147
**Question**: Summarize the compact PKM retrieval experiment in one sentence.
**GT entities**: Exp_0_2_CompactPKM
**Resolved**: Exp_0_2_CompactPKM, Experiment_0_Results_(poc_Validation), Changes_Made_For_Experiment_0.3, Retrieval_Diagnostics, Slot_Selector_(sam/model/slot_Selector.py)
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Changes_Made_For_Experiment_0.3, Retrieval_Diagnostics
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 8: q148
**Question**: What was the main finding of the PKM candidate generation experiment?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), summary_the, Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, summary_the
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 9: q149
**Question**: What research question did the PKM candidate generation experiment investigate?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 10: q150
**Question**: Which research phase does the PKM candidate generation experiment belong to?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 11: q151
**Question**: What problem or limitation from previous experiments did the PKM candidate generation experiment address?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Sam_Does_Not_Collapse_With_One_Distractor
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Sam_Does_Not_Collapse_With_One_Distractor
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 12: q152
**Question**: What was the significance of the PKM candidate generation experiment for the overall SAM research arc?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**Missed**: (none)
**Extra**: Bug_3:_Evaluation_Used_Wrong_Checkpoints_For_Sam_Modes, Experiment_0_Results_(poc_Validation), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 13: q153
**Question**: If the PKM candidate generation experiment had failed, what would have been the consequence?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Experiment_0.3
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Experiment_0.13a_—_Controlled_Noisy_Memory_Tolerance, Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 14: q154
**Question**: What experiment directly builds on the findings of the PKM candidate generation experiment?
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training, Experiment_0.3, Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, Experiment_0.12_—_Candidate_Selection_And_Memory_Use_Training
**GT intent**: , **Pred intent**: multi_hop ✗
**Encoder entities**: []

### Case 15: q155
**Question**: Summarize the PKM candidate generation experiment in one sentence.
**GT entities**: Exp_0_3_PKM_Candidates
**Resolved**: Exp_0_3_PKM_Candidates, Experiment_0_Results_(poc_Validation), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3, Slot_Selector_(sam/model/slot_Selector.py)
**Missed**: (none)
**Extra**: Experiment_0_Results_(poc_Validation), Slot_Selector_(sam/model/slot_Selector.py), Sam_Lm_Experiment_0.3_—_Pkm_Retrieval_Diagnosis_And_Repair, Experiment_0.3
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 16: q156
**Question**: What was the main finding of the dense dataset experiment?
**GT entities**: Exp_0_5_DenseDataset
**Resolved**: Exp_0_5_DenseDataset, Experiment_0_Results_(poc_Validation), Experiment_0_—_Pipeline_Diagnosis, Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), Gate_1_(rec@8_≥_80%):_Passed, Experiment_0_—_Pipeline_Diagnosis
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 17: q157
**Question**: What research question did the dense dataset experiment investigate?
**GT entities**: Exp_0_5_DenseDataset
**Resolved**: Exp_0_5_DenseDataset, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.6_—_Full_Validation, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.5_—_Dense_Dataset_Fix
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0.5_—_Dense_Dataset_Fix, Gate_1_(rec@8_≥_80%):_Passed, Experiment_0_—_Pipeline_Diagnosis
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 18: q158
**Question**: Which research phase does the dense dataset experiment belong to?
**GT entities**: Exp_0_5_DenseDataset
**Resolved**: Exp_0_5_DenseDataset, Experiment_0.6_—_Full_Validation, Original_Gates_(experiments_0.0–0.6), Decision_Gates, Gate_1
**Missed**: (none)
**Extra**: Decision_Gates, Original_Gates_(experiments_0.0–0.6), Experiment_0.6_—_Full_Validation, Gate_1
**GT intent**: , **Pred intent**: factual_lookup ✗
**Encoder entities**: []

### Case 19: q159
**Question**: What problem or limitation from previous experiments did the dense dataset experiment address?
**GT entities**: Exp_0_5_DenseDataset
**Resolved**: Exp_0_5_DenseDataset, Required_Slots, Experiment_0_—_Pipeline_Diagnosis, Experiment_0.6_—_Full_Validation, Randomly_Sampled_From_Live_Slots
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Required_Slots, Randomly_Sampled_From_Live_Slots, Experiment_0_—_Pipeline_Diagnosis
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []

### Case 20: q160
**Question**: What was the significance of the dense dataset experiment for the overall SAM research arc?
**GT entities**: Exp_0_5_DenseDataset
**Resolved**: Exp_0_5_DenseDataset, Experiment_0.6_—_Full_Validation, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants, Experiment_0_Results_(poc_Validation), Gate_1_(rec@8_≥_80%):_Passed
**Missed**: (none)
**Extra**: Experiment_0.6_—_Full_Validation, Experiment_0_Results_(poc_Validation), Gate_1_(rec@8_≥_80%):_Passed, Experiment_0.7_0.9_—_Retrieval_Interface_And_Selection_Variants
**GT intent**: , **Pred intent**: diagnostic ✗
**Encoder entities**: []
