# SynthesizingModel Relevance Audit

**Sample size**: 14 stratified by question_type
**Source**: `relevance_sample.json`
**Judge**: heuristic checklist (4-point rubric)

---

## Per-Case Analysis

### Case 1: q001 (factual)

**Question**: What was the overall accuracy of the SAM oracle memory experiment?

**Answer**:
> Regarding "overall accuracy of the SAM oracle memory experiment", the evidence reveals the following:
> 
> Oracle memory: 99.87%. Retrieved memory = core_only (68.74%). Query projection mismatch identified.
> 
> Experiment 0.6 â€” Full Validation â€” oracle_text_memory.
> 
> Dense dataset with 21.8 ex/slot -> 99.0% Rec@8. Gate 1 PASSED.
> 
> External text query bypasses hidden-state projection. Tested topK sweep.
> 
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.
> 
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.
> 
> Additional evidence suggests the following:
>   - Experiment 0.6 â€” Full Validation supports Experiment 0.6 â€” Full Validation â€” oracle_text_memory.
>   - Experiment 0.6 â€” Full Validation depends on Experiment 0.5 â€” Dense Dataset Fix.
>   - Experiment 0.6 â€” Full Validation is a dependency of Experiment 0.7 â€” External Text Query.
> 
> 
> These findings are drawn from 10 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 2: q002 (factual)

**Question**: What accuracy did the SAM core-only baseline achieve?

**Answer**:
> Regarding "What accuracy did the SAM core-only baseline achieve?", the evidence reveals the following:
> 
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.
> 
> Additional evidence suggests the following:
>   - Experiment 0.6 â€” Full Validation supports Experiment 0.6 â€” Full Validation â€” core_only.
>   - Experiment 0.11 â€” Chain-Aware Retrieval depends on Experiment 0.10 â€” Required-Set Retrieval Diagnostics.
>   - Experiment 0.11 â€” Chain-Aware Retrieval is a dependency of Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
> 
> 
> These findings are drawn from 10 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 3: q003 (factual)

**Question**: What was the breakthrough result of the chain-set BCE retriever?

**Answer**:
> Regarding "breakthrough result of the chain-set BCE retriever", the evidence reveals the following:
> 
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.
> 
> Chain-set BCE retriever achieves 100% all_required@32 â€” retrieval is solved.
> 
> All_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem.
> 
> Experiment 0.11 â€” Chain-Aware Retrieval â€” chain_set_bce.
> 
> Additional evidence suggests the following:
>   - Experiment 0.11 â€” Chain-Aware Retrieval validates Concept_ChainRetrieval.
>   - Experiment 0.11 â€” Chain-Aware Retrieval depends on Experiment 0.10 â€” Required-Set Retrieval Diagnostics.
>   - Experiment 0.11 â€” Chain-Aware Retrieval is a dependency of Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
> 
> 
> These findings are drawn from 9 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 4: q004 (factual)

**Question**: What was the precision of the learned slot selector in experiment 0.12?

**Answer**:
> Regarding "precision of the learned slot selector in experiment 0.12", the evidence reveals the following:
> 
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.
> 
> Learned selector precision is the critical bottleneck (50% precision, 96.6% recall).
> 
> Experiment 0.12 â€” Candidate Selection & Memory-Use Training â€” learned_selector.
> 
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.
> 
> Additional evidence suggests the following:
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training validates Concept_SelectorBottleneck.
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training supports Experiment 0.12 â€” Candidate Selection & Memory-Use Training â€” learned_selector.
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training depends on Experiment 0.11 â€” Chain-Aware Retrieval.
> 
> 
> These findings are drawn from 7 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 5: q005 (factual)

**Question**: How many random distractors can SAM tolerate while maintaining over 90% accuracy?

**Answer**:
> Regarding "How many random distractors can SAM tolerate while maintaining over 90% accuracy", the evidence reveals the following:
> 
> The evidence clearly indicates that SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck. Additionally, overall Accuracy. Additionally, 1-Hop Accuracy. Additionally, 2-Hop Accuracy. Furthermore, 3-Hop Accuracy.
> 
> Additional evidence suggests the following:
>   - Experiment 0.13A â€” Controlled Noisy Memory Tolerance supports Experiment 0.13A â€” Controlled Noisy Memory Tolerance â€” noise_+0.
>   - Experiment 0.13A â€” Controlled Noisy Memory Tolerance â€” noise_+0 supports Overall Accuracy.
>   - Experiment 0.13A â€” Controlled Noisy Memory Tolerance supports Experiment 0.13A â€” Controlled Noisy Memory Tolerance â€” noise_+0.
> 
> 
> These findings are drawn from 4 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 6: q015 (comparative)

**Question**: What was the 2-hop all_required@32 for the dual encoder vs chain-set BCE retriever?

**Answer**:
> Concept_ChainRetrieval achieved 100%, while Exp_0_11_ChainRetrieval achieved 100%. The difference is 0.0 percentage points in favor of Exp_0_11_ChainRetrieval.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 7: q019 (comparative)

**Question**: What was the accuracy of oracle text memory compared to oracle latent memory?

**Answer**:
> Regarding "accuracy of oracle text memory compared to oracle latent memory", the evidence reveals the following:
> 
> 16K PKM: 25.8% Rec@8. Oracle text: 100% â€” core CAN use memory.
> 
> Oracle memory: 99.87%. Retrieved memory = core_only (68.74%). Query projection mismatch identified.
> 
> External text query bypasses hidden-state projection. Tested topK sweep.
> 
> Proves SAM core CAN use external memory â€” 100% accuracy.
> 
> Dual encoder query projection mismatch prevents SAM from using retrieved memory.
> 
> Additional evidence suggests the following:
>   - Experiment 0.2 â€” Compact PKM Retrieval is a dependency of Experiment 0.3 â€” PKM Candidate Generation.
>   - Experiment 0.3 â€” PKM Candidate Generation is a dependency of Experiment 0.5 â€” Dense Dataset Fix.
>   - Experiment 0.5 â€” Dense Dataset Fix is a dependency of Experiment 0.6 â€” Full Validation.
> 
> 
> These findings are drawn from 14 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 8: q028 (comparative)

**Question**: How does SAM differ from RAG?

**Answer**:
> The evidence forms the following chain: Exp_0_13A_NoisyMemory depends on Exp_0_12_Selection, which in turn depends on Exp_0_12_Selection, which in turn depends on Exp_0_12_Selection, which in turn depends on Exp_0_12_Selection, which depends on Exp_0_12_Selection.

**Heuristic verdict**: `no` (score: 1/4)

**Reasons**:
- [FAIL] Focus entity present
- [FAIL] Asked metric present
- [PASS] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 9: q021 (diagnostic)

**Question**: Why did the dual encoder retriever fail to retrieve complete multi-hop chains?

**Answer**:
> Regarding "Why did the dual encoder retriever fail to retrieve complete multi-hop chains?", the evidence reveals the following:
> 
> The evidence clearly indicates that All_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem. Furthermore, sAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.
> 
> Additional evidence suggests the following:
>   - Experiment 0.10 â€” Required-Set Retrieval Diagnostics is a dependency of Experiment 0.11 â€” Chain-Aware Retrieval.
>   - Experiment 0.11 â€” Chain-Aware Retrieval is a dependency of Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training is a dependency of Experiment 0.13A â€” Controlled Noisy Memory Tolerance.
> 
> 
> These findings are drawn from 17 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 10: q022 (diagnostic)

**Question**: How did the dense dataset fix enable successful retrieval?

**Answer**:
> Regarding "dense dataset fix enable successful retrieval", the evidence reveals the following:
> 
> Found and fixed 3 critical bugs. Retrieval is the bottleneck.
> 
> 16K PKM: 25.8% Rec@8. Oracle text: 100% â€” core CAN use memory.
> 
> Dense dataset with 21.8 ex/slot -> 99.0% Rec@8. Gate 1 PASSED.
> 
> Additional evidence suggests the following:
>   - Experiment 0 â€” Pipeline Diagnosis is a dependency of Experiment 0.2 â€” Compact PKM Retrieval.
>   - Experiment 0.2 â€” Compact PKM Retrieval is a dependency of Experiment 0.3 â€” PKM Candidate Generation.
>   - Experiment 0.3 â€” PKM Candidate Generation is a dependency of Experiment 0.5 â€” Dense Dataset Fix.
> 
> 
> These findings are drawn from 22 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 11: q024 (diagnostic)

**Question**: Why does chain-set retrieval achieve 100% all_required@32 but SAM still equals core_only?

**Answer**:
> Regarding "Why does chain-set retrieval achieve 100% all_required@32 but SAM still equals c", the evidence reveals the following:
> 
> All_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem.
> 
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.
> 
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.
> 
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.
> 
> Testing in progress. Code implemented.
> 
> Additional evidence suggests the following:
>   - Experiment 0.10 â€” Required-Set Retrieval Diagnostics is a dependency of Experiment 0.11 â€” Chain-Aware Retrieval.
>   - Experiment 0.11 â€” Chain-Aware Retrieval is a dependency of Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training is a dependency of Experiment 0.13A â€” Controlled Noisy Memory Tolerance.
> 
> 
> These findings are drawn from 17 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 12: q023 (multi-hop)

**Question**: What chain of experiments led from the discovery of the retriever problem to solving it?

**Answer**:
> The evidence forms the following chain: Exp_0_12_Selection depends on Exp_0_11_ChainRetrieval, which depends on Exp_0_9_OracleFilter.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 13: q025 (multi-hop)

**Question**: What is the relationship between the selector bottleneck and the pivot to NEXUS?

**Answer**:
> Regarding "relationship between the selector bottleneck and the pivot to NEXUS", the evidence reveals the following:
> 
> Learned selector precision is the critical bottleneck (50% precision, 96.6% recall).
> 
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.
> 
> Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture.
> 
> Pivot from SAM (latent-vector associative memory) to NEXUS (graph-first reasoning).
> 
> Additional evidence suggests the following:
>   - Concept_SelectorBottleneck is validated by Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
>   - Concept_PivotToNEXUS is validated by Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
>   - Concept_SelectorBottleneck supports Decision_PivotToNEXUS.
> 
> 
> These findings are drawn from 5 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 14: q026 (multi-hop)

**Question**: Walk through the evolution from experiment 0.6 to the architecture pivot.

**Answer**:
> Regarding "Walk through the evolution from experiment 0.6 to the architecture pivot.", the evidence reveals the following:
> 
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.
> 
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.
> 
> Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture.
> 
> Additional evidence suggests the following:
>   - Experiment 0.12 â€” Candidate Selection & Memory-Use Training is a dependency of Experiment 0.13A â€” Controlled Noisy Memory Tolerance.
>   - Concept_PivotToNEXUS is validated by Experiment 0.12 â€” Candidate Selection & Memory-Use Training.
>   - Concept_PivotToNEXUS is validated by Experiment 0.13A â€” Controlled Noisy Memory Tolerance.
> 
> 
> These findings are drawn from 6 source document(s) in the knowledge graph.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [FAIL] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

## Aggregate Results

| Verdict | Count | % |
|---------|-------|---|
| yes | 0 | 0.0% |
| partial | 13 | 92.9% |
| no | 1 | 7.1% |

**Relevance rate**: 46.4%
  (Formula: % yes + 0.5 × % partial)

### Per Question Type

| Type | Yes | Partial | No | Rate |
|------|-----|---------|----|------|
| factual | 0 | 5 | 0 | 50.0% |
| comparative | 0 | 2 | 1 | 33.3% |
| diagnostic | 0 | 3 | 0 | 50.0% |
| multi-hop | 0 | 3 | 0 | 50.0% |

## ⚠️ Metric Caveat

The zero-LLM SynthesizingModel accuracy claim (39–44%) is **unvalidated** by this relevance audit. The key-fact-overlap metric in the verifier rewards evidence dumping even when the answer does not directly address the question. Heuristic relevance is below 70%, meaning fewer than 7 out of 10 answers are actually relevant to the asked question.

**Recommendation**: Replace or augment the accuracy metric with a relevance-gated accuracy score. A simple weighted score could be: `accuracy × relevance_rate`. This would place the true actionable SynthesizingModel accuracy at approximately 18.3% instead of the reported 39.33%.

