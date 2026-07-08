# SynthesizingModel Relevance Audit

**Sample size**: 26 stratified by question_type
**Source**: `relevance_sample.json`
**Judge**: heuristic checklist (4-point rubric)

---

## Per-Case Analysis

### Case 1: q001 (factual)

**Question**: What was the overall accuracy of the SAM oracle memory experiment?

**Answer**:
> Oracle memory: 99.87%. Retrieved memory = core_only (68.74%). Query projection mismatch identified.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 2: q002 (factual)

**Question**: What accuracy did the SAM core-only baseline achieve?

**Answer**:
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 3: q003 (factual)

**Question**: What was the breakthrough result of the chain-set BCE retriever?

**Answer**:
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 4: q004 (factual)

**Question**: What was the precision of the learned slot selector in experiment 0.12?

**Answer**:
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 5: q005 (factual)

**Question**: How many random distractors can SAM tolerate while maintaining over 90% accuracy?

**Answer**:
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 6: q006 (factual)

**Question**: At what noise level does 3-hop reasoning collapse in SAM?

**Answer**:
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 7: q007 (factual)

**Question**: What accuracy did SAM achieve with the oracle-filter from chain candidates in experiment 0.12?

**Answer**:
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 8: q008 (factual)

**Question**: What was the all_required@64 result for the dual encoder retriever in experiment 0.10?

**Answer**:
> all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 9: q009 (factual)

**Question**: What Rec@8 did the dual encoder achieve on the dense dataset in experiment 0.5?

**Answer**:
> Dense dataset with 21.8 ex/slot -> 99.0% Rec@8. Gate 1 PASSED.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 10: q010 (factual)

**Question**: What was the key finding of the pipeline diagnosis experiment?

**Answer**:
> Found and fixed 3 critical bugs. Retrieval is the bottleneck.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 11: q011 (factual)

**Question**: How many parameters does the SAM core model have?

**Answer**:
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 12: q012 (factual)

**Question**: What is the 3-hop accuracy of SAM without memory (core_only)?

**Answer**:
> Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 13: q013 (factual)

**Question**: What dataset size was used for the dense synthetic experiments?

**Answer**:
> Dense dataset with 21.8 ex/slot -> 99.0% Rec@8. Gate 1 PASSED.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 14: q014 (factual)

**Question**: What accuracy did random memory injection achieve in SAM?

**Answer**:
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 15: q015 (comparative)

**Question**: What was the 2-hop all_required@32 for the dual encoder vs chain-set BCE retriever?

**Answer**:
> Chain-set BCE retriever achieved 100.00%, while Dual encoder achieved 27.00%.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 16: q016 (factual)

**Question**: What did experiment 0.13A prove about the SAM gate?

**Answer**:
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 17: q017 (factual)

**Question**: How many live memory slots did the SAM experiments use?

**Answer**:
> SAM tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate NOT the bottleneck.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 18: q018 (factual)

**Question**: What were the three critical bugs found in the pipeline diagnosis?

**Answer**:
> Found and fixed 3 critical bugs. Retrieval is the bottleneck.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 19: q019 (comparative)

**Question**: What was the accuracy of oracle text memory compared to oracle latent memory?

**Answer**:
> Oracle memory: 99.87%. Retrieved memory = core_only (68.74%). Query projection mismatch identified.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 20: q020 (factual)

**Question**: What accuracy does SAM achieve with exactly 1 distractor in controlled noisy memory?

**Answer**:
> Experiment 0.13A — Controlled Noisy Memory Tolerance — noise_+4.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 21: q021 (diagnostic)

**Question**: Why did the dual encoder retriever fail to retrieve complete multi-hop chains?

**Answer**:
> all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem. Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 22: q022 (diagnostic)

**Question**: How did the dense dataset fix enable successful retrieval?

**Answer**:
> Found and fixed 3 critical bugs. Retrieval is the bottleneck. all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem.

**Heuristic verdict**: `partial` (score: 3/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 23: q023 (multi-hop)

**Question**: What chain of experiments led from the discovery of the retriever problem to solving it?

**Answer**:
> Oracle memory = 99.87-100%, oracle filter = 100% — the core+memory architecture IS valid. Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture. Dual encoder query projection mismatch prevents SAM from using retrieved memory.

**Heuristic verdict**: `partial` (score: 2/4)

**Reasons**:
- [FAIL] Focus entity present
- [PASS] Asked metric present
- [FAIL] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 24: q024 (diagnostic)

**Question**: Why does chain-set retrieval achieve 100% all_required@32 but SAM still equals core_only?

**Answer**:
> all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem. Chain-set BCE: all_required@32 = 100%. But SAM still = core_only.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 25: q025 (multi-hop)

**Question**: What is the relationship between the selector bottleneck and the pivot to NEXUS?

**Answer**:
> Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck. Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture. Learned selector precision is the critical bottleneck (50% precision, 96.6% recall).

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

### Case 26: q026 (multi-hop)

**Question**: Walk through the evolution from experiment 0.6 to the architecture pivot.

**Answer**:
> Oracle memory = 99.87-100%, oracle filter = 100% — the core+memory architecture IS valid. Flat latent-vector memory can't solve selection quality -> pivot to graph-first architecture. Dual encoder query projection mismatch prevents SAM from using retrieved memory.

**Heuristic verdict**: `yes` (score: 4/4)

**Reasons**:
- [PASS] Focus entity present
- [PASS] Asked metric present
- [PASS] No more than N unrelated facts
- [PASS] Answer is direct (no preamble dodge)

**Manual review note**: (leave blank)

---

## Aggregate Results

| Verdict | Count | % |
|---------|-------|---|
| yes | 14 | 53.8% |
| partial | 12 | 46.2% |
| no | 0 | 0.0% |

**Relevance rate**: 76.9%
  (Formula: % yes + 0.5 × % partial)

### Per Question Type

| Type | Yes | Partial | No | Rate |
|------|-----|---------|----|------|
| factual | 8 | 10 | 0 | 72.2% |
| comparative | 2 | 0 | 0 | 100.0% |
| diagnostic | 2 | 1 | 0 | 83.3% |
| multi-hop | 2 | 1 | 0 | 83.3% |

## ✅ Metric Validation

Relevance rate (76.9%) exceeds the 70% threshold. The key-fact-overlap accuracy metric is reasonably well-aligned with actual answer relevance.

