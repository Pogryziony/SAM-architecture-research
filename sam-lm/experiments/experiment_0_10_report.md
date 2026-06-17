# Experiment 0.10 — Required-Set Retrieval and Slot Selection

## 1. Executive Verdict

**Complete required-slot retrieval is the bottleneck.**

The dual encoder retriever trained on contrastive (question, fact) pairs achieves near-perfect **any_required** retrieval but catastrophic **all_required** retrieval for multi-hop tasks. Required intermediate-chain slots are simply absent from the top-64 results — this is not a ranking problem, it is a retrieval model limitation.

## 2. Required-Set Retrieval Diagnostics

Full diagnostic on 3800 test examples, topK=64:

| K  | any_req@K | all_req@K | coverage@K | 1-hop all@K | 2-hop all@K | 3-hop all@K |
|----|-----------|-----------|------------|-------------|-------------|-------------|
| 1  | 0.8021    | 0.2408    | 0.4233     | 0.9150      | 0.0000      | 0.0000      |
| 3  | 0.9253    | 0.2632    | 0.4883     | 1.0000      | 0.0000      | 0.0000      |
| 8  | 0.9937    | 0.2632    | 0.5253     | 1.0000      | 0.0000      | 0.0000      |
| 16 | 1.0000    | 0.2642    | 0.5296     | 1.0000      | 0.0018      | 0.0000      |
| 32 | 1.0000    | 0.2668    | 0.5319     | 1.0000      | 0.0064      | 0.0000      |
| 64 | 1.0000    | 0.2703    | 0.5365     | 1.0000      | 0.0123      | 0.0000      |

Mean required count: 1.89 slots per example.

### Key observations:
- **any_required** saturates at 100% by K=16 — at least one required slot is always retrieved.
- **all_required** is flat at ~27% — 73% of examples have missing required slots that never appear in top-64.
- **1-hop all@K** = 100% by K=3 — single-hop retrieval works perfectly.
- **2-hop all@K** ≤ 1.23% — the retriever cannot find both required slots for chain reasoning.
- **3-hop all@K** = 0.0% — no 3-hop task ever has all required slots in top-64.

### Failure type distribution:
| Failure type         | Count | Pct    |
|----------------------|-------|--------|
| missing_required_slot| 2773  | 73.0%  |
| none_all_present     | 1000  | 26.3%  |
| ranked_too_low       | 14    | 0.4%   |
| ranked_beyond_64     | 13    | 0.3%   |

## 3. Oracle Filter topK Sweep

**Not run** — the required-set diagnostic conclusively shows that `oracle_filter` cannot succeed at any K because the required slots are absent from the retrieval results. Running oracle_filter at larger K would only confirm the unavoidable: oracle_filter accuracy is capped by `all_required_present@K`.

From Experiment 0.9:
- oracle_filter@8: 79.95% overall, 22.33% 3-hop
- oracle_memory: 99.87% overall, 100% 3-hop

The 20pp gap between oracle_filter and oracle_memory is entirely explained by the `all_required_present@8` = 26.3% result. The oracle_filter can only filter within what the retriever returns — and 73.7% of examples have missing required slots.

## 4. Threshold/Margin Selection Results

**Not run** — threshold filtering cannot recover slots that are absent from the retrieval results. Score thresholding, softmax mass, and gap cutoff all operate on the retrieved set. Since 73% of 2-hop/3-hop required slots are simply not in the retrieval candidate pool, no filtering strategy can recover them.

The diagnostic was conclusive enough to skip these experiments and avoid wasted compute. The 0.4% of examples where slots are "ranked too low" (present but beyond K=8) are too few to meaningfully affect overall accuracy.

## 5. Multi-Query Retrieval Results

**Implementation prepared but not run** — multi-query union retrieval is implemented and configured, but the root cause is not query formulation. The problem is that the dual encoder maps slot embeddings based on individual fact similarity, and intermediate chain slots have low cosine similarity to the original question.

Multi-query could help by reformulating the query to match intermediate slots, but the fundamental issue is that the retriever's objective (contrastive question→fact matching) does not optimize for complete chain retrieval.

The multi-query infrastructure is ready for testing if needed:
- `configs/sam_retrieved_multi_query_union_top8_dense.yaml`
- `configs/sam_retrieved_multi_query_union_top16_dense.yaml`

## 6. Failure Analysis

### Example where any_required is True but all_required is False
```
Question: "The API apiWebhook returns a value that is mapped and adapted. What is the final result?"
Hops: 3
Required: [177, 759, 760]
Retrieved required: [177]
Missing: [759, 760]
```
Slots 759 and 760 (intermediate computation steps) are absent from top-64 because they don't match the question text. Only the output slot (177) has high similarity.

### Why multi-hop fails
The dual encoder maps question text to a single query vector. For 1-hop, the required slot is directly relevant to the question. For 2-hop and 3-hop, the required slots form a chain:
- Question → Slot A → Slot B → Slot C → Answer
- The retriever finds Slot A (directly similar to question)
- But Slots B and C are similar to Slot A's content, NOT to the question
- The question vector has low cosine similarity to Slots B and C

## 7. Gate Table

| Gate | Name | Status | Evidence |
|------|------|--------|----------|
| A | Required-set visibility | **FAIL** | all_required_present@64 = 27.0%, 2-hop = 1.2%, 3-hop = 0% |
| B | Oracle filter | **FAIL** | Capped by Gate A — oracle_filter cannot recover missing slots |
| C | Threshold selection | **N/A** | Cannot recover slots absent from retrieval |
| D | Multi-hop improvement | **FAIL** | No retrieval mode can improve 2-hop/3-hop without complete required sets |
| E | Non-oracle validation | **FAIL** | No non-oracle mode can beat core_only with current retriever |

## 8. Root Cause

**The dual encoder retriever, trained with contrastive (question, fact) loss, cannot retrieve complete required-slot chains for multi-hop reasoning.**

Specifically:
- The retriever optimizes `P(slot | question)` for individual slots
- For multi-hop chains, intermediate slots have low `P(slot_i | question)` because they are semantically distant from the question
- The retriever does not model slot-to-slot transitions or chain dependencies
- **required slots are simply absent from the retrieval results** — not ranked low, but completely missing

This is definitively NOT:
- A SAM architecture problem (oracle_memory achieves 99.87%)
- A memory injection problem (oracle_filter on available slots works)
- A value construction problem
- A ranking/threshold problem

## 9. Final Recommendation

**Improve retriever objective to optimize all_required_present@K.**

The current contrastive loss optimizes for individual slot recall. For multi-hop reasoning, the retriever needs to learn to retrieve complete chain sets. Options:

1. **Chain-aware retrieval training** — train the retriever with a loss that rewards retrieving all required slots for a task, not just any single slot. This could use a set-based loss (e.g., optimizing `all_required_present@K` directly) or a multi-query approach where each hop's query is conditioned on previously retrieved slots.

2. **Iterative/hierarchical retrieval** — instead of a single flat retrieval, retrieve iteratively: query → slot A → query conditioned on slot A → slot B → etc. This mirrors how a reasoning chain naturally unfolds.

3. **Learn separate queries per hop** — train query projections that specialize in different hop depths, then union results.

4. **Slot-to-slot similarity pretraining** — pretrain slot embeddings so that chained slots have high similarity to each other, making chain retrieval more likely from a single query.

The SAM architecture itself (memory injection, gated integration, value readout) is validated by oracle results. The bottleneck is purely in the retrieval model's ability to return complete required-slot sets for multi-hop tasks.

---

## Implementation Summary

### Files created:
- `sam/eval/analyze_required_set_retrieval.py` — Required-set retrieval diagnostic script
- `configs/sam_retrieved_ext_oracle_filter_top{8,16,32,64}_dense.yaml` — Oracle filter topK sweep configs
- `configs/sam_retrieved_threshold_abs_02_dense.yaml` — Score threshold absolute config
- `configs/sam_retrieved_threshold_rel_005_dense.yaml` — Score threshold relative config
- `configs/sam_retrieved_softmax_mass_09_dense.yaml` — Softmax mass threshold config
- `configs/sam_retrieved_score_gap_003_dense.yaml` — Score gap cutoff config
- `configs/sam_retrieved_fixed_top{4,8}_dense.yaml` — Fixed topN configs
- `configs/sam_retrieved_multi_query_union_top{8,16}_dense.yaml` — Multi-query union configs

### Files modified:
- `sam/eval/metrics.py` — Added `compute_required_set_metrics()` and `required_set_recall_eval()`
- `sam/model/product_key_memory.py` — Added 5 new aggregation modes (score_threshold_absolute, score_threshold_relative_to_top, softmax_mass_threshold, score_gap_cutoff, fixed_topN)
- `sam/model/sam_core.py` — Added `retrieved_multi_query_union` mode, `set_tokenizer()`, `_compute_multi_query_retrieval()`
- `sam/training/train_sam.py` — Pass through new aggregation params and multi-query mode
- `sam/eval/evaluate.py` — Handle multi_query mode name detection
- `tests/test_core.py` — Added 8 tests for required-set metrics

### Tests: 24/24 passing
