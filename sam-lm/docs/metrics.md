# SAM Experiment Metrics

## Core Accuracy Metrics

### accuracy_overall
Fraction of test examples where the generated answer token exactly matches the expected answer token.

```
accuracy_overall = correct / total
```

### accuracy_single_hop
Overall accuracy restricted to examples with `reasoning_hops == 1`.

**Diagnoses**: Whether the model can perform direct fact lookup. All variants should score well here.

### accuracy_two_hop
Overall accuracy restricted to examples with `reasoning_hops == 2`.

**Diagnoses**: Whether the model can chain two facts. This is the first real test of reasoning. The
bridge entity is intentionally not named in the question, so single-step retrieval from the
question alone cannot reach the answer-bearing fact. The model must either store the bridge
in weights (dense baseline) or retrieve and compose (SAM).

### accuracy_three_hop
Overall accuracy restricted to examples with `reasoning_hops == 3`.

**Diagnoses**: Whether the model can chain three facts with two implicit bridge entities.
This is the hardest reasoning test. Most models will score near zero unless they can
genuinely perform multi-step inference over externally stored knowledge.

### accuracy_by_task_type
Accuracy broken down by task type (single_fact_recall, two_hop_reasoning, three_hop_reasoning,
api_usage_reasoning, code_symbol_reasoning).

**Diagnoses**: Whether certain task structures are inherently harder. If SAM excels on
api_usage_reasoning but fails on code_symbol_reasoning, the synthetic task templates may
need adjustment.

## Retrieval Metrics

### memory_recall_at_k

For each test example, does the product-key memory top-k retrieval include at least one
of the required_slots?

```
recall_at_k = |{i : required_slots_i ∩ retrieved_top_k_i ≠ ∅}| / N
```

- **recall_at_1** (~5% is random baseline for 64 slots, ~0.0001% for 1M slots)
- **recall_at_8**: Critical. Used for Gate 1. Threshold: >= 80%.
- **recall_at_32**: If recall@8 is low but recall@32 is high, the retrieval signal is
  present but noisy. Top_k needs tuning.

**Diagnoses**:
- Low recall@8: query encoder is not learning useful representations, or key tables
  are not organizing semantically.
- recall@8 ~= random: keys have collapsed. Check key normalization, temperature, or
  add a spread loss.

## Required-Set Retrieval Metrics (Experiment 0.10+)

These metrics measure whether the retriever finds the *complete set* of required facts,
not just *any* required fact. Introduced in Experiment 0.10 when it was discovered that
the dual encoder achieved 99.3% any_required@8 but 0% all_required@8 for 3-hop tasks.

### any_required_present@K
Fraction of examples where at least one required slot appears in top-K retrieval.
```
any_required@K = |{i : required_i ∩ topK_i ≠ ∅}| / N
```
Equivalent to the original recall_at_k. Near-perfect results can mask catastrophic
all_required failures.

### all_required_present@K
Fraction of examples where ALL required slots appear in top-K retrieval.
```
all_required@K = |{i : required_i ⊆ topK_i}| / N
```
**This is the critical multi-hop retrieval metric.** A retriever that finds the output
slot but misses intermediate chain slots will have high any_required@K but low
all_required@K. The dual encoder baseline (Exp 0.10) achieved 0% all_required@K for
3-hop at all K values.

### required_slot_coverage@K
Fraction of total required slots retrieved across all examples.
```
coverage@K = Σ|required_i ∩ topK_i| / Σ|required_i|
```

### Per-hop all_required@K
all_required@K broken down by number of reasoning hops (1-hop, 2-hop, 3-hop).
```
all_required_single_hop@K: only examples with hops == 1
all_required_two_hop@K:   only examples with hops == 2
all_required_three_hop@K: only examples with hops == 3
```

### Rank Metrics
- **MRR_first_required@K**: Mean reciprocal rank of the first required slot in top-K
- **mean_rank_slot_N@K**: Mean rank of N-th required slot (1st, 2nd, 3rd)
- **mean_max_required_rank@K**: Mean of the maximum rank among all required slots

These diagnose whether required slots are present but ranked too low.
Example: if all_required@8 = 0 but all_required@32 = 1.0, the slots are ranked between
8-32 (ranked_too_low failure). If MRR is near 1.0, the first required slot is always top-ranked.

### Failure Type Distribution
- **none_all_present**: All required slots in top-K
- **ranked_too_low**: All required present at K=32 but not K=8
- **ranked_beyond_64**: All required present at K=64 but not K=32
- **missing_required_slot**: Some (but not all) required slots in top-K
- **no_required_in_topk**: Zero required slots in top-K

## Derived Metrics

### oracle_gap
```
oracle_gap = accuracy(oracle_memory) - accuracy(retrieved_memory)
```

The accuracy lost due to imperfect retrieval. If oracle_memory is perfect and
retrieved_memory is poor, retrieval is the bottleneck.

**Diagnoses**: Gap > 20pp means Gate 3 fails. Improve retrieval before scaling the core.

### memory_gain
```
memory_gain = accuracy(retrieved_memory) - accuracy(core_only)
```

The accuracy gained by adding (imperfect) retrieval to the core. If this is near zero,
memory is providing no benefit over the dense weights alone.

**Diagnoses**: If oracle memory beats core-only but retrieved doesn't, the retrieval
is failing. If oracle memory also doesn't beat core-only, the core cannot use memory.

### dense_gap
```
dense_gap = accuracy(retrieved_memory) - accuracy(dense_baseline)
```

The advantage (or disadvantage) of SAM over a same-size dense Transformer.
Gate 5 requires this to be positive.

**Diagnoses**:
- Positive: SAM is validated at this scale. Proceed to larger memory.
- Zero or negative: SAM is not competitive. Inspect whether retrieval or reasoning
  is the root cause before scaling.

### retrieval_gain (Experiment 0.11)
```
retrieval_gain = accuracy(chain_set_retrieved) - accuracy(dual_encoder_retrieved)
```

The accuracy gained by improving retrieval from any_required to all_required.
If this is zero despite perfect all_required retrieval, the bottleneck has shifted
from retrieval to reasoning capacity or memory integration.

## Training Metrics

### training_loss
Cross-entropy loss on the training set, measured during training.

### validation_loss
Cross-entropy loss on the validation set. Used for model selection (best checkpoint).

### parameter_count
Total number of unique trainable parameters. For SAM, this includes both core parameters
and memory parameters (key tables, value embeddings).

## System Metrics (optional)

### tokens_per_second
Training throughput. Not a research metric but useful for estimating experiment wall time.

### peak_ram
Memory usage. Important for understanding when to move to mmap-backed memory.

## Metric Interpretation Guide

| Pattern                                          | Diagnosis                                          |
|--------------------------------------------------|----------------------------------------------------|
| oracle >> core, retrieved ~= oracle               | SAM works perfectly; retrieval is great            |
| oracle >> core, retrieved >> dense                | SAM works; some retrieval noise but net positive    |
| oracle >> core, retrieved ~= dense                | Retrieval is the bottleneck                        |
| oracle ~= core                                    | Core cannot use memory; architecture problem       |
| oracle >> core, retrieved ~= core                 | Memory integration bottleneck (Exp 0.11)           |
| oracle ~= retrieved, both > dense                 | Retrieval works but memory not adding value        |
| all_required@32 = 1.0, SAM ~= core                | Retrieval solved; reasoning/memory new bottleneck  |
| any_required@K >> all_required@K                  | Retriever finds output slot, misses chain slots    |
| all variants similar                               | Task is too easy or models too similar              |
| dense > all SAM variants                           | SAM thesis rejected at this scale                  |
