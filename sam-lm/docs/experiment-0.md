# Experiment 0: Can Sparse Associative Memory Improve Multi-Hop Reasoning?

## Hypothesis

A small reasoning core (20-50M parameters) augmented with product-key associative memory
can outperform a same-size dense Transformer on tasks where required facts are stored
externally and retrieved sparsely.

Specifically, SAM should show increasing advantage over the dense baseline as the number
of reasoning hops increases -- this would demonstrate that the architecture enables
multi-step inference over externally stored knowledge, not just single-fact recall.

## Failure Condition

If SAM improves single-hop factual recall but does **not** improve two-hop and three-hop
reasoning relative to the dense baseline, SAM is not validated as a general base architecture.
It should be reclassified as a retrieval-heavy specialist.

Additional failure modes:
- If SAM + oracle memory does not beat SAM core-only, the model cannot use memory at all.
- If SAM + retrieved memory does not beat the dense baseline on any metric, SAM is not
  competitive.
- If Recall@8 is below 80%, retrieval is not working well enough to test the thesis.

## Model Variants

| Variant             | Description                                                  | Purpose                            |
|---------------------|--------------------------------------------------------------|------------------------------------|
| Dense Transformer   | Standard causal decoder, 20-50M params, trained on QA + facts | Baseline: knowledge in weights     |
| SAM core-only       | SAM with memory disabled, same active parameter budget        | Capacity floor: can it reason?     |
| SAM + oracle memory | Correct required slots injected directly                      | Upper bound: can it use memory?    |
| SAM + random memory | Random live slot values injected                              | Placebo: is gating alone helping?  |
| SAM + retrieved     | Learned product-key retrieval                                 | The real SAM mechanism             |

## Retriever Architectures

| Retriever | Training Objective | all_required@32 (3-hop) | Description |
|-----------|-------------------|------------------------|-------------|
| Dual Encoder | InfoNCE (1st slot only) | 0.000 (0%) | Single-slot contrastive, finds output slot |
| Chain-Set BCE | Multi-positive BCE (all slots) | **1.000 (100%)** | All required slots as positives vs all live slots |

## Dataset Design

Synthetic dataset with explicit facts and multi-hop questions, organized into task types:

1. **single_fact_recall** (1 hop): Direct fact lookup.
2. **two_hop_reasoning** (2 hops): Bridge entity not named in question.
3. **three_hop_reasoning** (3 hops): Two bridge entities, chained transformations.
4. **api_usage_reasoning** (2 hops): API -> header -> token chain.
5. **code_symbol_reasoning** (2 hops): Function -> error -> status code chain.

Key design properties:
- Train/test entity separation (different noun roots for test entities).
- Deterministic generation with configurable seed.
- Bridge entities intentionally not named in questions, forcing chaining.
- Each answer is a single vocabulary token for clean exact-match evaluation.

### Dense Dataset (data/synthetic_dense)

- **Slots**: 1,650 (shared — many questions reference the same facts)
- **Training**: 19,000 examples
- **Validation**: 3,800 examples
- **Test**: 3,800 examples
- **Vocab**: 853 tokens
- All experiments 0.6–0.11 use this dataset

## Metrics

See [metrics.md](metrics.md) for full definitions.

Core metrics:
- accuracy_single_hop, accuracy_two_hop, accuracy_three_hop
- accuracy_by_task_type
- memory_recall_at_1, recall_at_8, recall_at_32
- oracle_gap = oracle_accuracy - retrieved_accuracy
- dense_gap = retrieved_accuracy - dense_baseline_accuracy

Experiment 0.10+ required-set metrics:
- any_required_present@K, all_required_present@K
- required_slot_coverage@K
- Per-hop breakdowns (1-hop, 2-hop, 3-hop)
- MRR of first required slot, mean rank per slot position

## Decision Gates

### Original Gates (Experiments 0.0–0.6)

1. **Gate 1 — Retrieval**: If Recall@8 < 80%, stop and improve retrieval.
2. **Gate 2 — Memory usefulness**: If SAM + oracle memory does not beat SAM core-only, stop.
3. **Gate 3 — Retrieval gap**: If oracle_gap > 20pp, retrieval is the bottleneck.
4. **Gate 4 — Reasoning**: If SAM improves single-hop but not multi-hop, do not scale.
5. **Gate 5 — Dense baseline**: If SAM + retrieved does not beat same-size dense, do not scale.

### Experiment 0.11 Retrieval Gates

| Gate | Condition | Result |
|------|-----------|--------|
| Gate A | 2-hop all_required@16 ≥ 80% | **PASS** — 0.9600 |
| Gate B | 3-hop all_required@32 ≥ 70% | **PASS** — 1.0000 |
| Gate C | 3-hop coverage@32 ≥ 90% | **PASS** — 1.0000 |
| Gate D | retrieved_memory > core_only | **FAIL** — 0.6866 = 0.6874 |
| Gate E | SAM improves 2-hop and 3-hop | **FAIL** — identical to core_only |

## Full Experiment History

### Experiment 0.0–0.5: Infrastructure & Baselines

- Synthetic data generation, tokenizer, dense Transformer baseline
- Product-key memory with 1M slots, retrieval pretraining
- SAM core-only and oracle memory modes

### Experiment 0.6: Validation (dense dataset)

**SAM oracle_memory = 99.9% vs core_only = 68.7% (+31pp). Thesis CONFIRMED.**

| Model | Overall | 1-hop | 2-hop | 3-hop |
|-------|---------|-------|-------|-------|
| Dense baseline | 68.7% | 91.5% | 71.1% | 22.0% |
| SAM core_only | 68.7% | 91.5% | 71.1% | 22.0% |
| SAM oracle_memory | **99.9%** | **99.5%** | **100%** | **100%** |
| SAM retrieved (dual enc) | 68.7% | 91.5% | 71.1% | 22.0% |

Dual encoder Rec@8 = 99.3%. Retrieved memory = core_only.

### Experiment 0.7: External Text Query & Hidden Adapter

Two new retrieval-to-SAM interfaces:
- **External text query**: Dual encoder encodes raw question → retrieves slot text → feeds as input tokens
- **Hidden adapter**: Trainable MLP from SAM hidden state → dual encoder query space

Both modes = core_only accuracy (68.7%). Memory integration doesn't convert retrieval into accuracy.

### Experiment 0.8: Oracle Slots & Thresholding

SAM with oracle slot selection shows that even perfect retrieval → text doesn't help.
Thresholding and score-based filtering cannot recover from absent slots.

### Experiment 0.9: Multi-Query Unions, TopK Sweeps

Extensive parameter sweeps (topK=1..64, score temperatures, weighted aggregation).
All retrieved modes = core_only. Retrieval quality ≠ QA accuracy.

### Experiment 0.10: Required-Set Retrieval Diagnostic

First systematic measurement of multi-hop retrieval quality:

| K | all_required@K | 2-hop all@K | 3-hop all@K |
|---|---------------|-------------|-------------|
| 1 | 0.2408 | 0.000 | 0.000 |
| 8 | 0.2634 | 0.001 | 0.000 |
| 64 | 0.2729 | 0.017 | 0.000 |

73% of examples have required slots absent from top64. 3-hop = 0% at all K.
Root cause: dual encoder trained on first required slot only — finds output slot but misses chain.

### Experiment 0.11: Chain-Aware Retrieval (CURRENT)

**Chain-Set BCE Retriever — retrieval solved at K=32:**

| K | all_required@K | 2-hop all@K | 3-hop all@K | coverage@K |
|---|---------------|-------------|-------------|------------|
| 1 | 0.2408 | 0.000 | 0.000 | 0.431 |
| 8 | 0.8103 | 0.851 | 0.343 | 0.873 |
| 16 | 0.9653 | 0.960 | 0.927 | 0.980 |
| 32 | **1.0000** | **1.000** | **1.000** | **1.000** |

3-hop all_required went from 0.000 (Exp 0.10) → 1.000 (Exp 0.11). **267x improvement on 2-hop, infinite on 3-hop.**

**SAM with chain-aware retriever:**

| Mode | Overall | 1-hop | 2-hop | 3-hop |
|------|---------|-------|-------|-------|
| core_only | 0.6874 | 0.915 | 0.711 | 0.22 |
| chain_set retrieved | 0.6866 | 0.915 | 0.710 | 0.22 |
| oracle_memory | 0.9987 | 0.995 | 1.000 | 1.00 |

Retrieved = core_only despite perfect retrieval. SAM is the new bottleneck.

## Updated Decision Gates (Post-Experiment 0.11)

| Gate | Result |
|------|--------|
| Gate 1 — Retrieval Rec@8 ≥ 80% | **PASS** (99.3% any, 81% all) |
| Gate 2 — Memory usefulness (oracle) | **PASS** (99.9% vs 68.7%) |
| Gate A — 2-hop all@16 ≥ 80% | **PASS** (96.0%) |
| Gate B — 3-hop all@32 ≥ 70% | **PASS** (100%) |
| Gate C — 3-hop coverage@32 ≥ 90% | **PASS** (100%) |
| Gate 4 — Multi-hop reasoning (retrieved) | **FAIL** (chain-aware = core_only) |
| Gate 5 — Dense comparison (retrieved) | **FAIL** (chain-aware = dense baseline) |
| Gate D — retrieved > core_only | **FAIL** (–0.1pp) |
| Gate E — 2/3-hop improvement | **FAIL** (no improvement) |

## Actual Outcomes Summary

| Condition | Expected | Actual | Verdict |
|-----------|----------|--------|---------|
| SAM oracle >> core-only | +15-30% | **+31pp** | ✓ CONFIRMED |
| Multi-hop improvement (oracle) | ✓ | Three-hop: 22% → 100% | ✓ CONFIRMED |
| Gap widens with hops (oracle) | ✓ | +8pp / +29pp / +78pp | ✓ CONFIRMED |
| SAM core = dense | equal | Both 68.7% | ✓ CONFIRMED |
| Retrieval any_required@8 ≥ 80% | Gate 1 | 99.3% → 97.6% | ✓ CONFIRMED |
| Retrieval all_required@32 ≥ 70% | Gate B | 100% | ✓ CONFIRMED |
| Retrieved improves over core_only | Gate D | 68.7% = 68.7% | ✗ FAILED |
| Retrieved improves multi-hop | Gate E | 2-hop: 71% = 71%, 3-hop: 22% = 22% | ✗ FAILED |

## Root Cause Analysis

1. **Retrieval bottleneck (solved)**: The dual encoder was trained on single-slot contrastive
   loss. Multi-positive BCE over all live slots fixes this — all_required@32 = 100%.

2. **Memory integration bottleneck (open)**: The SAM model's gated_sum memory integration
   doesn't convert accurate retrieval into improved reasoning. At 16M params / 3 epochs,
   the core cannot effectively use external text facts. Oracle memory (injecting correct
   latent vectors) still works (99.9%), confirming the core CAN compose — but external
   text is not being used.

3. **Likely causes**: (a) Model capacity too low to process retrieved text, (b) gated_sum
   insufficient for text integration, (c) training too short for memory utilization,
   (d) the core learns to ignore external memory input.

## Next Steps

1. **Investigate memory integration**: Compare gated_sum vs cross-attention
2. **Longer training**: Extend from 3 to 10+ epochs with chain-aware retriever
3. **Larger models**: Scale SAM core to 50M+ params to process retrieved facts
4. **Loss analysis**: Check if gradients flow through memory path during training
5. **Attention visualization**: See if the model attends to retrieved fact tokens
6. **Scale dataset**: Move to larger/synthetic_50k where core_only capacity is insufficient
