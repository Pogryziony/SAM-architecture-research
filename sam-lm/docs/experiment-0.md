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

### Sizes

- Training: 10,000 examples (default)
- Validation: 1,000 examples
- Test: 1,000 examples

## Metrics

See [metrics.md](metrics.md) for full definitions.

Core metrics:
- accuracy_single_hop, accuracy_two_hop, accuracy_three_hop
- accuracy_by_task_type
- memory_recall_at_1, recall_at_8, recall_at_32
- oracle_gap = oracle_accuracy - retrieved_accuracy
- dense_gap = retrieved_accuracy - dense_baseline_accuracy

## Decision Gates

1. **Gate 1 -- Retrieval**: If Recall@8 < 80%, stop and improve retrieval before
   training SAM end-to-end.

2. **Gate 2 -- Memory usefulness**: If SAM + oracle memory does not beat SAM core-only,
   stop. The model is not using memory correctly.

3. **Gate 3 -- Retrieval gap**: If oracle_gap > 20 percentage points, retrieval is
   the bottleneck.

4. **Gate 4 -- Reasoning**: If SAM improves single-hop accuracy but not two-hop or
   three-hop, do not scale. The architecture is recall-only.

5. **Gate 5 -- Dense baseline**: If SAM + retrieved memory does not beat the same-size
   dense Transformer on knowledge-heavy tasks, do not scale.

## Expected Outcomes

If the thesis holds:
- SAM + oracle memory >> SAM core-only (+15-30% absolute on multi-hop).
- SAM + retrieved > dense baseline (+5-15% absolute on multi-hop).
- Gap widens with increasing hop count.

If the thesis fails:
- SAM + oracle approximately equal to SAM core-only: model cannot use memory.
- SAM + retrieved approximately equal to dense baseline: no advantage.
- SAM improves single-hop but not multi-hop: recall-only architecture.

## CLI Usage

```bash
# Generate data
python -m sam.data.synthetic_facts --output data/synthetic --train 10000 --val 1000 --test 1000 --seed 42

# Train dense baseline
python -m sam.training.train_dense --config configs/dense_tiny.yaml

# Train retrieval (Gate 1 diagnostic)
python -m sam.training.train_retrieval --config configs/retrieval_1m.yaml

# Train SAM variants
python -m sam.training.train_sam --mode core_only --config configs/sam_tiny.yaml
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny.yaml
python -m sam.training.train_sam --mode retrieved_memory --config configs/sam_tiny.yaml

# Evaluate
python -m sam.eval.evaluate --runs experiments/
```
