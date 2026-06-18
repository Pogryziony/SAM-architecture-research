# Roadmap

What's next for SAM research.

**Updated: 2026-06-18**

## Immediate next step: Experiment 0.13B

**Realistic Retrieval Distractor Replay and Post-Fix Non-Oracle Revalidation**

### Goal

Experiment 0.13A proved that SAM tolerates controlled random distractors well
(91.6% with +8 distractors, 99.8% with +1). Now test whether realistic
retrieval distractors are harder, and revalidate the non-oracle paths after
the padding bug fix.

### Steps

1. **Rerun non-oracle baselines post-fix**
   - Run `retrieved_memory_external_text_query` with chain-set retriever
   - Compare to pre-fix results (exp 0.11-0.12)
   - Did the padding bug explain the core_only result?

2. **Realistic distractor replay**
   - Use chain-set retriever to get top-K candidates
   - For each example: note required slots and distractor slots
   - Replay EXACT distractors via `oracle_plus_realistic_distractors`
   - Compare damage vs random distractors at same noise level

3. **Test topK caps** — configs prepared:
   - top4, top8, top16, top32, top64
   - Does capping how many retrieved slots enter memory help?

4. **Test chain-set with external text query** — post-padding fix
   - Does wiring the chain-set retriever through the external text query
     path work now that the padding bug is fixed?

### Decision rules

| If... | Then... |
|-------|---------|
| Realistic top8 beats core-only strongly | Continue selector/ranking optimization. Path is viable. |
| Random +8 works but realistic +8 fails | Distractor QUALITY is the problem → train on realistic hard negatives |
| Realistic replay works but actual retrieved path fails | Bug in the retrieved-memory wiring path → fix wire path |
| All realistic bounded-memory paths fail | Go deeper — implement slot-wise memory reader instead of flat averaging |
| Post-fix non-oracle works now | Previous failure was padding bug → everything before 0.13B must be rerun |
| Post-fix non-oracle still fails | Deeper issue than padding → focus on distractor quality and training dynamics |

## Medium-term next steps

### After 0.13B succeeds

If realistic retrieval works post-bugfix:

1. **Revalidate learned selector**
   - Retrain selector on fixed codebase
   - Test with chain-set candidates at various topK
   - Goal: >95% recall AND >80% precision

2. **Hard negative training**
   - Train selector with realistic distractors instead of random negatives
   - Generate hard negatives from retrieval-mined candidates
   - Goal: selector precision that holds on realistic distractors

3. **End-to-end pipeline validation**
   - Chain-set retriever → selector → SAM
   - Single-pass, no oracle shortcuts
   - Must beat core_only by a statistically significant margin

4. **Efficiency measurements**
   - Memory bandwidth used per token (reads from PKM)
   - Latency benchmarks (CPU, not just GPU)
   - Comparison to equivalent-size dense transformer
   - Comparison to small LLM + RAG

### After end-to-end pipeline works

1. **Slot-wise memory reader**
   - Replace flat aggregation with per-slot attention
   - Let the model attend to individual slots differently
   - This may handle noise better than averaging

2. **Iterative querying**
   - Instead of one query → one memory read, do:
     query → read → integrate → re-query → read more
   - Needed for longer chains (4+ hops)

3. **Scale test**
   - More slots (10K, 100K)
   - Larger core model (50M, 100M params)
   - More diverse dataset

## Longer-term vision items

These are the bigger architectural ideas from the thesis that are NOT yet
implemented. They depend on the basic pipeline working first.

| Item | What it means | Prerequisites |
|------|-------------|--------------|
| mmap-backed memory | Memory too large to fit in GPU VRAM | Pipeline works at scale |
| Ternary core quantization | 1.58-bit weights (not FP32) to reduce bandwidth | Core-only baseline established |
| Exact payload memory | Store fact text, not just vectors | Retrieval quality proven |
| Adaptive multi-hop | Iterative retrieve → use → retrieve again | Single-hop pipeline works |
| Surprise-gated patching | BLT-style token compression on input | Multi-hop works reliably |
| Real-world data | Code repos, API docs, logs instead of synthetic | All above items |

## What we will NOT do (near term)

- Scale to hundreds of millions of parameters just to "see if it works"
- Add retrieval complexity (more stages, ranking layers) before fixing the basic path
- Compare to GPT, DeepSeek, or production LLMs (premature and misleading)
- Claim readiness for any practical application
- Publish results as "validated architecture" before realistic retrieval works

## Timeline notes

No fixed timeline. Each experiment informs the next. The decision rules above
determine whether we proceed down a path or pivot.

The current cadence: ~2 experiments per week internally.

## How to propose a new experiment

1. Articulate exactly one question to test
2. Define what success means for that question
3. Define what failure would mean
4. Write the config and implement any needed code
5. Run and write a report
6. Integrate findings into this roadmap

---

*Last updated: 2026-06-18*
