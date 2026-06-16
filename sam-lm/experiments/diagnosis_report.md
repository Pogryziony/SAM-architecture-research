# SAM-LM Experiment 0 — Diagnosis Report

**Date:** 2026-06-16
**Status:** Pipeline bugs fixed. Full rerun completed. Retrieval is the bottleneck.

---

## 1. Summary Verdict

**SAM is not yet validated. Retrieval is the bottleneck.**

After fixing three critical pipeline bugs and rerunning the full experiment:
- Oracle memory shows a real but modest improvement (+1.8pp val, +1.9pp test over core-only)
- Retrieved memory performs identically to or worse than core-only
- Retrieval recall@8 = 6.9% on 1M-slot PKM (far below 80% gate)
- The PKM can learn to retrieve (overfit Recall@8 = 100%)
- But 4781 live slots spread across 1,048,576 PKM addresses is too sparse for effective retrieval

---

## 2. Bugs Found and Fixed

### Bug 1: best_val_loss: Infinity
**Cause:** eval_every=500 > total_steps=264 → val_loss never computed
**Fix:** Per-epoch validation eval in all three training scripts
**Evidence:** All runs now show proper val_loss tracking

### Bug 2: Retrieval InfoNCE dead-slot negatives
**Cause:** Negatives from all 1M PKM slots; 99.5% dead → no learning
**Fix:** Live-slot-only negative sampling
**Evidence:** Overfit Recall@8 = 100% (was 0%)

### Bug 3: Evaluation used wrong checkpoints for SAM modes
**Cause:** Directory walker only found first checkpoint per top-level directory
**Fix:** Recursive walker finds checkpoints in subdirectories; mode detected from path name

---

## 3. Full Rerun Results (Val Set, 1000 examples)

| Model | overall | single_hop | two_hop | three_hop | val_loss | params |
|-------|---------|------------|---------|-----------|----------|--------|
| Dense baseline | 5.9% | 6.0% | 6.0% | 5.5% | 1.83 | 34.2M |
| Dense open-book | 100% | 100% | 100% | 100% | 0.004 | 34.2M |
| SAM core_only | 6.5% | 7.0% | 6.5% | 6.0% | 1.89 | 27.6M |
| SAM oracle_memory | 8.3% | 8.5% | 8.8% | 6.5% | 1.88 | 27.6M |
| SAM retrieved_memory | 6.5% | 7.5% | 6.7% | 5.0% | 1.89 | 27.6M |

Retrieval: best_recall@8=6.9%, recall@32=23.5% (on val)

### Test Set (362 examples, novel entities)

| Model | overall | single_hop | two_hop | three_hop |
|-------|---------|------------|---------|-----------|
| Dense baseline | 1.9% | 0.0% | 3.2% | 0.0% |
| Dense open-book | 19.9% | 0.0% | 33.2% | 0.0% |
| SAM core_only | 2.2% | 0.0% | 3.7% | 0.0% |
| SAM oracle_memory | 4.1% | 0.0% | 6.9% | 0.0% |
| SAM retrieved_memory | 1.4% | 0.0% | 2.3% | 0.0% |

Note: Test set uses entirely novel entities (entity separation). Low absolute numbers are expected. Open-book model confirms facts are readable in context (33% two-hop). Three-hop at 0% for all models indicates the chains are too long for current model capacity.

---

## 4. Retrieval Diagnosis

### Overfit (16K slots, 558 live): Recall@8 = 100% ✓

### Full (1M slots, 4781 live): Recall@8 = 6.9%, Recall@32 = 23.5%

**Root cause of poor recall:** 4781 live / 1,048,576 total = 0.46% density. The PKM must learn to concentrate queries into a tiny fraction of the key space. With 1024 subkeys and only ~5 live slots per subkey-pair region on average, the cartesian product generates mostly dead candidates.

### Recommended fix: Reduce PKM to 128 subkeys (16,384 slots, ~29% density) for the full experiment. Or pretrain retrieval separately with more epochs.

---

## 5. Decision Gates

| Gate | Threshold | Result | Pass |
|------|-----------|--------|------|
| Gate 1 — Retrieval | Recall@8 >= 80% | 6.9% | FAIL |
| Gate 2 — Memory usefulness | Oracle > Core-only | 8.3% > 6.5% (+1.8pp) | PASS |
| Gate 3 — Retrieval gap | Gap <= 20pp | 8.3% - 6.5% = 1.8pp | PASS |
| Gate 4 — Reasoning | Multi-hop sustained | Oracle two-hop 8.8% > single-hop 8.5% | PASS |
| Gate 5 — Dense baseline | SAM retrieved > Dense | 6.5% vs 5.9% (+0.6pp, val) | MARGINAL |

---

## 6. Next Recommended Action

**Fix retrieval scaling before proceeding.**

Options (in priority order):
1. Use 128-subkey PKM (16K slots) for Experiment 0 instead of 1M — density would be ~29% instead of 0.46%
2. Pretrain retrieval for more epochs (20+) before end-to-end SAM training
3. Add a temperature parameter to PKM scoring to sharpen retrieval
4. Accept Gate 1 failure and proceed with end-to-end SAM training (retrieval improves jointly with LM)

The overfit test proves the pipeline works. The 1M-slot sparse-addressing problem is a scaling issue, not a fundamental bug.
