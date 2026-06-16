# SAM-LM Experiment 0.3 — PKM Retrieval Diagnosis and Repair

**Date:** 2026-06-16
**Status:** Diagnosis complete. Candidate generation solved. Final ranking + generalization are the bottlenecks.

---

## 1. Summary Verdict

**PKM retrieval fundamentally works but does not generalize.** With subkey auxiliary loss, the PKM includes the correct slot in candidates 100% of the time and achieves positive score margins. However, val Recall@8 peaks at 29.3% vs train 76.5% — a 47pp generalization gap. The PKM key tables (128x128) memorize training query patterns rather than learning generalizable key semantics.

---

## 2. Retrieval Mode Comparison

| Mode | Recall@8 | pos_in_cand | score_margin | cand_live_rate |
|------|----------|-------------|-------------|----------------|
| Compact 16K baseline (top32) | 25.8% | 78% | -1.07 | 36% |
| Compact 16K top64 | 27.2% | 98% | -0.01 | 26% |
| **Compact 16K + subkey loss** | **29.3%** | **100%** | **+1.30** | **86%** |

Subkey loss is the best config: 100% candidate inclusion, positive score margins, 86% live candidate rate. But val recall remains at 29%.

---

## 3. TopA/TopB Sweep Results

| top_a/b | Candidates | pos_in_cand | val Recall@8 | cand_live_rate |
|---------|-----------|-------------|-------------|----------------|
| 32/32 | 1024 | 78→98% | 25.8% | 36→26% |
| 64/64 | 4096 | 98% | 27.2% | 26% |
| 32/32 + subkey | 1024 | 100% | 29.3% | 86% |

Higher top_a/top_b alone doesn't help — candidate live rate drops because more dead space is included. Subkey loss improves candidate quality (86% live).

---

## 4. Detailed Diagnostics (subkey loss, step 600)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| k1_in_topA | 100% | k1 subkey correctly identified |
| k2_in_topB | 100% | k2 subkey correctly identified |
| pos_in_candidates | 100% | Required slot always in candidates |
| pos_rank_mean | 3.0 | When found, avg rank in top-32 is 3rd |
| pos_score_mean | 14.3 | Positive slot score |
| top_neg_score_mean | 13.0 | Highest negative score |
| score_margin | +1.30 | Positive scores ABOVE negatives |
| candidate_live_rate | 86% | Most candidates are live slots |
| score_std | 1.02 | Moderate score variation |
| score_entropy | 2.24 | Moderate distribution spread |
| **train Recall@8** | **76.5%** | Batch-level recall |
| **val Recall@8** | **28.9%** | **47pp generalization gap** |

---

## 5. Candidate Generation vs. Final Ranking

### Candidate generation: SOLVED
- With subkey loss: pos_in_candidates = 100%
- k1 and k2 subkey predictions are perfect (100% each)
- Candidate live rate = 86% (high quality candidates)

### Final ranking: BROKEN (generalization)
- Train Recall@8 = 76.5% → model CAN rank on seen queries
- Val Recall@8 = 28.9% → model CANNOT rank on unseen queries
- The PKM key tables overfit to training query patterns
- 128 subkeys x 128 dimensions is not enough capacity for 4781 live slots with generalization

---

## 6. Decision Gates

| Gate | Result |
|------|--------|
| Gate 1 (Retrieval) | **FAIL** — 29.3% < 80% |
| Candidate generation | **PASS** — 100% with subkey loss |
| Final ranking (train) | **PASS** — 76.5% |
| Final ranking (val) | **FAIL** — 29.3% |
| Generalization | **FAIL** — 47pp gap |

---

## 7. Root Cause Analysis

The PKM key tables (K1, K2: 128 x 128 each) have 16,384 parameters each. With 4781 live slots, the information per slot is ~3.4 dimensions. This is insufficient for learning generalizable key representations. The keys memorize coarse training query patterns but can't distinguish slots on unseen queries.

**Why candidate generation works but ranking fails:**
- Candidate generation is a coarser task: put the correct k1 and k2 in the top 32/128
- Final ranking is finer: distinguish the exact slot from 1024 candidates
- The keys have enough capacity for coarse routing but not fine-grained discrimination

---

## 8. Next Recommended Action

**Fix retrieval generalization before proceeding to SAM.**

Recommended approaches (priority order):
1. **Increase key dimensionality** — key_dim=128 → 256 or 512 (more slot discrimination capacity)
2. **Add slot-specific key embeddings** — each live slot gets a dedicated key vector (not just shared subkey tables)
3. **Use cosine similarity instead of dot product** — normalize queries and keys
4. **Implement a simpler retriever** — dense cosine baseline to determine if PKM specifically is the issue
5. **Data augmentation** — generate more diverse training queries (currently 2102)

Do NOT proceed to SAM retrieved-memory until val Recall@8 >= 80%.

---

## 9. Changes Made for Experiment 0.3

1. `sam/training/train_retrieval.py` — Added subkey_loss(), margin_loss(), retrieval_diagnostics()
2. `configs/retrieval_compact_16k_top16.yaml` — NEW: topA=16 sweep
3. `configs/retrieval_compact_16k_top32.yaml` — NEW: topA=32 sweep
4. `configs/retrieval_compact_16k_top64.yaml` — NEW: topA=64 sweep
5. `configs/retrieval_compact_16k_subkey_loss.yaml` — NEW: subkey auxiliary loss
6. `configs/retrieval_compact_16k_subkey_margin.yaml` — NEW: subkey + margin loss
7. Diagnostic metrics: k1/k2 hit rate, pos_in_candidates, score_margin, candidate_live_rate, score_entropy
