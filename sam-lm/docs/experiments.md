# Experiments

Overview of all experiments in the SAM project — what was tested, what was
found, and what changed next.

## Experiment timeline

```
Experiment 0 (Diagnosis) ──► 0.2-0.5 (Retrieval basics) ──► 0.6 (Validation)
    │                              │                              │
    Pipeline bugs fixed      Dense dataset fixes retrieval    Oracle works,
                             Rec@8: 6.9% → 99.0%             retrieved fails
                                    │
                                    ▼
0.10 (Required-set) ──► 0.11 (Chain retrieval) ──► 0.12 (Selection) ──► 0.13A (Noise tolerance)
       │                       │                        │                      │
  Multi-hop slots         all_required@32          Selector recall        +8 distractors
  missing from            = 100% —                = 96.6%,               = 91.6%, gate
  dual encoder            retrieval solved         precision = 50%        is NOT bottleneck
```

## Experiment 0 — Pipeline Diagnosis

**Question:** Is the experimental pipeline working correctly?

**Findings:**
- Three critical bugs found and fixed:
  1. `best_val_loss` was `Infinity` because validation never ran
  2. InfoNCE loss used dead slots as negatives → no learning
  3. Evaluation used wrong checkpoints for SAM modes
- After fixes, oracle_memory showed a real but small improvement (+1.8pp over core_only)
- Retrieval Rec@8 = 6.9% — far below the 80% gate

**What changed next:** Focus shifted to fixing retrieval.

## Experiment 0.5 — Dense Dataset Fix

**Question:** Can retrieval work if the dataset has better slot coverage?

**Findings:**
- Original dataset had only 1.5 examples per slot, 30% unseen slots in validation
- New dense dataset: 21.8 examples per slot, all 1,650 slots shared across splits
- Dual encoder retriever: 99.0% val Rec@8 (up from 6.9%)
- **Gate 1 (Rec@8 ≥ 80%): PASSED**

**What changed next:** With retrieval working, full SAM validation could proceed.

## Experiment 0.6 — Full Validation

**Question:** Does SAM work end-to-end with retrieval?

**Findings:**
- Oracle memory: **99.87%** — SAM core CAN use memory for reasoning ✓
- Oracle text memory: **100%** — text memory also works ✓
- Retrieved memory: **68.74%** — identical to core_only and random_memory ✗
- Random memory: **68.74%** — placebo control works as expected ✓
- Dense baseline: **68.74%** — SAM core matches dense transformer at equal params ✓

**Root cause:** The dual encoder retriever received SAM's intermediate hidden
states, not raw question tokens. The `query_proj` was trained for dual encoder
outputs, not transformer states — creating a projection mismatch.

**What changed next:** Switched to `retrieved_memory_external_text_query` mode,
where the retriever encodes the raw question text independently.

## Experiment 0.7-0.9 — Retrieval Interface and Selection Variants

**Questions tested:**
- Can external text query fix the projection mismatch?
- What aggregation modes work best?
- Can oracle filtering or multi-query improve results?

**Key findings:**
- External text query: still identical to core_only (retrieved slots too noisy)
- Tested: uniform_mean, score_weighted, threshold-based, softmax-mass, score-gap
- Oracle filter: 79.95% overall (shows the gap from distractors)
- Multi-query union: implemented but not yet effective

## Experiment 0.10 — Required-Set Retrieval Diagnostics

**Question:** Where exactly are the retrieval failures?

**Findings:**
- `any_required@K` saturates at 100% by K=16 — at least one required slot is found
- `all_required@K` is flat at ~27% — 73% of examples have missing required slots
- 2-hop `all@K` ≤ 1.2% — the retriever cannot find both required slots for chains
- 3-hop `all@K` = 0.0% — no 3-hop task ever has all required slots in top-64
- Intermediate chain slots are absent, not ranked low — a retrieval model limitation

**Root cause:** The dual encoder maps question text to slot similarity. For chains:
Question → Slot A (similar to question) ✓
Question → Slot B (similar to Slot A's content, not the question) ✗
Question → Slot C (similar to Slot B's content, not the question) ✗

**What changed next:** Redesign the retriever to optimize for complete chain sets.

## Experiment 0.11 — Chain-Aware Retrieval

**Question:** Can the retriever learn to retrieve complete required-slot chains?

**Findings:**

| K | all_required@K (dual encoder) | all_required@K (chain-set BCE) |
|---|------------------------------|-------------------------------|
| 8 | 26.3% | **81.0%** |
| 16 | 26.5% | **96.5%** |
| 32 | 26.8% | **100.0%** |

- Chain-set BCE: all_required@32 = 100% — retrieval is solved
- Multi-positive loss (reward complete sets) is the key innovation
- However: SAM retrieved-memory with chain-set retriever: **still = core_only (68.74%)**
- The retrieval improvement does not translate to QA improvement

**What changed next:** The bottleneck shifted from retrieval to slot **selection**
— the retriever returns 32+ candidates, and SAM must pick the right ones.

## Experiment 0.12 — Candidate Selection and Memory-Use Training

**Question:** Can SAM select the right slots from chain-set retrieval?

**Findings:**
- **Oracle filter** (only required slots from chain candidates): **100% accuracy**
  — proves the chain candidates are sufficient and the retrieved-memory path works
- **Learned selector**: recall 96.6%, precision 50%
  - Selects ~3.5 slots per example (vs ~1.89 required)
  - ~1.75 distractors injected into memory
  - QA accuracy: **68.74% — identical to core_only**
- Fixed top-by-hop: also = core_only (ranking alone insufficient)
- The gate learns to suppress memory completely when it's noisy

**Root cause hypothesis:** The selector's ~1.75 distractors cause the gate to
ignore memory entirely. The model never learns to use memory because it's
always noisy during training.

**What changed next:** Test whether SAM can tolerate any distractors at all
(Experiment 0.13A).

## Experiment 0.13A — Controlled Noisy Memory Tolerance

**Question:** How much memory noise can SAM tolerate? Does even one distractor
cause collapse?

**Findings:** [Detailed in experiment-0-13a-noisy-memory.md](experiment-0-13a-noisy-memory.md)

Contrary to the 0.12 hypothesis, SAM does NOT collapse with one distractor:

| Distractors | Overall | 1-hop | 2-hop | 3-hop |
|------------|---------|-------|-------|-------|
| 0 (oracle) | 100.00% | 100.00% | 100.00% | 100.00% |
| 1 | 99.82% | 99.90% | 99.86% | 99.50% |
| 2 | 99.39% | 99.80% | 99.55% | 98.17% |
| 4 | 97.63% | 98.10% | 98.14% | 95.00% |
| 8 | 91.58% | 95.90% | 92.95% | 79.33% |
| 16 | 75.42% | 92.80% | 77.45% | 39.00% |

**Key insight:** With +1-2 distractors (matching the selector's typical output),
SAM achieves 99.4-99.8% accuracy — far above core_only. The 0.12 selector
failure is NOT about distractor count (~1.75). It's about something else — likely
the qualitative nature of the selector's distractors (semantically misleading
rather than random), or training dynamics that prevented the gate from opening.

## Experiment 0.13B — Realistic Retrieval Distractor Replay *(in progress)*

**Question:** Are realistic retrieval distractors harder than random distractors?

**Goal:** Controlled random distractors worked (0.13A). Now test with distractors
from the actual retriever (chain-set top-K results).

**Status:** Code implemented. Realistic +1 experiment launched. Remaining
configs prepared (+2, +4, +8, +16, +32). TopK cap experiments (top4, top8,
top16, top32, top64) also prepared.

**Decision rules:**
- If realistic +8 beats core-only strongly → continue selector/ranking optimization
- If random +8 works but realistic +8 fails → the problem is distractor quality, train on hard negatives
- If realistic replay works but actual retrieved path fails → investigate wiring/path bug
- If all realistic bounded-memory paths fail → implement slot-wise memory reader

---

*Last updated: 2026-06-18*
