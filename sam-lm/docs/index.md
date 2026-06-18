# SAM Documentation

Welcome to the SAM (Sparse Associative Memory) project documentation.

## What to read first

If you're new to the project, we recommend reading in this order:

1. **[Thesis](thesis.md)** — Why SAM exists and what problem it tries to solve
2. **[Architecture](architecture.md)** — How SAM works technically
3. **[Concepts and Terms](glossary.md)** — Key terms explained
4. **[Current Status](current-status.md)** — What is confirmed and what is not
5. **[Experiments](experiments.md)** — What experiments have been run and what they found

For practical instructions, start with:

1. **[Getting Started](getting-started.md)** — Setup and first run
2. **[Repository Map](repository-map.md)** — Where each file lives
3. **[Troubleshooting](troubleshooting.md)** — Common problems and solutions

## Project maturity

**SAM is experimental research. It is not validated, not production-ready,
and not yet proven to work at scale.**

Current experiments use:
- A small synthetic dataset (19,000 examples, 1,650 memory slots)
- A tiny model (~16M parameters)
- CPU-only training
- Template-generated multi-hop QA tasks

The project is in an active exploration phase. Things may change, break, or be
redesigned. Negative results are expected and documented honestly.

## Confirmed

These things have been demonstrated on the current small-scale setup:

- SAM's core CAN use memory for reasoning: oracle memory achieves 100% accuracy
  on multi-hop QA, compared to 68.74% without memory.
- SAM tolerates controlled noisy memory: up to +8 random distractors, overall
  accuracy stays above 90%.
- One distractor does NOT collapse SAM: 99.82% accuracy with +1 distractor.
- Chain-set retrieval can find all required slots: all_required@32 = 100%.
- The product-key memory addressing mechanism works technically.

## Not yet confirmed

These things have NOT been demonstrated:

- Realistic non-oracle retrieved memory beats the core-only baseline
  (currently identical at 68.74%).
- Realistic hard distractors (from actual retrieval) are tolerable.
- The learned slot selector path is correct after recent bug fixes.
- SAM scales to larger models, larger memory banks, or harder datasets.
- SAM is more efficient than dense baselines in measured CPU/RAM terms.

## Known open questions

- Are realistic retrieval distractors qualitatively harder than random distractors?
- Does the selector's ~50% precision cause systematic failures beyond what
  distractor count alone would predict?
- Will the padding-bug fix change previous non-oracle baseline results?
- Is latent-vector averaging sufficient, or do we need a slot-wise memory reader?

## Next steps

See the **[Roadmap](roadmap.md)** for current and planned experiments.

The immediate next step is **Experiment 0.13B** — Realistic Retrieval Distractor
Replay and Post-Fix Non-Oracle Revalidation.

## Documentation pages

| Page | Description |
|------|-------------|
| [Getting Started](getting-started.md) | Setup and first run |
| [Thesis](thesis.md) | Why SAM exists |
| [Architecture](architecture.md) | How SAM works |
| [Concepts & Terms](glossary.md) | Key terms explained |
| [Experiments](experiments.md) | Experiment timeline and findings |
| [Experiment 0.13A](experiment-0-13a-noisy-memory.md) | Controlled noisy memory deep-dive |
| [Current Status](current-status.md) | Confirmed vs not-yet-confirmed |
| [Roadmap](roadmap.md) | Next research steps |
| [Glossary](glossary.md) | Alphabetical term definitions |
| [Repository Map](repository-map.md) | Codebase navigation |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |
| [Experiment Index](experiment-index.md) | Quick reference for all experiment reports |

---

*Last updated: 2026-06-18*
