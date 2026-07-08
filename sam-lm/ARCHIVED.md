# SAM Classic — ARCHIVED

This directory contains the original SAM (Sparse Associative Memory) experiments.
**This code is archived and no longer under active development.**

## Why archived?

The SAM architecture proved that:
1. A small reasoning core CAN use external memory (100% oracle accuracy)
2. Chain-set retrieval CAN find complete fact chains (100% all_required@32)
3. But realistic end-to-end retrieval FAILS — the selector bottleneck (50% precision) is a structural limitation of flat latent-vector memory

## What replaced it?

**NEXUS** (Non-Parametric Execution and Understanding System) — see `../nexus/` and `../README.md`.

NEXUS shifts from latent-vector memory to explicit graph memory:
- Knowledge = graph of entities, relations, sources (not floating-point vectors)
- Retrieval = graph traversal (not embedding similarity)
- Reasoning = path scoring + evidence building (not gated memory injection)

## What to keep from SAM

The key experimental findings that transfer to NEXUS are documented in `../ANALYSIS_AND_ROADMAP.md` §0.3.

## What NOT to use from SAM

- Do NOT use the PKM (product-key memory) for new work
- Do NOT use the dual encoder or chain-set retriever for new work
- Do NOT use the learned slot selector for new work
- Do NOT treat knowledge as latent vectors

The SAM code remains here for reference, reproducibility of published experiments, and as a record of the research arc.
