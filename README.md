# SAM-architecture-research

**SAM — Sparse Associative Memory Language Model**

*Experimental research project. Not production code. Not a validated architecture.*

## What SAM means in this repository

SAM explores whether useful language understanding, question answering, and
multi-hop reasoning can be built with **less dependence on huge dense weights**
and **repeated RAM/VRAM bandwidth**.

Instead of storing all knowledge inside dense neural network weights (like most
large language models do), SAM uses a **small active core** plus a **sparse,
selectively-accessed memory** that lives in RAM. The core stays small and
constant-cost per token, while knowledge capacity scales through the memory
bank.

**SAM is not "just a small LLM with RAG."** It is a different architecture
hypothesis: what if knowledge and computation are fundamentally separated?

## Current research status

| Area | Status |
|------|--------|
| Oracle (perfect) memory | ✅ Confirmed — 100% accuracy on multi-hop QA |
| Controlled noisy memory tolerance | ✅ Confirmed — tolerates up to +8 random distractors |
| Retrieval (finding correct slots) | ✅ Chain-set retriever works (100% all_required@32) |
| Learned slot selection | ⚠️ Precision bottleneck (~50%) |
| Realistic retrieval end-to-end | ❌ Not yet validated |
| Scaling to larger models/datasets | ❌ Not yet evaluated |
| CPU efficiency vs dense baselines | ❌ Not yet measured |

## Quick start

```bash
cd sam-lm
pip install -r requirements.txt
python -m sam.data.synthetic_facts --output data/synthetic --train 20000 --val 1000 --test 1000 --seed 42
python -m sam.training.train_dense --config configs/dense_tiny.yaml
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny.yaml
pytest -q
```

See [sam-lm/README.md](sam-lm/README.md) for detailed instructions.

## Key experimental results

| Experiment | Finding | Overall accuracy |
|-----------|---------|-----------------|
| Core-only baseline | SAM without memory | 68.74% |
| Oracle memory (perfect retrieval) | SAM CAN use memory for reasoning | 99.87% → 100.00% |
| Tracked noisy memory (+1 distractor) | SAM does NOT collapse with one distractor | 99.82% |
| Tracked noisy memory (+8 distractors) | SAM tolerates substantial noise | 91.58% |
| Tracked noisy memory (+16 distractors) | 3-hop reasoning collapses | 75.42% (3-hop: 39%) |
| Chain-set retrieval (all_required@32) | Retriever finds all required slots | 100% coverage |
| Learned slot selector | Finds required slots but picks distractors | Precision ~50% |

## Current next step

**Experiment 0.13B** — Realistic Retrieval Distractor Replay.

Controlled random distractors worked (0.13A). Now test whether realistic
retrieval distractors (hard negatives from actual retrieval) are harder, and
rerun non-oracle baselines after critical padding-bug fixes.

## Documentation

Full documentation is in [sam-lm/docs/](sam-lm/docs/):

- [Getting started](sam-lm/docs/getting-started.md)
- [Thesis explanation](sam-lm/docs/thesis.md)
- [Architecture](sam-lm/docs/architecture.md)
- [Glossary](sam-lm/docs/glossary.md)
- [Experiment history](sam-lm/docs/experiments.md)
- [Experiment 0.13A — Noisy Memory](sam-lm/docs/experiment-0-13a-noisy-memory.md)
- [Current research status](sam-lm/docs/current-status.md)
- [Roadmap](sam-lm/docs/roadmap.md)
- [Glossary](sam-lm/docs/glossary.md)
- [Repository map](sam-lm/docs/repository-map.md)
- [Troubleshooting](sam-lm/docs/troubleshooting.md)

## Warnings

- This is **experimental research**, not production software.
- SAM has not been validated at scale.
- SAM does not currently beat GPT, DeepSeek, or any production LLM.
- All results are on synthetic, small-scale datasets.
- The architecture is a work in progress — many pieces may change.

---

*Last updated: 2026-06-18*
