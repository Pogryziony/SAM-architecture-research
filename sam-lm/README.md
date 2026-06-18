# SAM-LM: Sparse Associative Memory Language Model

Research proof-of-concept testing whether decoupling knowledge (sparse product-key memory)
from computation (small dense core) enables multi-hop reasoning on a CPU memory budget.

**Status**: Through Experiment 0.13A. Oracle memory validated. Controlled noisy memory
tolerance confirmed. Realistic retrieval end-to-end NOT yet validated. Not production code.

## Thesis

Dense LLMs store reasoning and knowledge in the same streamed weights. SAM keeps reasoning
in a small dense core and stores knowledge in a sparse associative memory retrieved by lookup.
Per-token dense compute remains small and constant, while knowledge capacity scales through
RAM-backed memory slots.

SAM is **not "just a small LLM with RAG"** — memory is integrated into the model's internal
computation via learned gating, not simply prepended to the input text.

## Key Experimental Results

| Experiment | Finding | Key Metric |
|-----------|---------|-----------|
| **0.6 — Dense baseline** | SAM core ≈ dense transformer at equal params | 68.74% overall |
| **0.6 — Oracle memory** | SAM CAN use clean memory for reasoning | 99.87% → 100% |
| **0.10 — Required-set retrieval** | Dual encoder misses multi-hop required slots | all_required@64 = 27% |
| **0.11 — Chain-aware retrieval** | BCE chain-set retriever finds all required slots | all_required@32 = 100% |
| **0.12 — Slot selection** | Selector recall 96.6%, precision 50% — bottleneck | 0% improvement vs core_only |
| **0.13A — Controlled noise** | SAM tolerates +8 random distractors | 91.58% overall |
| **0.13A — Collapse point** | 3-hop collapses between +8 and +16 distractors | 79.3% → 39.0% |

### Core findings:
- **SAM oracle memory works** — 100% on multi-hop QA with clean memory
- **SAM tolerates controlled noise** — does NOT collapse with 1-2 distractors
- **Retrieval solved** — chain-set BCE finds 100% of required slots at top32
- **Selection is the bottleneck** — learned selector finds slots (96.6% recall) but picks distractors (50% precision)

## Quick Start

```bash
pip install -r requirements.txt

# Generate dataset
python -m sam.data.synthetic_facts --output data/synthetic_dense --train 19000 --val 3800 --test 3800 --seed 42

# Run tests
pytest -q

# Core-only baseline
python -m sam.training.train_sam --mode core_only --config configs/sam_tiny_dense.yaml

# Oracle memory (upper bound)
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny_dense.yaml
```

## Key experiments to reproduce

```bash
# Controlled noisy memory (0.13A)
python -m sam.training.train_sam --mode retrieved_memory_external_text_query \
  --config configs/sam_noise_oracle_plus_1_dense.yaml

# Chain-set retrieval
python -m sam.training.train_retrieval --config configs/retrieval_chain_set_bce_dense.yaml

# Learned selector
python -m sam.training.train_sam --mode retrieved_memory_external_text_query \
  --config configs/sam_chain_learned_selector_dense.yaml
```

## Diagnostics

```bash
# Inspect dataset
python -m sam.eval.inspect_dataset --data-dir data/synthetic_dense --limit 20

# Inspect memory slots
python -m sam.eval.inspect_slots --data-dir data/synthetic_dense --limit 20

# Required-set retrieval analysis
python -m sam.eval.analyze_required_set_retrieval \
  --retriever experiments/exp_0_11/chain_set_bce/checkpoint.pt \
  --data-dir data/synthetic_dense
```

## Documentation

Full documentation is in `docs/`:
- [Getting Started](docs/getting-started.md)
- [Thesis](docs/thesis.md)
- [Architecture](docs/architecture.md)
- [Experiments](docs/experiments.md)
- [Current Status](docs/current-status.md)
- [Roadmap](docs/roadmap.md)
- [Glossary](docs/glossary.md)
- [Repository Map](docs/repository-map.md)

## Experiment Reports

- `experiments/diagnosis_report.md` — Experiment 0 pipeline fixes
- `experiments/experiment_0_5_report.md` — Retrieval solved (dense dataset)
- `experiments/experiment_0_6_final_report.md` — Full validation: oracle works, retrieval fails
- `experiments/experiment_0_10_report.md` — Required-set retrieval diagnostics
- `experiments/experiment_0_11_report.md` — Chain-aware retrieval (multi-positive BCE)
- `experiments/experiment_0_12_report.md` — Candidate selection and memory-use training
- `experiments/experiment_0_13A_noisy_memory_report.md` — Controlled noisy memory tolerance

---

*Last updated: 2026-06-18*
