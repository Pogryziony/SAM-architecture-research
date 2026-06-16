# SAM-LM: Sparse Associative Memory Language Model

Research proof-of-concept testing whether decoupling knowledge (sparse product-key memory)
from computation (small dense core) improves multi-hop reasoning over external facts.

**Status**: Experiments 0–0.4 complete. SAM architecture validated (core CAN use memory). Retrieval is the bottleneck. Not production code.

## Thesis

Dense LLMs store reasoning and knowledge in the same streamed weights. SAM keeps reasoning
in a small dense core and stores knowledge in a sparse associative memory retrieved by lookup.
Per-token dense compute remains small and constant, while knowledge capacity scales through
RAM-backed memory slots.

## Key Results (Experiments 0–0.4)

| Experiment | Finding |
|-----------|---------|
| **0.0 — Dense baseline** | 5.9% val accuracy (closed-book). Open-book: 100% |
| **0.1 — SAM memory modes** | Oracle latent: 8.3% (+1.8pp over core-only). Retrieved: 6.5% (no benefit) |
| **0.2 — Compact retrieval** | 16K PKM: 25.8% Rec@8. Oracle text: 100% overfit (core CAN use memory) |
| **0.3 — PKM diagnostics** | Candidate gen: SOLVED (100%). Ranking: 87.5% train but 29% val (generalization gap) |
| **0.4 — Retrieval baselines** | Linear classifier: 16.5% val Rec@8 (task IS learnable but data-starved) |
| **0.4 — Dataset fix** | Fact pool: 15K examples, 6.4/slot. Contrastive retriever: testing now |

### Core findings:
- **SAM oracle memory works** — 8.3% vs 6.5% core-only on val (Gate 2 PASS)
- **SAM core CAN use memory** — 100% overfit with oracle text injection
- **PKM mechanism validated** — 100% candidate inclusion with subkey loss
- **Retrieval is the bottleneck** — Gate 1 (Rec@8 >= 80%) not reached with any retriever

## Quick Start (Smoke Test)

```bash
pip install -r requirements.txt
python -m sam.data.synthetic_facts --output data/synthetic --train 20000 --val 1000 --test 1000 --seed 42
python -m sam.training.train_dense --config configs/dense_tiny.yaml
python -m sam.training.train_retrieval --config configs/retrieval_compact_16k_subkey_loss.yaml
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny.yaml
python -m sam.eval.evaluate --runs experiments/
```

## Full Experiment Pipeline

```bash
# Generate data (fact-pool approach: ~15K train, 6+ examples per slot)
python -m sam.data.synthetic_facts --output data/synthetic --train 20000 --val 1000 --test 1000 --seed 42

# Train all baselines
python -m sam.training.train_dense --config configs/dense_tiny.yaml
python -m sam.training.train_dense --config configs/dense_openbook.yaml

# Retrieval experiments
python -m sam.training.train_retrieval --config configs/retrieval_compact_16k_subkey_loss.yaml
python -m sam.training.train_retrieval --config configs/retrieval_classifier_compact_16k.yaml
python -m sam.training.train_retrieval --config configs/retrieval_contrastive.yaml

# SAM memory modes
python -m sam.training.train_sam --mode core_only --config configs/sam_tiny.yaml
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny.yaml
python -m sam.training.train_sam --mode oracle_text_memory --config configs/sam_tiny.yaml
python -m sam.training.train_sam --mode retrieved_memory --config configs/sam_tiny.yaml

# Evaluation
python -m sam.eval.evaluate --runs experiments/
```

## Diagnostic Tools

```bash
python -m sam.eval.inspect_dataset --data-dir data/synthetic --limit 20
python -m sam.eval.inspect_slots --data-dir data/synthetic --limit 20
python -m sam.eval.inspect_retrieval_split --data-dir data/synthetic
```

## Experiment Reports

- `experiments/diagnosis_report.md` — Experiment 0 pipeline fixes
- `experiments/experiment_0_2_report.md` — Compact retrieval + oracle text
- `experiments/experiment_0_3_report.md` — PKM diagnostics + subkey loss
- `experiments/experiment_0_4_report.md` — Retrieval baselines + dataset diagnosis
