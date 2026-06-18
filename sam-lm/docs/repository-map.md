# Repository Map

Where everything is and what it does.

**Updated: 2026-06-18**

## Top-level structure

```
SAM-architecture-research/       ← Repository root
├── README.md                    ← High-level project entry point
├── sam-lm/                      ← The actual project code
│   ├── README.md                ← Practical setup & run instructions
│   ├── docs/                    ← Documentation (you are here)
│   ├── sam/                     ← Source code
│   │   ├── model/               ← Model architecture
│   │   ├── training/            ← Training scripts
│   │   ├── eval/                ← Evaluation & diagnostics
│   │   ├── data/                ← Dataset generation & loading
│   │   └── utils/               ← Shared utilities
│   ├── configs/                 ← ~90+ YAML configuration files
│   ├── experiments/             ← Experiment reports and output
│   ├── tests/                   ← Unit tests
│   ├── data/                    ← Datasets (generated, not committed)
│   └── run_logs/                ← Training logs
```

## `sam/model/` — Core architecture

| File | What it does | Status |
|------|-------------|--------|
| `sam_core.py` | Main `SamModel` class. Ties together transformer, memory, gate, retriever, noise modes. | Core implementation |
| `product_key_memory.py` | Product-key memory with codebooks, slot values, subkey addressing. | Core implementation |
| `slot_selector.py` | Learned 3-layer MLP predictor. Scores candidate slots for relevance. | Experimental |
| `transformer.py` | Decoder-only transformer. RMSNorm, SwiGLU feed-forward, causal attention. | Core implementation |
| `dense_transformer.py` | Baseline dense transformer (no memory). Same architecture, same params. | Baseline |
| `memory_utils.py` | Helpers for memory layer indices, memory block insertion. | Utility |

## `sam/training/` — Training infrastructure

| File | What it does | Status |
|------|-------------|--------|
| `train_sam.py` | Main SAM training loop. All memory modes, checkpointing, evaluation. | Core implementation |
| `train_retrieval.py` | Retrieval model training. Dual encoder, chain-set BCE, retrieval evaluation. | Core implementation |
| `train_dense.py` | Dense baseline training (no memory). | Baseline |
| `optimizers.py` | AdamW configuration, cosine warmup schedule. | Utility |

## `sam/eval/` — Evaluation and diagnostics

| File | What it does | Status |
|------|-------------|--------|
| `metrics.py` | Accuracy calculations (overall, by hop, recall@K). | Core implementation |
| `evaluate.py` | Load checkpoints, run evaluation over val/test sets. | Core implementation |
| `analyze_required_set_retrieval.py` | all_required@K, any_required@K diagnostics. | Diagnostic |
| `compare_retriever_interfaces.py` | Compare dual encoder vs chain-set vs oracle filter. | Diagnostic |
| `inspect_dataset.py` | Print dataset examples for debugging. | Diagnostic |
| `inspect_slots.py` | Slot content inspection tool. | Diagnostic |

## `sam/data/` — Data pipeline

| File | What it does | Status |
|------|-------------|--------|
| `synthetic_facts.py` | Generate synthetic multi-hop QA dataset from templates. | Core implementation |
| `dataset.py` | `QADataset` class, tokenization, collation, slot tracking. | Core implementation |

## `sam/utils/` — Utilities

| File | What it does |
|------|-------------|
| `config.py` | YAML config loading, model config classes. |
| `model_helpers.py` | Parameter counting, layer setup. |

## `configs/` — Experiment configurations

~90+ YAML files. Key groups:

| Group | Prefix | Description |
|-------|--------|-------------|
| SAM baselines | `sam_tiny_*.yaml` | Core SAM configs (dense variant) |
| Retriever training | `train_retriever_*.yaml` | Dual encoder and chain-set retriever configs |
| Noise experiments | `sam_noise_*.yaml` | Controlled noisy memory configs (0.13A) |
| Realistic replay | `sam_noise_realistic_*.yaml` | Realistic distractor configs (0.13B) |
| Chain retrieval | `sam_chain_*.yaml` | Chain-set retrieval topK variants |
| Forced gate | `*forced_gate*.yaml` | Forced gate configs |
| Oracle baselines | `sam_oracle_*.yaml` | Oracle text and memory baselines |

Configs follow this pattern:
```
configs/
  sam_tiny_dense.yaml               ← Main SAM config
  sam_noise_oracle_plus_0_dense.yaml ← 0.13A: 0 distractors
  sam_noise_oracle_plus_1_dense.yaml ← 0.13A: 1 distractor
  ...
  train_retriever_dense_dual_encoder.yaml
  train_retriever_dense_chain_set_bce.yaml
```

## `experiments/` — Reports and outputs

Reports (markdown):
```
experiments/
  diagnosis_report.md                              ← Early debugging
  experiment_0_5_report.md                         ← Dense dataset
  experiment_0_6_final_report.md                   ← Validation
  experiment_0_10_report.md                        ← Required-set retrieval
  experiment_0_11_report.md                        ← Chain-set retrieval
  experiment_0_12_report.md                        ← Slot selector
  experiment_0_13A_noisy_memory_report.md          ← Noise tolerance
```

Output directories (~20KB each, organized by experiment):
```
experiments/
  exp_0_5/           ← Dense dataset experiment outputs
  exp_0_6/           ← Validation experiment outputs
  exp_0_10/          ← Required-set experiment outputs
  exp_0_11/          ← Chain retrieval experiment outputs
  exp_0_12/          ← Selector experiment outputs
  exp_0_13/          ← Noise tolerance experiment outputs
  debug/             ← Shared debug/diagnostic outputs
```

## `tests/` — Unit tests

```
tests/
  test_sam_model.py         ← SAM model tests
  test_sam_training.py      ← Training infrastructure tests
  test_product_key_memory.py ← PKM retrieval tests
  test_retriever.py         ← Retriever tests
  conftest.py               ← Test fixtures
```

Run with: `pytest -q` (38 tests, all passing as of 2026-06-18)

## Key files for quick orientation

If you're new to the codebase, read these files in order:

1. `sam/config.py` — understand the config structure
2. `sam/data/synthetic_facts.py` — understand the dataset
3. `sam/model/sam_core.py` — the main model class (start from `forward()`)
4. `sam/model/product_key_memory.py` — the memory retrieval mechanism
5. `sam/model/transformer.py` — the transformer blocks
6. `sam/training/train_sam.py` — how training works
7. `sam/eval/metrics.py` — what metrics are computed
8. `configs/sam_tiny_dense.yaml` — the main SAM configuration

## What is committed vs generated

| Committed to git | Generated locally |
|-----------------|-------------------|
| All `.py` source files | `data/synthetic_dense/` (run `python -m sam.data.synthetic_facts`) |
| All `.yaml` configs | `experiments/exp_*/` outputs |
| All `.md` reports | `run_logs/` log files |
| All tests | `docs/` (generated documentation) |
| All docs | |

---

*Last updated: 2026-06-18*
