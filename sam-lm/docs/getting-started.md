# Getting Started

Step-by-step guide to setting up and running SAM experiments.

## Prerequisites

- **Python 3.10 or later** (recommended; earlier 3.x versions may work)
- **pip** (Python package manager)
- **8+ GB RAM** (for training; 16 GB recommended)
- **Git** (optional, for cloning the repository)
- **Operating system**: Linux, macOS, or Windows. The code runs on CPU by default.

No GPU is required.

## 1. Clone the repository

```bash
git clone https://github.com/Pogryziony/SAM-architecture-research.git
cd SAM-architecture-research/sam-lm
```

## 2. Create a virtual environment (recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## 4. Generate the dataset

```bash
python -m sam.data.synthetic_facts \
  --output data/synthetic_dense \
  --train 19000 \
  --val 3800 \
  --test 3800 \
  --seed 42
```

This creates `data/synthetic_dense/` with:
- Training, validation, and test examples in JSONL format
- A fact pool of 1,000 fact chains across 1,650 memory slots
- A tokenizer with 853 vocabulary tokens

Check the data:
```bash
python -m sam.eval.inspect_dataset --data-dir data/synthetic_dense --limit 5
```

## 5. Run the tests

```bash
pytest -q
```

You should see: `38 passed`

If tests fail, see [Troubleshooting](troubleshooting.md).

## 6. Run your first experiment

### Core-only baseline

The simplest experiment — SAM with no memory at all:

```bash
python -m sam.training.train_sam --mode core_only --config configs/sam_tiny_dense.yaml
```

This should complete in about 10-15 minutes on a modern CPU. Output goes to
`experiments/exp_sam_dense/core_only/`.

Expected result: ~68.74% overall accuracy on the validation set.

### Oracle memory (the validation experiment)

SAM with perfect memory — all required slots injected without noise:

```bash
python -m sam.training.train_sam --mode oracle_memory --config configs/sam_tiny_dense.yaml
```

Expected result: ~99.87% overall accuracy. This proves the core CAN use memory
for reasoning.

## 7. Reproduce key experiments

### Controlled noisy memory (0.13A)

Test how much memory noise SAM can tolerate:

```bash
# 1 distractor (mild noise)
python -m sam.training.train_sam \
  --mode retrieved_memory_external_text_query \
  --config configs/sam_noise_oracle_plus_1_dense.yaml

# 8 distractors (substantial noise)
python -m sam.training.train_sam \
  --mode retrieved_memory_external_text_query \
  --config configs/sam_noise_oracle_plus_8_dense.yaml
```

Expected: +1 distractor ≈ 99.8%, +8 distractors ≈ 91.6%.

### Train a retriever

```bash
# Dual encoder (basic retriever for single-hop)
python -m sam.training.train_retrieval --config configs/retrieval_dual_encoder_dense.yaml

# Chain-set BCE (multi-hop capable retriever)
python -m sam.training.train_retrieval --config configs/retrieval_chain_set_bce_dense.yaml
```

### Run the learned selector

After training the chain-set retriever:

```bash
python -m sam.training.train_sam \
  --mode retrieved_memory_external_text_query \
  --config configs/sam_chain_learned_selector_dense.yaml
```

## 8. View results

Each experiment creates a directory under `experiments/` with:
- `checkpoint_best.pt` — model weights at best validation step
- `detailed_metrics.json` — accuracy, gate stats, memory diagnostics
- `detailed_predictions.jsonl` — per-example results with diagnostics
- `train.log` — training progress

Check results:
```bash
# Quick overview of latest run
cat experiments/exp_0_13/sam_noise_oracle_plus_1_dense/detailed_metrics.json

# Browse predictions
head -5 experiments/exp_0_13/sam_noise_oracle_plus_1_dense/detailed_predictions.jsonl
```

## 9. Read the documentation

Recommended reading order:
1. [Thesis](thesis.md) — why SAM exists
2. [Architecture](architecture.md) — how it works
3. [Experiments](experiments.md) — what has been tested
4. [Current Status](current-status.md) — what we know and don't know

## Next steps after your first run

- Read the [Experiment 0.13A deep-dive](experiment-0-13a-noisy-memory.md)
- Check the [Roadmap](roadmap.md) for planned experiments
- Try modifying configs to test your own hypotheses
- Read the [Architecture](architecture.md) to understand the code

## Common first-time issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'sam'` | Run from the `sam-lm/` directory |
| `FileNotFoundError: data/synthetic_dense/` | Generate the dataset first (step 4) |
| Tests fail | Check [Troubleshooting](troubleshooting.md) |
| Training is slow | Expected on CPU; reduce `num_epochs` in config for testing |
| Out of memory | Reduce `batch_size` in config; close other programs |

---

*Last updated: 2026-06-18*
