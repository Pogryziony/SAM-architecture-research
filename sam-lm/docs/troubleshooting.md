# Troubleshooting

Common issues when working with the SAM codebase.

## Setup issues

### "No module named 'sam'"

You need to run from the `sam-lm/` directory or set `PYTHONPATH`:

```bash
cd sam-lm
python -m sam.training.train_sam ...
```

Or:
```bash
export PYTHONPATH=/path/to/SAM-architecture-research/sam-lm:$PYTHONPATH
```

### "I don't have a GPU"

The code has no GPU requirement. It runs on CPU if PyTorch is installed (CPU
version). Training is slower but works:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### PyTorch version mismatch

Recommended: PyTorch >= 2.0. The code uses features like `torch.compile`
(optional) and recent attention APIs. If you see import errors, update:
```bash
pip install --upgrade torch
```

## Dataset issues

### "data/synthetic_dense not found"

The dataset is not committed. You must generate it:
```bash
cd sam-lm
python -m sam.data.synthetic_facts
```

This creates `data/synthetic_dense/` with train/val/test splits.

### "Dataset has wrong number of slots"

If you changed `num_slots` in the config, regenerate:
```bash
rm -rf data/synthetic_dense
python -m sam.data.synthetic_facts
```

The generator respects the config's `num_slots` setting.

## Training issues

### PC freezes or becomes unresponsive during training

The current model is small (~16M parameters). It should not cause memory issues
on any modern system (4GB+ RAM).

If you experience freezing:
1. Check `nvidia-smi` if using GPU — ensure you're not accidentally running
   multiple training processes
2. Reduce `batch_size` in the config YAML
3. Set `num_workers: 0` in the config to avoid data-loader fork issues
4. On low-RAM systems, reduce `max_seq_len` in the config

### Training seems stuck in a loop

Check if `best_val_loss` is properly decreasing. If it oscillates wildly:
- The learning rate may be too high (reduce `lr` in config)
- The batch size may be too small for stable gradients
- Check `run_logs/` for anomaly warning messages

### "CUDA out of memory"

The model is designed to not need a GPU. If you're running on CUDA and getting
OOM:
```yaml
# In your config
device: cpu
```
Or set environment variable:
```bash
CUDA_VISIBLE_DEVICES="" python -m sam.training.train_sam ...
```

## Test issues

### Tests fail (non-passing)

First, ensure you're in the right directory:
```bash
cd sam-lm
pytest -v
```

Current status: 38 tests passing (as of 2026-06-18).

If tests fail:
1. Check `git status` — uncommitted changes may have broken something
2. Run `pytest -v --tb=short` for readable tracebacks
3. Run a specific failing test: `pytest -v tests/test_sam_model.py::test_name`

### Import errors in tests

Make sure you've installed the package in development mode:
```bash
cd sam-lm
pip install -e .
```

## Result interpretation issues

### "My results don't match the experiment reports"

This is expected if conditions differ:

1. **Different random seed** — results vary ±1-2% between seeds. Use the same
   seed as the experiment config if exact reproduction matters.

2. **Post-bugfix code** — Results from experiments 0.11 and earlier may not
   match after the padding bug fix (slot `-1` no longer clamped to 0). The
   current codebase is the correct baseline.

3. **Different PyTorch/NumPy versions** — Minor floating-point differences
   can accumulate. Match versions if reproducibility is critical.

4. **Incomplete training** — Check `run_logs/` to ensure training reached
   the expected number of steps. Early stopping may cut training short.

### "I ran a config and get core_only accuracy"

Common causes:

1. **Using retrieved_memory mode without a trained retriever** — The retriever
   checkpoint path in the config must match a valid checkpoint. If no retriever
   exists, train one first with `train_retrieval.py`.

2. **Padding bug from old code** — If you're running old checkpoints on new
   code, results may differ. Retrain with the current codebase.

3. **Retrieval mode mismatch** — `retrieved_memory` uses internal PKM lookup.
   `retrieved_memory_external_text_query` uses external retriever. Make sure
   the config matches your intent.

## Bug-related issues

### The "padding bug" (slot -1 clamped to 0)

**Symptom:** Memory path behaves as if random noise is injected.

**Cause:** In `product_key_memory.py`, `-1` padding slots were incorrectly
clamped to slot `0`, causing slot 0's value to be injected into every example
that had fewer than `topK` retrieved candidates.

**Fix:** Applied after Experiment 0.12. The current code correctly handles -1
padding.

**Affected:** Results from 0.11 (non-oracle baselines) and 0.12 (selector
training) may have been contaminated by this bug. Rerun these on the fixed
codebase.

### "Stale checkpoint from before bugfix"

If you have checkpoints saved before the padding bug was fixed, retrain from
scratch. Stale checkpoints will give misleading results.

Delete old checkpoints:
```bash
rm -rf experiments/exp_0_11/*.pt experiments/exp_0_12/*.pt
```
Then rerun the experiment.

## How to rerun cleanly

To ensure a clean state for rerunning an experiment:

```bash
cd sam-lm

# 1. Delete old experiment outputs
rm -rf experiments/exp_0_XX/

# 2. Check git status (ensure code is clean)
git status

# 3. Regenerate data if needed
rm -rf data/synthetic_dense
python -m sam.data.synthetic_facts

# 4. Run the experiment
python -m sam.training.train_sam --mode retrieved_memory_external_text_query --config configs/your_config.yaml
```

---

*Last updated: 2026-06-18*
