# Artifact governance

Large evaluation JSON, checkpoints, and training corpora should not grow
unbounded in ordinary Git history.

## Policy

1. **Code + small fixtures** stay in Git.
2. **Primary evidence packages** (≥2 MB JSON, model weights, corpora) belong in
   Git LFS **or** an immutable release asset referenced by SHA-256 from
   `benchmarks/results/evidence_manifest_v1.json`.
3. Metadata-only dataset rebinding is **forbidden**. Regenerate from the
   generating checkout via `benchmarks/regenerate_evidence_identity.py` and
   `benchmarks/run_phase4_arms.py`.
4. Historical invalid tooling: `benchmarks/_rebind_nexus_dataset.py` exits
   non-zero and must not be used.

## LFS patterns (recommended)

Do **not** enable these filters until `git lfs install` is adopted for all
contributors. Suggested patterns for a future `.gitattributes` migration:

```
benchmarks/results/phase4_*.json filter=lfs diff=lfs merge=lfs -text
models/**/*.bin filter=lfs diff=lfs merge=lfs -text
models/**/*.safetensors filter=lfs diff=lfs merge=lfs -text
training/**/*.jsonl filter=lfs diff=lfs merge=lfs -text
```

Until LFS is enabled repository-wide, new large arms should be published as
release assets and linked from the evidence manifest.

## Clean reproduction

```bash
docker build -t nexus-eval .
docker run --rm nexus-eval
python benchmarks/regenerate_evidence_identity.py
```
