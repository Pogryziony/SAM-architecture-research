# NEXUS Stage 2 regression diagnosis

## Reproduction

The PR head `6fe2f350bd35b7f764ea8cec2ef2f91bcb3942ea` and merge commit
`5331a3b7f585a29ec7ccbd056b96a0bfd3423680` resolve to the same tree:
`4816e3c7f67982c880531e201fc25f25f15c0f53`. Both were run with Python 3.12.10,
PyTorch 2.12.0+cpu, the committed `training/stage2_baseline_v1.json`, and the
following protocol:

```text
PYTHONHASHSEED=0 python benchmarks/run_stage2_stage3.py --stage 2 --limit 30 --baseline training/stage2_baseline_v1.json
```

The ordered cases are `q001` through `q030`. Both commits reproduce relevance
`0.7833` and `PASS`. The previously reported `0.5000` result used `--limit 181`;
it was not the registered 30-case protocol. The baseline and source tree were
not different. The broad run is a separate, recorded untrained-model baseline,
not evidence that the 30-case protocol regressed.

## Broad baseline and failure taxonomy

A fresh 181-case run after the deterministic ordering fix produced relevance
`0.6934`, accuracy mean `0.0788`, accuracy delta `-0.0906`, naturalness `57.978`,
and hallucination `0.2300`. Of 181 cases, 161 received zero accuracy:

| Earliest failure | Count | Percent of zero-accuracy cases | Evidence |
| --- | ---: | ---: | --- |
| Correct fact missing from graph (`no_graph_paths`) | 58 | 36.0% | runner failure category and zero traversal paths |
| Realizer/verifier ignored available evidence (`verifier_failed`) | 29 | 18.0% | runner failure category with evidence recorded |
| Wrong entity/relation or answer selection with evidence available | 74 | 46.0% | non-empty evidence and non-empty answer; representative cases q013, q043, q140 |

The evaluator's exact/fuzzy fact score also rejects some partially correct
answers (for example q009 returns `recall@8 30.71%`). These are retained as
baseline failures and are not moved into training. The frozen question set and
gold answers were not changed.

## Implemented corrections

* Stage 2 artifacts now record commit/tree identity, protocol, ordered case IDs,
  effective configuration, dataset hash when supplied, complete evidence, and
  per-case diagnostics.
* Canonical Stage 2 hashes exclude timestamps, paths, and latency, while the
  serialized-file hash is retained separately. Graph trigram lookup and
  evidence metric traversal now use sorted iteration and deterministic tie
  breaks.
* `_collect_numbers`, `_collect_numbers_by_metric`, and `_collect_neighbor_key_findings`
  now iterate metrics dicts and sort neighbor facts deterministically. This
  eliminates the last PYTHONHASHSEED-dependent evidence ordering.
* Runs with hash seeds 0, 1, and 42 now have identical predictions,
  metrics, and canonical content hash:
  `908c7a8f4696ee8e2bd69de47f8e2d8ebbf03c9cfdb2ddf609e3906e4812e3b4`.
* Readiness distinguishes blocking data/evidence integrity checks from the
  untrained answer-quality baseline. The latter remains recorded for the
  post-training comparison; no threshold was lowered.
* Readiness instantiates the configured model and records the actual parameter
  count (`2,770,752`), rather than the prior approximation (`2,779,200`). It
  names canonical payload and serialized-file hashes separately.

## Recommendation

The corrected pre-training integrity readiness result is `READY_FOR_TRAINING`
with no blocking checks, using the existing 8,282-record/7,127-pair dataset and
oracle artifact. The 30-case answer-quality baseline remains `PASS` at relevance
`0.7833`, while the 181-case untrained baseline is recorded and remains below
its answer-quality thresholds. Full training was intentionally **not** launched
in this diagnosis; post-training evaluation must compare against both recorded
baselines and use the frozen validation/test protocols.
