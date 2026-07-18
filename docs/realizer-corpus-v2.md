# NEXUS Realizer corpus v2 — large representative PL/EN dataset

**Status:** `REALIZER_CORPUS_V2_READY`

**Scope:** data acquisition, normalization, leakage prevention and frozen splits.

**Training:** not launched.

## Outcome

The previous repository-only search found just 69 unused two-claim
compositions. That is useful as an integrity smoke test, but it is not a
representative language-realization sample. Corpus v2 therefore uses pinned,
licensed, human-authored public QA sources and never expands one source record
into multiple training records.

| Split | Records | Intended use |
|---|---:|---|
| Train | 147,154 | parameter updates only |
| Validation | 3,660 | checkpoint selection and stopping only |
| Test | 12,126 | frozen final evaluation only |
| **Total** | **162,940** | |

The materialized dataset hash is
`b19deb25c586ca5f82fd318e68243bd46f1a68c9eef442329e5e2f80e20ea621`.
Raw and normalized corpora are intentionally external to git. The repository
stores the complete acquisition recipe, source hashes, contracts and summary.

## Training composition

| Dimension | Count | Share of train |
|---|---:|---:|
| English | 86,665 | 58.89% |
| Polish | 60,489 | 41.11% |
| Multi-hop composition | 75,737 | 51.47% |
| Extractive realization | 50,076 | 34.03% |
| Comparison | 10,928 | 7.43% |
| Abstention | 10,413 | 7.08% |

The operator is metadata supplied by verified source annotations. It is not a
label that the Realizer is expected to rediscover from unstructured text.

## Sources and roles

| Source | Language | Native role | Project role | License |
|---|---|---|---|---|
| [HotpotQA](https://hotpotqa.github.io/) | EN | multi-hop/comparison QA | train + frozen evaluation | CC BY-SA 4.0 |
| [MuSiQue](https://github.com/stonybrooknlp/musique) | EN | connected 2–4-hop QA | train + frozen evaluation | CC BY 4.0 |
| [PoQuAD](https://huggingface.co/datasets/clarin-pl/poquad) | PL | answerable/unanswerable QA | train + frozen evaluation | CC BY-SA 4.0 |
| [PolQA](https://huggingface.co/datasets/ipipan/polqa) | PL | open-domain QA with judged passages | train + frozen evaluation | CC BY-SA 4.0 |
| [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) | EN | cross-document news QA | **test only** | ODC-BY 1.0 |

Exact revisions, download URLs and SHA-256 values are registered in
`training/realizer_corpus_v2_sources.json`. Licenses and attribution must stay
with every redistributed derivative.

## What “unique” means

Corpus v2 enforces all of the following:

1. one native source record produces at most one normalized record;
2. no generated questions or answers;
3. no translation or paraphrase expansion;
4. exact normalized questions are unique across all splits;
5. normalized question/answer text is checked against both committed Realizer
   v1 datasets and cannot be reused;
6. document groups are disjoint between train, validation and test;
7. normalized evidence fingerprints are disjoint between the splits;
8. only native training splits can enter NEXUS train;
9. test records are processed first, so collisions are removed from validation
   or train rather than from the test benchmark.

The builder rejected 24,241 train candidates due to document overlap with
evaluation, 122 due to exact evidence overlap and 77 due to duplicate text.
It also removed 5,272 validation candidates that touched the protected test
surface. These removals are expected evidence that the leakage guard is doing
real work.

## Record contract

Each JSONL record contains:

- the original question and natural answer;
- language and native source split;
- only gold supporting evidence, with stable fingerprints and locators;
- document group identifiers used for split isolation;
- a verified semantic operator: `extract`, `compose_path`, `compare` or
  `abstain`;
- an explicit contract that upstream NEXUS owns reasoning and the Realizer may
  not change facts;
- source revision, license, URL and artifact hash.

This is a surface-realization corpus, not a replacement knowledge base. During
an actual NEXUS run, the equivalent evidence and semantic plan must come from
graph traversal, path selection and verification—not from dataset labels.

## Reproduction

Install the optional acquisition dependency, acquire pinned artifacts and build
to a new external directory:

```bash
pip install -e '.[data]'
python benchmarks/acquire_realizer_corpus_v2.py \
  --output-root /external/nexus-realizer-corpus-v2-sources
python benchmarks/build_realizer_corpus_v2.py \
  --source-root /external/nexus-realizer-corpus-v2-sources \
  --output /external/nexus-realizer-corpus-v2
```

Both commands refuse to overwrite unexpected files. Acquisition verifies every
source hash. The builder verifies every normalized record and the complete
manifest before reporting `REALIZER_CORPUS_V2_READY`.

## Alignment with the NEXUS vision

The project has not abandoned its graph-first direction, but the earlier
comparison checkpoint was close to doing so conceptually: it learned a
`SAME`/`DIFFERENT` control label even though symbolic code had already decided
the relation. Longer training on that task would measure label imitation, not
better language realization.

Corpus v2 restores the intended boundary:

1. graph traversal or a gold source annotation identifies supporting facts;
2. upstream logic creates a verified semantic plan;
3. the Realizer receives only that plan and evidence;
4. the Realizer produces natural PL/EN wording;
5. a verifier checks factual support and fails closed.

Neural realization is worth promoting only if it improves wording or
instruction adherence over deterministic Pointer/Copy and registered-template
baselines while preserving grounding. It must not receive authority to invent
facts, choose graph paths or silently override an upstream plan.

## Honest coverage limits

This corpus is representative of the next factual Realizer experiment, not of
all future NEXUS use cases. It covers two languages, extraction, abstention,
comparison, connected multi-hop and cross-document news. It is still weighted
toward short factual answers and Wikipedia-style evidence. Long explanations,
causal diagnostics, project-specific enterprise graphs and scientific prose
remain underrepresented.

These gaps must be measured as separate frozen evaluation slices. They must not
be filled by translating or paraphrasing existing records. A later source can
be admitted only with clear licensing, native records, supporting-fact
annotations and the same document-disjoint split policy.

## Conditions before the next pilot

Do not pass this dataset directly to the historical 256-byte serializer. Before
training:

1. implement a corpus-v2 serializer that preserves evidence boundaries,
   semantic operators, language and abstention;
2. define a token budget from measured length percentiles; use truncation only
   if it cannot remove gold evidence;
3. evaluate deterministic Pointer/Copy and template baselines on the new
   validation split;
4. run an untrained neural baseline and record it without making it a blocking
   data-integrity gate;
5. run a one-epoch pilot, then at most three epochs with generation-aware early
   stopping—never default to 50 epochs;
6. require per-language and per-operator grounding, answer F1, semantic
   similarity, abstention precision/recall, EOS, empty-output and repetition
   metrics;
7. keep the 12,126-record test split sealed until a checkpoint has been selected.

The first four preparation conditions are now implemented by
[`AnswerPlan v1`](realizer-answer-plan-v1.md): resolved facts and provenance are
compiled one-to-one, a train-only lossless tokenizer has been fitted, lengths
have been audited, deterministic baselines are frozen and the test split remains
sealed. The abstention gap has since been resolved by moving 344 complete,
deterministically selected document groups from train into a tokenizer-blind
holdout: 2,408 records, including 500 abstentions, with no duplication. Full
training remains blocked for a different reason: bounded neural pilots still
fail EOS, exact-match and token-F1 generation gates. The representative pilot
must not run until the 2,048-record stage passes.

## Related documents

- [Corpus Coverage Roadmap](corpus-coverage-roadmap.md) — planned data source
  additions for long explanations, causal reasoning, scientific prose,
  enterprise structured data and dialogue
- [Realizer AnswerPlan v1](realizer-answer-plan-v1.md) — data preparation,
  pilot protocol and test-seal policy
- [Test Evaluation Protocol](test-evaluation-protocol.md) — one-time test
  evaluation process with precondition gates, JSON schema and re-sealing
  rules
