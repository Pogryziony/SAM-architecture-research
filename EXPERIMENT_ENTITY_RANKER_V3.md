# EXPERIMENT: Entity Ranker V3 — Genuinely Question-Conditioned CPU-Only Encoder/Ranker

**Pre-registered**: 2026-07-10
**Status**: Preregistration — no implementation yet
**Repository**: SAM-architecture-research
**Reference commit**: `e204a31552f5774fa291ea0f0b346c9b2c14a69e`

---

## Summary

Build a genuinely question-conditioned, CPU-only entity encoder/ranker that reaches **>= 65% entity recall@10** on the unchanged frozen 225-question test split, with K <= 10 enforced inside the ranking path, without relaxing any gates, changing frozen labels, or tuning on the test set.

This experiment explicitly replaces the current Stage 1C/1D feature-logistic and encoder rankers ("Stage 1E" is prohibited by protocol — this is a separately registered experiment, not a Stage 1E).

---

## Honest Current Reference (from Stage 1C C4 artifact)

| Metric | Value |
|---|---|
| Feature-logistic recall@1 | 9.45% |
| Feature-logistic recall@5 | 37.82% |
| Feature-logistic recall@10 | 53.45% |
| Feature-logistic precision@10 | 6.56% |
| Trivial baseline recall@10 | 35.64% |
| Intent accuracy | 90.67% |
| Inference p50 | 20.8 ms |
| RSS | 47.3 MB |
| Frozen gate | recall@10 >= 65% |
| K max | **10** (enforced in-path) |

---

## Critical Defects Found (Code Inspection at `e204a31`)

### Defect 1: Non-Question-Conditioned Entity Scoring (CRITICAL)

**Location**: `stack/encoder/model.py` lines 133, 174-177
**Impact**: The neural encoder ranker scores entities independently of the question, making it a trivial feature ranker in practice

**Root cause**: The entity scorer is a single linear layer over concatenated question and entity vectors:

```python
self.entity_scorer = nn.Linear(combined_dim + embed_dim, 1)  # line 133
# ...
pair_feats = torch.cat([combined_expanded, candidate_entity_feats], dim=-1)  # line 176
entity_scores = self.entity_scorer(pair_feats).squeeze(-1)  # line 177
```

This produces `score(q,e) = W_q·q + W_e·e + b`. For a fixed question, the question term `W_q·q` is constant across all candidates (because `combined_expanded = combined.unsqueeze(1).expand(-1, K, -1)`). Therefore the relative ranking of any two entities is independent of the question. The encoder ranker scores 0.0 on all metrics because the trained weights collapsed to produce near-identical scores for all candidates.

**Confirmed by artifact**: `stage1c_full_selection_log.json` shows encoder recall@10 = 0.0 while feature-logistic achieves 71.32% on validation.

**Fix**: Replace with genuine interaction model (one of):
- Bilinear: `score(q,e) = q^T W e`
- Projected dot-product: `score(q,e) = (P_q·q) · (P_e·e)` or cosine similarity
- MLP: `MLP([q, e, q * e, |q - e|, cosine(q,e)])`

**Required test**: Unit test proving that changing the question changes the relative ordering of the same two entities.

### Defect 2: Validation Denominator Leakage (CRITICAL)

**Location**: `stack/encoder/c2_c3.py` lines 274-276
**Impact**: Validation questions where gold entity is missing from the candidate pool are silently dropped from evaluation

```python
group = _group(str(row["id"]), str(row["question"]), row["entities"], builder(row["question"]), "validation_candidate_pipeline")
if group is not None: val_groups.append(group)  # line 276
```

`_group()` returns `None` when `positive_ids` are not a subset of `candidate_ids` (line 54-55). The validation split has 150 questions but rankers were evaluated on only 127 groups (as confirmed in `stage1c_full_selection_log.json`: `"candidate_groups": 127`). 23 questions with absent gold entities were silently excluded.

**Fix**:
- Every validation question remains in the denominator (150 questions, 182 gold entities)
- Missing gold candidate produces zero candidate/ranker recall for that question
- Baseline and all rankers use identical 150-question populations
- Report: total questions, total gold entities, absent-gold count, candidate recall ceiling, recall@1/5/10, precision@10
- Reject artifacts whose denominators differ

**Required test**: Prove a missing gold candidate cannot disappear from evaluation.

### Defect 3: Provenance Mismatch

**Location**: `benchmarks/results/stage1c_full_selection_log.json`
**Impact**: The selection artifact records source SHA `6c0e25fe7b642747ab2dd7d59e7a85fe411b1eef` which predates the commit containing the evaluated C2/C3 implementation (commit `87d31c1`)

The baseline artifact `baseline_val_20260710T172932Z.json` records SHA `6be28b130a17894a3fc5b707cd8a78936ba73fe4` — these SHA values are inconsistent with the actual code paths executed.

**Fix**:
- Clean git working tree required before calibration or final evaluation
- Evaluation SHA must contain the exact code being executed
- Model config, weights, selection artifact, graph metadata, split hashes, and source SHA must agree
- Fail before evaluation when the worktree is dirty
- Never overwrite historical artifacts

### Defect 4: Impoverished Entity Representation

**Location**: `stack/encoder/c2_c3.py` line 216
**Impact**: Entities are embedded using only their node ID with underscores replaced by spaces:

```python
desc = [x.replace("_", " ") for x in group["candidate_ids"]]  # line 216
```

And in `train_encoder` (line 198):
```python
p_feats = model.embed_entities([pos[0].replace("_", " ")], tokenizer)
n_feats = model.embed_entities([neg[0].replace("_", " ")], tokenizer)
```

This discards aliases, node types, descriptions, key findings, provenance, and typed-relation summaries.

**Fix**: Entity text must include:
- node ID
- canonical/display name
- node type
- aliases
- description
- key_finding
- source/provenance
- short typed-relation summary (neighbor types + counts)

### Defect 5: Broken Encoder Training Loop

**Location**: `stack/encoder/c2_c3.py` lines 175-204
**Impact**: The training loop uses only the first positive entity, first negative entity, two epochs, and no batching

```python
pos = [x for x in group["candidate_ids"] if x in positives]   # line 194
neg = [x for x in group["candidate_ids"] if x not in positives]  # line 195
# ...
p_feats = model.embed_entities([pos[0].replace("_", " ")], tokenizer)  # pos[0] only
n_feats = model.embed_entities([neg[0].replace("_", " ")], tokenizer)  # neg[0] only
```

**Fix**:
- All positive entities per question (not just `pos[0]`)
- Multiple hard negatives per question
- Pairwise margin/logistic loss or listwise softmax over all candidates
- Deterministic mini-batches (grouped by source, balanced)
- Source-balanced sampling (50% real / 50% synthetic per batch)
- Early stopping using full validation recall@10
- Validation recall@5 as tie-break
- Fixed random seeds (20260710 or new preregistered seed)
- CPU-only execution (torch.set_num_threads(1))
- Training curves and checkpoint metadata saved

### Defect 6: Fake Hard Negatives

**Location**: `stack/encoder/c2_c3.py` lines 93-96
**Impact**: Hard negatives are simply the first 15 non-gold IDs from the candidate pool (lexical pipeline order), not genuinely confusing candidates

```python
ordered_non_gt = [x for x in group["candidate_ids"] if x not in positive_set]
hard = ordered_non_gt[: max(0, hard_negative_k)]  # line 94
```

**Fix**: True hard negatives must be the highest-scoring incorrect candidates from:
- The lexical baseline (highest BM25/overlap scores that are not gold)
- The current feature ranker
- The first encoder training pass
- Same-type negatives (same node type as gold, but wrong entity)
- Same-source negatives (same experiment/document, but wrong entity)
- High node-degree negatives (confusable hub nodes)
- Alias-confusable negatives (entities with overlapping aliases)
- Graph-neighbor negatives (1-hop neighbors of gold that are not gold)

Allow exactly one preregistered hard-negative refresh before final model selection.

### Defect 7: Synthetic Data Dominance

**Location**: `stack/encoder/c2_c3.py` line 112, `stage1c_full_selection_log.json`
**Impact**: 1557 graph-mined groups vs 375 real-question groups means synthetic templates dominate training 4:1

```
"sources": {
    "stage1d_graph_mined": 1557,
    "train_candidate_pipeline": 375
}
```

Additionally, `stage1c.py` generates questions using only repetitive `"What is <alias>?"` templates (line 92), `"What is the finding about <terms>?"` (line 103), and `"What is the <type> relation between <a> and <b>?"` (line 115).

**Fix**: Preregistered sampling policy:
- 50% real questions (train.jsonl)
- 25% natural graph-mined paraphrases (deterministic templates matching factual, diagnostic, comparison, and multi-hop styles)
- 15% alias/key-finding examples
- 10% relation examples

Per-source results must be reported.

Natural templates must include:
- Factual: "What was the finding of <experiment>?", "What is the significance of <concept>?"
- Diagnostic: "Why did <experiment> produce <result>?", "How does <concept> relate to <observation>?"
- Comparison: "Compare <experiment_a> and <experiment_b>", "What is the difference between <concept_a> and <concept_b>?"
- Multi-hop: "How does <concept_a> influence <concept_b> through <relation>?"

### Defect 8: Missing Canonical Entity Mapping

**Impact**: The graph contains granular entities such as `Mmap`, while frozen labels reference parent experiments such as `Exp_0_6_Validation`. Without canonical mapping, many gold entities are impossible to retrieve by their granular graph IDs.

**Fix**: Build a deterministic graph-derived canonical mapping using only:
- `derived_from` edges (child entity → parent experiment)
- `mentioned_in` / `source` / `provenance` properties
- Explicit parent experiment/document relations
- Do NOT inspect frozen test answers when creating the mapping

Apply canonical mapping before the K=10 cap and deduplicate mapped IDs.

Report both:
- Raw granular recall@K
- Canonical frozen-label recall@K

**Required tests**: One-to-one, many-to-one, missing-parent, cyclic, and ambiguous mappings.

### Defect 9: Limited Feature-Based Ranking

**Location**: `stack/encoder/c2_c3.py` lines 140-154
**Impact**: Only 6 features: bias, lexical_overlap, char_ngram_cosine, type_rule_intent, node_degree_log, alias_match

The feature ranker is missing several cheap and discriminative features.

**Fix**: Add optional CPU-cheap features:
- BM25 / TF-IDF token overlap (normalized)
- Exact alias phrase match (case-insensitive)
- Normalized alias overlap (Jaccard of alias tokens with question tokens)
- Node-type versus intent compatibility (expanded beyond binary)
- Provenance/source match
- Graph distance to top lexical candidates
- Relation to strong lexical candidates (shared graph neighbors)
- Property-field-specific overlap (key_finding vs description vs aliases weighted separately)
- Limit or regularize node-degree influence (log1p already used, verify calibration)

Compare four rankers:
1. Trivial lexical baseline (unchanged, non-parametric)
2. Corrected feature-logistic ranker (expanded features)
3. Neural encoder v3 (genuinely question-conditioned)
4. Hybrid encoder + graph features

### Defect 10: Paraphrase-Gate Semantics

**Location**: `benchmarks/stage1c_final.py` line 220
**Impact**: The artifact reports `drop_pp = -14.71` because paraphrases perform better than originals. The gate uses `abs(drop_pp) < 10` which causes FAIL.

The preregistered meaning of "drop" in the original experiment is:
> "paraphrase_30: entity_accuracy drop vs original < 10pp"

"Drop" implies degradation. When paraphrases outperform originals, the absolute-value gate creates a false negative.

**Fix** for this experiment:
- If only degradation is gated: use `drop_pp < 10` (one-sided, negative values pass)
- If absolute instability is intended: preregister that explicitly as `abs(drop_pp) < 10` in this document before evaluation
- **Preregistered decision**: This experiment interprets "drop" as degradation only. Paraphrase recall@10 must not decrease by >= 10 pp from original recall@10. Improvements pass unconditionally.

---

## Architecture Specification

### Model: Question-Conditioned Entity Ranker

```
Input: question text (q), entity description (e)
Output: scalar relevance score

Option A (preferred): Bilinear interaction
    score(q, e) = q^T @ W @ e
    where W ∈ R^(d×d), learnable

Option B: Projected dot-product
    proj_q = Linear(combined_dim → proj_dim)(q_encoding)
    proj_e = Linear(embed_dim → proj_dim)(e_embedding)
    score(q, e) = dot(proj_q, proj_e) / sqrt(proj_dim)

Option C: MLP interaction
    interaction = [q_encoding, e_embedding, q*e, abs(q-e), cosine(q,e)]
    score(q, e) = MLP(interaction)
```

The ranking for a single question is:
```
scores = [score(q, e_i) for e_i in candidates]
ranked = argsort(scores, descending=True)[:K]
```

### Entity Text Representation

Each entity is represented as a concatenated text string:
```
<node_id> | <type> | <display_name> | aliases: <alias1>, <alias2> | key_finding: <finding> | description: <desc> | source: <source> | relations: <type1>:<count1>, <type2>:<count2>
```

### Training Protocol

- Loss: Listwise softmax cross-entropy over all candidates per question
- Or pairwise margin ranking loss with multiple positives and hard negatives per question
- Optimizer: Adam with cosine annealing
- Batch: 16-32 groups per batch, balanced source sampling
- Epochs: Up to 50 with early stopping (patience=5 on validation recall@10)
- All computation on CPU (torch.set_num_threads(1))

### Hard-Negative Mining Protocol

Phase 1: Score all non-gold candidates using the trivial lexical baseline. Retain top 10 highest-scoring non-gold candidates per question as initial hard negatives.

Phase 2 (single refresh): After first encoder training pass, re-score all non-gold candidates with the trained model. Retain top 10 highest-scoring non-gold candidates. Combined with initial hard negatives, keeping at most 20 per question.

Hard negatives must include representatives from each category:
- Same-type as any gold entity
- Same-source as any gold entity
- High node-degree (top 10% of graph)
- Overlapping aliases with gold entities

### Synthetic Data Generation

Replace the repetitive `"What is <alias>?"` templates with deterministic natural templates:

**Factual templates**:
- "What was the result of <experiment_name>?"
- "What is <alias>?"
- "What did <experiment> demonstrate?"
- "What was the key finding of <experiment>?"
- "Describe <concept>."

**Diagnostic templates**:
- "Why did <experiment> produce <finding>?"
- "How does <concept_a> relate to <concept_b>?"
- "What is the significance of <finding>?"

**Comparison templates**:
- "Compare the results of <exp_a> and <exp_b>."
- "How does <exp_a> differ from <exp_b>?"

**Multi-hop templates**:
- "How does <concept_a> influence <concept_b>?"
- "What is the relationship between <exp_a> and <concept_b>?"

Sampling proportions (preregistered):
| Source | Ratio |
|---|---|
| Real questions (train.jsonl) | 50% |
| Natural graph-mined paraphrases | 25% |
| Alias / key-finding examples | 15% |
| Relation examples | 10% |

### Canonical Entity Mapping

Build a deterministic mapping from granular graph entities to frozen-label entities:

1. For each node, follow `derived_from` edges to find parent experiments
2. Check `source` and `provenance` properties for experiment/document references
3. Check `mentioned_in` properties
4. Use `Exp_*` and `Concept_*` regex patterns to identify canonical IDs
5. Build a deterministic mapping dict (no trainable parameters)
6. Apply before K=10 cap, deduplicate mapped IDs

---

## Validation and Selection Protocol

### Data
- Train: `stack/encoder/data/train.jsonl` (375 questions, 504 gold entities)
- Validation: `stack/encoder/data/val.jsonl` (150 questions, 182 gold entities)
- Test: `stack/encoder/data/test.jsonl` (225 questions, 275 gold entities) — **read exactly once**

### Validation Requirements
- Use only train.jsonl and val.jsonl for development
- Never read test.jsonl during development, feature selection, canonical mapping construction, threshold selection, or model selection
- Evaluate ALL 150 validation questions (no silently dropped groups)
- Compute candidate-recall ceiling before training (fraction of gold entities present in any candidate's pool)
- If validation candidate recall@10 cannot support 65%, fix candidate generation before training

### Selection Rule (mechanical)
1. Winner = argmax(recall@10), tie-break by recall@5, then recall@1, then latency, then RSS
2. Require validation recall@10 >= 70% before allowing frozen test evaluation
3. Require selected model to beat identical-population trivial baseline by >= 15 percentage points

### Final Frozen Evaluation
- K <= 10 enforced inside the ranking path (not post-hoc filtering)
- Unchanged 225-question frozen split
- Unchanged 65% entity recall@10 gate
- One final evaluation command
- Complete timestamped artifact
- Read-back validation
- Exact question and entity denominators
- Split SHA256, code SHA, model/config/weights hashes
- Per-question ranked entities and scores
- Candidate-stage diagnostics
- Latency and RSS
- Mechanical HONEST PASS or HONEST FAIL

### Gates (unchanged from Stage 1)
| Gate | Threshold |
|---|---|
| Entity recall@10 | >= 65% |
| Baseline gap (validation) | >= 15 pp |
| Intent accuracy | >= 85% |
| Paraphrase drop (degradation only) | < 10 pp |
| Resolution rate | No regression |
| Inference p50 | <= 50 ms |
| RSS delta | <= 150 MB |
| K cap | <= 10 |

---

## Required Tests

| # | Test | Description |
|---|---|---|
| T1 | Question conditions entity ordering | Prove changing question changes relative entity ranking |
| T2 | All validation questions in denominator | Missing gold counted as miss, no silent drops |
| T3 | Baseline and rankers use identical populations | Same 150 groups for all rankers |
| T4 | K cannot exceed 10 | In-path enforcement, not post-hoc |
| T5 | Canonical mapping is deterministic | Fixed seed, no training, no test inspection |
| T6 | Hard negatives are score-derived | Not just first-N from candidate pool |
| T7 | Synthetic/real sampling proportions enforced | Test with mocked data |
| T8 | Dirty worktree blocks evaluation | Raise error if uncommitted changes |
| T9 | Artifact decision is mechanically derived | All gates pass = PASS; any fail = FAIL |
| T10 | Nexus-only tests collect without PyTorch | Subprocess import check |
| T11 | Python 3.11 and 3.12 CI pass | Both interpreters |

---

## Files to Change

### New Files
- `EXPERIMENT_ENTITY_RANKER_V3.md` — this document
- `stack/encoder/entity_ranker_v3.py` — new ranker module
- `stack/encoder/canonical_mapping.py` — graph-derived canonical entity mapping
- `stack/encoder/hard_negative_miner.py` — score-derived hard negative mining
- `stack/encoder/natural_templates.py` — deterministic natural question templates
- `benchmarks/entity_ranker_v3_selection.py` — validation-only selection script
- `benchmarks/entity_ranker_v3_final.py` — single-read frozen evaluator
- `tests/test_entity_ranker_v3.py` — tests for new ranker
- `tests/test_canonical_mapping.py` — tests for canonical mapping (one-to-one, many-to-one, missing-parent, cyclic, ambiguous)
- `tests/test_hard_negative_miner.py` — tests for hard negative mining
- `tests/test_question_conditioned_scoring.py` — test proving question-conditional ranking
- `tests/test_validation_denominator.py` — test proving all 150 questions counted
- `tests/test_dirty_worktree_guard.py` — test for dirty worktree detection

### Modified Files
- `stack/encoder/model.py` — replace linear scorer with interaction model
- `stack/encoder/c2_c3.py` — fix denominator, fix hard negatives, improve features
- `stack/encoder/stage1c.py` — add natural templates, diverse question styles
- `README.md` — update status
- `STACK_RESULTS.md` — update results
- `PROTOCOL_VIOLATIONS.md` — document defect discovery
- `benchmarks/results/INDEX.md` — add new artifact entries

### Preserved (unchanged)
- All historical Stage 1, 1B, 1C, and 1D artifacts
- All `STAGE*_NEGATIVE.md` and `STAGE1D_RESULT.md` files
- All frozen split files (train.jsonl, val.jsonl, test.jsonl)
- EXPERIMENT_SAM_NEXUS_STACK.md immutable gate thresholds

---

## Final Report Requirements

The final report must include:
1. Exact files changed
2. Defects found and fixed
3. Explanation of the corrected interaction architecture
4. Training-data counts and source balance
5. Validation denominators (must be 150 questions, 182 gold)
6. Candidate recall ceiling
7. Baseline vs every ranker comparison
8. Recall@1/5/10 and precision@10
9. Paraphrase metrics (degradation-only gate)
10. Latency and RSS
11. Exact CLI commands
12. Artifact path and hashes
13. Final PASS/FAIL
14. Confirmation that no frozen labels, thresholds, K limits, or historical artifacts were changed

---

## Implementation Order

1. **Commit this document** (preregistration)
2. Write and pass all required tests (T1-T11)
3. Implement canonical entity mapping (graph-derived, no test inspection)
4. Implement improved entity text representation
5. Implement natural template generation with diverse question styles
6. Implement score-derived hard negative mining
7. Implement question-conditioned interaction model (bilinear/dot-product/MLP)
8. Implement corrected training loop with batching, all-positives, early stopping
9. Implement expanded feature-logistic ranker
10. Implement selection/evaluation scripts with provenance guards
11. Calibrate on validation (mechanical selection)
12. If validation recall@10 >= 70% and baseline gap >= 15 pp: run single frozen evaluation
13. Write final report

---

## Endpoint Rule

- PASS → document the result, preserve the artifact, no Stage 1E
- FAIL → document the result in ENTITY_RANKER_V3_NEGATIVE.md, preserve the artifact, no Stage 1E
- No automatic progression in either case
