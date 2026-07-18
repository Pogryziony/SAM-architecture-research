# Realizer AnswerPlan v1 — Test Evaluation Protocol

**Status:** `PROTOCOL_DEFINED`

**Scope:** the single permitted opening of the 12,126-record sealed test split after
a checkpoint passes all pilot generation gates.

**Sealed split:** `docs/realizer-answer-plan-v1.md` records the test data identity,
source hashes and split policy. This protocol governs the evaluation process
for that one-time window.

## 1. Preconditions

All of the following must be satisfied before the test split may be opened.
No partial or partial-failure evaluation is permitted.

### 1.1 Bounded pilot generation gates

| Gate | Exact match (≥) | Token F1 (≥) | EOS (≥) |
|---|---:|---:|---:|
| Overfit (64 records) | 0.80 | 0.85 | 0.95 |
| Small (2,048 stratified) | 0.50 | 0.70 | 0.95 |
| Representative (≤ 17,000) | 0.70 | 0.85 | 0.95 |

All three gates must pass on the **held-out validation split**
(3,660 records). Overfit and small gates report on their respective training
subsets; the representative gate reports on the full held-out validation.

### 1.2 Checkpoint selected

A single checkpoint is selected from the representative pilot run. Selection
criteria are policy-defined but must include at minimum:

- lowest validation loss among all checkpoints that satisfy the representative
  gate thresholds above;
- no single PL or EN slice falls below 80% of the overall token F1.

The checkpoint identity is pinned at selection time by its SHA-256 hash and
the configuration hash from `training/realizer_answer_plan_v1.json`.

### 1.3 Full-training readiness

`check_answer_plan_full_training_readiness.py` must report
`READY_FOR_FULL_TRAINING` (not `FULL_TRAINING_BLOCKED`). This check
aggregates the three pilot generation gates, checkpoint selection,
immutable preservation and unsupported number rate.

### 1.4 Immutable preservation and unsupported numbers

| Metric | Threshold |
|---|---|
| Immutable value preservation on validation | ≥ 0.99 |
| Unsupported number rate on validation | ≤ 0.01 |

These thresholds are checked on the full held-out validation split. The
immutable preservation metric counts the fraction of validation records where
every immutable value in the AnswerPlan appears verbatim in the generated
output. The unsupported number rate counts records where a number present in
the generated output was not present in the AnswerPlan.

### 1.5 Precondition checklist

Before proceeding, confirm every item:

- [ ] Overfit generation gate passes (EM ≥ 0.80, F1 ≥ 0.85, EOS ≥ 0.95)
- [ ] Small generation gate passes (EM ≥ 0.50, F1 ≥ 0.70, EOS ≥ 0.95)
- [ ] Representative generation gate passes (EM ≥ 0.70, F1 ≥ 0.85, EOS ≥ 0.95)
- [ ] Single checkpoint selected and pinned
- [ ] `READY_FOR_FULL_TRAINING` reported
- [ ] Immutable preservation ≥ 0.99 on validation
- [ ] Unsupported number rate ≤ 0.01 on validation

## 2. Evaluation identity

Before opening the test split, the evaluation identity must be pinned.
This triple binds the evaluation to an exact artifact state.

| Identity component | Source |
|---|---|
| Checkpoint SHA-256 | Hash of the selected checkpoint file |
| Configuration hash | SHA-256 of `training/realizer_answer_plan_v1.json` |
| Data readiness hash | `check_answer_plan_full_training_readiness.py` readiness report hash |

All three values are recorded in the evaluation result file and must not
change after the test split is opened. The data readiness hash is the same
value registered in `docs/realizer-answer-plan-v1.md`.

## 3. Evaluation process

### 3.1 Opening the test split

The test split is opened exactly once. The opening step:

1. verifies the test split's registered source hash against the seal in
   `docs/realizer-answer-plan-v1.md`;
2. loads the 12,126 test records without any filtering, sampling or
   modification;
3. verifies the test record count matches the sealed count (12,126).

### 3.2 Running evaluation

The pinned checkpoint is evaluated on the full test split using the same
generation configuration that passed the representative pilot gate.
Generation must use:

- the same beam search or decoding strategy;
- the same immutable-value constraint mechanism;
- the same verifier configuration that rejects neural output and falls
  back to AnswerPlan copy.

No hyperparameter may be changed between the pilot validation run and
the test run. The test run is inference-only—no gradient computation,
no weight update, no adaptation of any kind.

### 3.3 Metrics computed

| Metric | Description |
|---|---|
| Exact match | Fraction of outputs identical to the reference answer |
| Token F1 | Token-level precision/recall F1 vs reference |
| Immutable preservation | Fraction where all immutable values appear verbatim |
| Per-language slices | Separate EM and F1 for English and Polish |
| Per-operator slices | Separate EM and F1 for `extract`, `compose_path`, `compare`, `abstain` |
| EOS rate | Fraction of outputs ending with the end-of-sequence token |
| Empty output rate | Fraction of outputs that are empty or whitespace-only |
| Unsupported number rate | Fraction of outputs containing a number not in the AnswerPlan |
| Repetition rate | Fraction of outputs with repeated n-gram blocks (> 4 tokens) |

All metrics are reported for the full test split and for every slice.
Per-operator abstention metrics include precision, recall and F1 for the
binary abstain/generate decision.

## 4. Result artifact

### 4.1 Output file

Results are written to a single JSON file:

```
benchmarks/results/realizer/answer_plan_v1_test_evaluation.json
```

The file must be created atomically (write to a temporary file in the same
directory, then rename). It must never be overwritten by a later evaluation.

### 4.2 JSON schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AnswerPlan v1 Test Evaluation",
  "type": "object",
  "required": [
    "evaluation_id",
    "identity",
    "preconditions",
    "metrics",
    "slices",
    "verdict"
  ],
  "properties": {
    "evaluation_id": {
      "type": "string",
      "description": "SHA-256 of (checkpoint_hash + config_hash + readiness_hash)"
    },
    "identity": {
      "type": "object",
      "required": ["checkpoint_sha256", "config_sha256", "readiness_sha256", "evaluation_timestamp_utc"],
      "properties": {
        "checkpoint_sha256": { "type": "string" },
        "config_sha256": { "type": "string" },
        "readiness_sha256": { "type": "string" },
        "evaluation_timestamp_utc": { "type": "string", "format": "date-time" }
      }
    },
    "preconditions": {
      "type": "object",
      "required": [
        "overfit_gate", "small_gate", "representative_gate",
        "checkpoint_selected", "full_training_readiness",
        "immutable_preservation", "unsupported_number_rate"
      ],
      "properties": {
        "overfit_gate": { "$ref": "#/$defs/gate_result" },
        "small_gate": { "$ref": "#/$defs/gate_result" },
        "representative_gate": { "$ref": "#/$defs/gate_result" },
        "checkpoint_selected": { "type": "boolean" },
        "full_training_readiness": {
          "type": "string",
          "enum": ["READY_FOR_FULL_TRAINING"]
        },
        "immutable_preservation": {
          "type": "number", "minimum": 0.0, "maximum": 1.0
        },
        "unsupported_number_rate": {
          "type": "number", "minimum": 0.0, "maximum": 1.0
        }
      }
    },
    "metrics": {
      "type": "object",
      "required": [
        "exact_match", "token_f1", "immutable_preservation",
        "eos_rate", "empty_output_rate", "unsupported_number_rate",
        "repetition_rate", "test_record_count"
      ],
      "properties": {
        "exact_match": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "token_f1": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "immutable_preservation": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "eos_rate": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "empty_output_rate": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "unsupported_number_rate": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "repetition_rate": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "test_record_count": { "type": "integer", "const": 12126 }
      }
    },
    "slices": {
      "type": "object",
      "required": ["by_language", "by_operator"],
      "properties": {
        "by_language": {
          "type": "object",
          "required": ["en", "pl"],
          "properties": {
            "en": { "$ref": "#/$defs/slice_metrics" },
            "pl": { "$ref": "#/$defs/slice_metrics" }
          }
        },
        "by_operator": {
          "type": "object",
          "required": ["extract", "compose_path", "compare", "abstain"],
          "properties": {
            "extract": { "$ref": "#/$defs/slice_metrics" },
            "compose_path": { "$ref": "#/$defs/slice_metrics" },
            "compare": { "$ref": "#/$defs/slice_metrics" },
            "abstain": { "$ref": "#/$defs/abstain_slice_metrics" }
          }
        }
      }
    },
    "verdict": {
      "type": "object",
      "required": ["result", "reason"],
      "properties": {
        "result": {
          "type": "string",
          "enum": ["ACCEPTED", "REJECTED"]
        },
        "reason": { "type": "string" },
        "regression_details": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "metric": { "type": "string" },
              "validation_value": { "type": "number" },
              "test_value": { "type": "number" },
              "threshold": { "type": "number" }
            }
          }
        }
      }
    }
  },
  "$defs": {
    "gate_result": {
      "type": "object",
      "required": ["passed", "exact_match", "token_f1", "eos_rate"],
      "properties": {
        "passed": { "type": "boolean" },
        "exact_match": { "type": "number" },
        "token_f1": { "type": "number" },
        "eos_rate": { "type": "number" }
      }
    },
    "slice_metrics": {
      "type": "object",
      "required": ["record_count", "exact_match", "token_f1", "immutable_preservation"],
      "properties": {
        "record_count": { "type": "integer", "minimum": 0 },
        "exact_match": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "token_f1": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "immutable_preservation": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      }
    },
    "abstain_slice_metrics": {
      "allOf": [
        { "$ref": "#/$defs/slice_metrics" },
        {
          "type": "object",
          "required": ["abstain_precision", "abstain_recall", "abstain_f1"],
          "properties": {
            "abstain_precision": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "abstain_recall": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "abstain_f1": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
          }
        }
      ]
    }
  }
}
```

### 4.3 Baseline comparison

The result file must include a comparison against the deterministic validation
baselines registered in `docs/realizer-answer-plan-v1.md`:

| Baseline | Exact match | Token F1 |
|---|---:|---:|
| AnswerPlan copy | 85.58% | 89.08% |
| Registered PL/EN template | 0% | 72.74% |

The verdict section must note whether the neural checkpoint matches or exceeds
the AnswerPlan copy baseline on the relevant metrics. Falling below the copy
baseline on any slice is a regression signal but does not automatically reject
the checkpoint unless the policy-defined regression threshold is breached.

## 5. Post-evaluation rules

### 5.1 Re-sealing

After evaluation results are written, the test split must be re-sealed:

- no retraining on test records;
- no hyperparameter tuning after seeing test results;
- no further evaluation runs on the test split;
- the evaluation is **final and terminal** for this checkpoint.

A re-seal record is appended to the evaluation result file confirming the
test split was not used for training, tuning or repeated evaluation.

### 5.2 No test-guided improvement

Once the test results are seen, the checkpoint may not be modified or
retrained. No hyperparameter, architecture or data change may be justified
by test-set performance. This includes:

- adjusting decoding parameters after seeing test metrics;
- modifying the verifier threshold based on test output;
- re-running evaluation with a different beam width or sampling strategy;
- selecting a different checkpoint from the same run after seeing test results.

Violating this rule invalidates the evaluation and requires the test split
to be re-sealed indefinitely.

## 6. Failure handling

### 6.1 Regression threshold

The checkpoint is **rejected** if any of the following occur:

1. test exact match falls more than 5 percentage points below the validation
   exact match for the representative gate;
2. test token F1 falls more than 5 percentage points below the validation
   token F1 for the representative gate;
3. test immutable preservation falls below 0.99;
4. test EOS rate falls below 0.90;
5. any language slice (en or pl) falls more than 10 percentage points below
   the corresponding validation slice on token F1.

### 6.2 Rejected checkpoint

If the checkpoint is rejected:

- the test split stays sealed for a future attempt;
- the rejected checkpoint is archived externally with the evaluation result;
- the failure is documented with the specific metrics that regressed;
- `FULL_TRAINING_BLOCKED` is reasserted with the rejection reason added;
- a new checkpoint may be developed under a new configuration, but it must
  re-pass all pilot gates before the test split can be opened again.

### 6.3 Accepted checkpoint

If the checkpoint is accepted:

- results are published in `benchmarks/results/realizer/answer_plan_v1_test_evaluation.json`;
- the test split is permanently sealed—no further evaluations, no retraining;
- the accepted checkpoint becomes the reference Realizer model for AnswerPlan v1;
- all subsequent work on the Realizer must use a new AnswerPlan version or a
  new corpus version that re-inherits the sealed-test policy.

## 7. Related documents

- [Realizer AnswerPlan v1](realizer-answer-plan-v1.md) — data preparation,
  pilot protocol, seal record and readiness status
- [Realizer Corpus v2](realizer-corpus-v2.md) — corpus sources, split policy
  and leakage prevention
- [Corpus Coverage Roadmap](corpus-coverage-roadmap.md) — planned data source
  additions for future corpus versions
