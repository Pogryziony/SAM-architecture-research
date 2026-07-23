# Sealed external evaluator handoff package

**Status:** Package complete for an *independent* evaluator.  
**Sealed run itself:** `BLOCKED` until an independent evaluator and external corpus exist.

This directory is intentionally separate from internal `oracle_v1`. Do **not**
label development-controlled SAM questions as sealed.

## Contents

| Path | Purpose |
|------|---------|
| `PROTOCOL.md` | End-to-end sealed evaluation protocol |
| `ACCEPTED_SOURCE_FORMATS.md` | Allowed corpus formats |
| `HIDDEN_QUESTION_SCHEMA.md` | Hidden question JSONL schema |
| `RESULT_SCHEMA.md` | Pointer to `nexus-eval-result-v1` |
| `SYSTEM_CONFIG_REGISTRY.md` | How systems declare identity |
| `METRIC_DEFINITIONS.md` | Primary/secondary metrics |
| `PREREGISTRATION_TEMPLATE.md` | Fill before disclosure |
| `ADJUDICATION_RUBRIC.md` | Human dimensions |
| `LEAKAGE_CHECKLIST.md` | Pre-release leakage controls |
| `FINAL_RELEASE_PROCEDURE.md` | Signed/hash-identified release |
| `tools/hash_corpus.py` | Corpus hashing utility |
| `tools/validate_handoff.py` | Package self-check |

## Evaluator workflow (summary)

1. Freeze external corpora (`tools/hash_corpus.py`).
2. Create or hold hidden questions (schema above).
3. Disclose questions without gold to system owners (or run systems yourself).
4. Receive schema-valid `nexus-eval-result-v1` outputs.
5. Adjudicate (automated + blinded human where required).
6. Run paired statistics only on complete comparable arms.
7. Publish a hash-identified result package.

## Explicit non-claims

- Internal `oracle_v1` is **not** a sealed external holdout.
- Completing this package does **not** produce a sealed result by itself.
