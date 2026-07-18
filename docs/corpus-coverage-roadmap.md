# NEXUS Realizer — Corpus Coverage Roadmap

**Status:** `ROADMAP_DRAFT`

**Scope:** plan for extending the Realizer corpus with long explanation, causal
diagnostic and scientific prose data sources in future corpus versions.

**Current corpus:** `docs/realizer-corpus-v2.md` — this roadmap builds on the
existing 163K-record representative dataset without modifying or replacing it.

## 1. Current coverage inventory

| Attribute | Detail |
|---|---|
| Sources | 5 (HotpotQA, MuSiQue, PoQuAD, PolQA, MultiHop-RAG) |
| Languages | PL, EN |
| Answer style | Short factual answers (1–3 sentences typical) |
| Evidence domain | Wikipedia, news articles |
| Total records | 162,940 (147,154 train / 3,660 validation / 12,126 test) |
| Operators | `extract`, `compose_path`, `compare`, `abstain` |

The current corpus is representative for the next factual Realizer experiment
but is not representative of all future NEXUS use cases. Short factual answers
on Wikipedia-style evidence are a foundation, not a ceiling.

## 2. Gap categories

Gaps are ordered by priority. Priority reflects the expected impact on NEXUS
usability and the degree to which the gap limits the Realizer's ability to
serve downstream tasks.

### 2.1 Long explanations — priority: HIGH

**Problem:** the Realizer currently sees only short canonical answers
(median target length under 50 tokens). Multi-sentence explanatory answers
require a different generation surface: the model must produce coherent
paragraphs, maintain factual consistency across sentences and avoid
hallucinated elaboration.

**Required capability:** generate 3–8 sentence explanations from a set of
resolved facts, preserving all immutable values while producing fluent
connective prose. The explanation must not add claims beyond the evidence.

### 2.2 Causal reasoning — priority: HIGH

**Problem:** corpus v2 contains comparison (`SAME`/`DIFFERENT`) but no
explicit causal questions. "Why" questions with multi-step causal chains
stress the verifier and the generation pipeline differently: the Realizer
must express directional causal relationships ("X caused Y because Z")
without fabricating intermediate steps.

**Required capability:** express causal chains in natural language from a
symbolic causal graph provided by upstream NEXUS. Each claim in the output
must be traceable to a resolved fact in the AnswerPlan.

### 2.3 Scientific prose — priority: MEDIUM

**Problem:** corpus v2 evidence is drawn from Wikipedia and news. The
Realizer has never seen academic abstracts, clinical trial summaries,
materials science descriptions or peer-reviewed article excerpts.
Scientific text uses domain-specific vocabulary, passive constructions,
hedging and citation patterns that differ from encyclopedic prose.

**Required capability:** realize structured scientific findings (hypothesis,
method, result, conclusion) in fluent academic English. Must handle
measurement units, statistical claims and citation placeholders without
hallucination.

### 2.4 Enterprise structured data — priority: MEDIUM

**Problem:** the Realizer has no exposure to table-to-text generation,
structured record verbalization or enterprise knowledge-graph descriptions.
Downstream NEXUS use cases in finance, legal and internal knowledge
management require these capabilities.

**Required capability:** verbalize rows from structured tables, entity
profiles and property graphs. Must preserve numeric precision, unit
consistency and entity-name fidelity. The structured input format must
be compatible with the AnswerPlan contract.

### 2.5 Dialogue and conversational — priority: LOW

**Problem:** corpus v2 is single-turn QA. Multi-turn conversational
answers require the Realizer to maintain topical coherence across turns,
handle anaphora resolution and produce contextually appropriate responses.

**Required capability:** produce answers that are coherent within a
multi-turn dialogue history. The AnswerPlan must carry turn context and
the Realizer must not contradict previous turns.

## 3. Candidate sources

Each candidate is selected for native human-authored records, clear licensing
and compatibility with the supporting-fact annotation requirement. Sources
marked "needs survey" require further investigation before acquisition.

### 3.1 Long explanations

| Source | Language | Description | License | Status |
|---|---|---|---|---|
| [ELI5](https://huggingface.co/datasets/eli5_category) | EN | Long-form QA with multi-sentence answers sourced from Reddit's r/explainlikeimfive | MIT | Candidate |
| [MS MARCO NLGEN](https://github.com/microsoft/MSMARCO-Question-Answering) | EN | Passage reranking with human-authored natural language answers, includes longer responses | MIT | Candidate |
| [Natural Questions](https://ai.google.com/research/NaturalQuestions) (long-answer subset) | EN | Google QA with long-answer annotations extracted from Wikipedia paragraphs | CC BY-SA 4.0 | Candidate |
| [WebGPT comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) | EN | Human-written long-form answers with cited sources from web browsing | CC0 1.0 | Candidate |

**Priority source:** ELI5 — explicit explanatory structure, diverse domains,
clear license, and direct compatibility with the supporting-fact model
(each answer is grounded in source subreddit posts that can be treated
as evidence documents).

### 3.2 Causal reasoning

| Source | Language | Description | License | Status |
|---|---|---|---|---|
| [TemporalWiki / CaTeRS](https://github.com/StonyBrookNLP/causal-reasoning) | EN | Causal and temporal relation extraction from Wikipedia with annotated causal chains | CC BY 4.0 | Candidate |
| [e-CARE](https://github.com/Waste-Wood/e-CARE) | EN | Explainable causal reasoning dataset with human-annotated causal explanations | Apache 2.0 | Candidate |
| [COPA](https://people.ict.usc.edu/~gordon/copa.html) (augmented with explanations) | EN | Choice of plausible alternatives, extendable with human-written causal explanations | BSD-style | Needs survey |
| [BECauSE 2.0](https://github.com/duncanka/BECAUSE) | EN | Bank of effects and causes with annotated causal relations from multiple genres | CC BY 4.0 | Candidate |

**Priority source:** e-CARE — provides both causal questions and human-written
explanations with explicit causal chain structure. The explanations can be
aligned with supporting facts from the source documents.

### 3.3 Scientific prose

| Source | Language | Description | License | Status |
|---|---|---|---|---|
| [SciQ](https://allenai.org/data/sciq) | EN | Science exam questions with supporting paragraphs from textbooks | CC BY-NC 4.0 | Candidate (license constrains commercial use) |
| [PubMedQA](https://pubmedqa.github.io/) | EN | Biomedical QA with long answers and PubMed abstracts as context | MIT | Candidate |
| [Evidence Inference](https://evidence-inference.ebm-nlp.com/) | EN | Clinical trial report annotations with outcome-to-evidence mappings | Apache 2.0 | Candidate |
| [SciTLDR](https://github.com/allenai/scitldr) | EN | Extreme summarization of scientific papers (single-sentence TLDR from full text) | Apache 2.0 | Needs survey (summarization, not QA) |

**Priority source:** PubMedQA — human-authored long answers, PubMed abstracts as
evidence documents, clear license (MIT), and the biomedical domain is a
strong test of scientific prose capability. Non-commercial SciQ limitations
make PubMedQA the safer primary choice.

### 3.4 Enterprise structured data

| Source | Language | Description | License | Status |
|---|---|---|---|---|
| [ToTTo](https://github.com/google-research-datasets/ToTTo) | EN | Table-to-text generation with Wikipedia tables and human-authored descriptions | CC BY-SA 4.0 | Candidate |
| [WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) + human answers | EN | Question answering on Wikipedia tables with human-authored answers | CC BY-SA 4.0 | Needs survey |
| [LogicNLG](https://github.com/wenhuchen/LogicNLG) | EN | Table-to-text generation with logical fidelity constraints | MIT | Candidate |
| [DART](https://github.com/Yale-LILY/dart) | EN | Open-domain structured data record to text with human references | MIT | Candidate |

**Priority source:** ToTTo — explicit table-to-text task, Wikipedia domain
(consistent with existing corpus), human-authored descriptions with
supporting-fact annotations derivable from highlighted table cells.
LogicNLG adds logical fidelity constraints that align with the immutable
value preservation requirement.

### 3.5 Dialogue and conversational

| Source | Language | Description | License | Status |
|---|---|---|---|---|
| [CoQA](https://stanfordnlp.github.io/coqa/) | EN | Conversational QA with free-form answers grounded in passages | Apache 2.0 | Candidate |
| [QuAC](https://quac.ai/) | EN | Question answering in context with information-seeking dialogue | CC BY 4.0 | Candidate |
| [DoQA](https://github.com/RevanthRameshkumar/DoQA) | EN | Conversational QA with access to cooking/cosmos documents | CC BY-SA 4.0 | Needs survey |

**Priority source:** CoQA — conversational structure with passage-grounded
answers, clear license, and answer annotations that include both extractive
spans and free-form human responses. The dialogue history can be encoded in
an extended AnswerPlan format.

## 4. Admission criteria

Every new source must satisfy all criteria below. These reaffirm and extend
the policies established in `docs/realizer-corpus-v2.md`.

### 4.1 Mandatory criteria

| Criterion | Detail |
|---|---|
| Native human-authored | Records must be written by humans for their original purpose. No LLM-generated, no synthetic, no template-filled answers. |
| No translation or paraphrase | Source records must be used in their original language. No machine translation of EN sources into PL or vice versa. No paraphrase expansion of one record into multiple training records. |
| Supporting-fact annotations | Every record must include gold supporting evidence with stable identifiers or locators. Evidence must be traceable to source documents. |
| Document-disjoint splits | Train, validation and test splits must be document-disjoint. No document may appear in more than one split. Test records are processed first so overlaps are removed from train/validation. |
| Clear license | License must be explicitly stated, compatible with research use and distribution, and recorded in `training/realizer_corpus_v3_sources.json`. |
| Attribution preserved | Source name, URL, license, revision and artifact hash must be recorded and redistributed with every derivative. |
| Must extend, not replace | New sources are added to the existing corpus. No existing source is removed or replaced. Existing train/validation/test records are preserved unchanged. |

### 4.2 Corpus versioning

Each admission batch produces a new corpus version:

| Version | Status | Description |
|---|---|---|
| v2 | `REALIZER_CORPUS_V2_READY` | Current representative corpus (5 sources, 163K records) |
| v3 | Planned | v2 + long explanations + causal reasoning |
| v4 | Planned | v3 + scientific prose + enterprise structured data |
| v5 | Planned | v4 + dialogue and conversational |

Versions are cumulative. v3 contains everything in v2 plus the admitted
v3 sources. A new AnswerPlan version may be required when the record
schema changes between corpus versions.

## 5. Admission process

Each new source follows a six-step admission pipeline.

### Step 1 — Proposal

A short proposal document identifies the source, its gap category, license,
expected record count and language. The proposal must explain why the source
is needed and how it maps to the NEXUS AnswerPlan contract.

### Step 2 — License review

Verify the license permits research use, redistribution and derivative
works. Record the license SPDX identifier. Flag any restrictions (e.g.,
non-commercial only, attribution required, share-alike). The license must
be compatible with the existing corpus's license mix.

### Step 3 — Acquisition script

Write an acquisition script in `benchmarks/acquire_*_corpus_v3.py` that:

- downloads from the pinned URL;
- verifies the source artifact hash against the proposal;
- extracts native records without modification;
- filters records that lack supporting-fact annotations;
- applies the document-disjoint split policy (test first);
- writes normalized JSONL to an external corpus directory.

### Step 4 — Hash pinning

After acquisition, pin every artifact identity:

- source download SHA-256;
- normalized record manifest SHA-256;
- corpus builder output SHA-256.

All hashes are registered in `training/realizer_corpus_v3_sources.json`
and must never change after the corpus version is declared ready.

### Step 5 — Integration test

A new `benchmarks/build_realizer_corpus_v3.py` builder:

- ingests v2 records unchanged;
- appends v3 source records with the same schema;
- verifies the document-disjoint property across all splits;
- reports record counts, language distribution and operator distribution;
- runs the full leakage test suite from v2;
- declares `REALIZER_CORPUS_V3_READY`.

### Step 6 — Corpus rebuild

Run the v3 builder to produce the new external corpus artifact. The v2
corpus remains available and unchanged. v3 becomes the active corpus for
the next AnswerPlan version.

## 6. Related documents

- [Realizer Corpus v2](realizer-corpus-v2.md) — current corpus sources,
  split policy, leakage prevention and honest coverage limits
- [Realizer AnswerPlan v1](realizer-answer-plan-v1.md) — data preparation,
  pilot protocol and test-seal policy
- [Test Evaluation Protocol](test-evaluation-protocol.md) — one-time test
  evaluation process and post-evaluation re-sealing
