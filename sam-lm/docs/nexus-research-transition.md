# NEXUS Research Transition Notes

Status: research documentation only  
Date: 2026-06-18  
Branch: `research/nexus-architecture-docs`

## 1. Purpose

This document captures a research-oriented analysis of the current SAM architecture and a proposed transition direction toward a broader CPU-first architecture tentatively named **NEXUS**.

This is not a production implementation plan and does not introduce production code. The goal is to document research assumptions, findings, design considerations, trade-offs, open questions, and future implementation criteria.

The central research question is:

> Can the existing SAM work be reused as a foundation for a non-parametric, CPU-first reasoning architecture where knowledge lives primarily in external memory rather than dense model weights?

## 2. Repository Context Reviewed

The analysis was based on the current repository structure and the relevant documentation and modules already present in the repository.

Relevant documentation:

- [`../../README.md`](../../README.md) — root project overview and current status.
- [`README.md`](../README.md) — SAM-LM project instructions and experiment summary.
- [`thesis.md`](thesis.md) — motivation for separating knowledge from computation.
- [`architecture.md`](architecture.md) — current SAM architecture and memory modes.
- [`experiments.md`](experiments.md) — experiment chronology and findings.
- [`current-status.md`](current-status.md) — confirmed results, pending hypotheses, and risks.
- [`roadmap.md`](roadmap.md) — planned next experiments and decision rules.
- [`repository-map.md`](repository-map.md) — repository structure and module responsibilities.
- [`experiment-0-13a-noisy-memory.md`](experiment-0-13a-noisy-memory.md) — controlled noisy memory tolerance study.

Relevant implementation modules:

- [`../sam/model/sam_core.py`](../sam/model/sam_core.py) — main SAM model, memory modes, retrieval wrappers, memory integration, noisy memory paths, selector wiring.
- [`../sam/model/product_key_memory.py`](../sam/model/product_key_memory.py) — product-key memory implementation, subkey addressing, slot lookup, value reads.
- [`../sam/model/slot_selector.py`](../sam/model/slot_selector.py) — learned candidate-slot selector.
- [`../sam/model/transformer.py`](../sam/model/transformer.py) — decoder-only transformer core.
- [`../sam/model/dense_transformer.py`](../sam/model/dense_transformer.py) — dense baseline.
- [`../sam/data/synthetic_facts.py`](../sam/data/synthetic_facts.py) — synthetic multi-hop fact/question generator.
- [`../sam/data/dataset.py`](../sam/data/dataset.py) — tokenizer, QA dataset, slot tracking, KB tensor construction.
- [`../sam/training/train_sam.py`](../sam/training/train_sam.py) — SAM training loop and memory-mode orchestration.
- [`../sam/training/train_retrieval.py`](../sam/training/train_retrieval.py) — dual encoder and chain-set retrieval training.
- [`../sam/eval/metrics.py`](../sam/eval/metrics.py) — accuracy and retrieval metrics.
- [`../sam/eval/analyze_required_set_retrieval.py`](../sam/eval/analyze_required_set_retrieval.py) — required-slot retrieval diagnostics.

## 3. Existing SAM Thesis

SAM currently investigates whether useful language understanding, question answering, and multi-hop reasoning can be achieved with less dependence on large dense weights and repeated RAM/VRAM bandwidth.

The current thesis can be summarized as follows:

1. Dense LLMs entangle factual knowledge and reasoning capability inside the same weight matrices.
2. This causes every token prediction to stream all model weights, even when only a small subset of knowledge is relevant.
3. A smaller active core plus a large sparse associative memory may reduce bandwidth pressure.
4. Knowledge capacity should scale through memory slots, not through dense parameters.
5. Retrieval should select only the relevant memory entries per query.
6. The active core should remain small and comparatively constant-cost.

This thesis remains valuable, but the repository evidence suggests that the next research step should not be to scale the current latent-vector memory path directly. Instead, the architecture should evolve toward a more explicit CPU-native memory and reasoning engine.

## 4. Current Findings Reused by NEXUS

### 4.1 Oracle memory validates the value of external facts

The oracle memory experiments show that the small SAM core can reach near-perfect multi-hop QA accuracy when the required slots are injected directly.

Research implication:

- External memory is useful.
- The core can consume memory-derived information when it is clean.
- The failure is not simply that small models cannot use external facts.

NEXUS should preserve this result as an upper-bound test: if the reasoning engine provides the correct chain, the answer module should be able to produce the correct answer.

### 4.2 Chain-set retrieval solves coverage but not answer accuracy

The chain-set BCE retriever reaches full required-slot coverage at sufficiently high top-K in the synthetic setting. However, the retrieved-memory path still performs similarly to the core-only baseline.

Research implication:

- Retrieval coverage alone is insufficient.
- The architecture needs better structure between retrieval and answer generation.
- Passing a flat set of retrieved slots to the model is not equivalent to giving the model an executable reasoning path.

NEXUS should treat retrievers as candidate generators, not complete reasoning modules.

### 4.3 Learned slot selection is not yet usable

The learned selector has high recall but insufficient precision. It identifies many required slots but still selects too many distractors.

Research implication:

- Slot-level classification may be the wrong unit of decision.
- A chain-level reranker may be more appropriate than independent slot selection.
- Structured graph paths are likely easier to validate than isolated facts.

NEXUS should investigate chain scoring rather than only slot scoring.

### 4.4 Controlled random distractors are tolerated

Experiment 0.13A indicates that random distractors do not immediately collapse SAM. This weakens the hypothesis that any memory noise causes failure.

Research implication:

- The problem is probably not just distractor count.
- Realistic distractors may be qualitatively different because they are semantically close.
- Training dynamics, memory wiring, and aggregation strategy remain plausible failure sources.

NEXUS should explicitly distinguish random noise from hard, semantically plausible distractors.

### 4.5 Flat aggregation is a likely bottleneck

The current architecture often collapses retrieved memory into one vector through aggregation. This can destroy slot identity, relation order, and chain structure.

Research implication:

- Multi-hop reasoning should preserve structure.
- Slot-wise or chain-wise readers are a better research direction than simple averaging.
- CPU-side graph traversal can preserve explicit reasoning paths before any neural component sees the data.

## 5. Proposed NEXUS Direction

NEXUS is a proposed successor research direction, not a direct rename of all SAM components.

Working expansion:

> **NEXUS — Non-Parametric Execution and Understanding System**

The intended distinction from SAM-LM is that NEXUS should not primarily behave like a small LLM with attached memory. It should behave like a CPU-first memory and reasoning system with neural components used selectively.

### 5.1 Core architectural shift

Current SAM orientation:

```text
question
  -> transformer core
  -> memory retrieval
  -> slot aggregation
  -> gated residual injection
  -> answer
```

Proposed NEXUS orientation:

```text
question
  -> language interpreter
  -> query / reasoning plan
  -> CPU memory engine
  -> graph expansion / candidate chain generation
  -> chain validation / reranking
  -> working memory
  -> answer synthesis
```

The key shift is that CPU/RAM should not be passive storage for vectors. CPU/RAM should actively execute memory operations:

- lexical lookup,
- entity anchoring,
- graph traversal,
- relation expansion,
- candidate-chain generation,
- contradiction checks,
- deduplication,
- scoring,
- working-memory management,
- cache lookup.

The neural model should handle tasks that are hard to express deterministically:

- ambiguous language interpretation,
- semantic embedding,
- soft reranking,
- response synthesis,
- decision-making over uncertain candidates.

## 6. Reuse Strategy for Existing SAM Work

The existing SAM repository should not be discarded. Most of it can be reused as research infrastructure for NEXUS.

### 6.1 Product-key memory

Existing role:

```text
query vector -> product-key memory -> top-K slots -> value aggregation
```

Proposed NEXUS role:

```text
query / entity / relation -> associative candidate index -> memory nodes
```

`ProductKeyMemory` can remain an associative candidate retrieval mechanism, but it should not be expected to solve full reasoning by itself.

### 6.2 Synthetic facts dataset

The synthetic dataset is still valuable as a controlled reasoning benchmark.

It provides:

- one-hop questions,
- two-hop questions,
- three-hop questions,
- required slot labels,
- explicit answer tokens,
- known reasoning paths,
- controlled distractor conditions.

In NEXUS this dataset can be reused to validate graph traversal, candidate chain construction, chain ranking, and answer synthesis.

### 6.3 Oracle memory

Oracle memory should become the upper-bound test for the answer module.

If the reasoning engine supplies the correct chain, the answer module should reach high accuracy. Any failure under oracle chain input indicates a downstream synthesis or representation problem.

### 6.4 Chain-set retriever

The chain-set retriever should become a baseline candidate generator.

NEXUS should compare graph-based candidate expansion against chain-set retrieval on:

- anchor recall,
- complete-chain recall,
- candidate precision,
- realistic distractor rate,
- memory reads per query,
- CPU latency.

### 6.5 Slot selector

The current slot selector should not be discarded, but its target should be reconsidered.

Instead of classifying independent slots:

```text
slot -> required / not required
```

NEXUS should test chain-level ranking:

```text
question + candidate chain -> chain score
```

A chain-level scorer can use features that independent slot scoring loses:

- relation order,
- graph distance,
- start anchor match,
- terminal answer compatibility,
- consistency between adjacent facts,
- retrieval scores at each hop.

## 7. Proposed Research Architecture

The following architecture is proposed for research evaluation only.

```text
NEXUS
├── Language Interpreter
│   ├── parse query intent
│   ├── extract anchor candidates
│   └── identify expected relation pattern
│
├── Memory Engine
│   ├── lexical index
│   ├── semantic / PKM index
│   ├── graph index
│   └── cache index
│
├── Reasoning Engine
│   ├── anchor resolution
│   ├── graph expansion
│   ├── candidate-chain generation
│   ├── chain validation
│   └── contradiction / consistency checks
│
├── Working Memory
│   ├── selected facts
│   ├── candidate chains
│   ├── rejected distractors
│   └── reasoning trace
│
├── Neural Controller
│   ├── semantic reranking
│   ├── uncertainty handling
│   └── next-read decisions
│
└── Response Synthesizer
    ├── answer extraction
    └── final response generation
```

## 8. Proposed Memory Representation

The current value representation is intentionally minimal. For NEXUS, memory records should become structured nodes or edges rather than only latent slot values.

Proposed memory record:

```json
{
  "slot_id": 123,
  "subject": "createOrder",
  "relation": "returns",
  "object": "OrderId",
  "text": "Function createOrder returns OrderId .",
  "embedding": "optional_vector_reference",
  "source": "synthetic_dense",
  "confidence": 1.0,
  "version": 1
}
```

Proposed chain record:

```json
{
  "chain_id": 77,
  "slots": [12, 53, 91],
  "path": [
    ["apiOrder", "returns", "OrderId"],
    ["OrderId", "converted_to", "PaymentRef"],
    ["PaymentRef", "wrapped_as", "ContractKey"]
  ],
  "score": 0.94,
  "evidence": ["lexical_anchor", "graph_expansion", "semantic_rerank"]
}
```

This preserves information that flat vector aggregation loses.

## 9. Proposed Experiment 1.0

### Experiment 1.0 — NEXUS Graph Memory Reader

Objective:

Test whether structured CPU-side graph traversal can outperform flat retrieved-slot aggregation on the existing synthetic multi-hop task.

### 1.0A — Oracle anchor + graph traversal

Use the known addressable anchor from the dataset. Traverse the memory graph up to the required hop count.

Purpose:

- validate graph construction,
- validate traversal correctness,
- establish a graph-reader upper bound.

Success criteria:

- overall accuracy above core-only baseline,
- high complete-chain recovery,
- high three-hop recovery.

### 1.0B — Retrieved anchor + graph traversal

Use the existing retriever to find anchor candidates, then expand through graph edges.

Purpose:

- measure whether retrieval only needs to find the entry point,
- reduce the burden of retrieving every required slot directly.

### 1.0C — Retrieved anchor + realistic distractors

Add realistic distractors from retriever candidates and compare against random distractors.

Purpose:

- separate distractor quantity from distractor quality,
- quantify hard-negative damage.

### 1.0D — Chain reranker

Score full candidate chains instead of individual slots.

Purpose:

- test whether chain-level selection improves precision,
- preserve relation order and path consistency.

### 1.0E — Answer synthesizer integration

Pass selected chains into the existing answer path or a small answer module.

Purpose:

- determine whether selected structured evidence can drive final answer generation.

## 10. Suggested Metrics

Existing metrics should be retained:

- overall accuracy,
- accuracy by hop,
- Recall@K,
- all_required@K,
- any_required@K,
- validation loss.

New NEXUS metrics should be added:

- anchor_recall@K,
- complete_chain_recall@K,
- chain_precision,
- chain_path_accuracy,
- distractor_rejection_rate,
- hard_negative_failure_rate,
- average_memory_reads_per_query,
- graph_expansion_width,
- CPU_latency_ms,
- RAM_read_MB_per_query,
- cache_hit_rate,
- reasoning_trace_accuracy.

These metrics are required because NEXUS shifts the unit of evaluation from isolated slot retrieval to structured memory execution.

## 11. Trade-offs

### Advantages

- Better alignment with CPU/RAM strengths.
- Reduced dependence on large dense weights.
- Explicit reasoning traces.
- Easier debugging than latent-only retrieval.
- Better support for mutable knowledge.
- Natural fit for codebases, APIs, logs, documentation, and enterprise knowledge graphs.

### Risks

- Graph construction may be brittle on unstructured text.
- Rule-heavy execution may reduce flexibility.
- Candidate expansion can grow combinatorially.
- Neural reranking may still require careful training.
- The architecture may perform well on structured domains but poorly on open-ended language tasks.
- CPU efficiency claims remain unproven until measured.

### Key design tension

NEXUS should avoid recreating a full LLM in smaller form. The system should use neural models where they add value, but not use dense weights as the primary knowledge store.

## 12. Implementation Considerations

The next implementation should be strictly experimental.

Recommended module additions:

```text
sam-lm/sam/memory/graph_memory.py
sam-lm/sam/reasoning/graph_traversal.py
sam-lm/sam/reasoning/chain_reranker.py
sam-lm/sam/eval/graph_metrics.py
sam-lm/configs/nexus_graph_reader_*.yaml
sam-lm/experiments/experiment_1_0_nexus_graph_reader_report.md
```

Recommended constraints:

- do not modify the existing SAM baseline until graph reader behavior is validated,
- use oracle anchors first,
- keep all experiments reproducible through configs,
- report exact dataset, seed, model checkpoint, and retrieval checkpoint,
- compare against core_only, oracle_memory, chain-set retrieved memory, and selector-based memory.

## 13. Open Questions

1. Can graph traversal recover complete chains more efficiently than direct chain-set retrieval?
2. Is finding only the anchor easier and more robust than retrieving every required slot?
3. Do realistic distractors still damage the model when chain structure is preserved?
4. Is chain-level scoring easier than slot-level scoring?
5. Does slot-wise or chain-wise memory reading improve over flat vector aggregation?
6. Can the architecture preserve CPU-first efficiency after adding graph traversal and reranking?
7. How much of language understanding should live in the neural interpreter versus deterministic memory execution?
8. Can real code/API/log data be converted into useful memory graph records?
9. Does NEXUS remain useful outside highly structured domains?
10. What is the smallest neural component sufficient to interpret queries and synthesize answers?

## 14. Current Conclusion

The existing SAM work should be treated as a successful validation of one important sub-thesis: **external memory can be useful when the system supplies the correct facts**.

However, the current architecture should not be scaled directly as a small transformer with flat latent memory aggregation. The strongest research direction is to evolve the project toward **NEXUS**, a CPU-first, non-parametric reasoning system where:

- memory is structured,
- retrieval generates candidates,
- graph traversal constructs chains,
- reranking selects reasoning paths,
- working memory stores evidence,
- neural modules interpret and synthesize rather than store all knowledge.

The immediate next research step is Experiment 1.0: a graph memory reader evaluated on the existing synthetic multi-hop dataset before any production implementation is attempted.
